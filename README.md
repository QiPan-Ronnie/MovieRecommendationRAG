# KG-RAG Enhanced Movie Recommendation

A two-stage movie recommendation system on MovieLens 1M that combines **KG-enhanced ranking** with **Phase 2 explanation generation and faithfulness evaluation**. The repository now packages the recommendation stack (Phase 1), the four-way explanation comparison (`prompt-only`, `retrieval-only`, `KG-only`, `hybrid KG+RAG`), unified BERTScore reevaluations, significance tests, and larger perturbation follow-ups.

## What This Project Does

1. **Recall stage**: Item-CF generates top-70 candidates, KG (TransE) embeddings add top-50 more, merged to 100 candidates per user.
2. **Re-ranking stage**: LightGBM re-ranks candidates using CF scores, content similarity, and KG features (hand-crafted + RotatE embeddings).
3. **Explanation stage**: Generate recommendation explanations under four controlled settings: `prompt-only`, `retrieval-only RAG`, `KG-only`, and `hybrid KG+RAG`.
4. **Faithfulness evaluation**: Compare explanations using evidence overlap, ROUGE-L, semantic similarity, BERTScore, and perturbation-based stress tests (E1-E4).

## Key Results

### Phase 1: Recommendation (LambdaMART)

| Setting | NDCG@10 | Recall@10 | Hit@10 |
|--|--:|--:|--:|
| Recall-only (no re-ranking) | 0.1451 | 0.1817 | 0.5141 |
| V2: CF + Content features | 0.1365 | 0.1879 | 0.5253 |
| V3: + KG hand-crafted features | 0.1415 | 0.1948 | 0.5215 |
| **V4: + KG RotatE embeddings** | **0.1429** | **0.1986** | **0.5292** |

- KG features significantly improve re-ranking: V3 vs V2 p=0.026 (Pointwise), p=0.045 (LambdaMART).
- KG helps long-tail items **3.5x more** than head items (tail Recall@10: 0.033 -> 0.167).
- KG features account for **28.4%** of total LightGBM feature importance.

### Phase 2: Final Main Comparison (Unified BERTScore)

| Setting | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
|--|--:|--:|--:|--:|--:|
| Retrieval-only RAG | 59500 | 0.1452 | 0.1674 | 0.8871 | 0.8391 |
| Prompt-only (Retrieval companion) | 59500 | 0.0861 | 0.1368 | 0.9244 | 0.8196 |
| Hybrid KG+RAG | 59500 | 0.2024 | 0.2345 | 0.9088 | 0.8443 |
| Prompt-only (Hybrid companion) | 59500 | 0.0968 | 0.1705 | 0.9462 | 0.8135 |
| KG-only | 59481 | 0.2007 | 0.2112 | 0.9452 | 0.8835 |
| Prompt-only (KG companion) | 59481 | 0.0441 | 0.1063 | 0.9695 | 0.8279 |

- **Hybrid KG+RAG** is strongest on the most direct grounding indicators (`Overlap`, `ROUGE-L`), supporting the claim that structured KG evidence and textual retrieval are complementary. Relative to `Retrieval-only`, Hybrid improves overlap by **+0.0572** and ROUGE-L by **+0.0671**, while keeping BERTScore competitive.
- **KG-only** is also very strong, and leads on `Sem.Sim` / `BERTScore`, showing that KG paths alone provide highly informative and semantically coherent explanation evidence. It also clearly outperforms `Retrieval-only` on overlap- and ROUGE-based grounding metrics.
- Paired significance testing under `results/phase5_stats/` confirms the main Hybrid-vs-Retrieval and KG-vs-Retrieval improvements are statistically significant, which makes the overall comparison more reliable than reporting raw averages alone.

### Phase 2: Perturbation Follow-up (`p500`)

| Setting | E1 Overlap | E4 Overlap | E1 ROUGE-L | E4 ROUGE-L | E1 BERTScore | E4 BERTScore |
|--|--:|--:|--:|--:|--:|--:|
| Hybrid KG+RAG | 0.1962 | 0.0361 | 0.2311 | 0.1432 | 0.8434 | 0.7959 |
| Retrieval-only RAG | 0.1354 | 0.0274 | 0.1780 | 0.1191 | 0.8359 | 0.8008 |
| KG-only | 0.1984 | 0.0642 | 0.2125 | 0.1032 | 0.8833 | 0.8357 |

Across all three evidence-grounded settings, replacing relevant evidence with irrelevant evidence (`E4`) causes sharp degradation, reinforcing that the model is reacting to evidence quality rather than only generating generic recommendation language. The enlarged `p500` follow-up also makes the perturbation story more stable: evidence relevance matters much more than evidence order, while Hybrid and KG-only both remain clearly sensitive to irrelevant substitutions.

Taken together, the Phase 2 results support a consistent picture: `retrieval-only RAG`, `KG-only`, and `hybrid KG+RAG` all outperform their prompt-only counterparts, while `hybrid KG+RAG` achieves the strongest grounding on overlap- and ROUGE-based metrics. At the same time, `KG-only` remains highly competitive and achieves the best semantic-similarity and BERTScore values, indicating that structured KG evidence is already very informative on its own. Overall, the findings suggest that structured KG paths and textual retrieval provide complementary signals for faithful explanation generation, with Hybrid offering the best overall grounding trade-off and KG-only providing a strong structured baseline.

See [`results/RESULTS.md`](results/RESULTS.md) for Phase 1 details, [`results/PHASE5_EXPERIMENTS_INDEX.md`](results/PHASE5_EXPERIMENTS_INDEX.md) for the organized Phase 5 result layout, and [`results/phase5_stats/significance_summary.md`](results/phase5_stats/significance_summary.md) for the latest statistical comparison report.

## Setup

**Requirements**: Python >= 3.10, ~8GB RAM for Phase 1 analysis; GPU strongly recommended for explanation generation and evaluation.

```bash
# Create environment
python -m venv movie_env && source movie_env/bin/activate
pip install -r requirements.txt

# Download MovieLens 1M
# Get ml-1m.zip from https://grouplens.org/datasets/movielens/1m/
# Unzip to data/raw/ml-1m/ (should contain ratings.dat, movies.dat, users.dat)

# Set TMDB API key (for fetching movie metadata)
export TMDB_API_KEY=your_key_here
```

## Running Experiments

### Phase 1 recommendation pipeline

```bash
python run_all.py
```

### Run individual Phase 1 stages

```bash
python run_all.py --phase 0    # Data prep: parse ML-1M, fetch TMDB metadata
python run_all.py --phase 1    # Recall baselines: Item-CF, BPR-MF, LightGCN
python run_all.py --phase 2    # KG: build graph, TransE recall, RotatE features
python run_all.py --phase 3    # Ranker: LightGBM ablation (V1-V4, Pointwise + LambdaMART)
python run_all.py --phase 4    # Analysis: head/tail stratified evaluation
```

### Phase 2 canonical experiment scripts

Hybrid KG+RAG full run:

```bash
bash scripts/phase5_with_recommendations_v4\&KG_Path/phase52.sh
bash scripts/phase5_with_recommendations_v4\&KG_Path/phase53.sh
bash scripts/phase5_with_recommendations_v4\&KG_Path/phase54.sh
```

KG-only full run:

```bash
bash scripts/phase5_with_recommendations_KG_Only/phase52.sh
bash scripts/phase5_with_recommendations_KG_Only/phase53.sh
bash scripts/phase5_with_recommendations_KG_Only/phase54.sh
```

### Phase 2 follow-up evaluation scripts

These scripts produced the final packaged results under `results/phase5_with_recommendation_*_final/`:

```bash
# Unified BERTScore reevaluation (Phase 5.4 only)
bash scripts/phase5_with_recommendation_Hybrid_bertscore_unified/phase54.sh
bash scripts/phase5_with_recommendation_Retrieval_Only_bertscore_unified/phase54.sh
bash scripts/phase5_with_recommendation_KG_Only_bertscore_unified/phase54.sh

# Larger perturbation follow-up (p500)
bash scripts/phase5_with_recommendation_Hybrid_p500/phase53.sh
bash scripts/phase5_with_recommendation_Hybrid_p500/phase54.sh

# Statistical comparison
/root/miniconda3/bin/python analysis/phase5_significance.py
```

## Pipeline Architecture

```
[Phase 0] Data Preparation
    MovieLens 1M -> filter rating >= 4 -> per-user time-based split (70/10/20)
    TMDB API -> movie metadata (genres, actors, directors, year)

[Phase 1] Recall Baselines
    Item-CF / BPR-MF / LightGCN -> top-100 candidates per user
    Evaluated on full catalog (~3,125 movies)

[Phase 2] KG + Multi-Route Recall + Feature Engineering
    Build KG: 134K triples (co_liked, has_genre, acted_by, directed_by, released_in_decade)
    Train TransE (200 epochs) -> multi-recall: Item-CF top-70 + KG top-50 -> 100/user
    Train RotatE (300 epochs, balanced sampling) -> embedding features for ranker

[Phase 3] Ranker Ablation
    LightGBM re-ranking with distribution-matched training
    V1 (CF) -> V2 (+Content) -> V3 (+KG) -> V4 (+All)

[Phase 4] Long-tail Analysis
    Head/tail stratified Recall@10, user genre entropy analysis

[Phase 5] RAG-Based Explanation and Faithfulness Evaluation
    5.1 Build sentence-level evidence corpus from TMDB metadata
    5.2 Generate explanations under prompt-only / retrieval-only / KG-only / hybrid settings
    5.3 Run perturbation experiments (E1-E4)
    5.4 Evaluate with overlap, ROUGE-L, semantic similarity, and BERTScore
    5.4b Reevaluate with unified BERTScore configuration
    5.3b-5.4b Run larger perturbation follow-up with 500 samples and significance tests
```

## Project Structure

```
MovieRecommendation/
|- data_prep/
|- models/
|- kg/
|- ranker/
|- evaluation/
|- rag/                                 # Phase 5 retrieval / generation / evaluation code
|- analysis/                            # Significance testing and follow-up analysis
|- scripts/                             # Reusable Phase 5 run scripts
|- results/
|  |- RESULTS.md
|  |- PHASE5_EXPERIMENTS_INDEX.md
|  |- phase5_with_recommendation_Hybrid_final/
|  |- phase5_with_recommendation_Retrieval_Only_final/
|  |- phase5_with_recommendation_KG_Only_final/
|  |- phase5_stats/
|  |- recommendations_v2.csv
|  `- recommendations_v4.csv
|- run_all.py
|- export_phase1_for_rag.py
`- requirements.txt
```

## Research Questions

| RQ | Question | Status | Current Answer |
|----|----------|--------|----------------|
| **RQ1** | Does KG-enhanced re-ranking outperform CF/content baselines? | Done | Yes |
| **RQ2** | Does KG disproportionately help long-tail items? | Done | Yes |
| **RQ3** | Are evidence-conditioned explanations more faithful than prompt-only? | Done | Yes |
| **RQ4** | Are structured and unstructured knowledge complementary? | Done | Yes, Hybrid is strongest on grounding metrics while KG-only is also highly effective on semantic and BERT-based measures |
