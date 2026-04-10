# KG-RAG Enhanced Movie Recommendation

A two-stage movie recommendation system on MovieLens 1M that combines KG-enhanced ranking with Phase 2 explanation generation and faithfulness evaluation. The project now includes both the recommendation stack (Phase 1) and the completed explanation ablation suite (Phase 2): `prompt-only`, `retrieval-only RAG`, `KG-only`, and `hybrid KG+RAG`.

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

### Phase 2: Explanation Faithfulness

| Setting | Count | Overlap | ROUGE-L | Sem.Sim | BERTScore |
|--|--:|--:|--:|--:|--:|
| Retrieval-only RAG | 59,500 | 0.1452 | 0.1674 | 0.8871 | 0.8391 |
| Hybrid KG+RAG | 59,500 | **0.2024** | **0.2345** | 0.9088 | 0.8443 |
| KG-only | 59,481 | 0.2007 | 0.2112 | 0.9452 | **0.8835** |
| Prompt-only (KG-only companion) | 59,481 | 0.0441 | 0.1063 | 0.9695 | 0.8279 |

- `Hybrid KG+RAG` achieves the strongest overall grounding on overlap- and ROUGE-based metrics.
- `KG-only` also strongly outperforms `prompt-only`, showing that structured KG paths alone provide useful explanation evidence.
- Compared with `retrieval-only RAG`, `hybrid KG+RAG` improves overlap from `0.1452` to `0.2024` and ROUGE-L from `0.1674` to `0.2345`, indicating complementary value from structured KG evidence.
- Perturbation experiments show clear degradation under irrelevant evidence replacement (`E4`), supporting that the model is responding to evidence quality rather than generating generic recommendation language.

See [`results/RESULTS.md`](results/RESULTS.md) for full Phase 1 results and [`results/PHASE5_EXPERIMENTS_INDEX.md`](results/PHASE5_EXPERIMENTS_INDEX.md) for Phase 2 experiment packaging and source manifests.

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

### Phase 2 explanation pipeline

Hybrid KG+RAG scripts:

```bash
bash scripts/phase5_with_recommendations_v4\&KG_Path/phase52.sh
bash scripts/phase5_with_recommendations_v4\&KG_Path/phase53.sh
bash scripts/phase5_with_recommendations_v4\&KG_Path/phase54.sh
```

KG-only scripts:

```bash
bash scripts/phase5_with_recommendations_KG_Only/phase52.sh
bash scripts/phase5_with_recommendations_KG_Only/phase53.sh
bash scripts/phase5_with_recommendations_KG_Only/phase54.sh
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
|- scripts/                             # Reusable Phase 5 run scripts
|- tests/test_phase5_modes.py           # KG-only / phase-mode regression tests
|- results/
|  |- RESULTS.md
|  |- PHASE5_EXPERIMENTS_INDEX.md
|  |- phase5_with_recommendation_Hybrid/
|  |- phase5_with_recommendation_Retrieval_Only/
|  |- phase5_with_recommendation_Prompt_Only/
|  |- phase5_with_recommendation_KG_Only/
|  `- results_from_kg/
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
| **RQ4** | Are structured and unstructured knowledge complementary? | Done | Yes, hybrid KG+RAG performs best overall |
