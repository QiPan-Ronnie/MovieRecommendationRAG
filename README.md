# KG-RAG Enhanced Movie Recommendation

A two-stage (recall + re-rank) movie recommendation system on MovieLens 1M, enhanced by Knowledge Graph (KG) features and RAG-based explanations. Phase 1 (KG-enhanced recommendation) is complete; Phase 2 (RAG-based explanation) is planned.

## What This Project Does

1. **Recall stage**: Item-CF generates top-70 candidates, KG (RotatE) embeddings add top-50 more, merged to 100 candidates per user
2. **Re-ranking stage**: LightGBM re-ranks candidates using CF scores, content similarity, and KG features (hand-crafted + RotatE embeddings)
3. **Evaluation**: Ablation study comparing feature sets (V1→V4), statistical significance tests, long-tail analysis

## Key Results (Phase 1)

| | NDCG@10 | Recall@10 | Hit@10 |
|--|---------|-----------|--------|
| Recall-only (no re-ranking) | 0.1451 | 0.1817 | 0.5141 |
| V2: CF + Content features | 0.1365 | 0.1879 | 0.5253 |
| V3: + KG hand-crafted features | 0.1415 | 0.1948 | 0.5215 |
| **V4: + KG RotatE embeddings** | **0.1429** | **0.1986** | **0.5292** |

- KG features significantly improve re-ranking: V3 vs V2 p=0.026 (Pointwise), p=0.045 (LambdaMART)
- KG helps long-tail items **3.5x more** than head items (tail Recall@10: 0.033 → 0.167)
- KG features account for **28.4%** of total LightGBM feature importance

See [`results/RESULTS.md`](results/RESULTS.md) for full results, or [`results/RESULTS_ZH.md`](results/RESULTS_ZH.md) for Chinese version.

## Setup

**Requirements**: Python >= 3.10, ~8GB RAM. GPU optional (only accelerates Phase 1 baseline training).

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

### Full pipeline (from scratch, ~2 hours)

```bash
python run_all.py
```

### Run individual phases

```bash
python run_all.py --phase 0    # Data prep: parse ML-1M, fetch TMDB metadata
python run_all.py --phase 1    # Recall baselines: Item-CF, BPR-MF, LightGCN
python run_all.py --phase 2    # KG: build graph, train RotatE, multi-recall, features
python run_all.py --phase 3    # Ranker: LightGBM ablation (V1-V4, Pointwise + LambdaMART)
python run_all.py --phase 4    # Analysis: head/tail stratified evaluation
```

### Skip expensive steps (use pre-computed data)

If you have `data/` and `results/` directories pre-populated (e.g., from a shared archive), you can skip directly to the ranker and analysis:

```bash
python run_all.py --phase 3    # ~10 min
python run_all.py --phase 4    # ~5 min
```

### Interactive demo

```bash
streamlit run app.py
```

## Pipeline Architecture

```
[Phase 0] Data Preparation
    MovieLens 1M → filter rating >= 4 → per-user time-based split (70/10/20)
    TMDB API → movie metadata (genres, actors, directors, year)

[Phase 1] Recall Baselines
    Item-CF / BPR-MF / LightGCN → top-100 candidates per user
    Evaluated on full catalog (~3,125 movies)

[Phase 2] KG + Multi-Route Recall + Feature Engineering
    Build KG: 134K triples (co_liked, has_genre, acted_by, directed_by, released_in_decade)
    Train RotatE: 128-dim embeddings, balanced relation sampling, 300 epochs
    Multi-recall: Item-CF top-70 + KG top-50 → 100 candidates/user
    Features: content similarity (Sentence-Transformer), KG hand-crafted (IDF-weighted),
              KG embeddings (RotatE distance/cosine to user history)

[Phase 3] Ranker Ablation
    LightGBM re-ranking with distribution-matched training
    V1 (CF) → V2 (+Content) → V3 (+KG) → V3e (+KGEmb) → V4 (+All)
    Both Pointwise and LambdaMART objectives, paired t-tests for significance

[Phase 4] Long-tail Analysis
    Head/tail stratified Recall@10, user genre entropy analysis

[Phase 5] RAG-Based Explanation (Planned)
    Retrieve KG evidence + unstructured reviews → LLM-generated explanations
    Evaluate explanation faithfulness vs prompt-only baselines
```

## Project Structure

```
MovieRecommendation/
├── data_prep/
│   ├── parse_ml1m.py            # ML-1M parsing, filtering, 3-way split
│   └── fetch_tmdb.py            # TMDB metadata fetch (with checkpoint/resume)
├── models/
│   ├── item_cf.py               # Item-based Collaborative Filtering
│   ├── matrix_factorization.py  # BPR Matrix Factorization (PyTorch)
│   ├── lightgcn.py              # LightGCN (PyTorch)
│   └── multi_recall.py          # Multi-route recall: CF + KG candidates
├── kg/
│   ├── build_kg.py              # KG construction (metadata + collaborative edges)
│   ├── rotate.py                # RotatE embedding training (default)
│   ├── transe.py                # TransE embedding training (baseline comparison)
│   ├── kg_features.py           # Hand-crafted KG features (IDF-weighted)
│   ├── kg_embedding_features.py # KG embedding-based features
│   └── content_similarity.py    # Sentence-Transformer content similarity
├── ranker/
│   └── ranker.py                # LightGBM ranker (Pointwise + LambdaMART)
├── evaluation/
│   ├── metrics.py               # Hit@K, NDCG@K, Recall@K, MRR, Coverage
│   └── longtail_analysis.py     # Head/tail stratified evaluation
├── experiments/                 # Experiment scripts (KG recall ablations)
├── docs/
│   ├── PROJECT_OVERVIEW.md      # Detailed research design
│   ├── PROJECT_OVERVIEW_ZH.md   # Chinese version
│   └── EXPERIMENT_LOG.md        # Experiment log with negative results
├── results/
│   ├── RESULTS.md               # Full experiment results and analysis
│   └── RESULTS_ZH.md            # Chinese version
├── run_all.py                   # End-to-end pipeline (Phase 0-4)
├── run_baselines.py             # Baseline model runner
├── app.py                       # Streamlit interactive demo
└── requirements.txt
```

## Research Questions

| RQ | Question | Status | Answer |
|----|----------|--------|--------|
| **RQ1** | Does KG-enhanced re-ranking outperform CF/content baselines? | Done | Yes (p < 0.05) |
| **RQ2** | Does KG disproportionately help long-tail items? | Done | Yes (3.5x tail lift vs head) |
| **RQ3** | Are RAG explanations more faithful than prompt-only? | Planned | — |
| **RQ4** | Are structured + unstructured knowledge complementary? | Planned | — |
