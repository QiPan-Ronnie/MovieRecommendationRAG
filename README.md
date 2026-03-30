# KG-RAG Enhanced Movie Recommendation

> Investigating whether Knowledge Graphs and Retrieval-Augmented Generation improve recommendation accuracy and explanation faithfulness.

## Experiment Goals

This project builds a two-stage (recall + re-rank) movie recommendation system on MovieLens 1M, enhanced by Knowledge Graph (KG) features and RAG-based explanations. We address four research questions in two phases:

**Phase 1 — KG-Enhanced Recommendation (Complete)**
- **RQ1**: Does KG-enhanced re-ranking significantly outperform CF and content baselines?
- **RQ2**: Does KG disproportionately benefit long-tail items?

**Phase 2 — RAG-Based Explanation (Planned)**
- **RQ3**: Are RAG-generated explanations more faithful than prompt-only LLM explanations?
- **RQ4**: Are structured (KG) and unstructured (RAG) knowledge complementary for explainability?

## Key Results (Phase 1)

| Finding | Result |
|---------|--------|
| **RQ1**: KG vs CF baseline | V3 vs V2: p=0.026 (Pointwise), p=0.045 (LambdaMART) |
| **RQ1**: Best Recall@10 | V4 LambdaMART: **0.1976** (vs Recall-only 0.1817, +8.8%) |
| **RQ2**: Tail lift vs head lift | **3.5x** higher for tail items (V3 vs V1) |
| **RQ2**: Low-entropy user lift | **+52%** Recall@10 (vs +6% for high-entropy users) |
| KG feature importance share | **28.4%** of total LightGBM gain |

See [`results/RESULTS.md`](results/RESULTS.md) for full experimental results.

## Quick Start

### Option A: Use Pre-computed Data (Recommended)

Download [`data_release.tar.gz`](https://github.com/XqFeng-Josie/MovieRecommendation/releases) and extract:

```bash
# 1. Environment
python -m venv movie_env && source movie_env/bin/activate
pip install -r requirements.txt

# 2. Data: download ML-1M and extract pre-computed data
# Download ML-1M from https://grouplens.org/datasets/movielens/1m/ to data/raw/ml-1m/
tar xzf data_release.tar.gz

# 3. Run ranker ablation + long-tail analysis directly (skip Phase 0-2)
python run_all.py --phase 3    # Ranker ablation (~10 min)
python run_all.py --phase 4    # Long-tail analysis (~5 min)
```

### Option B: Run Full Pipeline from Scratch

```bash
# 1. Environment
python -m venv movie_env && source movie_env/bin/activate
pip install -r requirements.txt

# 2. Data: download ML-1M to data/raw/ml-1m/, set TMDB key
export TMDB_API_KEY=your_key_here

# 3. Run full pipeline
python run_all.py

# Or run individual phases
python run_all.py --phase 0    # Data prep
python run_all.py --phase 1    # Recall baselines (requires GPU)
python run_all.py --phase 2    # KG + multi-recall + features
python run_all.py --phase 3    # Ranker ablation
python run_all.py --phase 4    # Long-tail analysis
```

### Data Contents

The `data_release.tar.gz` (123MB) contains all intermediate outputs so you can skip expensive computation:

| Directory | Contents | Size |
|-----------|----------|------|
| `data/tmdb/` | TMDB metadata (3,652 movies) | 2MB |
| `data/processed/` | Train/val/test splits, recall candidates, content similarity, movie embeddings | 76MB |
| `data/kg/` | KG graph, triples, TransE embeddings, KG features | 184MB |
| `results/` | Recall model scores, ablation results, feature importance | 69MB |

## Pipeline Architecture

```
[Phase 0] Data Preparation
    Parse ML-1M, filter rating >= 4, time-based split (70/10/20)

[Phase 1] Recall Baselines
    Item-CF / BPR-MF / LightGCN -> top-100 candidates per user

[Phase 2] KG + Multi-Route Recall + Features
    Build KG (134K triples: co_liked, genre, actor, director, decade)
    Train TransE embeddings (128-dim, 5 relations)
    Multi-route recall: CF top-70 + KG top-50 -> 100 candidates
    Features: content similarity, KG (raw + IDF), KG embeddings

[Phase 3] Ranker Ablation
    V1 (CF) -> V2 (+Content) -> V3 (+KG) -> V3e (+KGEmb) -> V4 (+All)
    Pointwise + LambdaMART, statistical significance tests

[Phase 4] Long-tail Analysis (RQ2)
    Head/tail stratified Recall@10, user genre entropy analysis
```

## Project Structure

```
MovieRecommendation/
├── src/
│   ├── data_prep/
│   │   ├── parse_ml1m.py            # ML-1M parsing, filtering, 3-way split
│   │   └── fetch_tmdb.py            # TMDB metadata fetch (checkpoint/resume)
│   ├── models/
│   │   ├── item_cf.py               # Item-based Collaborative Filtering
│   │   ├── matrix_factorization.py  # BPR Matrix Factorization (PyTorch)
│   │   ├── lightgcn.py              # LightGCN (PyTorch)
│   │   └── multi_recall.py          # Multi-route recall: CF + KG candidates
│   ├── kg/
│   │   ├── build_kg.py              # KG construction (metadata + collaborative edges)
│   │   ├── transe.py                # TransE embedding training (PyTorch)
│   │   ├── kg_features.py           # Hand-crafted KG features (IDF-weighted)
│   │   ├── kg_embedding_features.py # TransE embedding-based features
│   │   └── content_similarity.py    # Sentence-Transformer content similarity
│   ├── ranker/
│   │   └── ranker.py                # LightGBM ranker (Pointwise + LambdaMART)
│   ├── evaluation/
│   │   ├── metrics.py               # Hit@K, NDCG@K, Recall@K, MRR, Coverage
│   │   └── longtail_analysis.py     # RQ2: head/tail stratified evaluation
│   └── run_baselines.py             # Baseline model orchestrator
├── results/
│   └── RESULTS.md                   # Experiment results and analysis
├── app.py                           # Streamlit interactive demo
├── run_all.py                       # End-to-end pipeline (Phase 0-4)
├── PROJECT_OVERVIEW.md              # Research design document
└── requirements.txt
```