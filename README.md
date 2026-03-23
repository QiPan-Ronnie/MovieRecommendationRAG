# KG-Enhanced Movie Recommendation

> Do Knowledge Graphs and Retrieval-Augmented Generation Improve Recommendation Performance and Explanation Faithfulness?

This project investigates whether combining Knowledge Graph (KG)-enhanced re-ranking with collaborative filtering baselines yields measurable improvements in recommendation accuracy, particularly for long-tail items.

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Does KG-enhanced re-ranking significantly outperform CF and content-based baselines? |
| RQ2 | Does KG provide disproportionate benefit for long-tail items? |
| RQ3 | Are RAG-generated explanations more faithful than prompt-only explanations? (Phase 2) |
| RQ4 | Are structured (KG) and unstructured (RAG) knowledge complementary? (Phase 2) |

## Architecture

Two-stage **recall + re-rank** pipeline:

```
[Stage 1: Recall]   Item-CF / BPR-MF / LightGCN  -->  top-100 candidates
[Stage 2: Re-rank]  LightGBM ranker (CF + Content + KG features)  -->  top-10
[Stage 3: Explain]  RAG module (Phase 2)
```

### Ablation Variants

| Variant | Features |
|---------|----------|
| V1 | CF score only |
| V2 | CF score + content similarity + popularity |
| V3 | CF score + content similarity + popularity + KG features |

Both **Pointwise** (binary classification) and **LambdaMART** (listwise NDCG optimization) ranking objectives are compared.

## Dataset

- **MovieLens 1M**: ~6,040 users, ~3,900 movies, ~1M ratings
- **TMDB API**: genres, actors, directors, overviews, vote counts
- **Knowledge Graph**: constructed from TMDB metadata (has_genre, acted_by, directed_by)

## Setup

### 1. Create Virtual Environment

```bash
python -m venv movie_env
source movie_env/bin/activate      # Linux / macOS
# movie_env\Scripts\activate       # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Preparation

1. Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and extract to `data/raw/ml-1m/`
2. Set your TMDB API key:
   ```bash
   export TMDB_API_KEY=your_key_here
   ```
3. Run the full pipeline:
   ```bash
   python run_all.py
   ```

   Or run individual phases:
   ```bash
   python run_all.py --phase 0              # Data preparation
   python run_all.py --phase 1              # Baseline models
   python run_all.py --phase 2              # KG construction + features
   python run_all.py --phase 3              # Ranker ablation
   python run_all.py --skip-tmdb            # Skip TMDB fetch (use cached)
   ```

## Project Structure

```
MovieRecommendation/
├── src/
│   ├── data_prep/
│   │   ├── parse_ml1m.py          # ML-1M parsing, filtering, splitting
│   │   └── fetch_tmdb.py          # TMDB metadata fetching with checkpoint/resume
│   ├── models/
│   │   ├── item_cf.py             # Item-based Collaborative Filtering
│   │   ├── matrix_factorization.py # BPR Matrix Factorization (PyTorch)
│   │   └── lightgcn.py            # LightGCN (PyTorch, custom implementation)
│   ├── kg/
│   │   ├── build_kg.py            # KG triple construction from TMDB
│   │   ├── kg_features.py         # KG feature engineering per (user, item) pair
│   │   └── content_similarity.py  # Sentence-Transformer content similarity
│   ├── ranker/
│   │   └── ranker.py              # LightGBM ranker (Pointwise + LambdaMART)
│   ├── evaluation/
│   │   └── metrics.py             # Hit@K, NDCG@K, Recall@K, MRR, Coverage
│   └── run_baselines.py           # Run all baseline models
├── app.py                         # Streamlit demo interface
├── run_all.py                     # End-to-end pipeline orchestrator
├── experiment_plan.md             # Detailed experiment design
├── PROJECT_OVERVIEW.md            # Full project description
├── REFLECTION.md                  # Consistency and rigor audit
└── requirements.txt
```

## Evaluation

- **Metrics**: Hit@K, NDCG@K, Recall@K, MRR, Coverage (K = 1, 5, 10)
- **Statistical significance**: Paired t-test on per-user NDCG@10 (p < 0.05)
- **Data split**: Per-user time-based 70/10/20 (train/val/test)
- **Positive threshold**: Only ratings >= 4 count as positive interactions

## Interactive Demo

```bash
streamlit run app.py
```

Features: KG subgraph explorer, recommendation comparison across models, KG-based explanations, experiment dashboard.

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Ranker | LightGBM |
| Knowledge Graph | NetworkX |
| Text Embeddings | Sentence-Transformers |
| Evaluation | scikit-learn, SciPy |
| Demo | Streamlit |
