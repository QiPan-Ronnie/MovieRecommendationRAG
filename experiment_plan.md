# Experiment Plan: KG-Enhanced Recommendation (Phase 1)

> Covers RQ1 (KG vs. baselines) and RQ2 (long-tail analysis).
> RAG experiments (RQ3, RQ4) will start after Phase 1 is validated.

---

## Revision Notes (2026-03-20)

The first round of experiments revealed several critical issues that this revised plan addresses:

| Issue | Severity | Description | Fix |
|-------|----------|-------------|-----|
| Ranker trained on test data | **Critical** | LightGBM was trained on 70% of test users and evaluated on the other 30% — direct data leakage | Ranker must train on training set candidates, evaluate on test set |
| Inconsistent evaluation protocol | **Critical** | Baselines evaluated on full catalog (~3700 items), ranker on small candidate pool (~164 items/user) — results not comparable | Unify: all models evaluated on the same candidate pool |
| Low ratings treated as positive | **Major** | Ratings 1-2 (~12% of data) were treated as positive interactions, injecting noise | Apply threshold: only rating >= 4 counts as positive |
| Popularity leaked from test set | **Major** | Popularity feature computed from test set distribution | Compute all features from training set only |
| Content similarity not implemented | **Major** | Used `vote_average / 10` instead of Sentence-Transformer cosine similarity | Implement proper content similarity |
| No hyperparameter search | **Minor** | All models used single fixed hyperparameters | Add grid search with validation |

---

## 1. Experimental Goals

| ID | Research Question | Hypothesis | Success Criterion |
|----|-------------------|------------|-------------------|
| RQ1 | Does KG-enhanced re-ranking significantly outperform CF and content baselines? | H1: V3 (CF+Content+KG) > V2 (CF+Content) on NDCG@10 and Recall@10 | Paired t-test p < 0.05 |
| RQ2 | Does KG disproportionately help long-tail items? | H2: Tail Recall@10 improvement (V3 vs V1) > Head Recall@10 improvement | Tail lift ratio > Head lift ratio |

---

## 2. Resources

| Item | Configuration |
|------|--------------|
| Dataset | MovieLens 1M (~6,040 users, ~3,900 movies, ~1M ratings) |
| Metadata | TMDB API (via IMDb ID mapping, checkpoint-resume) |
| GPU | 8 x V100 (16GB/32GB) |
| Key libraries | PyTorch, LightGBM, NetworkX, Sentence-Transformers, SciPy |

---

## 3. Architecture

```
MovieLens 1M + TMDB Metadata
    |
[Phase 0] Data Preparation
    |   - Clean, filter (rating >= 4 as positive)
    |   - Time-based split: train / validation / test
    |   - Negative sampling (separately for each split)
    |
[Phase 1] Recall Models
    |   - Item-CF / BPR-MF / LightGCN
    |   - Train on train set, tune on validation set
    |   - Output: top-N candidates + scores per user
    |
[Phase 2] KG Construction & Feature Engineering
    |   - Build KG triples from TMDB metadata
    |   - Compute KG features for (user, candidate) pairs
    |   - Compute content similarity via Sentence-Transformer
    |   - ALL features computed using training data only
    |
[Phase 3] Re-ranking + Ablation
    |   - LightGBM ranker trained on train candidates
    |   - Hyperparameter tuning on validation candidates
    |   - Final evaluation on test candidates
    |   - Ablation: V1 (CF) / V2 (CF+Content) / V3 (CF+Content+KG)
    |
[Phase 3] Analysis
    - Statistical significance tests
    - Long-tail analysis (RQ2)
    - Feature importance
    - Visualization
```

---

## 4. Detailed Execution Plan

### Phase 0: Data Preparation

#### 0.1 Environment

```bash
python -m venv movie_env
source movie_env/bin/activate
pip install -r requirements.txt
```

Project directory:
```
MovieRecommendation/
├── data/
│   ├── raw/ml-1m/           # ML-1M raw .dat files
│   ├── processed/           # Cleaned and split data
│   ├── tmdb/                # TMDB metadata + cache
│   └── kg/                  # KG triples and features
├── src/
│   ├── data_prep/           # Data processing scripts
│   │   ├── parse_ml1m.py
│   │   └── fetch_tmdb.py
│   ├── models/              # Recall models
│   │   ├── item_cf.py
│   │   ├── matrix_factorization.py
│   │   └── lightgcn.py
│   ├── kg/                  # KG construction and features
│   │   ├── build_kg.py
│   │   └── kg_features.py
│   ├── ranker/              # Re-ranking model
│   │   └── ranker.py
│   ├── evaluation/          # Metrics
│   │   └── metrics.py
│   └── run_baselines.py
├── results/
├── configs/
└── notebooks/
```

#### 0.2 Download and Parse MovieLens 1M

- [ ] Parse `ratings.dat`, `movies.dat`, `users.dat` into CSV
- [ ] Output: `data/processed/movies.csv`, `data/processed/users.csv`

#### 0.3 Fetch TMDB Metadata

- [ ] Map MovieLens movie_id to TMDB via IMDb ID (from `links.dat`)
- [ ] Checkpoint-resume: cache each response to `data/tmdb/cache/{movie_id}.json`
- [ ] Extract: genres, actors (top-5), directors, overview, keywords, vote_average, vote_count, release_year
- [ ] Output: `data/tmdb/tmdb_metadata.csv`
- [ ] Log coverage rate (expect 80-90%)

#### 0.4 Data Cleaning and Filtering

- [ ] Merge MovieLens ratings with TMDB metadata (left join on movie_id)
- [ ] **Positive interaction threshold**: only keep ratings >= 4 as positive interactions
  - Ratings 1-3 are discarded (not used as positive or negative — they are ambiguous)
  - This reduces the implicit feedback to clear "like" signals
- [ ] User minimum interaction filter: keep users with >= 10 positive interactions (after threshold)
- [ ] Item minimum interaction filter: keep items with >= 5 positive interactions
- [ ] Output: `data/processed/clean_ratings.csv`

> **Rationale**: In implicit feedback settings, treating rating=1 as "positive" corrupts the signal. Academic convention for MovieLens is to use rating >= 4 as positive. The threshold of >= 10 interactions per user (down from 20) compensates for the reduced data after filtering.

#### 0.5 Train / Validation / Test Split

**Three-way time-based split** (per user, sorted by timestamp):

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | First 70% of each user's interactions | Train recall models + ranker |
| Validation | Next 10% | Tune hyperparameters for recall models + ranker |
| Test | Last 20% | Final evaluation only |

```python
def split_train_val_test(ratings, train_ratio=0.7, val_ratio=0.1):
    """Per-user time-based split."""
    ratings = ratings.sort_values(["user_id", "timestamp"])
    train, val, test = [], [], []
    for user_id, group in ratings.groupby("user_id"):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = max(1, int(n * val_ratio))
        train.append(group.iloc[:n_train])
        val.append(group.iloc[n_train:n_train + n_val])
        test.append(group.iloc[n_train + n_val:])
    return pd.concat(train), pd.concat(val), pd.concat(test)
```

- [ ] Output: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`

#### 0.6 Negative Sampling

Generate negative samples **separately for each split**:

```python
def generate_negatives(positive_df, all_movie_ids, user_history, neg_ratio=4, seed=42):
    """
    For each positive interaction, sample neg_ratio movies the user has NOT interacted with.
    user_history: movies the user has interacted with UP TO this split (no future leakage).
    """
```

**Key rule**: When sampling negatives for validation, `user_history` = train interactions only. When sampling negatives for test, `user_history` = train + validation interactions.

- [ ] Output: `data/processed/train_with_neg.csv`, `data/processed/val_with_neg.csv`, `data/processed/test_with_neg.csv`
- [ ] All three files share the same format: `(user_id, movie_id, label)` where label = 1 (positive) or 0 (negative)

#### Phase 0 Checkpoint

- [ ] Report: user count, movie count, interaction count after filtering
- [ ] Report: train / val / test sizes and ratios
- [ ] Report: TMDB coverage rate
- [ ] Sanity check: no user appears in test but not in train
- [ ] Sanity check: no timestamp in test <= max timestamp in train (per user)

---

### Phase 1: Baseline Recall Models

#### 1.1 Evaluation Interface

All models use the same evaluation function:

```python
def evaluate_all(predictions, ground_truth, k=10, total_items=None):
    """
    predictions: dict[user_id] -> list[movie_id] (ranked by score, descending)
    ground_truth: dict[user_id] -> set(movie_id) (positive items)
    Returns: {Hit@K, NDCG@K, Recall@K, MRR, Coverage}, per_user_ndcg
    """
```

**Ground truth definition**: For baseline evaluation, ground_truth = positive items in the test set (rating >= 4, after our filtering).

#### 1.2 Item-CF (Baseline A)

- [ ] Build item-item cosine similarity matrix from **train set** interaction vectors
- [ ] For each user, score all candidate items by weighted sum of similarities with train history
- [ ] Exclude train items from candidates
- [ ] Output top-100 candidates per user
- [ ] Save: `results/cf_scores.csv` (user_id, movie_id, cf_score)

#### 1.3 BPR-MF (Baseline B)

- [ ] Implement BPR Matrix Factorization in PyTorch
- [ ] Hyperparameter search (select best config on **validation set**):

| Hyperparameter | Search Range |
|----------------|-------------|
| embedding_dim | {32, 64, 128} |
| learning_rate | {1e-3, 5e-4, 1e-4} |
| regularization | {1e-4, 1e-5} |
| epochs | early stopping on val loss, patience=5 |

- [ ] Train on train set, evaluate each config on validation set, select best
- [ ] Final evaluation on test set with best config
- [ ] Save: `results/mf_scores.csv`

#### 1.4 LightGCN (Baseline C)

- [ ] Implement LightGCN in PyTorch (custom implementation, no PyG dependency)
- [ ] Hyperparameter search on validation set:

| Hyperparameter | Search Range |
|----------------|-------------|
| num_layers | {2, 3} |
| embedding_dim | {64, 128} |
| learning_rate | {1e-3, 5e-4} |

- [ ] Leverage multi-GPU for parallel hyperparameter search
- [ ] Save: `results/lightgcn_scores.csv`

#### 1.5 Baseline Summary

- [ ] Fill results table (all evaluated on test set, full catalog, top-10):

| Model | Hit@10 | NDCG@10 | Recall@10 | MRR | Coverage |
|-------|--------|---------|-----------|-----|----------|
| Item-CF | | | | | |
| BPR-MF (best) | | | | | |
| LightGCN (best) | | | | | |

#### Phase 1 Checkpoint

Expected NDCG@10 ranges for ML-1M (implicit, rating >= 4, full catalog):
- Item-CF: 0.03 - 0.08
- BPR-MF: 0.04 - 0.08
- LightGCN: 0.05 - 0.10

If results are far outside these ranges, debug:
- [ ] Check that ground truth only contains rating >= 4 items
- [ ] Check that train items are excluded from candidate set
- [ ] Check that evaluation is over full catalog, not a small candidate pool
- [ ] Print sample predictions for a few users to visually verify

---

### Phase 2: Knowledge Graph Construction & Feature Engineering

#### 2.1 KG Triple Construction

Extract three relation types from TMDB metadata:

| Relation | Example | Source |
|----------|---------|--------|
| `has_genre` | (Toy Story, has_genre, Animation) | TMDB genres |
| `acted_by` | (Toy Story, acted_by, Tom Hanks) | TMDB cast (top-5) |
| `directed_by` | (Toy Story, directed_by, John Lasseter) | TMDB crew |

- [ ] Build triples: `data/kg/triples.csv` (head, relation, tail)
- [ ] Build entity mapping: `data/kg/entity2id.csv`
- [ ] Build NetworkX graph: `data/kg/kg_graph.pkl`
- [ ] Report: node count, edge count, relation type distribution

#### 2.2 KG Feature Engineering

For each (user, candidate_movie) pair, compute features based on user's **training history** (rating >= 4 movies):

```python
def compute_user_candidate_features(G, user_history_movies, candidate_movie_id):
    """
    user_history_movies: movies the user liked in TRAINING set only
    Returns aggregated KG features.
    """
```

| Feature | Type | Aggregation | Description |
|---------|------|-------------|-------------|
| `kg_shared_actor_count_sum` | float | sum over history | Total shared actors between candidate and all history movies |
| `kg_shared_actor_count_max` | float | max over history | Max shared actors with any single history movie |
| `kg_same_director_max` | binary | max | Whether candidate shares a director with any history movie |
| `kg_same_genre_count_sum` | float | sum | Total shared genres |
| `kg_same_genre_count_max` | float | max | Max shared genres with any single history movie |
| `kg_shortest_path_min` | int | min | Shortest path in KG to nearest history movie (capped at 5) |

**Critical**: User history for KG features must come from **training set only**, never from validation or test.

- [ ] Compute KG features for all (user, candidate) pairs in train_with_neg, val_with_neg, and test_with_neg
- [ ] For all three sets, user history = training set interactions only
- [ ] Limit user history to last 20 movies (by timestamp) for efficiency
- [ ] Output: `data/kg/kg_features_train.csv`, `data/kg/kg_features_val.csv`, `data/kg/kg_features_test.csv`

#### 2.3 Content Similarity (Sentence-Transformer)

This must be a **personalized** feature reflecting how similar the candidate is to the user's taste, not a global quality score.

```python
def compute_content_similarity(user_history_movies, candidate_movie, movie_embeddings):
    """
    1. Encode each movie: text = genre_list + " " + overview
    2. For each (user, candidate) pair:
       sim = mean cosine_similarity(candidate_embedding, history_movie_embeddings)
    """
```

- [ ] Encode all movies using Sentence-Transformer (e.g., `all-MiniLM-L6-v2`)
- [ ] Pre-compute and cache movie embeddings: `data/processed/movie_embeddings.npy`
- [ ] For each (user, candidate) pair, compute mean cosine similarity with user's train history
- [ ] Output: `content_similarity` column added to feature CSVs

#### 2.4 Popularity Feature (from Training Set)

```python
# CORRECT: compute from training set
movie_popularity = train_df.groupby("movie_id").size().reset_index(name="popularity")

# WRONG (previous code): computed from test set
# movie_pop = test_df.groupby("movie_id").size()
```

- [ ] Compute movie popularity (interaction count) from **training set** only
- [ ] Optionally add: `vote_count` from TMDB (this is external data, not leakage)

#### 2.5 KG Analysis & Visualization

- [ ] KG graph statistics (nodes, edges, degree distribution)
- [ ] Relation type distribution bar chart
- [ ] Example KG subgraph visualization (select 3-5 popular movies)
- [ ] KG feature coverage: what fraction of (user, candidate) pairs have nonzero KG features?

#### Phase 2 Checkpoint

- [ ] KG feature nonzero rate should be > 50% (otherwise TMDB coverage is too low)
- [ ] Content similarity distribution should be roughly normal, centered around 0.3-0.7
- [ ] Verify: KG features for val/test pairs use only train history (no leakage)

---

### Phase 3: Re-ranking + Ablation

#### 3.1 Ranker Training (CORRECTED)

**Previous (WRONG)**: Trained LightGBM on 70% of test users, evaluated on 30%.

**Corrected approach**:

```
Training data:    train_with_neg.csv  + features from train set
Validation data:  val_with_neg.csv    + features from train set
Test data:        test_with_neg.csv   + features from train set
```

```python
def train_ranker(train_features_df, val_features_df, feature_cols):
    """
    Train LightGBM on TRAINING candidates, tune on VALIDATION candidates.
    """
    X_train = train_features_df[feature_cols]
    y_train = train_features_df["label"]
    X_val = val_features_df[feature_cols]
    y_val = val_features_df["label"]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(20)]
    )
    return model


def evaluate_ranker(model, test_features_df, feature_cols, k=10):
    """
    Predict on TEST candidates and evaluate.
    """
    test_features_df["pred_score"] = model.predict(test_features_df[feature_cols])
    # Group by user, rank by pred_score, compute metrics
    ...
```

#### 3.2 Candidate Pool Unification

**Problem**: Baselines and ranker must be evaluated on the same candidate pool for results to be comparable.

**Two valid approaches** (choose one):

**Option A: Evaluate everything on the negative-sampled candidate pool**
- For baselines: instead of ranking all ~3700 items, rank only the items in `test_with_neg.csv` for each user
- For ranker: same as before
- Pro: Directly comparable. Con: Metrics will be inflated vs. literature.

**Option B: Evaluate everything on full catalog (preferred)**
- For baselines: rank all items (current approach) — keep as-is
- For ranker: after training on train_with_neg, at test time generate predictions for ALL items
  - Score all candidate items using the trained LightGBM
  - This requires computing features for all (user, item) pairs at test time
  - Expensive but most faithful evaluation
- Pro: Comparable to literature. Con: Feature computation is expensive.

**Option C: Evaluate on recall model's candidate set (practical compromise)**
- Recall models generate top-100 candidates per user
- Ranker re-ranks these 100 candidates
- Evaluate top-10 from these 100
- Both baseline and ranker are evaluated on the same 100-item pool
- Pro: Realistic two-stage pipeline evaluation. Con: Ranker is bottlenecked by recall quality.

**Recommended: Option C** — this is the standard two-stage evaluation protocol in industry and academia.

```python
# Step 1: Recall model generates top-100 candidates
recall_candidates = {}  # user_id -> list of 100 (movie_id, recall_score)

# Step 2: Compute features for these 100 candidates per user
# (CF score, content similarity, KG features, popularity)

# Step 3: Ranker re-ranks the 100 candidates
# Evaluate top-10 from the re-ranked list

# Step 4: Also evaluate recall model's top-10 from the same 100
# This gives a fair comparison: same candidate pool, different ranking
```

- [ ] Implement Option C evaluation pipeline
- [ ] Both recall baselines and ranker variants evaluated on the same top-100 candidate pool

#### 3.3 Ranking Objective Comparison: Pointwise vs LambdaMART

In addition to the KG feature ablation (V1/V2/V3), we compare two fundamentally different ranking objectives to study how the loss function interacts with KG features.

| Method | Objective | How It Works | LightGBM Setting |
|--------|-----------|-------------|-----------------|
| **Pointwise** | Binary classification | Each (user, item) pair is treated independently; predicts P(relevant) | `objective="binary"`, `metric="binary_logloss"` |
| **LambdaMART** | Listwise ranking | Optimizes NDCG directly over the ranked list within each user group | `objective="lambdarank"`, `metric="ndcg"` |

**Why this comparison matters:**
- Pointwise treats each candidate independently — it doesn't know about the relative ordering of items for the same user.
- LambdaMART uses lambda gradients derived from NDCG to focus training on item pairs where swapping order would most improve the ranking. It sees the full candidate list per user.
- If KG features are more useful for distinguishing between items of similar CF scores (fine-grained re-ordering), LambdaMART may amplify the KG benefit compared to pointwise.
- LambdaMART requires a `group` parameter: the number of candidates per user, so data must be sorted by user_id.

**Experimental design**: Cross V1/V2/V3 with both objectives (6 runs total):

| | V1 (CF) | V2 (CF+Content) | V3 (CF+Content+KG) |
|---|---------|-----------------|---------------------|
| Pointwise | baseline | +content | +KG |
| LambdaMART | baseline | +content | +KG |

**Key questions this answers:**
1. Does LambdaMART consistently outperform pointwise across all feature sets?
2. Is the KG improvement (V3 vs V2) larger under LambdaMART than under pointwise? (i.e., does listwise optimization amplify KG signal?)
3. Which combination achieves the best overall NDCG@10?

**Statistical tests:**
- Paired t-test: LambdaMART V3 vs Pointwise V3 (ranking method effect, same features)
- Paired t-test: LambdaMART V3 vs LambdaMART V2 (KG effect under LambdaMART)
- Paired t-test: Pointwise V3 vs Pointwise V2 (KG effect under pointwise)

#### 3.4 Feature Sets for Ablation

| Variant | Features | Purpose |
|---------|----------|---------|
| **Recall-only** | Top-10 from recall model's ranking of 100 candidates | Stage 1 baseline |
| **V1** | cf_score | Ranker with CF signal only |
| **V2** | cf_score + content_similarity + popularity + vote_count | CF + content features |
| **V3** | cf_score + content_similarity + popularity + vote_count + KG features | Full KG-enhanced |

#### 3.5 LightGBM Hyperparameter Search

Grid search on **validation set**, apply best config to all three variants:

```python
param_grid = {
    "num_leaves": [31, 63],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200, 500],
    "min_child_samples": [20, 50],
}
# Select config with best NDCG@10 on validation set
# Apply same config to V1, V2, V3 for fair comparison
```

- [ ] Run grid search for V3 (most features) on validation set
- [ ] Use winning params for all three variants
- [ ] Report: best params and validation metrics

#### 3.6 Ablation Results (RQ1)

- [ ] Fill final test set results (2 x 3 = 6 ranker variants + 1 baseline):

| Variant | Method | Hit@10 | NDCG@10 | Recall@10 | MRR |
|---------|--------|--------|---------|-----------|-----|
| Recall-only | — | | | | |
| V1 (CF) | Pointwise | | | | |
| V2 (CF+Content) | Pointwise | | | | |
| V3 (CF+Content+KG) | Pointwise | | | | |
| V1 (CF) | LambdaMART | | | | |
| V2 (CF+Content) | LambdaMART | | | | |
| V3 (CF+Content+KG) | LambdaMART | | | | |

- [ ] Statistical significance: paired t-test on per-user NDCG@10

```python
from scipy.stats import ttest_rel

# KG contribution (within each method)
ttest_rel(pw_v3_ndcg, pw_v2_ndcg)   # Pointwise: V3 vs V2
ttest_rel(lm_v3_ndcg, lm_v2_ndcg)   # LambdaMART: V3 vs V2

# Ranking method comparison (same features)
ttest_rel(lm_v3_ndcg, pw_v3_ndcg)   # LambdaMART vs Pointwise on V3
ttest_rel(lm_v2_ndcg, pw_v2_ndcg)   # LambdaMART vs Pointwise on V2

# V3 vs Recall-only (full re-ranking benefit)
t_stat, p_value = ttest_rel(v3_ndcg_per_user, recall_ndcg_per_user)
```

- [ ] Plot: ablation bar chart with error bars

#### 3.7 Long-tail Analysis (RQ2)

- [ ] Define long-tail: movies in bottom 50% by **training set** interaction count
- [ ] For each variant, compute Recall@10 separately on head and tail test items

| Variant | Head Recall@10 | Tail Recall@10 | Tail Lift vs V1 |
|---------|---------------|----------------|-----------------|
| V1 | | | baseline |
| V2 | | | |
| V3 | | | |

- [ ] Plot: popularity vs recall curve (bin movies by popularity, plot recall per bin)
- [ ] User interest entropy analysis:
  - Compute genre entropy for each user (based on train history)
  - Bucket users into low/medium/high diversity
  - Compare V1 vs V3 improvement across buckets

#### 3.8 Feature Importance

- [ ] Extract LightGBM feature importance (gain and split)
- [ ] Plot: feature importance bar chart for V3
- [ ] Analyze: which KG features contribute most? Is `shared_actor` more important than `same_genre`?
- [ ] Partial dependence plots for top KG features (optional)

#### Phase 3 Checkpoint

- [ ] V3 NDCG@10 > V2 NDCG@10 (if not, investigate KG feature quality)
- [ ] Statistical significance achieved (p < 0.05)
- [ ] Tail Recall improvement from V3 > Head Recall improvement (supports H2)
- [ ] KG features appear in top-5 feature importance list

**If KG does not help:**
1. Check KG feature coverage (too many zeros?)
2. Check if content_similarity already captures genre overlap (redundancy with KG genre feature)
3. Consider adding more KG relations: `collaborated_with`, keyword-based relations
4. Consider KG embeddings (TransE / ComplEx) as alternative to hand-crafted features

---

## 5. Data Flow Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │           clean_ratings.csv                 │
                    │        (only rating >= 4)                   │
                    └──────────┬──────────┬──────────┬────────────┘
                               │          │          │
                    ┌──────────▼──┐ ┌─────▼─────┐ ┌─▼──────────┐
                    │  train.csv  │ │  val.csv   │ │  test.csv  │
                    │   (70%)     │ │   (10%)    │ │   (20%)    │
                    └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
                           │              │              │
              ┌────────────▼──────────┐   │              │
              │ Recall Models train   │   │              │
              │ (CF / MF / LightGCN)  │   │              │
              └────────────┬──────────┘   │              │
                           │              │              │
                           │    ┌─────────▼─────────┐    │
                           │    │ HP tuning (recall) │    │
                           │    └───────────────────┘    │
                           │                             │
              ┌────────────▼──────────────────────────────▼────────┐
              │        Generate top-100 candidates per user        │
              │        (for train / val / test users)              │
              └────────────┬──────────────────────────────┬────────┘
                           │                              │
              ┌────────────▼──────────┐    ┌──────────────▼───────┐
              │ Compute features      │    │ Compute features     │
              │ (train + val cands)   │    │ (test candidates)    │
              │ CF score, content sim │    │ Same feature set     │
              │ popularity, KG feats  │    │ User history = train │
              └────────────┬──────────┘    └──────────────┬───────┘
                           │                              │
              ┌────────────▼──────────┐                   │
              │ LightGBM train        │                   │
              │ (train cands)         │                   │
              │ HP tune (val cands)   │                   │
              └────────────┬──────────┘                   │
                           │                              │
                           │         ┌────────────────────▼───────┐
                           └────────►│ Final Evaluation           │
                                     │ (test candidates)          │
                                     │ Metrics: Hit/NDCG/Recall   │
                                     └───────────────────────────┘
```

---

## 6. Timeline

```
Week 1:
  ├── [Phase 0] Fix data pipeline
  │     - Apply rating >= 4 threshold
  │     - Implement 3-way split (train/val/test)
  │     - Fix negative sampling (per-split, no leakage)
  │
  └── [Phase 0] TMDB data acquisition (if not done)
        - Checkpoint-resume fetch
        - Verify coverage

Week 2:
  ├── [Phase 1] Re-run recall models with corrected data
  │     - Item-CF, BPR-MF, LightGCN
  │     - Hyperparameter search on validation set
  │
  └── [Phase 1] Baseline evaluation on test set
        - Verify metrics in expected range
        - Save top-100 candidates per user for all splits

Week 3:
  ├── [Phase 2] KG construction (if not done, reuse existing)
  │
  ├── [Phase 2] Fix feature engineering
  │     - Implement real content similarity (Sentence-Transformer)
  │     - Compute popularity from training set
  │     - Compute KG features for train/val/test candidates
  │     - Verify no leakage: features only use train history
  │
  └── [Phase 2] KG analysis & visualization

Week 4:
  ├── [Phase 3] Fix ranker pipeline
  │     - Train LightGBM on train candidates
  │     - Tune on validation candidates
  │     - Evaluate on test candidates (Option C: top-100 pool)
  │
  ├── [Phase 3] Ablation experiments (V1/V2/V3)
  │     - Statistical significance tests
  │
  ├── [Phase 3] Long-tail analysis
  │     - Head vs tail recall
  │     - User interest entropy buckets
  │
  └── [Phase 3] Feature importance + final results
        - Feature importance plots
        - Compile all results into tables
        - Write up findings
```

---

## 7. Code Changes Required

Below is a summary of the specific code modifications needed to implement this plan:

### 7.1 `src/data_prep/parse_ml1m.py`

| Change | Description |
|--------|-------------|
| Add positive threshold | Filter to rating >= 4 before splitting |
| 3-way split | Implement `split_train_val_test()` with 70/10/20 ratio |
| Fix negative sampling | Generate negatives per split; user_history only includes prior splits |
| Output val files | Add `val.csv`, `val_with_neg.csv` |

### 7.2 `src/models/*.py` (all recall models)

| Change | Description |
|--------|-------------|
| Add validation | Use validation set for early stopping / HP selection |
| Keep train-only data | Ensure models only train on `train.csv` |

### 7.3 `src/kg/kg_features.py`

| Change | Description |
|--------|-------------|
| Separate per-split output | Generate features for train/val/test candidate sets separately |
| Enforce train history | User history always from training set only |

### 7.4 `src/ranker/ranker.py` (major rewrite)

| Change | Description |
|--------|-------------|
| **Train on train candidates** | Replace test-data training with train_with_neg training |
| **Tune on val candidates** | Use val_with_neg for early stopping and HP search |
| **Evaluate on test candidates** | Final metrics from test_with_neg |
| **Fix popularity** | Compute from training set |
| **Fix content similarity** | Replace `vote_average/10` with Sentence-Transformer cosine similarity |
| **Option C evaluation** | Re-rank recall model's top-100 candidates |
| **HP search** | Grid search LightGBM params on validation set |

### 7.5 `src/evaluation/metrics.py`

| Change | Description |
|--------|-------------|
| No changes needed | Metric implementations are correct |

---

## 8. Expected Outcome

After corrections, we expect:
- **Baseline metrics** to be lower than before (because ground truth is now rating >= 4 only, and evaluation is on full catalog)
- **Ranker metrics** to be much lower than the inflated v1 results (no more training-on-test or small candidate pool)
- **V3 vs V2 improvement** to be modest (typical KG improvement on ML-1M is 2-5% relative NDCG)
- **Long-tail improvement** to be more pronounced than head improvement

If V3 does not improve over V2, this is still a valid research finding — it means hand-crafted KG features from TMDB are not sufficient for this dataset, and more sophisticated KG methods (embeddings, path reasoning) should be explored.

---

*Plan revised: 2026-03-20*
*Previous version: 2026-03-19*
*Current phase: Phase 0 (re-run with corrected data pipeline)*
