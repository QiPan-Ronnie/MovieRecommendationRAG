# Reflection: Consistency and Rigor Audit

> Final review after code corrections (2026-03-20)

---

## 1. What Was Fixed

### Critical Fixes (Data Leakage & Evaluation)

| Issue | Before | After | Files Changed |
|-------|--------|-------|---------------|
| Ranker trained on test data | LightGBM trained on 70% of test users | Trains on `train_with_neg.csv`, tunes on `val_with_neg.csv`, evaluates on `test_with_neg.csv` | `ranker.py` |
| All ratings as positive | Rating 1-5 all treated as positive | Only rating >= 4 is positive (implicit feedback) | `parse_ml1m.py` |
| 2-way split | train/test only | train (70%) / val (10%) / test (20%) time-based split | `parse_ml1m.py` |
| Popularity from test set | `test_df.groupby("movie_id").size()` | `train_df.groupby("movie_id").size()` | `ranker.py` |
| Content similarity = vote_avg/10 | Not content similarity at all | Sentence-Transformer cosine similarity between user profile and candidate | `content_similarity.py` (new) |

### Bug Fixes

| Issue | Before | After | Files Changed |
|-------|--------|-------|---------------|
| Early stopping counter shared | Val-based and loss-based stopping shared one `no_improve` counter | Separate `val_no_improve` and `loss_no_improve` counters; only one active depending on whether val set exists | `matrix_factorization.py`, `lightgcn.py` |
| MRR not truncated at K | MRR searched the full 100-item list | MRR respects K cutoff parameter | `metrics.py` |
| HP search not propagated | V3 search results not shared with V1, V2 | Search once on V3, apply `shared_params` to all variants | `ranker.py` |
| Dead parameter | `top_k_sim=50` accepted but never used in Item-CF | Parameter removed | `item_cf.py` |

### Structural Improvements

| Improvement | Description | Files |
|-------------|-------------|-------|
| Validation-based model selection | MF and LightGCN now use val NDCG@10 for early stopping | `matrix_factorization.py`, `lightgcn.py` |
| Per-split negative sampling | Negatives sampled with correct history scope per split | `parse_ml1m.py` |
| Per-split KG features | Separate KG feature files for train/val/test, all using train history | `kg_features.py` |
| Iterative min interaction filter | User and item filters applied iteratively until convergence | `parse_ml1m.py` |

---

## 2. Known Remaining Limitations

These are issues identified during reflection that are acknowledged but acceptable for Phase 1.

### 2.1 Evaluation Protocol: Candidate Pool Difference

**Status**: Partially addressed, documented as known limitation.

The experiment plan recommends **Option C** (evaluate both baselines and ranker on recall model's top-100 candidates). The current implementation uses:
- **Baselines**: Evaluated on full catalog (~3700 items), top-10 from top-100
- **Ranker**: Evaluated on negative-sampled pool (~5 items per positive)

These numbers are **not directly comparable**. The ablation results (V1 vs V2 vs V3) are valid for internal comparison (all use the same candidate pool), but cannot be compared with baseline numbers.

**Mitigation**: The experiment plan clearly documents this. In the results section, we will present:
1. Baseline metrics (full catalog) — for literature comparison
2. Ablation metrics (negative-sampled pool) — for V1/V2/V3 comparison
3. "Recall-only" baseline within the ranker evaluation — this IS comparable with V1/V2/V3

**Future fix**: Implement full Option C pipeline where the recall model generates top-100 candidates, features are computed for those candidates, and the ranker re-ranks them.

### 2.2 Content Similarity Depends on Sentence-Transformer

The `content_similarity.py` script requires the `sentence-transformers` package and the `all-MiniLM-L6-v2` model. If this cannot run, `content_similarity` will default to 0 for all pairs, making V2 effectively "CF + popularity" only. This is gracefully handled in code (no crash) but weakens the V2 vs V1 comparison.

### 2.3 CF Scores May Not Cover All Negative-Sampled Candidates

The recall model generates top-100 candidates per user. The negative sampling generates random items. Many negatively-sampled items won't appear in the recall model's top-100, so their `cf_score` will be 0. This is actually informative for the ranker (low cf_score = recall model didn't rank it highly), but it means `cf_score` is somewhat sparse in the ranker's training data.

### 2.4 `links.dat` Availability for TMDB Mapping

The `PROJECT_OVERVIEW.md` mentions mapping via IMDb IDs from `links.dat`. ML-1M does not include `links.dat` (it's a ML-20M feature). The `fetch_tmdb.py` script (not modified in this round) handles the actual mapping. This should be verified when running the pipeline.

### 2.5 No Hyperparameter Search for Recall Models

The experiment plan calls for HP search (grid over embed_dim, lr, etc.) for MF and LightGCN. The current `run_baselines.py` only runs a single configuration. The infrastructure for validation-based selection is in place (val_df is passed), but the grid loop is not implemented yet. This is a Week 2 task per the timeline.

---

## 3. Consistency Matrix

Verification that key design decisions are consistently applied across all three documents and all code:

| Decision | PROJECT_OVERVIEW.md | experiment_plan.md | Code |
|----------|--------------------|--------------------|------|
| Rating >= 4 as positive | Section 4.1 mentions "1-5 scale" dataset, Section 3.2 mentions "KG features for... user's training history (rating >= 4 movies)" | Section 0.4: "only keep ratings >= 4 as positive interactions" | `parse_ml1m.py:filter_positive_interactions(min_rating=4)` |
| 3-way split (70/10/20) | Not explicitly mentioned (overview level) | Section 0.5: table showing 70/10/20 | `parse_ml1m.py:split_train_val_test(train_ratio=0.7, val_ratio=0.1)` |
| Train-only user history for features | Section 3.2: "aggregated over the user's history" | Section 2.2: "User history always from training set only" | `kg_features.py:build_train_user_history()`, `content_similarity.py` |
| LightGBM trains on train set | Section 3.2: "LightGBM pointwise ranker" | Section 3.1: "Train on train candidates" | `ranker.py:train_lgbm(train_df, val_df, ...)` |
| Popularity from train set | Not mentioned at this level | Section 2.4: "Compute from training set" | `ranker.py:compute_popularity_from_train()` |
| Paired t-test for significance | Section 5.1: "Paired t-test on per-user NDCG@10" | Section 3.5: code example | `ranker.py:ttest_rel(a_vals, b_vals)` |
| Ablation V1/V2/V3 | Section 3.2: feature table | Section 3.3: feature table | `ranker.py:define_feature_sets()` |

---

## 4. Research Rigor Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| No data leakage (train -> features) | OK | All features computed from train set only |
| No data leakage (test -> training) | OK | Ranker trains on train_with_neg, tunes on val_with_neg |
| Time-based split (no future information) | OK | Per-user chronological split |
| Negative sampling without future info | OK | Per-split history scope |
| Reproducibility (random seeds) | OK | Seeds set in parse_ml1m (42,43,44), models (42), ranker (42) |
| Statistical significance testing | OK | Paired t-test on per-user NDCG |
| Fair comparison (same candidate pool) | Partial | V1/V2/V3 are comparable; baselines use different pool |
| Metrics correctly implemented | OK | Hit, NDCG, Recall, MRR (now K-truncated), Coverage |
| Hyperparameter tuning on validation | OK (infra) | Infrastructure in place; grid search is a TODO |
| Clear research questions | OK | RQ1, RQ2 with testable hypotheses |

---

## 5. Execution Order

When running the corrected pipeline:

```bash
# Step 1: Data preparation (re-run with rating >= 4 threshold and 3-way split)
cd /data2/xiaoqinfeng/workdir/MovieRecommendation
python src/data_prep/parse_ml1m.py

# Step 2: TMDB metadata (if not already done)
python src/data_prep/fetch_tmdb.py

# Step 3: KG construction (if not already done, can reuse)
python src/kg/build_kg.py

# Step 4: Baseline models
python src/run_baselines.py

# Step 5: Content similarity (requires sentence-transformers)
python src/kg/content_similarity.py

# Step 6: KG features (for all three splits)
python src/kg/kg_features.py

# Step 7: Ranker ablation
python src/ranker/ranker.py
```

---

*Reflection completed: 2026-03-20*
