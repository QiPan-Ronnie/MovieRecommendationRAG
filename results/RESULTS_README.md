# Results Directory

## Key Result Files

| File | Description |
|------|-------------|
| `RESULTS.md` | Full experiment results and analysis (English) |
| `RESULTS_ZH.md` | Chinese version |
| `ablation_results.json` | All variant metrics (NDCG, Recall, Hit, MRR @1/5/10) |
| `baseline_results.json` | Stage 1 recall baseline metrics (Item-CF, BPR-MF, LightGCN) |
| `feature_importance.json` | LightGBM feature importance (gain) per variant |
| `longtail_analysis.json` | Head/tail stratified metrics |

## Phase 2 RAG Inputs

| File | Description |
|------|-------------|
| `recommendations_v4.csv` | Per-user top-10 from V4 LambdaMART (CF+Content+KG+Emb, best model) |
| `recommendations_v2.csv` | Per-user top-10 from V2 LambdaMART (CF+Content, no KG — for RQ4 comparison) |
| `rag_eval_set.json` | 200 sampled users with watch history (10 movies) + V4 recommendations + movie overviews + gold labels |

Recommendation CSV columns: `user_id, movie_id, rank, pred_score, label` (5,950 users x 10 = 59,500 rows)

## Recall Model Scores

| File | Description |
|------|-------------|
| `cf_scores.csv` | Item-CF top-100 candidates per user |
| `mf_scores.csv` | BPR-MF top-100 candidates per user |
| `lightgcn_scores.csv` | LightGCN top-100 candidates per user |
| `multi_recall_scores.csv` | Multi-route recall: CF top-70 + KG top-50 merged to 100 per user |

## KG Recall Experiment Results

`kg_recall_experiments/` contains results from 5 ablation experiments on KG recall:

| Directory | Experiment | Key Finding |
|-----------|-----------|-------------|
| `baseline/` | Original TransE | KG score discrimination = 0.00113 |
| `A_balanced_coliked30k/` | co_liked=30K + balanced sampling | +20% KG hit rate |
| `AD_balanced_improved_profile/` | + IDF weighting + full history | +12% KG hit rate |
| `ADB_rotate_balanced_improved/` | + RotatE (co_liked=30K) | +77% KG hit rate, but E2E -5% |
| `ADB_e2e/` | A+D+B end-to-end ranker | Candidate pool change hurts E2E |
| `rotate_original_kg/` | RotatE on original KG | +99% KG hit rate, but E2E -11% |
| `rotate_features_only/` | RotatE as ranker features only | **V3e +7.4%, V4 +0.5% (adopted)** |

See `docs/EXPERIMENT_LOG.md` for detailed analysis of each experiment.
