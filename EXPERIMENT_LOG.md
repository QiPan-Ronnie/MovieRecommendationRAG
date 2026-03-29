# Experiment Log

Record of experimental attempts, negative results, and design decisions for future reference.

---

## Experiment 1: LightGCN vs Item-CF as Recall Base (2026-03-29)

### Motivation

Item-CF has only 13.5% catalog coverage vs LightGCN's 28.5%. We hypothesized that higher coverage would produce better candidates and improve re-ranking performance.

### Setup

- Replaced Item-CF (cf_scores.csv) with LightGCN (lightgcn_scores.csv) as the recall base in multi-route recall
- Multi-route: LightGCN top-70 + KG (TransE) top-50 → 100 candidates/user
- All other settings identical (same KG, same ranker, same features)

### Results

| Metric | Item-CF Recall | LightGCN Recall | Change |
|--------|---------------|-----------------|--------|
| V3 LambdaMART NDCG@10 | **0.1438** | 0.1355 | -5.8% |
| V3 LambdaMART Recall@10 | **0.1982** | 0.1842 | -7.1% |
| V3 Pointwise NDCG@10 | **0.1390** | 0.1261 | -9.3% |
| V3 Pointwise Recall@10 | **0.1899** | 0.1725 | -9.2% |
| Recall-only NDCG@10 | **0.1462** | 0.1336 | -8.6% |
| Recall-only Recall@10 | **0.1835** | 0.1685 | -8.2% |
| V3 Tail Recall@10 | **0.1316** | 0.0535 | -59.3% |
| V4 Tail Recall@10 | **0.2368** | 0.1062 | -55.1% |
| Movies covered | 1,340 | 2,144 | +60% |

### Statistical Significance (LightGCN recall)

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| LambdaMART V3 vs V2 | **0.000012** | Yes (stronger than Item-CF's p=0.007) |
| Pointwise V3 vs V2 | 0.762 | No (was p=0.0007 with Item-CF) |

### Analysis

1. **Higher coverage does not mean better candidates.** LightGCN covers 2,144 movies vs Item-CF's 1,340, but this extra coverage dilutes candidate quality. Many LightGCN candidates are weakly relevant, making the re-ranking task harder.

2. **Item-CF's similarity scores are more informative for LightGBM.** Item-CF's cosine similarity is a direct measure of item co-occurrence strength, which LightGBM can easily exploit. LightGCN's dot-product scores have a different distribution that may be less discriminative as a feature.

3. **KG's relative value increases on harder candidate pools.** LambdaMART V3 vs V2 became much more significant (p=0.000012 vs p=0.007), suggesting KG features are more valuable when the base CF signal is weaker. This supports the theoretical argument that KG addresses CF limitations.

4. **Long-tail performance dropped dramatically.** V3 tail Recall went from 0.1316 to 0.0535. The KG+TransE multi-recall with LightGCN didn't compensate — likely because LightGCN already covers more movies inherently, but the KG-only recall route contributes fewer *unique* tail items.

5. **Pointwise ranking lost significance.** Pointwise V3 vs V2 went from p=0.0007 to p=0.762. Pointwise may be more sensitive to candidate pool quality since it treats each pair independently. LambdaMART's listwise optimization appears more robust.

### Decision

**Reverted to Item-CF as recall base.** Item-CF produces better absolute results across all metrics. The finding that KG becomes relatively more valuable on harder candidate pools is interesting but doesn't justify worse overall performance.

### Future Directions

- Try a **hybrid approach**: Item-CF top-100 as primary candidates, but add LightGCN scores as an additional re-ranking feature (alongside cf_score)
- Investigate whether **LightGCN with hyperparameter tuning** (embed_dim, n_layers, lr grid search) produces better candidates
- Consider using LightGCN as the recall model but with a **larger candidate pool** (top-200 → re-rank to 100) to offset quality dilution

---

## Experiment 2: LightGBM Hyperparameter Search (2026-03-29)

### Motivation

All ablation results used default LightGBM parameters (num_leaves=31, learning_rate=0.1, min_child_samples=20). V3 improves Recall@10 but shows NDCG@10 regression vs recall-only baseline. Hyperparameter tuning might address this.

### Setup

Grid search: num_leaves={31,63}, learning_rate={0.05,0.1}, min_child_samples={20,50} (8 combinations per method).

### Result

**Not completed.** The search ran for 75+ CPU hours without finishing. The bottleneck was evaluate_ranker() being called after each parameter combination on the full validation set (~120K pairs). LightGBM training itself is fast, but the per-user grouping and metric computation in Python is slow.

### Future Directions

- **Reduce validation set size** for HP search (subsample 20% of users)
- **Use LightGBM's built-in NDCG metric** for early stopping instead of custom evaluate_ranker()
- **Parallelize** with joblib or similar
- Try **Optuna** with pruning instead of exhaustive grid search
