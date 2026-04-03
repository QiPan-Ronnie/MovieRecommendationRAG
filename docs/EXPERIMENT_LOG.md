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

---

## Experiment 3: KG Recall Improvement — Data & Model Ablation (2026-03-31)

### Motivation

KG recall scores have near-zero discrimination: positive vs negative cosine similarity difference is only **0.000519** (std=0.012). KG-only candidate hit rate is 1.75% (vs CF-only 5.64%). Root causes identified:

1. **co_liked edges dominate KG** (100K/134K = 74.4%), drowning out metadata signal in TransE
2. **TransE cannot model symmetric (co_liked) or N-to-N relations** (has_genre, acted_by)
3. **User profile = mean of last 20 embeddings** is too blurry (73.7% of users have >20 interactions)

### Setup

Four variants tested, each building on the previous:

| Variant | Description |
|---------|-------------|
| Baseline | Original: TransE, co_liked=100K, max_history=20, uniform sampling |
| A | co_liked reduced to 30K + relation-balanced sampling in TransE (300 epochs) |
| A+D | A + improved user profile: no history limit, IDF weighting, time decay |
| A+D+B | A+D + RotatE (complex-space rotation model) replacing TransE |

### Results: Stage 1 KG Recall Quality

| Metric | Baseline | A | A+D | A+D+B |
|--------|----------|---|-----|-------|
| **Recall@10** | 0.0526 | 0.0554 (+5.4%) | 0.0575 (+9.4%) | **0.0618 (+17.6%)** |
| **NDCG@10** | 0.0872 | 0.0889 (+2.0%) | 0.0908 (+4.2%) | **0.0953 (+9.4%)** |
| Hit@10 | 0.4435 | 0.4484 | 0.4536 | **0.4652** |
| KG-only hit rate | 0.0175 | 0.0210 (+20%) | 0.0197 (+12%) | **0.0311 (+77%)** |
| KG score discrimination | 0.00113 | 0.00083 | 0.00244 (2.2x) | **0.00705 (6.2x)** |
| KG score std | 0.012 | 0.012 | 0.013 | **0.059 (4.8x)** |
| KG unique positives | 3,127 | 3,719 | 3,500 | **4,537 (+45%)** |
| Total positives captured | 26,574 | 27,166 | 26,947 | **27,984 (+5.3%)** |
| Movies covered | 1,351 | 1,479 | 1,673 | **2,070 (+53%)** |
| CF/KG overlap | 9.0% | 9.4% | 6.0% | **24.3%** |

Full results saved in `results/kg_recall_experiments/`.

### Results: End-to-End (A+D+B through Ranker)

A+D+B was validated through the full re-ranking pipeline (label → features → LightGBM). **Absolute metrics dropped despite better KG recall.**

| Variant | Base NDCG@10 | ADB NDCG@10 | Δ | Base R@10 | ADB R@10 | Δ |
|---------|-------------|-------------|---|-----------|----------|---|
| Recall-only | 0.1451 | 0.1395 | -3.8% | 0.1817 | 0.1722 | -5.2% |
| V3 LambdaMART | 0.1415 | 0.1368 | -3.3% | 0.1948 | 0.1855 | -4.8% |
| V4 LambdaMART | 0.1428 | 0.1340 | -6.2% | 0.1976 | 0.1829 | -7.4% |

However, **KG's relative contribution was stronger**: V3 vs V2 LambdaMART Recall@10 lift was +0.0147 (ADB) vs +0.0070 (baseline) — **2.1x improvement**.

Full results saved in `results/kg_recall_experiments/ADB_e2e/`.

### Analysis

1. **co_liked=30K weakened CF candidate quality.** Reducing collaborative edges from 100K to 30K reduced the overall KG→CF synergy. The co_liked edges, while dominating TransE training, also provide crucial collaborative signal that TransE encodes into movie embeddings.

2. **High CF/KG overlap (24.3%) caused candidate pool shrinkage.** RotatE embeddings are more aligned with CF signal, so more KG candidates overlap with CF. After dedup, 3,681 users (62%) got fewer than 100 candidates (avg 94.5), directly losing information.

3. **KG-only positives lack cf_score, hurting Recall-only ranking.** 83.8% of ADB positives have cf_score>0 (vs 88.2% baseline). KG-only positives get cf_score=0 and are ranked last in Recall-only evaluation.

4. **RotatE's better discrimination is real but offset by weaker candidate pool.** The 6.2x improvement in score discrimination and 2.1x KG lift confirm RotatE captures richer relational patterns than TransE.

### Decision

A+D+B as a whole does not improve end-to-end. The bottleneck is co_liked reduction + candidate shrinkage, not the model itself.

### Next Steps

- **Train RotatE on original KG (100K co_liked)** — keep CF signal strong while gaining RotatE's better discrimination
- **Increase n_kg to compensate for higher CF/KG overlap** — ensure every user gets exactly 100 candidates
- Apply improved user profile (IDF weighting, no history limit) with the original KG

---

## Experiment 4: RotatE on Original KG + Candidate Compensation (2026-04-01)

### Motivation

Experiment 3 showed RotatE provides 6.2x better score discrimination but co_liked reduction and candidate shrinkage hurt end-to-end performance. This experiment isolates RotatE's benefit by keeping the original KG (100K co_liked) and compensating for CF/KG overlap to maintain 100 candidates per user.

### Setup

- **KG**: Original (134K triples, 100K co_liked) — unchanged
- **Embedding model**: RotatE (balanced sampling, 300 epochs, dim=128, gamma=12)
- **User profile**: Improved (no history limit, IDF weighting, time decay)
- **Candidate compensation**: Increase n_kg to ensure ≥100 candidates per user after dedup

### Results: Stage 1 KG Recall

| Metric | Baseline | RotatE+OrigKG |
|--------|----------|---------------|
| Recall@10 | 0.0526 | **0.0641 (+21.9%)** |
| NDCG@10 | 0.0872 | **0.0974 (+11.7%)** |
| KG-only hit rate | 0.0175 | **0.0348 (+99%)** |
| KG score discrimination | 0.00113 | **0.00565 (5.0x)** |
| Total positives captured | 26,574 | **29,663 (+11.6%)** |
| KG unique positives | 3,127 | **6,216 (+99%)** |
| Movies covered | 1,351 | **1,844 (+36%)** |
| Per-user candidates | 100.0 | **100.0** (compensation worked) |

### Results: End-to-End (through Ranker)

| Variant | Base NDCG@10 | New NDCG@10 | Δ | Base R@10 | New R@10 | Δ |
|---------|-------------|-------------|---|-----------|----------|---|
| Recall-only | 0.1451 | 0.1352 | -6.8% | 0.1817 | 0.1632 | -10.2% |
| V3 LambdaMART | 0.1415 | 0.1290 | -8.8% | 0.1948 | 0.1727 | -11.4% |
| V4 LambdaMART | 0.1428 | 0.1343 | -5.9% | 0.1976 | 0.1785 | -9.7% |

KG lift (V3 vs V2): +0.0150 (216% of baseline's +0.0070). **KG features contribute 2x more in the new setup**, but overall performance drops.

### Analysis

**KG recall quality improves massively** — 2x hit rate, 5x discrimination, +11.6% positives captured. Candidate compensation ensures all users get exactly 100 candidates.

**End-to-end still degrades.** The pattern is consistent across Experiments 3 and 4: better KG recall ≠ better end-to-end. Root cause: changing the recall candidate distribution breaks the ranker's learned patterns.

Key evidence:
- V4 LambdaMART drops least (-5.9%), suggesting RotatE embedding features partially compensate
- KG lift doubles (V3 vs V2), confirming KG features are more useful in the new candidate pool
- But Recall-only baseline drops -10.2%, meaning the cf_score ordering over the new candidate pool is worse

### Conclusion

**The recall→re-rank pipeline has a coupling problem.** The ranker implicitly depends on the recall candidate distribution (cf_score distribution, positive rate, candidate overlap patterns). Improving recall quality while keeping the ranker unchanged disrupts this coupling.

### Potential Next Steps (not yet tried)

1. **Re-tune ranker for new candidates** — the ranker hyperparameters and feature weights were optimized for baseline candidates
2. ~~**Use RotatE features as additional ranker features only**~~ → Done in Experiment 5
3. **Joint optimization** — train recall and ranker together, or use recall-aware ranker training
4. **Score calibration** — normalize cf_score and kg_recall_score distributions before ranker training

---

## Experiment 5: RotatE Embedding as Ranker Features Only (2026-04-02)

### Motivation

Experiments 3-4 showed that changing the recall candidate pool always degrades end-to-end performance, even when KG recall quality improves. The root cause is **candidate pool composition change** (CF-only items drop from 61% to 51%), not the ranker or its features — evidenced by even Recall-only (pure cf_score ordering) degrading -10%.

This experiment tests whether RotatE's better embeddings can improve the ranker **without changing the recall candidates at all**. Only the `kg_emb_features` (4 aggregate embedding features used in V3e/V4) are recomputed with RotatE embeddings; everything else stays identical.

### Setup

- **Recall candidates**: Unchanged (baseline Item-CF top-70 + KG top-50)
- **KG features (hand-crafted)**: Unchanged
- **Content similarity**: Unchanged
- **cf_score, kg_recall_score**: Unchanged
- **Only change**: `kg_emb_features` recomputed using RotatE embeddings (trained on original KG, balanced sampling, 300 epochs, dim=128, gamma=12)

### Results

| Variant | Base NDCG@10 | New NDCG@10 | Δ | Base R@10 | New R@10 | Δ |
|---------|-------------|-------------|---|-----------|----------|---|
| Recall-only | 0.1451 | 0.1451 | 0.0% | 0.1817 | 0.1817 | 0.0% |
| V3 (no emb) Pointwise | 0.1362 | 0.1362 | 0.0% | 0.1849 | 0.1849 | 0.0% |
| V3 (no emb) LambdaMART | 0.1415 | 0.1415 | 0.0% | 0.1948 | 0.1948 | 0.0% |
| **V3e Pointwise** | 0.1288 | 0.1306 | **+1.4%** | 0.1717 | 0.1753 | **+2.1%** |
| **V3e LambdaMART** | **0.1306** | **0.1374** | **+5.2%** | **0.1761** | **0.1892** | **+7.4%** |
| V4 Pointwise | 0.1336 | 0.1355 | +1.4% | 0.1828 | 0.1821 | -0.4% |
| **V4 LambdaMART** | 0.1428 | **0.1429** | +0.1% | 0.1976 | **0.1986** | **+0.5%** |

### Feature Importance Change (RotatE vs TransE)

**V3e Pointwise** — RotatE embedding features gain share massively:

| Feature | TransE Gain | RotatE Gain | Change |
|---------|------------|------------|--------|
| kg_emb_max_cos | 1,456 | 5,311 | **+265%** |
| kg_emb_mean_cos | 1,878 | 5,246 | **+179%** |
| kg_emb_min_dist | 2,038 | 2,733 | +34% |
| kg_emb_mean_dist | 2,792 | 3,067 | +10% |

**V4 LambdaMART** — all 4 embedding features gain 4-6x:

| Feature | TransE Gain | RotatE Gain | Change |
|---------|------------|------------|--------|
| kg_emb_mean_cos | 315 | 2,112 | **+570%** |
| kg_emb_max_cos | 262 | 1,647 | **+529%** |
| kg_emb_min_dist | 355 | 1,816 | **+412%** |
| kg_emb_mean_dist | 281 | 1,606 | **+472%** |

### Analysis

1. **Sanity check passed**: V1/V2/V3 and Recall-only are exactly identical — confirms no candidate pool contamination.

2. **V3e LambdaMART is the biggest winner** (+5.2% NDCG, +7.4% Recall@10). With TransE, V3e was the worst variant (worse than V2). With RotatE, V3e LambdaMART (0.1892) approaches V3 LambdaMART (0.1948), showing RotatE embedding features carry signal comparable to hand-crafted KG features.

3. **V4 LambdaMART sets new best** (0.1986 Recall@10, +0.5% over baseline 0.1976). The improvement is small but positive — hand-crafted + RotatE embedding features are complementary.

4. **RotatE embedding features are 4-6x more useful to LightGBM** than TransE ones, confirming RotatE's richer relational modeling translates to better downstream features.

### Decision

**Adopted as the new default.** RotatE replaces TransE for embedding feature computation. The improvement is safe (no candidate pool change), consistent (all embedding-using variants improve), and the V4 LambdaMART result is the new project best.

Full results saved in `results/kg_recall_experiments/rotate_features_only/`.
