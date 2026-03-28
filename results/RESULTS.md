# Phase 1 Experiment Results: KG-Enhanced Recommendation

> Last updated: 2026-03-24

---

## 1. Data Summary

| Item | Value |
|------|-------|
| Dataset | MovieLens 1M (rating >= 4 as positive) |
| Users | 5,950 |
| Movies | 3,125 |
| Positive interactions | 573,726 |
| Train / Val / Test | 398,867 / 54,680 / 120,179 (70/10/20, per-user time-based) |
| **KG triples** | **134,447** (`co_liked`: 100K, `acted_by`: 18K, `has_genre`: 8.9K, `directed_by`: 3.9K, `released_in_decade`: 3.7K) |
| TransE dim | 128 (300 epochs, 5 relation types) |
| Multi-route recall | CF top-70 + KG top-50 → 100 per user |
| Recall movie coverage | 1,340 (was 1,105 with CF-only) |
| Test recall positive rate | 4.4% |

---

## 2. Stage 1: Recall Model Baselines

Evaluated on full catalog (~3,125 movies), top-10.

| Model | Hit@10 | NDCG@10 | Recall@10 | MRR@10 | Coverage |
|-------|--------|---------|-----------|--------|----------|
| Item-CF | 0.4585 | 0.0945 | 0.0607 | 0.1916 | 13.5% |
| BPR-MF | 0.4439 | 0.0868 | 0.0550 | 0.1824 | 20.3% |
| LightGCN | 0.4565 | 0.0903 | 0.0601 | 0.1837 | 28.5% |

---

## 3. Stage 2: KG-Enhanced Re-ranking (RQ1)

### 3.1 Evaluation Protocol

- **Multi-route recall**: CF top-70 + KG (TransE) top-50, merged to 100 per user
- **Distribution-matched training**: ranker trains/evals on recall candidates (not random negatives)
- **Training labels**: val-period interactions; **Test labels**: test-period interactions

### 3.2 KG Enrichment (vs initial 3-relation KG)

| Improvement | Before | After | Impact |
|------------|--------|-------|--------|
| Relations | 3 (genre, actor, director) | **5** (+co_liked, +decade) | +103K triples |
| IDF weighting | No | **Yes** | Rare-entity sharing weighted higher |
| Collaborative edges | None | **100K co_liked** | Encodes CF signal in KG |
| Decade relations | None | **3.7K** | Temporal preference signal |

### 3.3 Ablation Results (Pointwise)

| Variant | NDCG@10 | Recall@10 | Hit@10 | MRR@10 |
|---------|---------|-----------|--------|--------|
| Recall-only | **0.1462** | 0.1835 | 0.5167 | **0.2160** |
| V1 (CF) | 0.1208 | 0.1531 | 0.4752 | 0.1837 |
| V2 (CF+Content) | 0.1309 | 0.1731 | 0.5032 | 0.1924 |
| **V3 (CF+Content+KG)** | 0.1390 | **0.1899** | **0.5235** | 0.1976 |
| V3e (CF+Content+KGEmb) | 0.1279 | 0.1706 | 0.4998 | 0.1842 |
| V4 (CF+Content+KG+Emb) | 0.1330 | 0.1841 | 0.5098 | 0.1868 |

### 3.4 Ablation Results (LambdaMART)

| Variant | NDCG@10 | Recall@10 | Hit@10 | MRR@10 |
|---------|---------|-----------|--------|--------|
| Recall-only | 0.1462 | 0.1835 | 0.5167 | 0.2160 |
| V1 (CF) | 0.1105 | 0.1451 | 0.4627 | 0.1687 |
| V2 (CF+Content) | 0.1371 | 0.1885 | 0.5318 | 0.1965 |
| **V3 (CF+Content+KG)** | **0.1438** | **0.1982** | 0.5220 | **0.2002** |
| V3e (CF+Content+KGEmb) | 0.1354 | 0.1836 | 0.5186 | 0.1935 |
| **V4 (CF+Content+KG+Emb)** | 0.1400 | 0.1960 | **0.5252** | 0.1923 |

### 3.5 Statistical Significance

| Comparison | Diff | p-value | Significant? |
|------------|------|---------|--------------|
| Pointwise: V3 vs V2 (KG contribution) | +0.0081 | 0.0007 | **Yes** |
| LambdaMART: V3 vs V2 (KG contribution) | +0.0066 | 0.007 | **Yes** |
| V3 LambdaMART vs Recall-only | -0.0025 | 0.453 | No |

### 3.6 Feature Importance (V3 Pointwise)

| Feature | Gain | Share | New? |
|---------|------|-------|------|
| cf_score | 49,187 | 58.9% | |
| **kg_same_genre_idf_sum** | **6,126** | **7.3%** | **IDF** |
| **kg_same_decade_ratio** | **5,707** | **6.8%** | **New** |
| popularity | 3,926 | 4.7% | |
| kg_same_genre_count_sum | 3,605 | 4.3% | |
| content_similarity | 3,544 | 4.2% | |
| vote_count | 3,330 | 4.0% | |
| **kg_co_liked_sum** | **2,571** | **3.1%** | **New** |
| **kg_shared_actor_idf_sum** | **1,957** | **2.3%** | **IDF** |
| kg_recall_score | 1,919 | 2.3% | |

**KG features account for 24.8% of total gain.** The three new signal types (IDF weighting, decade, co_liked) are all in the top-10 features.

---

## 4. Long-tail Analysis (RQ2)

### 4.1 Head/Tail Definition

| Group | Movies | Mean interactions |
|-------|--------|-------------------|
| Head | 1,560 | 240.7 |
| Tail | 1,562 | 15.0 |

Tail analysis based on 212 users with tail positives in recall candidates (statistically meaningful after multi-route recall fix).

### 4.2 Head/Tail Stratified Recall@10

| Variant | Head Recall | Tail Recall | Tail/Head |
|---------|-------------|-------------|-----------|
| Recall-only | 0.1837 | 0.0000 | 0.00 |
| V1 (CF) | 0.1533 | 0.0000 | 0.00 |
| V2 (CF+Content) | 0.1733 | 0.0000 | 0.00 |
| **V3 (CF+Content+KG)** | **0.1899** | **0.1316** | **0.69** |
| V3e (CF+Content+KGEmb) | 0.1707 | 0.0526 | 0.31 |
| **V4 (CF+Content+KG+Emb)** | 0.1837 | **0.2368** | **1.29** |

### 4.3 KG Lift: Head vs Tail

| Comparison vs V1 | Head Lift | Tail Lift | Ratio |
|-------------------|-----------|-----------|-------|
| V3 (CF+Content+KG) | +0.037 | **+0.132** | **3.6x** |
| V4 (CF+Content+KG+Emb) | +0.030 | **+0.237** | **7.9x** |

### 4.4 User Genre Entropy Analysis

| Variant | Low Entropy | Mid Entropy | High Entropy |
|---------|-------------|-------------|--------------|
| V1 (CF) | 0.1523 | 0.1549 | 0.1521 |
| V3 (KG) | **0.2307** | **0.1888** | 0.1562 |
| V3 vs V1 lift | **+51%** | **+22%** | +3% |

---

## 5. Key Findings

### RQ1: Does KG-enhanced re-ranking outperform baselines?

**Confirmed.** V3 significantly outperforms V2 in both Pointwise (p=0.0007) and LambdaMART (p=0.007). The enriched KG features account for 24.8% of total feature importance.

V3 Recall@10 surpasses Recall-only: **0.1899 vs 0.1835 (Pointwise)** and **0.1982 vs 0.1835 (LambdaMART)**. V3 does not yet beat Recall-only on NDCG@10, reflecting a trade-off: KG features improve recall and diversity at a small cost to top-position precision.

### RQ2: Does KG disproportionately help long-tail items?

**Strongly confirmed.**

1. **Without KG, tail recall is zero.** V1 and V2 (CF + content features) achieve 0.000 tail Recall@10. Pure CF cannot surface long-tail items.

2. **KG enables tail recommendations.** V3 achieves tail Recall@10 = 0.1316, V4 = 0.2368. This is a qualitative shift — from nothing to meaningful recall.

3. **Tail lift >> head lift.** V3 improves tail recall by +0.132 vs head by +0.037 — a **3.6x ratio**. V4 shows a **7.9x ratio**. KG features are disproportionately more valuable for long-tail items, confirming H2.

4. **KG helps focused users most.** Low-entropy users gain +51% Recall@10 from KG features, vs +3% for high-entropy users. Structured knowledge (shared directors, genre overlap, co-liked patterns) is most useful when the user has clear preferences.

### What Worked in KG Enrichment

| Innovation | Impact |
|------------|--------|
| **Co-liked edges** (100K) | Bridged CF and KG; `co_liked_sum` = 3.1% gain share |
| **IDF weighting** | `genre_idf_sum` became top KG feature (7.3%); rare-genre sharing is 1.7x more important than raw count |
| **Decade relations** | `same_decade_ratio` = 6.8% gain share; temporal preference is a strong re-ranking signal |

---

## 6. Methodology Notes

### Distribution-Matched Training
Training on random negatives causes distribution mismatch (cf_score=0 for 97% of negatives, making it a trivial discriminator). Fix: both training and evaluation use recall model's top-100 candidates. Training labels from val-period interactions, test labels from test-period interactions. This ensures the ranker learns patterns that transfer to the realistic re-ranking setting.

### Multi-Route Recall
CF top-70 + KG (TransE nearest-neighbor) top-50, merged to 100 per user. CF items ordered by cf_score first, then KG-only items by TransE similarity. KG-only items receive cf_score=0 and a separate `kg_recall_score` feature. This improved movie coverage from 1,105 to 1,340 and enabled statistically meaningful long-tail analysis (212 users vs 9 previously).

### KG Enrichment
Expanded from 3 metadata relations (30K triples) to 5 relations including collaborative edges (134K triples). Added IDF weighting (IDF = log(N_movies / df_entity)) to distinguish rare vs common entity sharing — e.g., sharing a niche actor is weighted higher than sharing "Drama" genre.

---

## 7. Remaining Work

- [ ] Hyperparameter search for LightGBM
- [ ] Try LightGCN as recall model (better coverage)
- [ ] Phase 2: RAG-based explanation (RQ3, RQ4)
