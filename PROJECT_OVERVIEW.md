# Research Design: KG-RAG Enhanced Recommendation with Faithful Explanations

> Do Knowledge Graphs and Retrieval-Augmented Generation Improve Recommendation Performance and Explanation Faithfulness?

---

## 1. Research Motivation

Traditional collaborative filtering (CF) recommender systems rely solely on user-item interaction patterns and suffer from two well-known limitations:

1. **Cold-start and long-tail problem**: Items with few interactions receive poor recommendations because CF has no signal to work with.
2. **Lack of explainability**: CF models produce scores but cannot articulate *why* an item is recommended in a way that is both faithful to the model's reasoning and informative to the user.

Knowledge Graphs (KGs) offer structured, human-interpretable relationships (e.g., shared actors, same director, genre overlap) that could address both problems. Meanwhile, Retrieval-Augmented Generation (RAG) provides a mechanism to generate natural-language explanations grounded in retrieved textual evidence, potentially improving explanation faithfulness over prompt-only LLM generation.

This project investigates whether combining KG-enhanced recommendation with RAG-based explanation yields measurable improvements in both recommendation accuracy and explanation quality.

---

## 2. Research Questions and Hypotheses

| ID  | Research Question | Hypothesis |
|-----|-------------------|------------|
| RQ1 | Does KG-enhanced re-ranking significantly outperform CF and content-based baselines on recommendation accuracy? | H1: Adding KG features to the ranker significantly improves NDCG@K and Recall@K. |
| RQ2 | Does KG provide disproportionate benefit for long-tail items and users with concentrated interests? | H2: KG features improve long-tail item recall more than head item recall. |
| RQ3 | Are RAG-generated explanations more faithful to retrieved evidence than prompt-only explanations? | H3: RAG explanations exhibit higher evidence overlap and semantic consistency than prompt-only explanations. |
| RQ4 | Are structured knowledge (KG) and unstructured knowledge (RAG) complementary in explainable recommendation? | H4: Combining KG-based and RAG-based explanations yields better coverage, diversity, and user trust than either alone. |

**Phased execution**: Phase 1 addresses RQ1 + RQ2 (KG-enhanced recommendation). Phase 2 addresses RQ3 + RQ4 (RAG-based explanation).

---

## 3. System Architecture

The system follows a three-stage **recall → re-rank → explain** pipeline:

```
User Input (history, query)
    |
    v
[Stage 1: Recall] ---- Item-CF / BPR-MF / LightGCN
    |                   Generate top-100 candidate items with scores
    v
[Stage 2: Re-rank] --- LightGBM ranker
    |                   Features: CF score + content similarity + KG features
    |                   Output: re-ranked top-K list
    v
[Stage 3: Explain] --- RAG module (Phase 2)
    |                   Retrieve evidence -> Prompt LLM -> Generate explanation
    v
Final Output: Ranked recommendations with explanations
```

### 3.1 Stage 1: Recall Models

Three baseline recall models, each representing a different paradigm:

| Model | Paradigm | Method |
|-------|----------|--------|
| **Item-CF** | Memory-based CF | Item-item cosine similarity on user interaction vectors; score = weighted sum of similarities with user's history |
| **BPR-MF** | Model-based CF | User/item embeddings trained with BPR loss; score = dot product of user and item embeddings |
| **LightGCN** | Graph-based CF | Multi-layer graph convolution on user-item bipartite graph; embeddings refined by neighbor aggregation |

All three models train on the training set only, output top-100 candidate items per user, and are evaluated with the same metrics on the same test set.

### 3.2 Stage 2: KG-Enhanced Re-ranking

A LightGBM ranker takes candidate items from Stage 1 and re-ranks them using enriched features. Two ranking objectives are compared:

- **Pointwise** (binary classification): treats each (user, item) pair independently, predicts P(relevant)
- **LambdaMART** (listwise ranking): optimizes NDCG directly over per-user ranked lists

**Evaluation protocol (Option C — distribution-matched training)**:

A critical design choice is that both training and evaluation use the recall model's top-100 candidates as the candidate pool. Training on random negatives creates a distribution mismatch: random negatives have cf_score = 0 (trivially distinguishable), causing the model to learn shortcuts (e.g., popularity bias) that fail on actual recall candidates. Distribution-matched training resolves this by ensuring the ranker sees realistic hard negatives during training.

- **Ranker training**: recall model's top-100 candidates, labeled with validation-period interactions (80% of users)
- **Ranker validation**: same candidates, remaining 20% of users (for early stopping)
- **Ranker evaluation**: recall model's top-100 candidates, labeled with test-period interactions
- **Feature computation**: all features (cf_score, content similarity, popularity, KG features) use training-period data only — no leakage

**Ablation variants**:

| Variant | Features | Purpose |
|---------|----------|---------|
| **V1** | CF score only | Pure CF baseline for ranker |
| **V2** | CF score + content similarity + popularity + vote_count | CF + content features |
| **V3** | CF score + content similarity + popularity + vote_count + KG features | Full KG-enhanced model |

**KG features** for each (user, candidate_movie) pair, aggregated over the user's training history:

| Feature | Description | Aggregation |
|---------|-------------|-------------|
| `shared_actor_count` | Number of actors shared | sum, max |
| `shared_actor_idf` | IDF-weighted shared actors (rare actors weighted higher) | sum |
| `same_director` / `same_director_idf` | Shared director (binary + IDF-weighted) | max |
| `same_genre_count` / `same_genre_idf` | Shared genres (count + IDF-weighted) | sum, max |
| `co_liked_sum` | Number of history movies co-liked with candidate (collaborative KG) | sum |
| `same_decade_ratio` | Fraction of history movies in the same decade as candidate | ratio |

**IDF weighting**: Sharing a rare entity (e.g., niche actor appearing in 2 movies) is weighted higher than sharing a common entity (e.g., "Drama" genre appearing in 50% of movies). IDF = log(N_movies / df_entity).

**Content similarity**: Cosine similarity between Sentence-Transformer embeddings of candidate movie's metadata (genre + overview) and the mean embedding of user's history movies.

### 3.3 Stage 3: RAG Explanation (Phase 2)

For each recommended item, generate a natural-language explanation:

1. **Query construction**: User history (top-K liked movies) + candidate movie
2. **Dual retrieval**: Dense (Sentence-Transformer + FAISS) + Sparse (BM25), with hybrid scoring
3. **Evidence filtering**: Deduplicate, remove irrelevant passages, enforce token limit
4. **Prompt design**: Structured prompt requiring explicit evidence citation, prohibiting fabrication
5. **Generation**: Small instruction-tuned LLM (e.g., Llama 2 7B / Qwen)

---

## 4. Datasets

### 4.1 User Interaction Data: MovieLens 1M

| Property | Value |
|----------|-------|
| Users (after filtering) | 5,950 |
| Movies (after filtering) | 3,125 |
| Positive interactions (rating >= 4) | 573,726 |
| Train / Val / Test | 398,867 / 54,680 / 120,179 |

**Data processing**: Only ratings >= 4 are treated as positive interactions. Per-user time-based split: train (70%) / validation (10%) / test (20%). Minimum interaction filter (iterative): users >= 10 positive interactions, items >= 5 positive interactions.

### 4.2 Movie Metadata: TMDB API

| Field | Description | Used For |
|-------|-------------|----------|
| genres | Movie genres | KG construction, content features |
| actors (top-5) | Lead cast members | KG construction |
| directors | Director(s) | KG construction |
| overview | Plot synopsis | Content similarity, RAG evidence |
| keywords | Thematic tags | RAG evidence |
| vote_average, vote_count | TMDB community ratings | Popularity features |

### 4.3 Knowledge Graph

Constructed from TMDB metadata + user co-interaction patterns + temporal data:

| Relation | Count | Source | Description |
|----------|-------|--------|-------------|
| `co_liked` | 100,000 | Training data | Movies co-liked by ≥10 users (collaborative edges) |
| `acted_by` | 18,044 | TMDB cast | (Toy Story, acted_by, Tom Hanks) |
| `has_genre` | 8,871 | TMDB genres | (Toy Story, has_genre, Animation) |
| `directed_by` | 3,880 | TMDB crew | (Toy Story, directed_by, John Lasseter) |
| `released_in_decade` | 3,652 | TMDB year | (Toy Story, released_in_decade, 1990s) |
| **Total** | **134,447** | | |

The collaborative `co_liked` edges bridge CF signal into the KG, enabling TransE embeddings to capture both structural metadata and user behavior patterns.

### 4.4 RAG Evidence Corpus (Phase 2)

| Source | Content |
|--------|---------|
| TMDB overview | Plot synopsis per movie |
| TMDB tagline + keywords | Short descriptive phrases |
| IMDb plot summary (optional) | More detailed plot descriptions |

Processing: sentence-level segmentation -> deduplication -> Sentence-Transformer embedding -> FAISS index + BM25 index.

---

## 5. Evaluation

### 5.1 Recommendation Metrics (RQ1, RQ2)

All metrics are computed on the recall model's top-100 candidates per user (Option C evaluation).

| Metric | Definition |
|--------|------------|
| **Hit@K** | Fraction of users whose test items appear in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain at K, position-aware |
| **Recall@K** | Fraction of test items that appear in top-K |
| **MRR** | Mean Reciprocal Rank of first relevant item |
| **Coverage** | Fraction of all items that appear in any user's top-K |

**Statistical significance**: Paired t-test on per-user NDCG@10 (p < 0.05).

### 5.2 Long-tail Analysis (RQ2)

- **Definition**: Movies in the bottom 50% by training set interaction count are "tail"; the rest are "head".
- **Metrics**: Recall@10 computed separately for head and tail items.
- **User interest entropy**: Users bucketed by genre entropy; analyze whether KG benefits concentrated-interest users more.

### 5.3 RAG Faithfulness Metrics (RQ3, Phase 2)

| Metric | Description |
|--------|------------|
| **Evidence Overlap Score** | Token/phrase overlap between generated explanation and retrieved evidence |
| **Semantic Similarity** | Embedding cosine similarity between explanation and evidence |
| **Human Judgment (1-5)** | Human raters assess faithfulness, informativeness, relevance |

**Perturbation experiments** (4 conditions):

| Condition | Operation | Expected Outcome |
|-----------|-----------|-----------------|
| E1 | Original evidence | Baseline explanation quality |
| E2 | Remove key evidence sentences | Explanation should degrade |
| E3 | Shuffle evidence order | Minor degradation |
| E4 | Replace with irrelevant evidence | Explanation should change substantially |

### 5.4 Structure vs. Text Analysis (RQ4, Phase 2)

Compare three explanation modes:

| Mode | Knowledge Source | Characteristics |
|------|----------------|-----------------|
| **KG-only** | Graph paths, shared entities | Concise, traceable, logically clear |
| **RAG-only** | Retrieved text passages | Natural language, richer |
| **KG+RAG** | Both | Expected to combine traceability with richness |

---

## 6. Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Graph Neural Network | PyTorch (custom LightGCN) |
| Ranker | LightGBM |
| KG Storage & Query | NetworkX |
| Text Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (Phase 2) |
| Sparse Search | BM25 (Phase 2) |
| LLM Generation | Llama 2 7B / Qwen (Phase 2) |
| Frontend | Streamlit |
| Evaluation | scikit-learn, SciPy |
