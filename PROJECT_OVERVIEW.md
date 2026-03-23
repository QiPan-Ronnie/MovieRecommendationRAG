# Project Overview: KG-Enhanced Recommendation with RAG Explanations

> Do Knowledge Graphs and Retrieval-Augmented Generation Improve Recommendation Performance and Explanation Faithfulness?

---

## 1. Research Motivation

Traditional collaborative filtering (CF) recommender systems rely solely on user-item interaction patterns. They suffer from two well-known limitations:

1. **Cold-start and long-tail problem**: Items with few interactions receive poor recommendations, because CF has no signal to work with.
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

**Current scope**: Phase 1 (RQ1 + RQ2) is the immediate focus. Phase 2 (RQ3 + RQ4) will begin after Phase 1 is validated.

---

## 3. System Architecture

The system follows a two-stage **recall + re-rank** pipeline, with optional RAG explanation generation:

```
User Input (history, query)
    |
    v
[Stage 1: Recall] ---- Item-CF / MF / LightGCN
    |                   Generate top-N candidate items with scores
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
| **Matrix Factorization (BPR-MF)** | Model-based CF | User/item embeddings trained with BPR loss; score = dot product of user and item embeddings |
| **LightGCN** | Graph-based CF | Multi-layer graph convolution on user-item bipartite graph; embeddings refined by neighbor aggregation |

All three models:
- Train on the training set only
- Output top-100 candidate items per user with scores
- Are evaluated with the same metrics on the same test set

### 3.2 Stage 2: KG-Enhanced Re-ranking

A LightGBM pointwise ranker takes candidate items from Stage 1 and re-ranks them using enriched features:

| Variant | Features | Purpose |
|---------|----------|---------|
| **V1** | CF score only | Pure CF baseline for ranker |
| **V2** | CF score + content similarity + popularity | CF + content features |
| **V3** | CF score + content similarity + popularity + KG features | Full KG-enhanced model |

**KG features** for each (user, candidate_movie) pair, aggregated over the user's history:

| Feature | Description | Aggregation |
|---------|-------------|-------------|
| `shared_actor_count` | Number of actors shared between candidate and history movies | sum, max |
| `same_director` | Whether candidate shares a director with any history movie | max (binary) |
| `same_genre_count` | Number of genres shared | sum, max |
| `shortest_path_len` | Shortest path in KG between candidate and nearest history movie | min |

**Content similarity**: Cosine similarity between Sentence-Transformer embeddings of candidate movie's metadata (genre + overview) and the user's history movie embeddings.

### 3.3 Stage 3: RAG Explanation (Phase 2)

For each recommended item, generate a natural-language explanation:

1. **Query construction**: User history (top-K liked movies) + candidate movie
2. **Dual retrieval**: Dense (Sentence-Transformer + FAISS) + Sparse (BM25), with hybrid scoring: `score = alpha * dense + beta * sparse`
3. **Evidence filtering**: Deduplicate, remove irrelevant passages, enforce token limit
4. **Prompt design**: Structured prompt requiring explicit evidence citation, prohibiting fabrication
5. **Generation**: Small instruction-tuned LLM (e.g., Llama 2 7B / Qwen)

---

## 4. Datasets

### 4.1 User Interaction Data: MovieLens 1M

| Property | Value |
|----------|-------|
| Users | ~6,040 |
| Movies | ~3,900 |
| Ratings | ~1,000,000 (1-5 scale) |
| Density | ~4.2% |

Used for: training CF/MF/LightGCN, constructing train/test splits, evaluating recommendation metrics.

### 4.2 Movie Metadata: TMDB API

Obtained by mapping MovieLens movie IDs to TMDB via movie title and year matching (ML-1M does not include `links.dat`; that file is only in ML-20M/Latest).

| Field | Description | Used For |
|-------|-------------|----------|
| genres | Movie genres | KG construction, content features |
| actors (top-5) | Lead cast members | KG construction |
| directors | Director(s) | KG construction |
| overview | Plot synopsis | Content similarity, RAG evidence |
| keywords | Thematic tags | RAG evidence |
| vote_average, vote_count | TMDB community ratings | Popularity features |
| release_year | Release year | Metadata |

Caching with checkpoint-resume: each movie's API response is cached to `data/tmdb/cache/{movie_id}.json`, with failed IDs logged to `data/tmdb/failed_ids.txt`.

### 4.3 Knowledge Graph

Constructed from MovieLens genres + TMDB metadata:

| Relation | Example | Source |
|----------|---------|--------|
| `has_genre` | (Toy Story, has_genre, Animation) | MovieLens + TMDB |
| `acted_by` | (Toy Story, acted_by, Tom Hanks) | TMDB (top-5 cast) |
| `directed_by` | (Toy Story, directed_by, John Lasseter) | TMDB |

Stored as: NetworkX undirected graph + CSV triples. Future extension: `collaborated_with` (actor-actor via shared movies), Wikidata entities.

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

| Metric | Definition |
|--------|------------|
| **Hit@K** | Fraction of users whose test items appear in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain at K, position-aware |
| **Recall@K** | Fraction of test items that appear in top-K |
| **MRR** | Mean Reciprocal Rank of first relevant item |
| **Coverage** | Fraction of all items that appear in any user's top-K |

**Statistical significance**: Paired t-test on per-user NDCG@10 between V2 and V3 (p < 0.05).

### 5.2 Long-tail Analysis (RQ2)

- **Definition**: Movies in the bottom 50% by interaction count are "tail"; the rest are "head".
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
| E2 | Remove key evidence sentences | Explanation should degrade (tests dependence on evidence) |
| E3 | Shuffle evidence order | Minor degradation (tests robustness) |
| E4 | Replace with irrelevant evidence | Explanation should change substantially (tests faithfulness) |

### 5.4 Structure vs. Text Analysis (RQ4, Phase 2)

Compare three explanation modes:

| Mode | Knowledge Source | Characteristics |
|------|----------------|-----------------|
| **KG-only** | Graph paths, shared entities | Concise, traceable, logically clear, but potentially shallow |
| **RAG-only** | Retrieved text passages | Natural language, richer, but potentially unfaithful |
| **KG+RAG** | Both | Expected to combine traceability with richness |

Measurable differences: Intra-list diversity, genre coverage, user trust scores.

---

## 6. Team Roles

| Team | Responsibility | Key Outputs |
|------|---------------|-------------|
| **A: Data & Baselines** | MovieLens processing, TMDB data acquisition, train/test split, negative sampling, Item-CF / MF / LightGCN implementation, evaluation metrics | `train.csv`, `test.csv`, baseline metric tables |
| **B: Knowledge Graph** | KG triple construction from TMDB, NetworkX graph, KG feature engineering (shared actors, directors, genres, shortest paths), path analysis and visualization | `triples.csv`, `kg_features.csv`, KG statistics, feature importance plots |
| **C: Ranking & Fusion** | LightGBM ranker (V1/V2/V3 ablation), unified candidate set and negative sampling, long-tail analysis, user interest entropy analysis | Ablation result tables, statistical significance tests, long-tail comparison |
| **D: RAG & Faithfulness** | Text evidence corpus construction (FAISS + BM25), hybrid retrieval, prompt design (prompt-only vs. RAG-enabled), faithfulness perturbation experiments (E1-E4) | Faithfulness score tables, perturbation comparison, example case studies |
| **E: Demo, Evaluation & Report** | Streamlit/Gradio frontend integrating recommendations + explanations + KG subgraph visualization, user study design (questionnaire), final report and presentation | Demo interface, user study results, final paper/PPT |

---

## 7. Deliverables

1. Reproducible code repository with clear documentation
2. Offline metric comparison tables (baselines + ablation)
3. Ablation bar charts (V1 vs V2 vs V3)
4. Faithfulness experiment curves (E1-E4 perturbation results)
5. User study statistics (questionnaire analysis)
6. Interactive demo interface (Streamlit/Gradio)
7. Final report and presentation slides

---

## 8. Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep learning | PyTorch |
| Graph neural network | PyTorch (custom LightGCN, no PyG dependency) |
| Ranker | LightGBM |
| KG storage & query | NetworkX |
| Text embeddings | Sentence-Transformers |
| Vector search | FAISS |
| Sparse search | BM25 (rank_bm25) |
| LLM generation | Llama 2 7B / Qwen (local inference) |
| Frontend | Streamlit or Gradio |
| Evaluation | scikit-learn, SciPy (statistical tests) |
| Visualization | Matplotlib, Seaborn |
| Hardware | 8x V100 GPU (16GB/32GB) |

---

*Document generated: 2026-03-20*
*Current phase: Phase 1 (Data -> KG -> Ablation experiments)*
