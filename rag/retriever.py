"""
Hybrid retriever: Dense (FAISS) + Sparse (BM25) dual-path retrieval.

For a given query (user history + candidate movie info), retrieve the most
relevant evidence chunks from the RAG corpus using a weighted combination
of dense cosine similarity and BM25 sparse scores.

score = alpha * dense_score + (1 - alpha) * normalized_bm25_score
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict


class HybridRetriever:
    """
    Dual-path retriever combining FAISS (dense) and BM25 (sparse) search.
    """

    def __init__(self, corpus_dir: str = "data/rag",
                 model_name: str = "all-MiniLM-L6-v2",
                 alpha: float = 0.6):
        """
        Args:
            corpus_dir: directory containing corpus.jsonl, faiss_index.bin,
                        bm25_index.pkl, doc_id_to_movie.json
            model_name: Sentence-Transformer model for query encoding
            alpha: weight for dense score (1-alpha for BM25)
        """
        self.corpus_dir = corpus_dir
        self.alpha = alpha
        self._model = None
        self._model_name = model_name

        # Load corpus
        self.corpus = []
        corpus_path = os.path.join(corpus_dir, "corpus.jsonl")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                self.corpus.append(json.loads(line))

        # Load doc_id → movie_id mapping
        with open(os.path.join(corpus_dir, "doc_id_to_movie.json"), "r") as f:
            self.doc_to_movie = {int(k): v for k, v in json.load(f).items()}

        # Build movie_id → doc_ids reverse mapping
        self.movie_to_docs = defaultdict(list)
        for doc_id, mid in self.doc_to_movie.items():
            self.movie_to_docs[mid].append(doc_id)

        # Load FAISS index
        import faiss
        self.faiss_index = faiss.read_index(
            os.path.join(corpus_dir, "faiss_index.bin"))

        # Load BM25 index
        with open(os.path.join(corpus_dir, "bm25_index.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)

        print(f"HybridRetriever loaded: {len(self.corpus)} docs, "
              f"alpha={alpha}")

    @property
    def encoder(self):
        """Lazy-load sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into a normalized dense vector."""
        emb = self.encoder.encode([query], normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 10,
                 candidate_movie_id: int = None,
                 history_movie_ids: list[int] = None) -> list[dict]:
        """
        Retrieve top_k evidence chunks for the given query.

        Args:
            query: natural language query describing user preference + candidate
            top_k: number of results to return
            candidate_movie_id: if set, boost docs about this movie
            history_movie_ids: if set, include docs about user history movies

        Returns:
            List of dicts: {"doc_id", "movie_id", "text", "source",
                            "dense_score", "bm25_score", "score"}
        """
        # Determine how many candidates to consider from each path
        fetch_k = min(top_k * 5, len(self.corpus))

        # --- Dense retrieval ---
        query_vec = self._encode_query(query)
        dense_scores_arr, dense_ids = self.faiss_index.search(query_vec, fetch_k)
        dense_scores = {}
        for score, doc_id in zip(dense_scores_arr[0], dense_ids[0]):
            if doc_id >= 0:
                dense_scores[int(doc_id)] = float(score)

        # --- Sparse retrieval (BM25) ---
        tokenized_query = query.lower().split()
        bm25_raw = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
        bm25_norm = bm25_raw / bm25_max

        # Get top BM25 doc ids
        bm25_top_ids = np.argsort(bm25_norm)[-fetch_k:][::-1]
        sparse_scores = {}
        for doc_id in bm25_top_ids:
            if bm25_norm[doc_id] > 0:
                sparse_scores[int(doc_id)] = float(bm25_norm[doc_id])

        # --- Merge candidates ---
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        # Optionally include docs about the candidate movie and history
        if candidate_movie_id and candidate_movie_id in self.movie_to_docs:
            all_doc_ids.update(self.movie_to_docs[candidate_movie_id])
        if history_movie_ids:
            for mid in history_movie_ids[:5]:  # limit to top-5 history
                if mid in self.movie_to_docs:
                    all_doc_ids.update(self.movie_to_docs[mid])

        # Score all candidates
        results = []
        for doc_id in all_doc_ids:
            d_score = dense_scores.get(doc_id, 0.0)
            s_score = sparse_scores.get(doc_id, 0.0)
            # If not in dense/sparse top-k, compute on the fly
            if doc_id not in dense_scores:
                doc_text = self.corpus[doc_id]["text"]
                doc_emb = self.encoder.encode([doc_text],
                                              normalize_embeddings=True)
                d_score = float(np.dot(query_vec[0], doc_emb[0]))
            if doc_id not in sparse_scores:
                s_score = float(bm25_norm[doc_id])

            combined = self.alpha * d_score + (1 - self.alpha) * s_score
            results.append({
                "doc_id": doc_id,
                "movie_id": self.corpus[doc_id]["movie_id"],
                "text": self.corpus[doc_id]["text"],
                "source": self.corpus[doc_id]["source"],
                "dense_score": round(d_score, 4),
                "bm25_score": round(s_score, 4),
                "score": round(combined, 4),
            })

        # Sort by combined score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate (same text)
        seen_texts = set()
        unique_results = []
        for r in results:
            if r["text"] not in seen_texts:
                seen_texts.add(r["text"])
                unique_results.append(r)

        return unique_results[:top_k]


def build_query(candidate_title: str, candidate_genres: str,
                history_titles: list[str], max_history: int = 5) -> str:
    """
    Construct a retrieval query from user history and candidate movie info.

    Args:
        candidate_title: title of the recommended movie
        candidate_genres: genres of the candidate (pipe-separated)
        history_titles: list of movie titles the user liked
        max_history: max number of history titles to include
    """
    history_str = ", ".join(history_titles[:max_history])
    genres_str = candidate_genres.replace("|", ", ") if candidate_genres else ""

    query = (f"Why should someone who enjoyed {history_str} "
             f"watch {candidate_title}? "
             f"The movie is a {genres_str} film." if genres_str
             else f"Why should someone who enjoyed {history_str} "
                  f"watch {candidate_title}?")
    return query
