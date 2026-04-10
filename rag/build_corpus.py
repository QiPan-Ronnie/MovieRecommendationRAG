"""
Build the RAG evidence corpus from TMDB metadata.

Pipeline:
  1. Load TMDB metadata (overview, tagline, keywords, genres, actors, directors)
  2. For each movie, produce sentence-level text chunks (documents)
  3. Encode chunks with Sentence-Transformer → dense embeddings
  4. Build FAISS index (dense retrieval) + BM25 index (sparse retrieval)
  5. Save corpus, indices, and mapping files

Output files (in data/rag/):
  - corpus.jsonl            : each line = {"doc_id", "movie_id", "text", "source"}
  - faiss_index.bin         : FAISS flat inner-product index
  - corpus_embeddings.npy   : dense embeddings matrix (N x D)
  - bm25_index.pkl          : serialized BM25 index
  - doc_id_to_movie.json    : mapping from doc_id to movie_id
"""
import os
import json
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: Build text chunks from TMDB metadata
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex heuristics."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Merge very short fragments (< 20 chars) with the previous sentence
    merged = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if merged and len(s) < 20:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def build_text_chunks(tmdb_path: str = "data/tmdb/tmdb_metadata.csv",
                      movies_path: str = "data/processed/movies.csv") -> list[dict]:
    """
    Build sentence-level text chunks from TMDB metadata.
    Each chunk is a dict: {"movie_id", "text", "source"}.
    Sources: overview, tagline, genre_desc, cast_desc
    """
    tmdb_df = pd.read_csv(tmdb_path)
    movies_df = pd.read_csv(movies_path)

    # Map movie_id -> title for richer context
    mid_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))

    chunks = []

    for _, row in tqdm(tmdb_df.iterrows(), total=len(tmdb_df), desc="Building chunks"):
        mid = row["movie_id"]
        title = mid_to_title.get(mid, row.get("title", f"Movie {mid}"))

        # 1) Overview sentences
        overview = str(row.get("overview", ""))
        if overview and overview != "nan":
            for sent in _split_sentences(overview):
                chunks.append({
                    "movie_id": mid,
                    "text": f"{title}: {sent}",
                    "source": "overview",
                })

        # 2) Tagline (short, keep as single chunk)
        tagline = str(row.get("tagline", ""))
        if tagline and tagline != "nan" and len(tagline.strip()) > 3:
            chunks.append({
                "movie_id": mid,
                "text": f"{title} — \"{tagline.strip()}\"",
                "source": "tagline",
            })

        # 3) Genre description
        genres = str(row.get("genres", ""))
        if genres and genres != "nan":
            genre_list = genres.replace("|", ", ")
            chunks.append({
                "movie_id": mid,
                "text": f"{title} is a {genre_list} film.",
                "source": "genre",
            })

        # 4) Cast / director description
        actors = str(row.get("actors", ""))
        directors = str(row.get("directors", ""))
        if actors and actors != "nan":
            actor_list = actors.replace("|", ", ")
            desc = f"{title} stars {actor_list}"
            if directors and directors != "nan":
                desc += f", directed by {directors.replace('|', ', ')}"
            desc += "."
            chunks.append({
                "movie_id": mid,
                "text": desc,
                "source": "cast",
            })
        elif directors and directors != "nan":
            chunks.append({
                "movie_id": mid,
                "text": f"{title} is directed by {directors.replace('|', ', ')}.",
                "source": "cast",
            })

    # Deduplicate by exact text
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique_chunks.append(c)

    print(f"Built {len(unique_chunks)} unique text chunks "
          f"from {tmdb_df['movie_id'].nunique()} movies")
    return unique_chunks


# ---------------------------------------------------------------------------
# Step 2: Encode chunks → dense embeddings + build FAISS index
# ---------------------------------------------------------------------------

def build_dense_index(chunks: list[dict],
                      model_name: str = "all-MiniLM-L6-v2",
                      output_dir: str = "data/rag"):
    """
    Encode all chunks with Sentence-Transformer and build a FAISS index.
    Returns (embeddings, faiss_index).
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    texts = [c["text"] for c in chunks]

    print(f"Encoding {len(texts)} chunks with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS inner-product index (cosine similarity on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    np.save(os.path.join(output_dir, "corpus_embeddings.npy"), embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return embeddings, index


# ---------------------------------------------------------------------------
# Step 3: Build BM25 sparse index
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[dict], output_dir: str = "data/rag"):
    """Build a BM25 index from chunk texts."""
    from rank_bm25 import BM25Okapi

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 index built: {len(tokenized)} documents")
    return bm25


# ---------------------------------------------------------------------------
# Step 4: Save corpus and mappings
# ---------------------------------------------------------------------------

def save_corpus(chunks: list[dict], output_dir: str = "data/rag"):
    """Save corpus as JSONL and create doc_id → movie_id mapping."""
    os.makedirs(output_dir, exist_ok=True)

    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    mapping = {}

    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id, chunk in enumerate(chunks):
            record = {"doc_id": doc_id, **chunk}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            mapping[doc_id] = chunk["movie_id"]

    with open(os.path.join(output_dir, "doc_id_to_movie.json"), "w") as f:
        json.dump(mapping, f)

    print(f"Corpus saved: {len(chunks)} documents -> {corpus_path}")
    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tmdb_path: str = "data/tmdb/tmdb_metadata.csv",
         movies_path: str = "data/processed/movies.csv",
         output_dir: str = "data/rag",
         model_name: str = "all-MiniLM-L6-v2"):
    """Build the full RAG evidence corpus and indices."""

    # Step 1: Build text chunks
    chunks = build_text_chunks(tmdb_path, movies_path)

    # Step 2: Save corpus
    save_corpus(chunks, output_dir)

    # Step 3: Build dense index (FAISS)
    build_dense_index(chunks, model_name, output_dir)

    # Step 4: Build sparse index (BM25)
    build_bm25_index(chunks, output_dir)

    print(f"\nRAG corpus ready at {output_dir}/")
    print(f"  - corpus.jsonl ({len(chunks)} docs)")
    print(f"  - faiss_index.bin")
    print(f"  - corpus_embeddings.npy")
    print(f"  - bm25_index.pkl")
    print(f"  - doc_id_to_movie.json")


if __name__ == "__main__":
    main()
