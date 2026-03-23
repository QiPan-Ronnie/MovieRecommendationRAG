"""
Compute content similarity between user history and candidate movies
using Sentence-Transformer embeddings.

For each (user, candidate) pair:
  content_similarity = mean cosine_similarity(candidate_embedding, history_embeddings)

User history is always derived from TRAINING set only.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def compute_movie_embeddings(tmdb_path="data/tmdb/tmdb_metadata.csv",
                             movies_path="data/processed/movies.csv",
                             output_path="data/processed/movie_embeddings.npy",
                             model_name="all-MiniLM-L6-v2"):
    """
    Encode each movie as a text embedding using Sentence-Transformer.
    Text = genres + " " + overview (from TMDB).
    """
    from sentence_transformers import SentenceTransformer

    # Load metadata
    movies_df = pd.read_csv(movies_path)
    tmdb_df = None
    if os.path.exists(tmdb_path):
        tmdb_df = pd.read_csv(tmdb_path)

    # Build text for each movie
    movie_texts = {}
    for _, row in movies_df.iterrows():
        mid = row["movie_id"]
        text_parts = []

        # Genres from MovieLens
        if pd.notna(row.get("genres")):
            text_parts.append(str(row["genres"]).replace("|", ", "))

        # Overview from TMDB
        if tmdb_df is not None:
            tmdb_row = tmdb_df[tmdb_df["movie_id"] == mid]
            if len(tmdb_row) > 0:
                overview = tmdb_row.iloc[0].get("overview", "")
                if pd.notna(overview) and str(overview) != "nan":
                    text_parts.append(str(overview))

        movie_texts[mid] = " ".join(text_parts) if text_parts else "unknown movie"

    # Sort by movie_id for consistent ordering
    movie_ids = sorted(movie_texts.keys())
    texts = [movie_texts[mid] for mid in movie_ids]

    print(f"Encoding {len(texts)} movies with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
    embeddings = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Save
    np.save(output_path, embeddings)
    movie_id_path = output_path.replace(".npy", "_ids.npy")
    np.save(movie_id_path, np.array(movie_ids))

    print(f"Embeddings saved to {output_path} (shape: {embeddings.shape})")
    print(f"Movie IDs saved to {movie_id_path}")
    return embeddings, movie_ids


def compute_content_sim_for_split(split_with_neg_path, train_path,
                                  embeddings, movie_ids, output_path,
                                  max_history=20):
    """
    Compute content similarity for all (user, candidate) pairs in a split.

    content_similarity = mean cosine_similarity(candidate_emb, history_emb)
    where history comes from TRAINING set only.
    """
    # Build movie_id -> embedding index
    mid_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    # Build user history from training set
    train_df = pd.read_csv(train_path)
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values("timestamp")

    user_history = defaultdict(list)
    for _, row in train_df.iterrows():
        uid, mid = row["user_id"], row["movie_id"]
        if mid in mid_to_idx:
            user_history[uid].append(mid)

    for uid in user_history:
        user_history[uid] = user_history[uid][-max_history:]

    # Pre-compute user profile embeddings (mean of history embeddings)
    user_profiles = {}
    for uid, history in user_history.items():
        indices = [mid_to_idx[mid] for mid in history if mid in mid_to_idx]
        if indices:
            user_profiles[uid] = embeddings[indices].mean(axis=0)

    # Compute similarity for each (user, candidate) pair
    split_df = pd.read_csv(split_with_neg_path)
    print(f"Computing content similarity for {split_with_neg_path} ({len(split_df)} pairs)...")

    similarities = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Content Sim"):
        uid, mid = row["user_id"], row["movie_id"]
        sim = 0.0
        if uid in user_profiles and mid in mid_to_idx:
            candidate_emb = embeddings[mid_to_idx[mid]]
            sim = float(np.dot(user_profiles[uid], candidate_emb))
        similarities.append({
            "user_id": uid,
            "movie_id": mid,
            "content_similarity": sim
        })

    result_df = pd.DataFrame(similarities)
    result_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Mean similarity: {result_df['content_similarity'].mean():.4f}")
    print(f"  Nonzero: {(result_df['content_similarity'] != 0).sum()} "
          f"({100*(result_df['content_similarity'] != 0).mean():.1f}%)")
    return result_df


def main():
    # Step 1: Compute movie embeddings
    emb_path = "data/processed/movie_embeddings.npy"
    ids_path = "data/processed/movie_embeddings_ids.npy"

    if os.path.exists(emb_path) and os.path.exists(ids_path):
        print("Loading cached embeddings...")
        embeddings = np.load(emb_path)
        movie_ids = np.load(ids_path).tolist()
    else:
        embeddings, movie_ids = compute_movie_embeddings()

    # Step 2: Compute content similarity for each split
    train_path = "data/processed/train.csv"
    splits = [
        ("data/processed/train_with_neg.csv", "data/processed/content_sim_train.csv"),
        ("data/processed/val_with_neg.csv", "data/processed/content_sim_val.csv"),
        ("data/processed/test_with_neg.csv", "data/processed/content_sim_test.csv"),
    ]

    for split_path, output_path in splits:
        if os.path.exists(split_path):
            compute_content_sim_for_split(
                split_path, train_path, embeddings, movie_ids, output_path
            )
        else:
            print(f"  Skipping {split_path} (not found)")


if __name__ == "__main__":
    main()
