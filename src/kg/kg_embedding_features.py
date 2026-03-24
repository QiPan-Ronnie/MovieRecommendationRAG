"""
Compute KG embedding-based features for (user, candidate_movie) pairs.

Uses TransE entity embeddings to compute similarity between candidate movies
and user history in the KG embedding space.

User history is always derived from TRAINING set only.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def load_transe_embeddings(
    emb_path="data/kg/transe_entity_emb.npy",
    entity2id_path="data/kg/entity2id.csv",
):
    """Load TransE entity embeddings and entity-to-index mapping."""
    embeddings = np.load(emb_path)
    entity2id_df = pd.read_csv(entity2id_path)
    entity2id = dict(zip(entity2id_df["entity"], entity2id_df["entity_id"]))

    # Build movie_id -> embedding index (movie nodes are "movie_{id}")
    movie2idx = {}
    for entity, idx in entity2id.items():
        if entity.startswith("movie_"):
            try:
                movie_id = int(entity.replace("movie_", ""))
                movie2idx[movie_id] = idx
            except ValueError:
                pass

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_normed = embeddings / norms

    print(f"TransE embeddings: {embeddings.shape}, movies mapped: {len(movie2idx)}")
    return embeddings, emb_normed, movie2idx


def compute_emb_features_for_split(
    embeddings, emb_normed, movie2idx, user_history,
    split_path, output_path, max_history=20,
):
    """
    Compute KG embedding features for all (user, candidate) pairs in a split.

    Features:
      - kg_emb_mean_dist: L2 distance between candidate and mean of user history embeddings
      - kg_emb_mean_cos: cosine similarity between candidate and mean of user history embeddings
      - kg_emb_min_dist: min L2 distance between candidate and any single history movie
      - kg_emb_max_cos: max cosine similarity between candidate and any single history movie
    """
    split_df = pd.read_csv(split_path)
    print(f"Computing KG embedding features for {split_path} ({len(split_df)} pairs)...")

    dim = embeddings.shape[1]

    # Pre-compute user profile embeddings
    user_profiles = {}       # mean of raw embeddings
    user_profiles_norm = {}  # mean of normalized embeddings (for cosine)
    user_history_embs = {}   # list of raw embeddings per user

    for uid, history in user_history.items():
        valid = [movie2idx[mid] for mid in history[-max_history:] if mid in movie2idx]
        if valid:
            embs = embeddings[valid]
            user_profiles[uid] = embs.mean(axis=0)
            embs_n = emb_normed[valid]
            user_profiles_norm[uid] = embs_n.mean(axis=0)
            # Normalize the mean profile for cosine
            norm = np.linalg.norm(user_profiles_norm[uid])
            if norm > 0:
                user_profiles_norm[uid] /= norm
            user_history_embs[uid] = embs

    records = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="KG Emb Features"):
        uid, mid = row["user_id"], row["movie_id"]

        feat = {
            "user_id": uid,
            "movie_id": mid,
            "kg_emb_mean_dist": 0.0,
            "kg_emb_mean_cos": 0.0,
            "kg_emb_min_dist": 0.0,
            "kg_emb_max_cos": 0.0,
        }

        if uid in user_profiles and mid in movie2idx:
            cand_emb = embeddings[movie2idx[mid]]
            cand_emb_n = emb_normed[movie2idx[mid]]

            # Mean distance/similarity
            feat["kg_emb_mean_dist"] = float(np.linalg.norm(cand_emb - user_profiles[uid]))
            feat["kg_emb_mean_cos"] = float(np.dot(user_profiles_norm[uid], cand_emb_n))

            # Min distance / max cosine to any single history movie
            hist_embs = user_history_embs[uid]
            dists = np.linalg.norm(hist_embs - cand_emb, axis=1)
            feat["kg_emb_min_dist"] = float(dists.min())

            hist_embs_n = emb_normed[[movie2idx[m] for m in user_history[uid][-max_history:] if m in movie2idx]]
            cosines = hist_embs_n @ cand_emb_n
            feat["kg_emb_max_cos"] = float(cosines.max())

        records.append(feat)

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    for col in ["kg_emb_mean_dist", "kg_emb_mean_cos", "kg_emb_min_dist", "kg_emb_max_cos"]:
        nonzero = (result_df[col] != 0).sum()
        print(f"  {col}: mean={result_df[col].mean():.4f}, nonzero={nonzero} ({nonzero/len(result_df)*100:.1f}%)")

    return result_df


def build_train_user_history(train_path="data/processed/train.csv", max_history=20):
    """Build user history from TRAINING set only."""
    train_df = pd.read_csv(train_path)
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values("timestamp")

    user_history = defaultdict(list)
    for _, row in train_df.iterrows():
        user_history[row["user_id"]].append(row["movie_id"])

    for uid in user_history:
        user_history[uid] = user_history[uid][-max_history:]

    return user_history


def main():
    embeddings, emb_normed, movie2idx = load_transe_embeddings()
    user_history = build_train_user_history()

    splits = [
        ("data/processed/train_with_neg.csv", "data/kg/kg_emb_features_train.csv"),
        ("data/processed/val_with_neg.csv", "data/kg/kg_emb_features_val.csv"),
        ("data/processed/test_with_neg.csv", "data/kg/kg_emb_features_test.csv"),
        ("data/processed/test_recall_candidates.csv", "data/kg/kg_emb_features_test_recall.csv"),
    ]

    for split_path, output_path in splits:
        if os.path.exists(split_path):
            compute_emb_features_for_split(
                embeddings, emb_normed, movie2idx, user_history,
                split_path, output_path,
            )
        else:
            print(f"  Skipping {split_path} (not found)")


if __name__ == "__main__":
    main()
