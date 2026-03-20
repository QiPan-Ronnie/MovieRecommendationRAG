"""
Item-based Collaborative Filtering.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import evaluate_all, print_results


def build_interaction_matrix(train_df, user2idx, movie2idx):
    """Build sparse user-item interaction matrix."""
    rows = [user2idx[u] for u in train_df["user_id"]]
    cols = [movie2idx[m] for m in train_df["movie_id"]]
    vals = np.ones(len(rows))
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user2idx), len(movie2idx)))
    return matrix


def train_item_cf(train_df, top_k_sim=50):
    """
    Train Item-CF model.
    Returns: item similarity matrix (sparse, top-k per item).
    """
    # Build mappings
    all_users = sorted(train_df["user_id"].unique())
    all_movies = sorted(train_df["movie_id"].unique())
    user2idx = {u: i for i, u in enumerate(all_users)}
    movie2idx = {m: i for i, m in enumerate(all_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    # Build interaction matrix (users x items)
    ui_matrix = build_interaction_matrix(train_df, user2idx, movie2idx)

    # Compute item-item cosine similarity (items x items)
    print("Computing item-item similarity...")
    item_matrix = ui_matrix.T  # items x users
    sim_matrix = cosine_similarity(item_matrix, dense_output=False)

    # For efficiency, keep only top-k similar items per item
    print(f"Keeping top-{top_k_sim} similar items per item...")
    n_items = sim_matrix.shape[0]

    return {
        "sim_matrix": sim_matrix,
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "idx2movie": idx2movie,
        "ui_matrix": ui_matrix,
        "all_movies": all_movies,
    }


def predict_item_cf(model, user_id, train_user_items, n_candidates=100):
    """
    Predict scores for all items for a given user.
    Returns: list of (movie_id, score) sorted by score descending.
    """
    user2idx = model["user2idx"]
    movie2idx = model["movie2idx"]
    idx2movie = model["idx2movie"]
    sim_matrix = model["sim_matrix"]

    if user_id not in user2idx:
        return []

    # Get user's interacted item indices
    interacted_indices = [movie2idx[m] for m in train_user_items if m in movie2idx]
    if not interacted_indices:
        return []

    # Score each candidate item
    scores = {}
    interacted_set = set(interacted_indices)

    for item_idx in range(sim_matrix.shape[0]):
        if item_idx in interacted_set:
            continue
        # Score = sum of similarities with user's interacted items
        sim_scores = sim_matrix[item_idx, interacted_indices].toarray().flatten()
        scores[item_idx] = np.sum(sim_scores)

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_candidates]
    return [(idx2movie[idx], score) for idx, score in ranked]


def run_item_cf(train_path="data/processed/train.csv", test_path="data/processed/test.csv",
                output_path="results/cf_scores.csv", k=10):
    """Run full Item-CF pipeline."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Training Item-CF...")
    model = train_item_cf(train_df)

    # Build user -> train items mapping
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row["user_id"]].add(row["movie_id"])

    # Build ground truth
    ground_truth = defaultdict(set)
    for _, row in test_df.iterrows():
        ground_truth[row["user_id"]].add(row["movie_id"])

    # Predict for all test users
    print("Generating predictions...")
    predictions = {}
    all_scores = []
    test_users = list(ground_truth.keys())

    for i, uid in enumerate(test_users):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(test_users)} users")
        ranked = predict_item_cf(model, uid, train_user_items[uid])
        predictions[uid] = [movie_id for movie_id, _ in ranked]
        for movie_id, score in ranked:
            all_scores.append({"user_id": uid, "movie_id": movie_id, "cf_score": score})

    # Save scores
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

    # Evaluate
    total_items = len(model["all_movies"])
    results, per_user_ndcg = evaluate_all(predictions, ground_truth, k=k, total_items=total_items)
    print_results(results, "Item-CF")

    return results, per_user_ndcg


if __name__ == "__main__":
    run_item_cf()
