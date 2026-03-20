"""
Compute KG-based features for (user, candidate_movie) pairs.
Features: shared_actor_count, same_director, same_genre_count, shortest_path_len
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_kg_graph(path="data/kg/kg_graph.pkl"):
    """Load the NetworkX KG graph."""
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def get_movie_neighbors(G, movie_node, relation):
    """Get neighbors of a movie node by relation type."""
    neighbors = set()
    if movie_node not in G:
        return neighbors
    for neighbor in G.neighbors(movie_node):
        edge_data = G.edges[movie_node, neighbor]
        if edge_data.get("relation") == relation:
            neighbors.add(neighbor)
    return neighbors


def compute_pairwise_features(G, movie_a_node, movie_b_node, max_path_len=4):
    """
    Compute KG features between two movies.
    Returns dict of features.
    """
    features = {
        "shared_actor_count": 0,
        "same_director": 0,
        "same_genre_count": 0,
        "shortest_path_len": max_path_len + 1,  # default: unreachable
    }

    if movie_a_node not in G or movie_b_node not in G:
        return features

    # Shared actors
    actors_a = get_movie_neighbors(G, movie_a_node, "acted_by")
    actors_b = get_movie_neighbors(G, movie_b_node, "acted_by")
    features["shared_actor_count"] = len(actors_a & actors_b)

    # Same director
    directors_a = get_movie_neighbors(G, movie_a_node, "directed_by")
    directors_b = get_movie_neighbors(G, movie_b_node, "directed_by")
    features["same_director"] = 1 if len(directors_a & directors_b) > 0 else 0

    # Shared genres
    genres_a = get_movie_neighbors(G, movie_a_node, "has_genre")
    genres_b = get_movie_neighbors(G, movie_b_node, "has_genre")
    features["same_genre_count"] = len(genres_a & genres_b)

    # Shortest path
    try:
        path_len = nx.shortest_path_length(G, movie_a_node, movie_b_node)
        features["shortest_path_len"] = min(path_len, max_path_len + 1)
    except nx.NetworkXNoPath:
        pass
    except nx.NodeNotFound:
        pass

    return features


def compute_user_candidate_features(G, user_history_movies, candidate_movie_id, max_path_len=4):
    """
    Compute aggregated KG features for a (user, candidate) pair.
    Aggregates over user's history movies.
    """
    candidate_node = f"movie_{candidate_movie_id}"

    if not user_history_movies:
        return {
            "kg_shared_actor_count_sum": 0,
            "kg_shared_actor_count_max": 0,
            "kg_same_director_max": 0,
            "kg_same_genre_count_sum": 0,
            "kg_same_genre_count_max": 0,
            "kg_shortest_path_min": max_path_len + 1,
        }

    all_features = []
    for hist_movie_id in user_history_movies:
        hist_node = f"movie_{hist_movie_id}"
        feat = compute_pairwise_features(G, hist_node, candidate_node, max_path_len)
        all_features.append(feat)

    # Aggregate
    result = {
        "kg_shared_actor_count_sum": sum(f["shared_actor_count"] for f in all_features),
        "kg_shared_actor_count_max": max(f["shared_actor_count"] for f in all_features),
        "kg_same_director_max": max(f["same_director"] for f in all_features),
        "kg_same_genre_count_sum": sum(f["same_genre_count"] for f in all_features),
        "kg_same_genre_count_max": max(f["same_genre_count"] for f in all_features),
        "kg_shortest_path_min": min(f["shortest_path_len"] for f in all_features),
    }
    return result


def main():
    """Compute KG features for all (user, candidate) pairs in test set."""
    # Load data
    train_df = pd.read_csv("data/processed/train.csv")
    test_with_neg = pd.read_csv("data/processed/test_with_neg.csv")

    # Load KG
    print("Loading KG graph...")
    G = load_kg_graph()
    print(f"KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build user -> history movies (from training data, use top-rated or recent)
    print("Building user history...")
    user_history = defaultdict(list)
    # Sort by timestamp if available, take recent movies
    if "timestamp" in train_df.columns:
        train_sorted = train_df.sort_values("timestamp")
    else:
        train_sorted = train_df

    for _, row in train_sorted.iterrows():
        uid = row["user_id"]
        # Only keep movies with rating >= 4 (positive signal)
        if "rating" in row and row["rating"] >= 4:
            user_history[uid].append(row["movie_id"])

    # Limit to last 20 history movies per user (for efficiency)
    for uid in user_history:
        user_history[uid] = user_history[uid][-20:]

    print(f"Users with history: {len(user_history)}")

    # Compute features for test pairs
    print("Computing KG features for test pairs...")
    records = []
    total = len(test_with_neg)

    for idx, row in tqdm(test_with_neg.iterrows(), total=total, desc="KG Features"):
        uid = row["user_id"]
        mid = row["movie_id"]

        hist_movies = user_history.get(uid, [])
        features = compute_user_candidate_features(G, hist_movies, mid)
        features["user_id"] = uid
        features["movie_id"] = mid
        records.append(features)

    # Save
    features_df = pd.DataFrame(records)
    out_path = "data/kg/kg_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"\nKG features saved to {out_path}: {len(features_df)} rows")

    # Quick stats
    print("\nFeature statistics:")
    for col in features_df.columns:
        if col not in ["user_id", "movie_id"]:
            print(f"  {col}: mean={features_df[col].mean():.3f}, max={features_df[col].max()}, nonzero={( features_df[col] != 0).sum()}")


if __name__ == "__main__":
    main()
