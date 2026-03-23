"""
Compute KG-based features for (user, candidate_movie) pairs.
Features: shared_actor_count, same_director, same_genre_count, shortest_path_len

IMPORTANT: User history is always derived from TRAINING set only,
regardless of which split we are computing features for.
"""
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm import tqdm


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
    """Compute KG features between two movies."""
    features = {
        "shared_actor_count": 0,
        "same_director": 0,
        "same_genre_count": 0,
        "shortest_path_len": max_path_len + 1,
    }

    if movie_a_node not in G or movie_b_node not in G:
        return features

    actors_a = get_movie_neighbors(G, movie_a_node, "acted_by")
    actors_b = get_movie_neighbors(G, movie_b_node, "acted_by")
    features["shared_actor_count"] = len(actors_a & actors_b)

    directors_a = get_movie_neighbors(G, movie_a_node, "directed_by")
    directors_b = get_movie_neighbors(G, movie_b_node, "directed_by")
    features["same_director"] = 1 if len(directors_a & directors_b) > 0 else 0

    genres_a = get_movie_neighbors(G, movie_a_node, "has_genre")
    genres_b = get_movie_neighbors(G, movie_b_node, "has_genre")
    features["same_genre_count"] = len(genres_a & genres_b)

    try:
        path_len = nx.shortest_path_length(G, movie_a_node, movie_b_node)
        features["shortest_path_len"] = min(path_len, max_path_len + 1)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    return features


def compute_user_candidate_features(G, user_history_movies, candidate_movie_id,
                                    max_path_len=4):
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

    result = {
        "kg_shared_actor_count_sum": sum(f["shared_actor_count"] for f in all_features),
        "kg_shared_actor_count_max": max(f["shared_actor_count"] for f in all_features),
        "kg_same_director_max": max(f["same_director"] for f in all_features),
        "kg_same_genre_count_sum": sum(f["same_genre_count"] for f in all_features),
        "kg_same_genre_count_max": max(f["same_genre_count"] for f in all_features),
        "kg_shortest_path_min": min(f["shortest_path_len"] for f in all_features),
    }
    return result


def build_train_user_history(train_path="data/processed/train.csv", max_history=20):
    """
    Build user history from TRAINING set only.
    Only includes positive interactions (the train set already contains only
    rating >= 4 after our data prep).
    Returns dict[user_id] -> list[movie_id] (last max_history items by time).
    """
    train_df = pd.read_csv(train_path)

    user_history = defaultdict(list)
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values("timestamp")

    for _, row in train_df.iterrows():
        user_history[row["user_id"]].append(row["movie_id"])

    # Limit to last max_history movies per user
    for uid in user_history:
        user_history[uid] = user_history[uid][-max_history:]

    print(f"Users with train history: {len(user_history)}")
    return user_history


def compute_features_for_split(G, user_history, split_with_neg_path, output_path):
    """
    Compute KG features for all (user, candidate) pairs in a split.
    User history always comes from training set.
    """
    split_df = pd.read_csv(split_with_neg_path)
    print(f"Computing KG features for {split_with_neg_path} ({len(split_df)} pairs)...")

    records = []
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc="KG Features"):
        uid = row["user_id"]
        mid = row["movie_id"]

        hist_movies = user_history.get(uid, [])
        features = compute_user_candidate_features(G, hist_movies, mid)
        features["user_id"] = uid
        features["movie_id"] = mid
        records.append(features)

    features_df = pd.DataFrame(records)
    features_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}: {len(features_df)} rows")

    # Stats
    for col in features_df.columns:
        if col.startswith("kg_"):
            nonzero = (features_df[col] != 0).sum()
            print(f"  {col}: mean={features_df[col].mean():.3f}, "
                  f"nonzero={nonzero} ({100*nonzero/len(features_df):.1f}%)")

    return features_df


def main():
    """Compute KG features for train, val, and test splits."""
    os.makedirs("data/kg", exist_ok=True)

    # Load KG
    print("Loading KG graph...")
    G = load_kg_graph()
    print(f"KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build user history from training set ONLY
    user_history = build_train_user_history("data/processed/train.csv", max_history=20)

    # Compute features for each split
    splits = [
        ("data/processed/train_with_neg.csv", "data/kg/kg_features_train.csv"),
        ("data/processed/val_with_neg.csv", "data/kg/kg_features_val.csv"),
        ("data/processed/test_with_neg.csv", "data/kg/kg_features_test.csv"),
    ]

    for split_path, output_path in splits:
        if os.path.exists(split_path):
            compute_features_for_split(G, user_history, split_path, output_path)
        else:
            print(f"  Skipping {split_path} (not found)")


if __name__ == "__main__":
    main()
