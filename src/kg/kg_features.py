"""
Compute KG-based features for (user, candidate_movie) pairs.

Features include:
  - shared_actor_count (raw + IDF-weighted)
  - same_director (raw + IDF-weighted)
  - same_genre_count (raw + IDF-weighted)
  - co_liked_count (collaborative KG edges)
  - same_decade (temporal match)

IDF weighting: sharing a rare entity (e.g., niche actor) is more informative
than sharing a common entity (e.g., "Drama" genre). IDF = log(N / df) where
N = total movies and df = number of movies connected to that entity.

IMPORTANT: User history is always derived from TRAINING set only.
"""
import os
import pickle
import math
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


class KGIndex:
    """
    Pre-computed index for fast KG feature computation.

    Instead of querying the NetworkX graph per call (O(degree) per relation),
    pre-builds dictionaries for O(1) set lookups. Critical for dense graphs
    with co_liked edges where movie nodes can have hundreds of neighbors.
    """

    def __init__(self, G):
        self.movie_actors = defaultdict(set)
        self.movie_directors = defaultdict(set)
        self.movie_genres = defaultdict(set)
        self.movie_co_liked = defaultdict(set)
        self.movie_decade = {}
        self.entity_idf = {}

        # Pre-compute neighbor sets per relation
        # Note: nx.Graph is undirected, so edge (u,v) order is arbitrary.
        # We must check which node is the movie node for asymmetric relations.
        for u, v, data in G.edges(data=True):
            rel = data.get("relation", "")
            # Determine movie node vs entity node
            u_is_movie = u.startswith("movie_")
            v_is_movie = v.startswith("movie_")

            if rel == "acted_by":
                movie, entity = (u, v) if u_is_movie else (v, u)
                self.movie_actors[movie].add(entity)
            elif rel == "directed_by":
                movie, entity = (u, v) if u_is_movie else (v, u)
                self.movie_directors[movie].add(entity)
            elif rel == "has_genre":
                movie, entity = (u, v) if u_is_movie else (v, u)
                self.movie_genres[movie].add(entity)
            elif rel == "co_liked":
                self.movie_co_liked[u].add(v)
                self.movie_co_liked[v].add(u)
            elif rel == "released_in_decade":
                movie, entity = (u, v) if u_is_movie else (v, u)
                self.movie_decade[movie] = entity

        # Compute IDF
        n_movies = sum(1 for n in G.nodes() if n.startswith("movie_"))
        entity_movie_count = defaultdict(int)
        for node in G.nodes():
            if node.startswith("movie_"):
                continue
            for n in G.neighbors(node):
                if n.startswith("movie_"):
                    entity_movie_count[node] += 1

        for entity, count in entity_movie_count.items():
            self.entity_idf[entity] = math.log(n_movies / count) if count > 0 else 0.0

        print(f"KGIndex built: {len(self.movie_actors)} movies with actors, "
              f"{len(self.movie_co_liked)} movies with co_liked, "
              f"{len(self.entity_idf)} entity IDFs")

    def pairwise_features(self, movie_a, movie_b):
        """Compute KG features between two movie nodes using pre-computed index."""
        shared_actors = self.movie_actors.get(movie_a, set()) & self.movie_actors.get(movie_b, set())
        shared_directors = self.movie_directors.get(movie_a, set()) & self.movie_directors.get(movie_b, set())
        shared_genres = self.movie_genres.get(movie_a, set()) & self.movie_genres.get(movie_b, set())
        co_liked = 1 if movie_b in self.movie_co_liked.get(movie_a, set()) else 0
        same_decade = 1 if (self.movie_decade.get(movie_a) and
                            self.movie_decade.get(movie_a) == self.movie_decade.get(movie_b)) else 0

        return {
            "shared_actor_count": len(shared_actors),
            "same_director": 1 if shared_directors else 0,
            "same_genre_count": len(shared_genres),
            "shared_actor_idf": sum(self.entity_idf.get(a, 0) for a in shared_actors),
            "same_director_idf": sum(self.entity_idf.get(d, 0) for d in shared_directors),
            "same_genre_idf": sum(self.entity_idf.get(g, 0) for g in shared_genres),
            "co_liked_count": co_liked,
            "same_decade": same_decade,
        }


def compute_user_candidate_features(kg_index, user_history_movies, candidate_movie_id):
    """Compute aggregated KG features for a (user, candidate) pair."""
    candidate_node = f"movie_{candidate_movie_id}"

    zero_result = {
        "kg_shared_actor_count_sum": 0, "kg_shared_actor_count_max": 0,
        "kg_same_director_max": 0,
        "kg_same_genre_count_sum": 0, "kg_same_genre_count_max": 0,
        "kg_shared_actor_idf_sum": 0.0, "kg_same_director_idf_max": 0.0,
        "kg_same_genre_idf_sum": 0.0,
        "kg_co_liked_sum": 0, "kg_same_decade_ratio": 0.0,
    }

    if not user_history_movies:
        return zero_result

    all_features = []
    for hist_movie_id in user_history_movies:
        hist_node = f"movie_{hist_movie_id}"
        feat = kg_index.pairwise_features(hist_node, candidate_node)
        all_features.append(feat)

    n_hist = len(all_features)
    return {
        "kg_shared_actor_count_sum": sum(f["shared_actor_count"] for f in all_features),
        "kg_shared_actor_count_max": max(f["shared_actor_count"] for f in all_features),
        "kg_same_director_max": max(f["same_director"] for f in all_features),
        "kg_same_genre_count_sum": sum(f["same_genre_count"] for f in all_features),
        "kg_same_genre_count_max": max(f["same_genre_count"] for f in all_features),
        "kg_shared_actor_idf_sum": sum(f["shared_actor_idf"] for f in all_features),
        "kg_same_director_idf_max": max(f["same_director_idf"] for f in all_features),
        "kg_same_genre_idf_sum": sum(f["same_genre_idf"] for f in all_features),
        "kg_co_liked_sum": sum(f["co_liked_count"] for f in all_features),
        "kg_same_decade_ratio": sum(f["same_decade"] for f in all_features) / n_hist,
    }


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

    print(f"Users with train history: {len(user_history)}")
    return user_history


def compute_features_for_split(kg_index, user_history, split_path, output_path):
    """Compute KG features for all (user, candidate) pairs in a split."""
    split_df = pd.read_csv(split_path)
    print(f"Computing KG features for {split_path} ({len(split_df)} pairs)...")

    records = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="KG Features"):
        uid, mid = row["user_id"], row["movie_id"]
        hist_movies = user_history.get(uid, [])
        features = compute_user_candidate_features(kg_index, hist_movies, mid)
        features["user_id"] = uid
        features["movie_id"] = mid
        records.append(features)

    features_df = pd.DataFrame(records)
    features_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}: {len(features_df)} rows")

    for col in features_df.columns:
        if col.startswith("kg_"):
            nonzero = (features_df[col] != 0).sum()
            print(f"  {col}: mean={features_df[col].mean():.3f}, "
                  f"nonzero={nonzero} ({100*nonzero/len(features_df):.1f}%)")

    return features_df


def main():
    """Compute KG features for train, val, and test splits."""
    os.makedirs("data/kg", exist_ok=True)

    print("Loading KG graph...")
    G = load_kg_graph()
    print(f"KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Building KG index...")
    kg_index = KGIndex(G)

    user_history = build_train_user_history("data/processed/train.csv", max_history=20)

    splits = [
        ("data/processed/train_with_neg.csv", "data/kg/kg_features_train.csv"),
        ("data/processed/val_with_neg.csv", "data/kg/kg_features_val.csv"),
        ("data/processed/test_with_neg.csv", "data/kg/kg_features_test.csv"),
        ("data/processed/test_recall_candidates.csv", "data/kg/kg_features_test_recall.csv"),
    ]

    for split_path, output_path in splits:
        if os.path.exists(split_path):
            compute_features_for_split(kg_index, user_history, split_path, output_path)
        else:
            print(f"  Skipping {split_path} (not found)")


if __name__ == "__main__":
    main()
