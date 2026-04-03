"""
Export Phase 1 outputs needed by Phase 2 (RAG explanation).

Generates:
  1. results/recommendations_v4.csv — Per-user top-10 ranked recommendations (best model)
  2. results/rag_eval_set.json — Sampled evaluation set for explanation quality assessment
  3. data/kg/kg_paths_for_recommendations.json — KG paths between user history and recommendations
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import lightgbm as lgb
from collections import defaultdict
from tqdm import tqdm


def export_recommendations(k=10):
    """
    Re-run best ranker (V4 LambdaMART with RotatE embeddings) and export
    per-user top-K recommendations with scores and ground truth labels.
    """
    print("=" * 60)
    print("  Exporting top-K recommendations")
    print("=" * 60)

    from ranker.ranker import (
        load_recall_scores, compute_popularity_from_train,
        load_content_similarity, load_tmdb_features,
        build_feature_df, train_lambdamart, load_kg_emb_features,
    )

    scores_path = "results/multi_recall_scores.csv"
    cf_scores_df = load_recall_scores(scores_path)
    popularity_df = compute_popularity_from_train()
    tmdb_df = load_tmdb_features()
    content_sim = load_content_similarity("test_recall")
    kg_features = pd.read_csv("data/kg/kg_features_test_recall.csv") if os.path.exists("data/kg/kg_features_test_recall.csv") else None
    kg_emb_features = load_kg_emb_features("data/kg/kg_emb_features_test_recall.csv")

    # Load candidate sets
    train_cands = pd.read_csv("data/processed/train_recall_candidates.csv")
    val_cands = pd.read_csv("data/processed/val_recall_candidates.csv")
    test_cands = pd.read_csv("data/processed/test_recall_candidates.csv")

    # Build feature DataFrames
    train_feat, kg_cols, kg_emb_cols = build_feature_df(
        train_cands, cf_scores_df, kg_features, content_sim, popularity_df, tmdb_df, kg_emb_features)
    val_feat, _, _ = build_feature_df(
        val_cands, cf_scores_df, kg_features, content_sim, popularity_df, tmdb_df, kg_emb_features)
    test_feat, _, _ = build_feature_df(
        test_cands, cf_scores_df, kg_features, content_sim, popularity_df, tmdb_df, kg_emb_features)

    # V4 features (all features)
    v4_features = ["cf_score", "kg_recall_score", "content_similarity", "popularity", "vote_count"]
    v4_features += kg_cols + kg_emb_cols
    v4_features = [f for f in v4_features if f in train_feat.columns]
    print(f"  V4 features ({len(v4_features)}): {v4_features}")

    # Train V4 LambdaMART
    print("  Training V4 LambdaMART...")
    model = train_lambdamart(train_feat, val_feat, v4_features)

    # Predict on test set
    test_feat = test_feat.copy()
    test_feat["pred_score"] = model.predict(test_feat[v4_features].values)

    # Build per-user top-K
    records = []
    for uid, group in test_feat.groupby("user_id"):
        sorted_g = group.sort_values("pred_score", ascending=False).head(k)
        for rank, (_, row) in enumerate(sorted_g.iterrows(), 1):
            records.append({
                "user_id": int(uid),
                "movie_id": int(row["movie_id"]),
                "rank": rank,
                "pred_score": round(float(row["pred_score"]), 6),
                "label": int(row["label"]),
            })

    rec_df = pd.DataFrame(records)
    output_path = "results/recommendations_v4.csv"
    rec_df.to_csv(output_path, index=False)

    n_users = rec_df.user_id.nunique()
    n_hits = (rec_df.label == 1).sum()
    print(f"\n  Saved: {output_path}")
    print(f"  {len(rec_df)} recommendations for {n_users} users")
    print(f"  Hits in top-{k}: {n_hits} ({n_hits/n_users:.2f} per user)")

    return rec_df


def export_eval_set(rec_df, n_users=200, seed=42):
    """
    Build evaluation set for RAG explanation quality assessment.

    Samples n_users test users, includes their training history and
    top-10 recommendations with ground truth labels.
    """
    print("\n" + "=" * 60)
    print("  Building RAG evaluation set")
    print("=" * 60)

    # Load auxiliary data
    train_df = pd.read_csv("data/processed/train.csv")
    movies_df = pd.read_csv("data/processed/movies.csv")
    tmdb_df = pd.read_csv("data/tmdb/tmdb_metadata.csv")

    movie_info = {}
    for _, row in movies_df.iterrows():
        mid = int(row["movie_id"])
        movie_info[mid] = {"title": row["title"], "genres": str(row.get("genres", ""))}

    # Add TMDB overview
    for _, row in tmdb_df.iterrows():
        mid = int(row["movie_id"])
        if mid in movie_info:
            movie_info[mid]["overview"] = str(row.get("overview", ""))

    # Build user histories (last 10 movies per user, sorted by time)
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values("timestamp")
    user_history = defaultdict(list)
    for _, row in train_df.iterrows():
        uid, mid = int(row["user_id"]), int(row["movie_id"])
        user_history[uid].append(mid)

    # Sample users who have at least 1 hit in top-10 (more interesting for explanation)
    users_with_hits = rec_df[rec_df.label == 1].user_id.unique()
    rng = np.random.RandomState(seed)
    if len(users_with_hits) >= n_users:
        sampled = rng.choice(users_with_hits, n_users, replace=False)
    else:
        # Fill with random users
        all_users = rec_df.user_id.unique()
        extra = rng.choice([u for u in all_users if u not in users_with_hits],
                           n_users - len(users_with_hits), replace=False)
        sampled = np.concatenate([users_with_hits, extra])[:n_users]

    eval_set = []
    for uid in sorted(sampled):
        uid = int(uid)
        # User history (last 10)
        history = user_history.get(uid, [])[-10:]
        history_items = []
        for mid in history:
            info = movie_info.get(mid, {})
            history_items.append({
                "movie_id": mid,
                "title": info.get("title", "Unknown"),
                "genres": info.get("genres", ""),
            })

        # Recommendations
        user_recs = rec_df[rec_df.user_id == uid].sort_values("rank")
        recs = []
        for _, row in user_recs.iterrows():
            mid = int(row["movie_id"])
            info = movie_info.get(mid, {})
            recs.append({
                "movie_id": mid,
                "title": info.get("title", "Unknown"),
                "genres": info.get("genres", ""),
                "overview": info.get("overview", ""),
                "rank": int(row["rank"]),
                "pred_score": round(float(row["pred_score"]), 6),
                "relevant": int(row["label"]),
            })

        eval_set.append({
            "user_id": uid,
            "history": history_items,
            "recommendations": recs,
        })

    output_path = "results/rag_eval_set.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    n_with_hits = sum(1 for u in eval_set if any(r["relevant"] for r in u["recommendations"]))
    print(f"  Saved: {output_path}")
    print(f"  {len(eval_set)} users, {n_with_hits} with at least 1 relevant recommendation")


def export_kg_paths(rec_df, max_hops=2, max_paths_per_pair=5):
    """
    Pre-compute KG connection paths between user history movies and
    recommended movies. These paths serve as structured evidence for
    KG-based explanations.

    For each (user, recommended_movie) pair, find paths like:
      history_movie --has_genre--> genre <--has_genre-- recommended_movie
      history_movie --acted_by--> actor <--acted_by-- recommended_movie
    """
    print("\n" + "=" * 60)
    print("  Extracting KG paths for recommendations")
    print("=" * 60)

    # Load KG
    with open("data/kg/kg_graph.pkl", "rb") as f:
        G = pickle.load(f)

    # Load user histories
    train_df = pd.read_csv("data/processed/train.csv")
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values("timestamp")
    user_history = defaultdict(list)
    for _, row in train_df.iterrows():
        user_history[int(row["user_id"])].append(int(row["movie_id"]))

    # For each user, take last 10 history movies
    for uid in user_history:
        user_history[uid] = user_history[uid][-10:]

    all_paths = {}
    users = rec_df.user_id.unique()

    for uid in tqdm(users, desc="KG paths"):
        uid = int(uid)
        history_movies = user_history.get(uid, [])
        if not history_movies:
            continue

        user_recs = rec_df[rec_df.user_id == uid].sort_values("rank")

        for _, row in user_recs.iterrows():
            rec_mid = int(row["movie_id"])
            rec_node = f"movie_{rec_mid}"
            if rec_node not in G:
                continue

            pair_paths = []
            for hist_mid in history_movies:
                hist_node = f"movie_{hist_mid}"
                if hist_node not in G:
                    continue

                # Find short paths via shared entities
                try:
                    for path in nx.all_simple_paths(G, hist_node, rec_node, cutoff=max_hops):
                        if len(path) < 2:
                            continue
                        # Build readable path with relation types
                        path_desc = []
                        for i in range(len(path) - 1):
                            edge_data = G.edges[path[i], path[i+1]]
                            rel = edge_data.get("relation", "connected")
                            path_desc.append({
                                "from": path[i],
                                "relation": rel,
                                "to": path[i+1],
                            })
                        pair_paths.append({
                            "history_movie": hist_mid,
                            "path": path_desc,
                        })
                        if len(pair_paths) >= max_paths_per_pair:
                            break
                except nx.NetworkXError:
                    continue

                if len(pair_paths) >= max_paths_per_pair:
                    break

            if pair_paths:
                key = f"{uid}_{rec_mid}"
                all_paths[key] = pair_paths

    output_path = "data/kg/kg_paths_for_recommendations.json"
    with open(output_path, "w") as f:
        json.dump(all_paths, f)

    n_pairs_with_paths = len(all_paths)
    total_paths = sum(len(v) for v in all_paths.values())
    print(f"  Saved: {output_path}")
    print(f"  {n_pairs_with_paths} (user, movie) pairs with KG paths")
    print(f"  {total_paths} total paths ({total_paths/max(n_pairs_with_paths,1):.1f} avg per pair)")


def main():
    # 1. Export recommendations
    rec_df = export_recommendations(k=10)

    # 2. Build RAG evaluation set
    export_eval_set(rec_df, n_users=200)

    # 3. Extract KG paths
    export_kg_paths(rec_df)

    print("\n" + "=" * 60)
    print("  Phase 1 exports for RAG complete!")
    print("=" * 60)
    print("  results/recommendations_v4.csv        — Per-user top-10 ranked list")
    print("  results/rag_eval_set.json           — 200-user evaluation set")
    print("  data/kg/kg_paths_for_recommendations.json — KG explanation paths")


if __name__ == "__main__":
    main()
