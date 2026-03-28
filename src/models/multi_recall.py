"""
Multi-route recall: CF + KG-based candidate generation.

Route 1 (CF): Item-CF top-K candidates — captures collaborative signal, biased toward popular items.
Route 2 (KG): TransE nearest neighbors — captures KG structure, better coverage for tail items.

Merged into top-100 candidates per user with both cf_score and kg_recall_score.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def load_transe_for_recall(
    emb_path="data/kg/transe_entity_emb.npy",
    entity2id_path="data/kg/entity2id.csv",
):
    """Load TransE embeddings and build movie_id -> embedding index mapping."""
    embeddings = np.load(emb_path)
    entity2id_df = pd.read_csv(entity2id_path)
    entity2id = dict(zip(entity2id_df["entity"], entity2id_df["entity_id"]))

    movie2idx = {}
    for entity, idx in entity2id.items():
        if entity.startswith("movie_"):
            try:
                movie_id = int(entity.replace("movie_", ""))
                movie2idx[movie_id] = idx
            except ValueError:
                pass

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_normed = embeddings / norms

    return emb_normed, movie2idx


def kg_recall_for_user(user_history, movie2idx, emb_normed,
                       all_movie_ids, exclude_set, n_candidates=50):
    """
    KG-based recall for a single user using TransE embeddings.

    Returns list of (movie_id, transe_score) sorted by similarity desc.
    """
    # Build user profile: mean of history movie embeddings
    valid_indices = [movie2idx[mid] for mid in user_history if mid in movie2idx]
    if not valid_indices:
        return []

    user_profile = emb_normed[valid_indices].mean(axis=0)
    norm = np.linalg.norm(user_profile)
    if norm > 0:
        user_profile /= norm

    # Compute cosine similarity with all movies
    candidate_mids = [mid for mid in all_movie_ids if mid not in exclude_set and mid in movie2idx]
    if not candidate_mids:
        return []

    candidate_indices = [movie2idx[mid] for mid in candidate_mids]
    candidate_embs = emb_normed[candidate_indices]
    scores = candidate_embs @ user_profile

    # Sort and return top-N
    top_indices = np.argsort(scores)[::-1][:n_candidates]
    results = [(candidate_mids[i], float(scores[i])) for i in top_indices]
    return results


def _auto_detect_recall_scores():
    """Auto-detect best available recall model scores (prefer LightGCN > MF > CF)."""
    candidates = [
        ("results/lightgcn_scores.csv", "lgcn_score", "LightGCN"),
        ("results/mf_scores.csv", "mf_score", "BPR-MF"),
        ("results/cf_scores.csv", "cf_score", "Item-CF"),
    ]
    for path, score_col, name in candidates:
        if os.path.exists(path):
            print(f"  Using recall scores from {name}: {path}")
            return path, score_col
    raise FileNotFoundError("No recall score files found in results/")


def generate_multi_recall(
    cf_scores_path=None,
    train_path="data/processed/train.csv",
    output_path="results/multi_recall_scores.csv",
    n_cf=70,
    n_kg=50,
    n_total=100,
    max_history=20,
):
    """
    Generate multi-route recall candidates for all users.

    Combines recall model top-n_cf with KG TransE top-n_kg, deduplicates,
    and outputs top-n_total candidates per user.
    """
    # Auto-detect best recall model if not specified
    if cf_scores_path is None:
        cf_scores_path, auto_score_col = _auto_detect_recall_scores()
    else:
        auto_score_col = None

    print(f"Multi-route recall: CF top-{n_cf} + KG top-{n_kg} -> {n_total} per user")

    # Load recall scores and normalize column name to "cf_score"
    cf_df = pd.read_csv(cf_scores_path)
    if auto_score_col and auto_score_col in cf_df.columns and "cf_score" not in cf_df.columns:
        cf_df = cf_df.rename(columns={auto_score_col: "cf_score"})
    cf_by_user = {}
    for uid, group in cf_df.groupby("user_id"):
        sorted_g = group.sort_values("cf_score", ascending=False)
        cf_by_user[uid] = list(zip(sorted_g["movie_id"], sorted_g["cf_score"]))

    # Load TransE
    emb_normed, movie2idx = load_transe_for_recall()
    all_kg_movies = sorted(movie2idx.keys())
    print(f"  CF covers {cf_df.movie_id.nunique()} movies, KG covers {len(all_kg_movies)} movies")

    # Build user histories
    train_df = pd.read_csv(train_path)
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values("timestamp")

    user_history = defaultdict(list)
    user_train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        uid, mid = row["user_id"], row["movie_id"]
        user_history[uid].append(mid)
        user_train_items[uid].add(mid)

    # Limit history for KG profile
    for uid in user_history:
        user_history[uid] = user_history[uid][-max_history:]

    all_users = sorted(cf_by_user.keys())

    # Stats tracking
    records = []
    source_stats = {"cf_only": 0, "kg_only": 0, "both": 0}
    kg_only_movies = set()

    for uid in tqdm(all_users, desc="Multi-recall"):
        exclude = user_train_items.get(uid, set())

        # Route 1: CF candidates
        cf_cands = cf_by_user.get(uid, [])[:n_cf]
        cf_dict = {mid: score for mid, score in cf_cands}

        # Route 2: KG candidates
        history = user_history.get(uid, [])
        kg_cands = kg_recall_for_user(
            history, movie2idx, emb_normed, all_kg_movies, exclude, n_kg
        )
        kg_dict = {mid: score for mid, score in kg_cands}

        # Merge: CF first (by cf_score), then KG-only (by kg_score)
        merged = {}
        for mid, cf_score in cf_cands:
            kg_score = kg_dict.get(mid, 0.0)
            merged[mid] = {"cf_score": cf_score, "kg_recall_score": kg_score, "source": "both" if mid in kg_dict else "cf"}

        for mid, kg_score in kg_cands:
            if mid not in merged:
                merged[mid] = {"cf_score": 0.0, "kg_recall_score": kg_score, "source": "kg"}

        # Sort: CF items by cf_score desc, then KG-only items by kg_score desc
        cf_items = [(mid, d) for mid, d in merged.items() if d["source"] != "kg"]
        kg_items = [(mid, d) for mid, d in merged.items() if d["source"] == "kg"]
        cf_items.sort(key=lambda x: x[1]["cf_score"], reverse=True)
        kg_items.sort(key=lambda x: x[1]["kg_recall_score"], reverse=True)

        final = (cf_items + kg_items)[:n_total]

        for mid, d in final:
            records.append({
                "user_id": uid,
                "movie_id": mid,
                "cf_score": d["cf_score"],
                "kg_recall_score": d["kg_recall_score"],
            })
            if d["source"] == "cf":
                source_stats["cf_only"] += 1
            elif d["source"] == "kg":
                source_stats["kg_only"] += 1
                kg_only_movies.add(mid)
            else:
                source_stats["both"] += 1

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_path, index=False)

    total = sum(source_stats.values())
    print(f"\n  Output: {len(result_df)} candidates, {result_df.user_id.nunique()} users")
    print(f"  Per user: {len(result_df) / result_df.user_id.nunique():.0f} candidates")
    print(f"  Source breakdown:")
    print(f"    CF-only:  {source_stats['cf_only']:>8} ({source_stats['cf_only']/total*100:.1f}%)")
    print(f"    KG-only:  {source_stats['kg_only']:>8} ({source_stats['kg_only']/total*100:.1f}%)")
    print(f"    Both:     {source_stats['both']:>8} ({source_stats['both']/total*100:.1f}%)")
    print(f"  Unique movies: {result_df.movie_id.nunique()} "
          f"(was {cf_df.movie_id.nunique()} with CF-only)")
    print(f"  KG-only unique movies: {len(kg_only_movies)}")

    return result_df


def main():
    generate_multi_recall()


if __name__ == "__main__":
    main()
