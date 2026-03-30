"""
Long-tail analysis (RQ2): Does KG disproportionately help long-tail items?

Analyzes:
1. Head/tail stratified Recall@10
2. V1 vs V3 improvement for head vs tail
3. User interest entropy analysis
"""
import os
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from evaluation.metrics import recall_at_k, ndcg_at_k, evaluate_all
from ranker.ranker import (
    load_recall_scores, compute_popularity_from_train,
    load_content_similarity, load_tmdb_features,
    build_feature_df, define_feature_sets, train_lgbm,
    build_group_array,
)


def define_head_tail(train_path="data/processed/train.csv"):
    """
    Define head/tail movies by training interaction count.
    Tail = bottom 50% by interaction count (the many movies with few interactions).
    """
    train_df = pd.read_csv(train_path)
    movie_counts = train_df.groupby("movie_id").size()
    median_count = movie_counts.median()

    tail_movies = set(movie_counts[movie_counts <= median_count].index)
    head_movies = set(movie_counts[movie_counts > median_count].index)

    print(f"Head/tail split (threshold: >{median_count:.0f} interactions):")
    print(f"  Head: {len(head_movies)} movies, "
          f"mean interactions={movie_counts[movie_counts > median_count].mean():.1f}")
    print(f"  Tail: {len(tail_movies)} movies, "
          f"mean interactions={movie_counts[movie_counts <= median_count].mean():.1f}")

    return head_movies, tail_movies, median_count


def stratified_recall(predictions, ground_truth, head_movies, tail_movies, k=10):
    """Compute Recall@K separately for head and tail items."""
    head_recalls = []
    tail_recalls = []

    for uid in predictions:
        gt = ground_truth[uid]
        ranked = predictions[uid]

        gt_head = gt & head_movies
        gt_tail = gt & tail_movies

        if gt_head:
            head_recalls.append(recall_at_k(ranked, gt_head, k))
        if gt_tail:
            tail_recalls.append(recall_at_k(ranked, gt_tail, k))

    return {
        "head_recall": np.mean(head_recalls) if head_recalls else 0,
        "tail_recall": np.mean(tail_recalls) if tail_recalls else 0,
        "head_users": len(head_recalls),
        "tail_users": len(tail_recalls),
    }


def compute_user_genre_entropy(train_path, movies_path="data/processed/movies.csv"):
    """Compute genre entropy for each user from training history."""
    train_df = pd.read_csv(train_path)
    movies_df = pd.read_csv(movies_path)

    movie_genres = {}
    for _, row in movies_df.iterrows():
        if pd.notna(row.get("genres")):
            movie_genres[row["movie_id"]] = str(row["genres"]).split("|")

    user_entropy = {}
    for uid, group in train_df.groupby("user_id"):
        genres = []
        for mid in group["movie_id"]:
            if mid in movie_genres:
                genres.extend(movie_genres[mid])
        if genres:
            counts = Counter(genres)
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            user_entropy[uid] = entropy
        else:
            user_entropy[uid] = 0.0

    return user_entropy


def entropy_stratified_recall(predictions, ground_truth, user_entropy, k=10):
    """Compute Recall@K stratified by user genre entropy (low/mid/high terciles)."""
    entropies = sorted(user_entropy.values())
    t1 = np.percentile(entropies, 33.3)
    t2 = np.percentile(entropies, 66.7)

    buckets = {"low": [], "mid": [], "high": []}
    for uid in predictions:
        if uid not in user_entropy or uid not in ground_truth:
            continue
        gt = ground_truth[uid]
        if not gt:
            continue
        r = recall_at_k(predictions[uid], gt, k)
        e = user_entropy[uid]
        if e <= t1:
            buckets["low"].append(r)
        elif e <= t2:
            buckets["mid"].append(r)
        else:
            buckets["high"].append(r)

    result = {}
    for bucket, recalls in buckets.items():
        result[bucket] = {
            "recall": np.mean(recalls) if recalls else 0,
            "users": len(recalls),
        }
    return result, t1, t2


def run_longtail_analysis(
    cf_scores_path="results/cf_scores.csv",
    train_recall_path="data/processed/train_recall_candidates.csv",
    val_recall_path="data/processed/val_recall_candidates.csv",
    test_recall_path="data/processed/test_recall_candidates.csv",
    kg_features_test_path="data/kg/kg_features_test_recall.csv",
    kg_emb_features_test_path="data/kg/kg_emb_features_test_recall.csv",
    train_path="data/processed/train.csv",
    tmdb_path="data/tmdb/tmdb_metadata.csv",
    k=10,
):
    """Run full long-tail analysis."""
    print("=" * 70)
    print("  Long-tail Analysis (RQ2)")
    print("=" * 70)

    # Define head/tail
    head_movies, tail_movies, threshold = define_head_tail(train_path)

    # Load features
    print("\nLoading features...")
    cf_scores_df = load_recall_scores(cf_scores_path)
    popularity_df = compute_popularity_from_train(train_path)
    tmdb_df = load_tmdb_features(tmdb_path)
    content_sim_recall = load_content_similarity("test_recall")

    kg_test = pd.read_csv(kg_features_test_path) if os.path.exists(kg_features_test_path) else None

    kg_emb_test = None
    kg_emb_cols = []
    if os.path.exists(kg_emb_features_test_path):
        kg_emb_test = pd.read_csv(kg_emb_features_test_path)
        kg_emb_cols = [c for c in kg_emb_test.columns if c.startswith("kg_emb_")]

    train_cands = pd.read_csv(train_recall_path)
    val_cands = pd.read_csv(val_recall_path)
    test_cands = pd.read_csv(test_recall_path)

    # Build feature DataFrames
    train_feat, kg_cols, kg_emb_cols_built = build_feature_df(
        train_cands, cf_scores_df, kg_test, content_sim_recall, popularity_df, tmdb_df, kg_emb_test
    )
    val_feat, _, _ = build_feature_df(
        val_cands, cf_scores_df, kg_test, content_sim_recall, popularity_df, tmdb_df, kg_emb_test
    )
    test_feat, _, _ = build_feature_df(
        test_cands, cf_scores_df, kg_test, content_sim_recall, popularity_df, tmdb_df, kg_emb_test
    )

    if kg_emb_cols_built:
        kg_emb_cols = kg_emb_cols_built

    # Define feature sets
    feature_sets = define_feature_sets(kg_cols, kg_emb_cols if kg_emb_cols else None)

    # Train each variant and collect predictions
    variant_predictions = {}
    variant_ground_truth = {}

    # Recall-only baseline
    print("\nTraining and evaluating variants...")
    test_copy = test_feat.copy()
    preds_recall = defaultdict(list)
    gt_recall = defaultdict(set)
    for uid, group in test_copy.groupby("user_id"):
        sorted_g = group.sort_values("cf_score", ascending=False)
        preds_recall[uid] = sorted_g["movie_id"].tolist()
        gt_recall[uid] = set(group[group["label"] == 1]["movie_id"])
    variant_predictions["Recall-only"] = dict(preds_recall)
    variant_ground_truth["Recall-only"] = dict(gt_recall)

    for variant_name, features in feature_sets.items():
        available = [f for f in features if f in train_feat.columns]
        full_name = f"{variant_name} [Pointwise]"

        model = train_lgbm(train_feat, val_feat, available, method="pointwise")

        test_copy = test_feat.copy()
        test_copy["pred_score"] = model.predict(test_copy[available].values)

        preds = defaultdict(list)
        gt = defaultdict(set)
        for uid, group in test_copy.groupby("user_id"):
            sorted_g = group.sort_values("pred_score", ascending=False)
            preds[uid] = sorted_g["movie_id"].tolist()
            gt[uid] = set(group[group["label"] == 1]["movie_id"])

        variant_predictions[full_name] = dict(preds)
        variant_ground_truth[full_name] = dict(gt)
        print(f"  {full_name}: trained")

    # 1. Head/tail stratified recall
    print("\n" + "=" * 70)
    print("  Head/Tail Stratified Recall@10")
    print("=" * 70)

    header = f"{'Variant':<45} {'Head':>8} {'Tail':>8} {'Delta':>8} {'Tail/Head':>10}"
    print(header)
    print("-" * len(header))

    stratified_results = {}
    recall_only_strat = None

    for variant_name in variant_predictions:
        preds = variant_predictions[variant_name]
        gt = variant_ground_truth[variant_name]
        strat = stratified_recall(preds, gt, head_movies, tail_movies, k)
        stratified_results[variant_name] = strat

        if variant_name == "Recall-only":
            recall_only_strat = strat

        ratio = strat["tail_recall"] / strat["head_recall"] if strat["head_recall"] > 0 else 0
        print(f"{variant_name:<45} "
              f"{strat['head_recall']:>8.4f} "
              f"{strat['tail_recall']:>8.4f} "
              f"{strat['tail_recall'] - strat['head_recall']:>+8.4f} "
              f"{ratio:>10.2f}")

    # 2. V1 vs V3 (and V3e, V4) improvement comparison
    print("\n" + "=" * 70)
    print("  KG Lift: Head vs Tail")
    print("=" * 70)
    print(f"{'Comparison':<50} {'Head Lift':>10} {'Tail Lift':>10} {'Tail > Head?':>14}")
    print("-" * 84)

    v1_key = "V1 (CF) [Pointwise]"
    for compare_key in sorted(variant_predictions.keys()):
        if compare_key in (v1_key, "Recall-only"):
            continue
        if v1_key in stratified_results and compare_key in stratified_results:
            v1_s = stratified_results[v1_key]
            vx_s = stratified_results[compare_key]
            head_lift = vx_s["head_recall"] - v1_s["head_recall"]
            tail_lift = vx_s["tail_recall"] - v1_s["tail_recall"]
            result = "Yes" if tail_lift > head_lift else "No"
            print(f"{compare_key + ' vs V1':<50} "
                  f"{head_lift:>+10.4f} "
                  f"{tail_lift:>+10.4f} "
                  f"{result:>14}")

    # 3. User interest entropy analysis
    print("\n" + "=" * 70)
    print("  User Genre Entropy Analysis")
    print("=" * 70)

    user_entropy = compute_user_genre_entropy(train_path)

    print(f"\n{'Variant':<45} {'Low Ent':>9} {'Mid Ent':>9} {'High Ent':>9}")
    print("-" * 72)

    entropy_results = {}
    for variant_name in variant_predictions:
        preds = variant_predictions[variant_name]
        gt = variant_ground_truth[variant_name]
        ent_strat, t1, t2 = entropy_stratified_recall(preds, gt, user_entropy, k)
        entropy_results[variant_name] = ent_strat

        print(f"{variant_name:<45} "
              f"{ent_strat['low']['recall']:>9.4f} "
              f"{ent_strat['mid']['recall']:>9.4f} "
              f"{ent_strat['high']['recall']:>9.4f}")

    print(f"\n  Entropy terciles: low<={t1:.2f}, mid<={t2:.2f}, high>{t2:.2f}")
    print(f"  Users per bucket: low={entropy_results.get('Recall-only',{}).get('low',{}).get('users',0)}, "
          f"mid={entropy_results.get('Recall-only',{}).get('mid',{}).get('users',0)}, "
          f"high={entropy_results.get('Recall-only',{}).get('high',{}).get('users',0)}")

    # Save results
    os.makedirs("results", exist_ok=True)
    save_data = {
        "head_tail_threshold": float(threshold),
        "head_movies_count": len(head_movies),
        "tail_movies_count": len(tail_movies),
        "stratified_recall": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                   for kk, vv in v.items()}
                              for k, v in stratified_results.items()},
        "entropy_results": {k: {kk: {kkk: float(vvv) if isinstance(vvv, (float, np.floating)) else vvv
                                       for kkk, vvv in vv.items()}
                                 for kk, vv in v.items()}
                            for k, v in entropy_results.items()},
    }
    with open("results/longtail_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print("\nResults saved to results/longtail_analysis.json")


if __name__ == "__main__":
    run_longtail_analysis()
