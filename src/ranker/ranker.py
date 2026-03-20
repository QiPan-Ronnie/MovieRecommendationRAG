"""
LightGBM Ranker for combining CF scores, content similarity, and KG features.
Runs ablation experiments: V1 (CF only), V2 (CF+Content), V3 (CF+Content+KG).
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ttest_rel
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import evaluate_all, print_results


def load_features(cf_scores_path, kg_features_path, test_with_neg_path,
                  movies_path="data/processed/movies.csv",
                  tmdb_path="data/tmdb/tmdb_metadata.csv"):
    """
    Load and merge all features into a single dataframe.
    """
    # Test data with labels
    test_df = pd.read_csv(test_with_neg_path)

    # CF scores
    cf_df = pd.read_csv(cf_scores_path)
    # Detect score column name
    score_cols = [c for c in cf_df.columns if c not in ["user_id", "movie_id"]]
    cf_df = cf_df.rename(columns={score_cols[0]: "cf_score"})

    # Merge CF scores
    merged = test_df.merge(cf_df[["user_id", "movie_id", "cf_score"]], on=["user_id", "movie_id"], how="left")
    merged["cf_score"] = merged["cf_score"].fillna(0)

    # Popularity (from test data or movies)
    movie_pop = test_df.groupby("movie_id").size().reset_index(name="popularity")
    merged = merged.merge(movie_pop, on="movie_id", how="left")
    merged["popularity"] = merged["popularity"].fillna(0)

    # Content similarity placeholder (TODO: compute from embeddings)
    # For now use a simple genre-based Jaccard similarity
    if os.path.exists(tmdb_path):
        tmdb_df = pd.read_csv(tmdb_path)
        # Use vote_average as a proxy feature
        merged = merged.merge(
            tmdb_df[["movie_id", "vote_average", "vote_count"]].drop_duplicates(),
            on="movie_id", how="left"
        )
        merged["vote_average"] = merged["vote_average"].fillna(0)
        merged["vote_count"] = merged["vote_count"].fillna(0)
        merged["content_similarity"] = merged["vote_average"] / 10.0  # normalize
    else:
        merged["content_similarity"] = 0
        merged["vote_average"] = 0
        merged["vote_count"] = 0

    # KG features
    if os.path.exists(kg_features_path):
        kg_df = pd.read_csv(kg_features_path)
        merged = merged.merge(kg_df, on=["user_id", "movie_id"], how="left")
        kg_cols = [c for c in kg_df.columns if c.startswith("kg_")]
        for col in kg_cols:
            merged[col] = merged[col].fillna(0)
    else:
        kg_cols = []

    return merged, kg_cols


def define_feature_sets(kg_cols):
    """Define feature sets for each variant."""
    v1_features = ["cf_score"]
    v2_features = ["cf_score", "content_similarity", "popularity", "vote_average", "vote_count"]
    v3_features = v2_features + kg_cols

    return {
        "V1 (CF)": v1_features,
        "V2 (CF+Content)": v2_features,
        "V3 (CF+Content+KG)": v3_features,
    }


def train_and_evaluate(merged_df, feature_cols, variant_name, k=10):
    """Train LightGBM and evaluate."""
    # Split: use a portion for LightGBM training, rest for eval
    # Group by user, use 70% users for train, 30% for eval
    all_users = merged_df["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(all_users)
    split_idx = int(len(all_users) * 0.7)
    train_users = set(all_users[:split_idx])
    eval_users = set(all_users[split_idx:])

    train_data = merged_df[merged_df["user_id"].isin(train_users)]
    eval_data = merged_df[merged_df["user_id"].isin(eval_users)]

    X_train = train_data[feature_cols].values
    y_train = train_data["label"].values
    X_eval = eval_data[feature_cols].values
    y_eval = eval_data["label"].values

    # Train LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    deval = lgb.Dataset(X_eval, label=y_eval, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[deval],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    # Predict on eval set
    eval_data = eval_data.copy()
    eval_data["pred_score"] = model.predict(X_eval)

    # Build predictions and ground truth
    predictions = defaultdict(list)
    ground_truth = defaultdict(set)

    for uid, group in eval_data.groupby("user_id"):
        sorted_group = group.sort_values("pred_score", ascending=False)
        predictions[uid] = sorted_group["movie_id"].tolist()

        pos_items = group[group["label"] == 1]["movie_id"].tolist()
        ground_truth[uid] = set(pos_items)

    total_items = merged_df["movie_id"].nunique()
    results, per_user_ndcg = evaluate_all(predictions, ground_truth, k=k, total_items=total_items)
    print_results(results, variant_name)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))

    return results, per_user_ndcg, importance, model


def run_ablation(cf_scores_path="results/cf_scores.csv",
                 kg_features_path="data/kg/kg_features.csv",
                 test_with_neg_path="data/processed/test_with_neg.csv",
                 k=10):
    """Run full ablation experiment."""
    print("Loading and merging features...")
    merged, kg_cols = load_features(cf_scores_path, kg_features_path, test_with_neg_path)
    print(f"Total samples: {len(merged)}, Features available: {list(merged.columns)}")

    feature_sets = define_feature_sets(kg_cols)
    all_results = {}
    all_per_user = {}
    all_importance = {}

    for variant_name, features in feature_sets.items():
        available_features = [f for f in features if f in merged.columns]
        print(f"\n{'='*60}")
        print(f"  {variant_name}: {available_features}")
        print(f"{'='*60}")

        results, per_user, importance, _ = train_and_evaluate(
            merged, available_features, variant_name, k=k
        )
        all_results[variant_name] = results
        all_per_user[variant_name] = per_user
        all_importance[variant_name] = importance

    # Summary table
    print("\n" + "=" * 80)
    print("  ABLATION RESULTS")
    print("=" * 80)
    header = f"{'Variant':<25} {'Hit@10':>8} {'NDCG@10':>8} {'Recall@10':>10} {'MRR':>8}"
    print(header)
    print("-" * len(header))
    for variant, metrics in all_results.items():
        print(f"{variant:<25} {metrics.get(f'Hit@{k}', 0):>8.4f} {metrics.get(f'NDCG@{k}', 0):>8.4f} {metrics.get(f'Recall@{k}', 0):>10.4f} {metrics.get('MRR', 0):>8.4f}")

    # Statistical significance: V2 vs V3
    if "V2 (CF+Content)" in all_per_user and "V3 (CF+Content+KG)" in all_per_user:
        v2_ndcg = all_per_user["V2 (CF+Content)"]
        v3_ndcg = all_per_user["V3 (CF+Content+KG)"]
        common_users = set(v2_ndcg.keys()) & set(v3_ndcg.keys())
        v2_vals = [v2_ndcg[u] for u in common_users]
        v3_vals = [v3_ndcg[u] for u in common_users]

        t_stat, p_value = ttest_rel(v3_vals, v2_vals)
        print(f"\n  Statistical Test (V3 vs V2):")
        print(f"    Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")
        print(f"    {'Significant (p < 0.05)' if p_value < 0.05 else 'Not significant (p >= 0.05)'}")

    # Feature importance for V3
    if "V3 (CF+Content+KG)" in all_importance:
        print(f"\n  Feature Importance (V3):")
        sorted_imp = sorted(all_importance["V3 (CF+Content+KG)"].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp:
            print(f"    {feat:<30s}: {imp:.1f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open("results/ablation_per_user.pkl", "wb") as f:
        pickle.dump(all_per_user, f)

    with open("results/feature_importance.json", "w") as f:
        json.dump(all_importance, f, indent=2, default=float)

    print("\nResults saved to results/")


if __name__ == "__main__":
    run_ablation()
