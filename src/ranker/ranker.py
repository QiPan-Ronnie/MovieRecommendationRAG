"""
LightGBM Ranker: re-ranks recall model's top-100 candidates using
CF scores + content similarity + KG features.

Supports two ranking objectives:
  - Pointwise (binary classification): treats each (user, item) independently
  - LambdaMART (listwise): optimizes NDCG directly over per-user ranked lists

Evaluation protocol (Option C):
  - Train on training set candidates (random negatives), tune on validation
  - Evaluate on recall model's top-100 candidates per user (realistic re-ranking)
  - All features computed from training data only (no leakage)
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ttest_rel
from collections import defaultdict
from itertools import product

from src.evaluation.metrics import evaluate_all, print_results


# ---------------------------------------------------------------------------
# Feature loading and merging
# ---------------------------------------------------------------------------

def load_recall_scores(scores_path):
    """Load recall model scores (user_id, movie_id, cf_score, kg_recall_score)."""
    df = pd.read_csv(scores_path)
    # Handle both old format (cf_score only) and new format (cf_score + kg_recall_score)
    if "cf_score" not in df.columns:
        score_cols = [c for c in df.columns if c not in ["user_id", "movie_id", "kg_recall_score"]]
        if score_cols:
            df = df.rename(columns={score_cols[0]: "cf_score"})
    if "kg_recall_score" not in df.columns:
        df["kg_recall_score"] = 0.0
    return df[["user_id", "movie_id", "cf_score", "kg_recall_score"]]


def compute_popularity_from_train(train_path="data/processed/train.csv"):
    """Compute movie popularity from training set ONLY."""
    train_df = pd.read_csv(train_path)
    movie_pop = train_df.groupby("movie_id").size().reset_index(name="popularity")
    return movie_pop


def load_content_similarity(split_name, content_sim_path="data/processed"):
    """Load pre-computed content similarity for a split."""
    path = os.path.join(content_sim_path, f"content_sim_{split_name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_tmdb_features(tmdb_path="data/tmdb/tmdb_metadata.csv"):
    """Load TMDB vote_count as an additional feature."""
    if not os.path.exists(tmdb_path):
        return None
    tmdb_df = pd.read_csv(tmdb_path)
    if "vote_count" in tmdb_df.columns:
        return tmdb_df[["movie_id", "vote_count"]].drop_duplicates()
    return None


def build_feature_df(candidates_with_labels, cf_scores_df, kg_features_df,
                     content_sim_df, popularity_df, tmdb_df,
                     kg_emb_features_df=None):
    """Merge all features into a single DataFrame for a given split."""
    merged = candidates_with_labels.copy()

    # CF scores + KG recall score
    if cf_scores_df is not None:
        merged = merged.merge(cf_scores_df, on=["user_id", "movie_id"], how="left")
        merged["cf_score"] = merged["cf_score"].fillna(0)
        merged["kg_recall_score"] = merged["kg_recall_score"].fillna(0)
    else:
        merged["cf_score"] = 0
        merged["kg_recall_score"] = 0

    # Popularity (from training set)
    if popularity_df is not None:
        merged = merged.merge(popularity_df, on="movie_id", how="left")
        merged["popularity"] = merged["popularity"].fillna(0)
    else:
        merged["popularity"] = 0

    # Content similarity
    if content_sim_df is not None:
        merged = merged.merge(
            content_sim_df[["user_id", "movie_id", "content_similarity"]],
            on=["user_id", "movie_id"], how="left"
        )
        merged["content_similarity"] = merged["content_similarity"].fillna(0)
    else:
        merged["content_similarity"] = 0

    # TMDB vote_count
    if tmdb_df is not None:
        merged = merged.merge(tmdb_df, on="movie_id", how="left")
        merged["vote_count"] = merged["vote_count"].fillna(0)
    else:
        merged["vote_count"] = 0

    # KG features (hand-crafted)
    kg_cols = []
    if kg_features_df is not None:
        merged = merged.merge(kg_features_df, on=["user_id", "movie_id"], how="left")
        kg_cols = [c for c in kg_features_df.columns if c.startswith("kg_")]
        for col in kg_cols:
            merged[col] = merged[col].fillna(0)

    # KG embedding features
    kg_emb_cols = []
    if kg_emb_features_df is not None:
        merged = merged.merge(kg_emb_features_df, on=["user_id", "movie_id"], how="left")
        kg_emb_cols = [c for c in kg_emb_features_df.columns if c.startswith("kg_emb_")]
        for col in kg_emb_cols:
            merged[col] = merged[col].fillna(0)

    return merged, kg_cols, kg_emb_cols


# ---------------------------------------------------------------------------
# Build recall test candidates (Option C)
# ---------------------------------------------------------------------------

def build_recall_test_candidates(
    cf_scores_path="results/cf_scores.csv",
    test_path="data/processed/test.csv",
    output_path="data/processed/test_recall_candidates.csv",
):
    """
    Build test candidate pool from recall model's top-100 candidates.

    Instead of evaluating on randomly-sampled negatives (which inflates metrics
    and makes cf_score trivially dominant), we evaluate on the recall model's
    actual top-100 candidates — the realistic re-ranking scenario.

    Label = 1 if the movie appears in the user's test positive items, else 0.
    """
    cf_scores = pd.read_csv(cf_scores_path)
    test_df = pd.read_csv(test_path)

    # Build test ground truth: user -> set of positive movie_ids
    test_positives = test_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Label recall candidates
    cf_scores["label"] = cf_scores.apply(
        lambda r: 1 if r["movie_id"] in test_positives.get(r["user_id"], set()) else 0,
        axis=1
    )

    # Save (user_id, movie_id, label) — cf_score merged later via build_feature_df
    result = cf_scores[["user_id", "movie_id", "label"]]
    result.to_csv(output_path, index=False)

    n_pos = (result["label"] == 1).sum()
    n_neg = (result["label"] == 0).sum()
    n_users = result["user_id"].nunique()
    print(f"Recall test candidates: {len(result)} rows")
    print(f"  Positive: {n_pos}, Negative: {n_neg}")
    print(f"  Positive rate: {n_pos / len(result) * 100:.1f}%")
    print(f"  Users: {n_users}, Candidates/user: {len(result) / n_users:.0f}")

    return result


def build_recall_train_val_candidates(
    cf_scores_path="results/cf_scores.csv",
    val_path="data/processed/val.csv",
    train_output="data/processed/train_recall_candidates.csv",
    val_output="data/processed/val_recall_candidates.csv",
    val_user_ratio=0.2,
    seed=42,
):
    """
    Build ranker training and validation data from recall candidates.

    Key insight: training the ranker on random negatives teaches it shortcuts
    (e.g., "high popularity → positive") that don't transfer to the evaluation
    setting where all candidates are recall-model outputs with non-zero cf_scores.

    Fix: label recall candidates using val.csv (observed next-period interactions).
    Split users 80/20 for ranker train/val.

    Features (content_sim, KG) already computed for these candidates can be reused.
    """
    cf_scores = pd.read_csv(cf_scores_path)
    val_df = pd.read_csv(val_path)

    # Label recall candidates with val positives
    val_positives = val_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    cf_scores["label"] = cf_scores.apply(
        lambda r: 1 if r["movie_id"] in val_positives.get(r["user_id"], set()) else 0,
        axis=1
    )

    # Split users for ranker train/val
    all_users = cf_scores["user_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(all_users)
    n_val = int(len(all_users) * val_user_ratio)
    val_users = set(all_users[:n_val])
    train_users = set(all_users[n_val:])

    train_df = cf_scores[cf_scores["user_id"].isin(train_users)][["user_id", "movie_id", "label"]]
    val_df_out = cf_scores[cf_scores["user_id"].isin(val_users)][["user_id", "movie_id", "label"]]

    train_df.to_csv(train_output, index=False)
    val_df_out.to_csv(val_output, index=False)

    for name, df in [("Ranker train", train_df), ("Ranker val", val_df_out)]:
        n_pos = (df["label"] == 1).sum()
        print(f"{name}: {len(df)} rows, {n_pos} pos ({n_pos / len(df) * 100:.1f}%), "
              f"{df['user_id'].nunique()} users")

    return train_df, val_df_out


# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------

def define_feature_sets(kg_cols, kg_emb_cols=None):
    """Define feature sets for each ablation variant."""
    if kg_emb_cols is None:
        kg_emb_cols = []

    v1_features = ["cf_score", "kg_recall_score"]
    v2_features = ["cf_score", "kg_recall_score", "content_similarity", "popularity", "vote_count"]
    v3_features = v2_features + kg_cols

    feature_sets = {
        "V1 (CF)": v1_features,
        "V2 (CF+Content)": v2_features,
        "V3 (CF+Content+KG)": v3_features,
    }

    if kg_emb_cols:
        feature_sets["V3e (CF+Content+KGEmb)"] = v2_features + kg_emb_cols
        feature_sets["V4 (CF+Content+KG+Emb)"] = v3_features + kg_emb_cols

    return feature_sets


# ---------------------------------------------------------------------------
# Group construction for LambdaMART
# ---------------------------------------------------------------------------

def build_group_array(df, max_per_group=5000):
    """
    Build the group array for LambdaMART.
    Data must be sorted by user_id. Each entry = number of candidates for that user.
    Groups exceeding max_per_group are truncated (keep all positives + sample negatives).
    Returns (sorted_df, group_array).
    """
    chunks = []
    for uid, group in df.groupby("user_id"):
        if len(group) <= max_per_group:
            chunks.append(group)
        else:
            pos = group[group["label"] == 1]
            neg = group[group["label"] == 0]
            n_neg = max_per_group - len(pos)
            if n_neg > 0 and len(neg) > n_neg:
                neg = neg.sample(n=n_neg, random_state=42)
            chunks.append(pd.concat([pos, neg]))

    sorted_df = pd.concat(chunks).sort_values("user_id").reset_index(drop=True)
    group_sizes = sorted_df.groupby("user_id").size().values
    return sorted_df, group_sizes


# ---------------------------------------------------------------------------
# Training: Pointwise and LambdaMART
# ---------------------------------------------------------------------------

def train_pointwise(train_df, val_df, feature_cols, params=None):
    """Train LightGBM with pointwise (binary) objective."""
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "min_child_samples": 20,
            "verbose": -1,
            "seed": 42,
        }

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    return model


def train_lambdamart(train_df, val_df, feature_cols, params=None):
    """
    Train LightGBM with LambdaMART (lambdarank) objective.

    Key differences from pointwise:
      - objective = "lambdarank"
      - metric = "ndcg" (directly optimizes NDCG)
      - requires group parameter: number of candidates per query (user)
      - data must be sorted by user_id within each split
      - label can be binary (0/1) or graded relevance
    """
    if params is None:
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "num_leaves": 31,
            "learning_rate": 0.1,
            "min_child_samples": 20,
            "verbose": -1,
            "seed": 42,
        }

    # Sort by user and build group arrays
    train_sorted, train_group = build_group_array(train_df)
    val_sorted, val_group = build_group_array(val_df)

    X_train = train_sorted[feature_cols].values
    y_train = train_sorted["label"].values
    X_val = val_sorted[feature_cols].values
    y_val = val_sorted["label"].values

    dtrain = lgb.Dataset(X_train, label=y_train, group=train_group)
    dval = lgb.Dataset(X_val, label=y_val, group=val_group, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    return model


def train_lgbm(train_df, val_df, feature_cols, params=None, method="pointwise"):
    """Unified training entry point."""
    if method == "lambdamart":
        return train_lambdamart(train_df, val_df, feature_cols, params=params)
    else:
        return train_pointwise(train_df, val_df, feature_cols, params=params)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ranker(model, test_df, feature_cols, k=10):
    """Evaluate ranker on test candidates."""
    test_df = test_df.copy()
    X_test = test_df[feature_cols].values
    test_df["pred_score"] = model.predict(X_test)

    predictions = defaultdict(list)
    ground_truth = defaultdict(set)

    for uid, group in test_df.groupby("user_id"):
        sorted_group = group.sort_values("pred_score", ascending=False)
        predictions[uid] = sorted_group["movie_id"].tolist()
        pos_items = group[group["label"] == 1]["movie_id"].tolist()
        ground_truth[uid] = set(pos_items)

    total_items = test_df["movie_id"].nunique()
    results, per_user_ndcg = evaluate_all(
        predictions, ground_truth, k=k, total_items=total_items, ks=[1, 5, 10]
    )
    return results, per_user_ndcg


def evaluate_recall_baseline(test_df, k=10):
    """Evaluate recall model's ranking on the same candidate pool."""
    test_df = test_df.copy()

    predictions = defaultdict(list)
    ground_truth = defaultdict(set)

    for uid, group in test_df.groupby("user_id"):
        sorted_group = group.sort_values("cf_score", ascending=False)
        predictions[uid] = sorted_group["movie_id"].tolist()
        pos_items = group[group["label"] == 1]["movie_id"].tolist()
        ground_truth[uid] = set(pos_items)

    total_items = test_df["movie_id"].nunique()
    results, per_user_ndcg = evaluate_all(
        predictions, ground_truth, k=k, total_items=total_items, ks=[1, 5, 10]
    )
    return results, per_user_ndcg


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def hp_search(train_df, val_df, feature_cols, method="pointwise", k=10):
    """Grid search LightGBM hyperparameters on validation set."""
    param_grid = {
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
        "min_child_samples": [20, 50],
    }

    keys = list(param_grid.keys())
    best_ndcg = -1
    best_params = None

    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        params["verbose"] = -1
        params["seed"] = 42

        if method == "lambdamart":
            params["objective"] = "lambdarank"
            params["metric"] = "ndcg"
            params["ndcg_eval_at"] = [10]
        else:
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"

        model = train_lgbm(train_df, val_df, feature_cols,
                           params=params, method=method)
        results, _ = evaluate_ranker(model, val_df, feature_cols, k=k)
        ndcg = results[f"NDCG@{k}"]

        print(f"    {dict(zip(keys, values))} -> Val NDCG@{k}: {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_params = params.copy()

    print(f"  Best: {best_params}, Val NDCG@{k}: {best_ndcg:.4f}")
    return best_params


def load_kg_emb_features(path):
    """Load pre-computed KG embedding features."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def run_ablation_matched(
    cf_scores_path="results/cf_scores.csv",
    train_recall_path="data/processed/train_recall_candidates.csv",
    val_recall_path="data/processed/val_recall_candidates.csv",
    test_recall_path="data/processed/test_recall_candidates.csv",
    kg_features_path="data/kg/kg_features_test_recall.csv",
    kg_emb_features_path="data/kg/kg_emb_features_test_recall.csv",
    train_path="data/processed/train.csv",
    tmdb_path="data/tmdb/tmdb_metadata.csv",
    k=10,
    do_hp_search=False,
    methods=None,
):
    """
    Run ablation with distribution-matched training.

    Key difference from run_ablation: both training AND evaluation use recall
    model's top-100 candidates, eliminating the train/test distribution mismatch
    where random negatives teach the model shortcuts that fail on hard negatives.

    Training labels come from val.csv (next-period interactions).
    Test labels come from test.csv.
    Features are the same for both (same recall candidates, same user history).
    """
    if methods is None:
        methods = ["pointwise", "lambdamart"]

    print("Loading data...")
    cf_scores_df = load_recall_scores(cf_scores_path) if os.path.exists(cf_scores_path) else None
    popularity_df = compute_popularity_from_train(train_path)
    tmdb_df = load_tmdb_features(tmdb_path)

    # Same features for train/val/test (same recall candidates, different labels)
    content_sim_recall = load_content_similarity("test_recall")
    kg_recall = pd.read_csv(kg_features_path) if os.path.exists(kg_features_path) else None
    kg_emb_recall = load_kg_emb_features(kg_emb_features_path)

    train_cands = pd.read_csv(train_recall_path)
    val_cands = pd.read_csv(val_recall_path)
    test_cands = pd.read_csv(test_recall_path)

    print("Building feature DataFrames...")
    train_feat, kg_cols, kg_emb_cols = build_feature_df(
        train_cands, cf_scores_df, kg_recall, content_sim_recall, popularity_df, tmdb_df, kg_emb_recall
    )
    val_feat, _, _ = build_feature_df(
        val_cands, cf_scores_df, kg_recall, content_sim_recall, popularity_df, tmdb_df, kg_emb_recall
    )
    test_feat, _, _ = build_feature_df(
        test_cands, cf_scores_df, kg_recall, content_sim_recall, popularity_df, tmdb_df, kg_emb_recall
    )

    print(f"  Train: {len(train_feat)} samples ({(train_feat['label']==1).sum()} pos)")
    print(f"  Val:   {len(val_feat)} samples ({(val_feat['label']==1).sum()} pos)")
    print(f"  Test:  {len(test_feat)} samples ({(test_feat['label']==1).sum()} pos)")
    print(f"  KG columns: {kg_cols}")
    if kg_emb_cols:
        print(f"  KG embedding columns: {kg_emb_cols}")

    feature_sets = define_feature_sets(kg_cols, kg_emb_cols)

    # Recall-only baseline
    print("\n" + "=" * 60)
    print("  Recall-only baseline (rank by cf_score)")
    print("=" * 60)
    recall_results, recall_per_user = evaluate_recall_baseline(test_feat, k=k)
    print_results(recall_results, "Recall-only")

    all_results = {"Recall-only": recall_results}
    all_per_user = {"Recall-only": recall_per_user}
    all_importance = {}

    for method in methods:
        method_label = "Pointwise" if method == "pointwise" else "LambdaMART"
        print(f"\n{'#'*70}")
        print(f"  RANKING METHOD: {method_label}")
        print(f"{'#'*70}")

        shared_params = None
        if do_hp_search:
            v3_features = [f for f in feature_sets["V3 (CF+Content+KG)"]
                           if f in train_feat.columns]
            print(f"\n  HP search for {method_label}...")
            shared_params = hp_search(train_feat, val_feat, v3_features,
                                      method=method, k=k)

        for variant_name, features in feature_sets.items():
            available = [f for f in features if f in train_feat.columns]
            full_name = f"{variant_name} [{method_label}]"

            print(f"\n{'='*60}")
            print(f"  {full_name}: {available}")
            print(f"{'='*60}")

            model = train_lgbm(train_feat, val_feat, available,
                               params=shared_params, method=method)
            results, per_user = evaluate_ranker(model, test_feat, available, k=k)
            print_results(results, full_name)

            all_results[full_name] = results
            all_per_user[full_name] = per_user

            importance = dict(zip(
                available, model.feature_importance(importance_type="gain")
            ))
            all_importance[full_name] = importance

    # Summary table
    print("\n" + "=" * 90)
    print("  FULL COMPARISON (Distribution-Matched Training)")
    print("=" * 90)
    header = (f"{'Variant':<40} "
              f"{'NDCG@1':>8} {'NDCG@5':>8} {'NDCG@10':>8} "
              f"{'Recall@1':>9} {'Recall@5':>9} {'Recall@10':>10} "
              f"{'Hit@10':>8} {'MRR@10':>8}")
    print(header)
    print("-" * len(header))
    for variant, metrics in all_results.items():
        print(
            f"{variant:<40} "
            f"{metrics.get('NDCG@1', 0):>8.4f} "
            f"{metrics.get('NDCG@5', 0):>8.4f} "
            f"{metrics.get('NDCG@10', 0):>8.4f} "
            f"{metrics.get('Recall@1', 0):>9.4f} "
            f"{metrics.get('Recall@5', 0):>9.4f} "
            f"{metrics.get('Recall@10', 0):>10.4f} "
            f"{metrics.get('Hit@10', 0):>8.4f} "
            f"{metrics.get('MRR@10', 0):>8.4f}"
        )

    # Statistical tests
    print("\n  Statistical Tests:")
    test_pairs = [
        ("V3 (CF+Content+KG) [Pointwise]", "V2 (CF+Content) [Pointwise]",
         "Pointwise: KG contribution"),
        ("V3 (CF+Content+KG) [LambdaMART]", "V2 (CF+Content) [LambdaMART]",
         "LambdaMART: KG contribution"),
        ("V3 (CF+Content+KG) [Pointwise]", "Recall-only",
         "V3 Pointwise vs Recall-only"),
        ("V3 (CF+Content+KG) [LambdaMART]", "Recall-only",
         "V3 LambdaMART vs Recall-only"),
    ]
    for name_a, name_b, desc in test_pairs:
        if name_a in all_per_user and name_b in all_per_user:
            a_ndcg = all_per_user[name_a]
            b_ndcg = all_per_user[name_b]
            common = set(a_ndcg.keys()) & set(b_ndcg.keys())
            if len(common) > 1:
                a_vals = [a_ndcg[u] for u in common]
                b_vals = [b_ndcg[u] for u in common]
                t_stat, p_value = ttest_rel(a_vals, b_vals)
                sig = "Significant" if p_value < 0.05 else "Not significant"
                mean_diff = np.mean(a_vals) - np.mean(b_vals)
                print(f"    {desc}:")
                print(f"      diff={mean_diff:+.4f}, t={t_stat:.4f}, "
                      f"p={p_value:.6f} -> {sig}")

    # Feature importance
    for key in all_importance:
        if "V3" in key:
            print(f"\n  Feature Importance ({key}):")
            sorted_imp = sorted(all_importance[key].items(),
                                key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_imp:
                print(f"    {feat:<35s}: {imp:.1f}")

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
    run_ablation_matched()
