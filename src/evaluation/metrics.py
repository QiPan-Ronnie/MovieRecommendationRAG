"""
Unified evaluation metrics for recommendation models.
Supports multiple K values (e.g., @1, @5, @10).
"""
import numpy as np
from collections import defaultdict


def hit_at_k(ranked_list, ground_truth, k=10):
    """1 if any ground truth item appears in top-k, else 0."""
    return 1.0 if len(set(ranked_list[:k]) & ground_truth) > 0 else 0.0


def ndcg_at_k(ranked_list, ground_truth, k=10):
    """Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    n_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_list, ground_truth, k=10):
    """Fraction of ground truth items that appear in top-k."""
    if len(ground_truth) == 0:
        return 0.0
    hits = len(set(ranked_list[:k]) & ground_truth)
    return hits / len(ground_truth)


def mrr(ranked_list, ground_truth, k=None):
    """Mean Reciprocal Rank: 1/rank of the first relevant item within top-k."""
    cutoff = k if k is not None else len(ranked_list)
    for i, item in enumerate(ranked_list[:cutoff]):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def coverage(all_recommendations, total_items):
    """Fraction of total items that appear in any recommendation list."""
    recommended = set()
    for rec_list in all_recommendations:
        recommended.update(rec_list)
    return len(recommended) / total_items if total_items > 0 else 0.0


def evaluate_all(predictions, ground_truth, k=10, total_items=None, ks=None):
    """
    Evaluate all metrics at one or more K values.

    Args:
        predictions: dict[user_id] -> list of movie_ids (ranked by score, descending)
        ground_truth: dict[user_id] -> set of movie_ids (positive items in test)
        k: primary cutoff (used for per_user_ndcg and coverage)
        total_items: total number of items (for coverage)
        ks: list of K values to evaluate (e.g., [1, 5, 10]).
            If None, defaults to [k].

    Returns:
        dict of metric_name -> value (includes all K values)
        dict of user_id -> per-user NDCG at primary k (for statistical tests)
    """
    if ks is None:
        ks = [k]
    # Ensure primary k is included
    if k not in ks:
        ks.append(k)
    ks = sorted(set(ks))

    # Per-K accumulators
    per_k = {kv: {"hits": [], "ndcgs": [], "recalls": [], "mrrs": []} for kv in ks}
    per_user_ndcg = {}
    all_recs = []

    users = set(predictions.keys()) & set(ground_truth.keys())

    for uid in users:
        pred = predictions[uid]
        gt = ground_truth[uid]
        if len(gt) == 0:
            continue

        for kv in ks:
            per_k[kv]["hits"].append(hit_at_k(pred, gt, kv))
            per_k[kv]["ndcgs"].append(ndcg_at_k(pred, gt, kv))
            per_k[kv]["recalls"].append(recall_at_k(pred, gt, kv))
            per_k[kv]["mrrs"].append(mrr(pred, gt, kv))

        # Per-user NDCG at primary k
        per_user_ndcg[uid] = ndcg_at_k(pred, gt, k)
        all_recs.append(pred[:k])

    results = {}
    for kv in ks:
        results[f"Hit@{kv}"] = np.mean(per_k[kv]["hits"])
        results[f"NDCG@{kv}"] = np.mean(per_k[kv]["ndcgs"])
        results[f"Recall@{kv}"] = np.mean(per_k[kv]["recalls"])
        results[f"MRR@{kv}"] = np.mean(per_k[kv]["mrrs"])

    if total_items is not None:
        results["Coverage"] = coverage(all_recs, total_items)

    return results, per_user_ndcg


def print_results(results, model_name="Model"):
    """Pretty print evaluation results, grouped by K."""
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    for metric, value in results.items():
        print(f"  {metric:15s}: {value:.4f}")
    print(f"{'='*50}")
