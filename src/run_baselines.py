"""
Run all baseline models and collect results.
Uses train/val/test split with validation-based model selection.
"""
import json
import os
import pickle

from src.models.item_cf import run_item_cf
from src.models.matrix_factorization import run_mf
from src.models.lightgcn import run_lightgcn


def main():
    os.makedirs("results", exist_ok=True)
    all_results = {}
    all_per_user = {}

    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"

    # 1. Item-CF (no validation needed, non-parametric)
    print("\n" + "=" * 60)
    print("  Running Item-CF")
    print("=" * 60)
    results, per_user = run_item_cf(
        train_path=train_path, test_path=test_path
    )
    all_results["Item-CF"] = results
    all_per_user["Item-CF"] = per_user

    # 2. Matrix Factorization (with validation-based early stopping)
    print("\n" + "=" * 60)
    print("  Running Matrix Factorization")
    print("=" * 60)
    results, per_user = run_mf(
        train_path=train_path, val_path=val_path, test_path=test_path,
        embed_dim=64, lr=1e-3, epochs=50
    )
    all_results["MF-64"] = results
    all_per_user["MF-64"] = per_user

    # 3. LightGCN (with validation-based early stopping)
    print("\n" + "=" * 60)
    print("  Running LightGCN")
    print("=" * 60)
    results, per_user = run_lightgcn(
        train_path=train_path, val_path=val_path, test_path=test_path,
        embed_dim=64, n_layers=3, lr=1e-3, epochs=100
    )
    all_results["LightGCN"] = results
    all_per_user["LightGCN"] = per_user

    # Summary
    print("\n" + "=" * 60)
    print("  BASELINE COMPARISON")
    print("=" * 60)

    header = (f"{'Model':<15} "
              f"{'NDCG@1':>8} {'NDCG@5':>8} {'NDCG@10':>8} "
              f"{'Recall@1':>9} {'Recall@5':>9} {'Recall@10':>10} "
              f"{'Hit@10':>8} {'MRR@10':>8} {'Coverage':>10}")
    print(header)
    print("-" * len(header))
    for model_name, metrics in all_results.items():
        print(
            f"{model_name:<15} "
            f"{metrics.get('NDCG@1', 0):>8.4f} "
            f"{metrics.get('NDCG@5', 0):>8.4f} "
            f"{metrics.get('NDCG@10', 0):>8.4f} "
            f"{metrics.get('Recall@1', 0):>9.4f} "
            f"{metrics.get('Recall@5', 0):>9.4f} "
            f"{metrics.get('Recall@10', 0):>10.4f} "
            f"{metrics.get('Hit@10', 0):>8.4f} "
            f"{metrics.get('MRR@10', 0):>8.4f} "
            f"{metrics.get('Coverage', 0):>10.4f}"
        )

    # Save results
    with open("results/baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results/baseline_results.json")

    with open("results/per_user_ndcg.pkl", "wb") as f:
        pickle.dump(all_per_user, f)
    print("Per-user NDCG saved to results/per_user_ndcg.pkl")


if __name__ == "__main__":
    main()
