"""
Run all baseline models and collect results.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.item_cf import run_item_cf
from models.matrix_factorization import run_mf
from models.lightgcn import run_lightgcn


def main():
    os.makedirs("results", exist_ok=True)
    all_results = {}
    all_per_user = {}

    # 1. Item-CF
    print("\n" + "=" * 60)
    print("  Running Item-CF")
    print("=" * 60)
    results, per_user = run_item_cf()
    all_results["Item-CF"] = results
    all_per_user["Item-CF"] = per_user

    # 2. Matrix Factorization
    print("\n" + "=" * 60)
    print("  Running Matrix Factorization")
    print("=" * 60)
    results, per_user = run_mf(embed_dim=64, lr=1e-3, epochs=50)
    all_results["MF-64"] = results
    all_per_user["MF-64"] = per_user

    # 3. LightGCN
    print("\n" + "=" * 60)
    print("  Running LightGCN")
    print("=" * 60)
    results, per_user = run_lightgcn(embed_dim=64, n_layers=3, lr=1e-3, epochs=100)
    all_results["LightGCN"] = results
    all_per_user["LightGCN"] = per_user

    # Summary
    print("\n" + "=" * 60)
    print("  BASELINE COMPARISON")
    print("=" * 60)

    header = f"{'Model':<15} {'Hit@10':>8} {'NDCG@10':>8} {'Recall@10':>10} {'MRR':>8} {'Coverage':>10}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in all_results.items():
        print(f"{model_name:<15} {metrics.get('Hit@10', 0):>8.4f} {metrics.get('NDCG@10', 0):>8.4f} {metrics.get('Recall@10', 0):>10.4f} {metrics.get('MRR', 0):>8.4f} {metrics.get('Coverage', 0):>10.4f}")

    # Save results
    with open("results/baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results/baseline_results.json")

    # Save per-user NDCG for later statistical tests
    import pickle
    with open("results/per_user_ndcg.pkl", "wb") as f:
        pickle.dump(all_per_user, f)
    print("Per-user NDCG saved to results/per_user_ndcg.pkl")


if __name__ == "__main__":
    main()
