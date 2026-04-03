"""
Experiment 5: RotatE embedding as ranker features only.

Key idea: Keep baseline recall candidates UNCHANGED. Only replace
TransE embeddings with RotatE embeddings for kg_emb_features computation.
This tests whether RotatE's better discrimination improves V3e/V4 ranker
variants without disrupting the candidate pool.

Changes vs baseline:
  - data/kg/transe_entity_emb.npy temporarily replaced with RotatE embeddings
  - kg_emb_features recomputed for train/val/test recall candidates
  - Ranker re-run with new embedding features
  - Everything else (recall candidates, labels, cf_score, KG hand-crafted features,
    content similarity) stays exactly the same
"""
import os
import shutil
import json

RESULTS_DIR = "results/kg_recall_experiments/rotate_features_only"
BACKUP_DIR = "backup/e2e_temp_exp5"

# Only these files change
FILES_TO_BACKUP = [
    "data/kg/transe_entity_emb.npy",
    "data/kg/transe_relation_emb.npy",
    "data/kg/transe_relation2id.json",
    "data/kg/kg_emb_features_train_recall.csv",
    "data/kg/kg_emb_features_val_recall.csv",
    "data/kg/kg_emb_features_test_recall.csv",
    "results/ablation_results.json",
    "results/ablation_per_user.pkl",
    "results/feature_importance.json",
]


def backup_files():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for f in FILES_TO_BACKUP:
        if os.path.exists(f):
            dest = os.path.join(BACKUP_DIR, f.replace("/", "__"))
            shutil.copy2(f, dest)
    print(f"Backed up {len(os.listdir(BACKUP_DIR))} files.")


def restore_files():
    if not os.path.exists(BACKUP_DIR):
        return
    for fname in os.listdir(BACKUP_DIR):
        original_path = fname.replace("__", "/")
        src = os.path.join(BACKUP_DIR, fname)
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        shutil.copy2(src, original_path)
    shutil.rmtree(BACKUP_DIR)
    print("Restored all original files.")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/5] Backing up...")
    backup_files()

    try:
        # =========================================
        # Step 2: Train RotatE on original KG
        # Saves to data/kg/transe_entity_emb.npy (same filename, different content)
        # =========================================
        print("\n[2/5] Training RotatE on original KG (balanced, 300 epochs)...")
        from kg.rotate import train_rotate
        train_rotate(balanced=True, epochs=300, dim=128, gamma=12.0, lr=0.001)

        # =========================================
        # Step 3: Recompute ONLY kg_emb_features with RotatE embeddings
        # Recall candidates, labels, cf_score, KG hand-crafted features all unchanged
        # =========================================
        print("\n[3/5] Recomputing KG embedding features with RotatE...")
        from kg.kg_embedding_features import main as kg_emb_main
        kg_emb_main()

        # =========================================
        # Step 4: Run ranker ablation
        # Uses existing recall candidates + existing labels + existing features
        # Only kg_emb_features are different
        # =========================================
        print("\n[4/5] Running ranker ablation...")
        scores_path = "results/multi_recall_scores.csv"
        from ranker.ranker import run_ablation_matched
        run_ablation_matched(cf_scores_path=scores_path, do_hp_search=False)

        # =========================================
        # Step 5: Save results
        # =========================================
        print("\n[5/5] Saving results...")
        for f in ["results/ablation_results.json", "results/feature_importance.json",
                   "results/ablation_per_user.pkl"]:
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(RESULTS_DIR, os.path.basename(f)))

    finally:
        print("\nRestoring original files...")
        restore_files()

    # =========================================
    # Compare
    # =========================================
    print("\n" + "=" * 80)
    print("  COMPARISON: Baseline (TransE emb) vs RotatE emb (features only)")
    print("=" * 80)

    with open("results/ablation_results.json") as f:
        baseline = json.load(f)
    with open(os.path.join(RESULTS_DIR, "ablation_results.json")) as f:
        rotate = json.load(f)

    key_variants = [
        "Recall-only",
        "V1 (CF) [Pointwise]",
        "V2 (CF+Content) [Pointwise]",
        "V3 (CF+Content+KG) [Pointwise]",
        "V3e (CF+Content+KGEmb) [Pointwise]",
        "V4 (CF+Content+KG+Emb) [Pointwise]",
        "V1 (CF) [LambdaMART]",
        "V2 (CF+Content) [LambdaMART]",
        "V3 (CF+Content+KG) [LambdaMART]",
        "V3e (CF+Content+KGEmb) [LambdaMART]",
        "V4 (CF+Content+KG+Emb) [LambdaMART]",
    ]

    print(f"\n{'Variant':<42} {'Base NDCG':>10} {'New NDCG':>10} {'Δ':>8} | "
          f"{'Base R@10':>10} {'New R@10':>10} {'Δ':>8}")
    print("-" * 110)

    for v in key_variants:
        b = baseline.get(v, {})
        a = rotate.get(v, {})
        if not b or not a:
            continue
        bn, an = b.get('NDCG@10', 0), a.get('NDCG@10', 0)
        br, ar = b.get('Recall@10', 0), a.get('Recall@10', 0)
        nd = (an - bn) / max(bn, 1e-9) * 100
        rd = (ar - br) / max(br, 1e-9) * 100
        # Highlight V3e and V4 (the ones that use embedding features)
        marker = " **" if "Emb" in v or "V4" in v else ""
        print(f"{v:<42} {bn:>10.4f} {an:>10.4f} {nd:>+7.1f}% | "
              f"{br:>10.4f} {ar:>10.4f} {rd:>+7.1f}%{marker}")

    # Sanity checks: V1/V2/V3 should be identical (no embedding features)
    print("\n  Sanity check (V1/V2/V3 should be unchanged - they don't use emb features):")
    for v in ["V3 (CF+Content+KG) [LambdaMART]", "Recall-only"]:
        b = baseline.get(v, {})
        a = rotate.get(v, {})
        diff = abs(b.get('NDCG@10', 0) - a.get('NDCG@10', 0))
        status = "OK (identical)" if diff < 0.0001 else f"CHANGED by {diff:.4f}"
        print(f"    {v}: {status}")

    # Feature importance comparison for V3e and V4
    base_imp_path = "results/feature_importance.json"
    new_imp_path = os.path.join(RESULTS_DIR, "feature_importance.json")
    if os.path.exists(base_imp_path) and os.path.exists(new_imp_path):
        with open(base_imp_path) as f:
            base_imp = json.load(f)
        with open(new_imp_path) as f:
            new_imp = json.load(f)

        for variant in ["V3e (CF+Content+KGEmb) [Pointwise]", "V4 (CF+Content+KG+Emb) [LambdaMART]"]:
            if variant in base_imp and variant in new_imp:
                print(f"\n  Feature Importance Change ({variant}):")
                bi = base_imp[variant]
                ni = new_imp[variant]
                all_feats = sorted(set(list(bi.keys()) + list(ni.keys())),
                                   key=lambda x: ni.get(x, 0), reverse=True)
                for feat in all_feats[:10]:
                    bv = bi.get(feat, 0)
                    nv = ni.get(feat, 0)
                    change = (nv - bv) / max(bv, 1) * 100
                    marker = " <-- emb" if "emb" in feat else ""
                    print(f"    {feat:<30s}: {bv:>8.1f} -> {nv:>8.1f} ({change:>+6.1f}%){marker}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
