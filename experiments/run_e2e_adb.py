"""
End-to-end evaluation of A+D+B (RotatE) variant through the full ranker pipeline.

Steps:
  1. Backup current intermediate files
  2. Rebuild KG with co_liked=30K
  3. Train RotatE with balanced sampling
  4. Generate improved multi-recall candidates
  5. Label candidates for ranker train/val/test
  6. Compute features (content similarity, KG features, KG embedding features)
  7. Run ranker ablation (Pointwise + LambdaMART)
  8. Save results to results/kg_recall_experiments/ADB_e2e/
  9. Restore original files
"""
import os
import sys
import shutil
import json

BACKUP_DIR = "backup/e2e_temp"
ADB_RESULTS_DIR = "results/kg_recall_experiments/ADB_e2e"

# Files that get overwritten during pipeline
INTERMEDIATE_FILES = [
    "results/multi_recall_scores.csv",
    "data/processed/train_recall_candidates.csv",
    "data/processed/val_recall_candidates.csv",
    "data/processed/test_recall_candidates.csv",
    "data/processed/content_sim_train_recall.csv",
    "data/processed/content_sim_val_recall.csv",
    "data/processed/content_sim_test_recall.csv",
    "data/kg/kg_features_train_recall.csv",
    "data/kg/kg_features_val_recall.csv",
    "data/kg/kg_features_test_recall.csv",
    "data/kg/kg_emb_features_train_recall.csv",
    "data/kg/kg_emb_features_val_recall.csv",
    "data/kg/kg_emb_features_test_recall.csv",
    "results/ablation_results.json",
    "results/ablation_per_user.pkl",
    "results/feature_importance.json",
]

KG_FILES = [
    "data/kg/triples.csv",
    "data/kg/entity2id.csv",
    "data/kg/kg_graph.pkl",
    "data/kg/transe_entity_emb.npy",
    "data/kg/transe_relation_emb.npy",
    "data/kg/transe_relation2id.json",
]


def backup_files():
    """Backup all files that will be overwritten."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for f in INTERMEDIATE_FILES + KG_FILES:
        if os.path.exists(f):
            dest = os.path.join(BACKUP_DIR, f.replace("/", "__"))
            shutil.copy2(f, dest)
    print(f"Backed up {len(os.listdir(BACKUP_DIR))} files to {BACKUP_DIR}/")


def restore_files():
    """Restore all backed-up files."""
    if not os.path.exists(BACKUP_DIR):
        print("No backup found, skipping restore.")
        return
    for fname in os.listdir(BACKUP_DIR):
        original_path = fname.replace("__", "/")
        src = os.path.join(BACKUP_DIR, fname)
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        shutil.copy2(src, original_path)
    shutil.rmtree(BACKUP_DIR)
    print("Restored all original files.")


def main():
    os.makedirs(ADB_RESULTS_DIR, exist_ok=True)

    # =========================================
    # Step 1: Backup
    # =========================================
    print("\n[1/9] Backing up current files...")
    backup_files()

    try:
        # =========================================
        # Step 2: Rebuild KG with co_liked=30K
        # =========================================
        print("\n[2/9] Rebuilding KG with co_liked=30K...")
        from run_kg_recall_experiments import build_kg_with_coliked_limit
        build_kg_with_coliked_limit(max_coliked=30000)

        # =========================================
        # Step 3: Train RotatE with balanced sampling
        # =========================================
        print("\n[3/9] Training RotatE (balanced, 300 epochs)...")
        from kg.rotate import train_rotate
        train_rotate(balanced=True, epochs=300, dim=128, gamma=12.0, lr=0.001)

        # =========================================
        # Step 4: Generate improved multi-recall
        # =========================================
        print("\n[4/9] Generating improved multi-recall candidates...")
        from run_kg_recall_experiments import generate_multi_recall_improved
        generate_multi_recall_improved(
            output_path="results/multi_recall_scores.csv",
            max_history=None,
            use_idf_weighting=True,
            use_time_decay=True,
        )

        # =========================================
        # Step 5: Label candidates
        # =========================================
        print("\n[5/9] Labeling recall candidates for ranker...")
        scores_path = "results/multi_recall_scores.csv"
        from ranker.ranker import (
            build_recall_test_candidates,
            build_recall_train_val_candidates,
        )
        build_recall_test_candidates(
            cf_scores_path=scores_path,
            test_path="data/processed/test.csv",
            output_path="data/processed/test_recall_candidates.csv",
        )
        build_recall_train_val_candidates(
            cf_scores_path=scores_path,
            val_path="data/processed/val.csv",
            train_output="data/processed/train_recall_candidates.csv",
            val_output="data/processed/val_recall_candidates.csv",
        )

        # =========================================
        # Step 6: Compute features
        # =========================================
        print("\n[6/9] Computing content similarity...")
        from kg.content_similarity import main as content_sim_main
        content_sim_main()

        print("\n[7/9] Computing KG hand-crafted features...")
        from kg.kg_features import main as kg_feat_main
        kg_feat_main()

        print("\n[8/9] Computing KG embedding features...")
        from kg.kg_embedding_features import main as kg_emb_main
        kg_emb_main()

        # =========================================
        # Step 7: Run ranker ablation
        # =========================================
        print("\n[9/9] Running ranker ablation...")
        from ranker.ranker import run_ablation_matched
        run_ablation_matched(
            cf_scores_path=scores_path,
            do_hp_search=False,
        )

        # =========================================
        # Save A+D+B results
        # =========================================
        print("\nSaving A+D+B end-to-end results...")
        for f in ["results/ablation_results.json", "results/feature_importance.json",
                   "results/ablation_per_user.pkl"]:
            if os.path.exists(f):
                dest = os.path.join(ADB_RESULTS_DIR, os.path.basename(f))
                shutil.copy2(f, dest)

        print(f"\nA+D+B results saved to {ADB_RESULTS_DIR}/")

    finally:
        # =========================================
        # Restore original files
        # =========================================
        print("\nRestoring original files...")
        restore_files()

    # =========================================
    # Compare with baseline
    # =========================================
    print("\n" + "=" * 70)
    print("  COMPARISON: Baseline vs A+D+B (RotatE)")
    print("=" * 70)

    baseline_path = "results/ablation_results.json"  # restored
    adb_path = os.path.join(ADB_RESULTS_DIR, "ablation_results.json")

    if os.path.exists(baseline_path) and os.path.exists(adb_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        with open(adb_path) as f:
            adb = json.load(f)

        key_variants = [
            "Recall-only",
            "V3 (CF+Content+KG) [Pointwise]",
            "V3 (CF+Content+KG) [LambdaMART]",
            "V4 (CF+Content+KG+Emb) [Pointwise]",
            "V4 (CF+Content+KG+Emb) [LambdaMART]",
        ]

        print(f"\n{'Variant':<42} {'NDCG@10':>9} {'Recall@10':>10} {'Hit@10':>8}")
        print("-" * 75)

        for v in key_variants:
            if v in baseline:
                b = baseline[v]
                print(f"{'[Base] ' + v:<42} {b.get('NDCG@10',0):>9.4f} "
                      f"{b.get('Recall@10',0):>10.4f} {b.get('Hit@10',0):>8.4f}")
            if v in adb:
                a = adb[v]
                print(f"{'[ADB]  ' + v:<42} {a.get('NDCG@10',0):>9.4f} "
                      f"{a.get('Recall@10',0):>10.4f} {a.get('Hit@10',0):>8.4f}")
                if v in baseline:
                    b = baseline[v]
                    ndcg_diff = (a.get('NDCG@10',0) - b.get('NDCG@10',0)) / b.get('NDCG@10',1) * 100
                    recall_diff = (a.get('Recall@10',0) - b.get('Recall@10',0)) / b.get('Recall@10',1) * 100
                    print(f"{'  >> Change':<42} {ndcg_diff:>+8.1f}% {recall_diff:>+9.1f}%")
            print()


if __name__ == "__main__":
    main()
