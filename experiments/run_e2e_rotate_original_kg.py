"""
Experiment 4: RotatE on original KG (100K co_liked) + candidate compensation.

Key differences from Experiment 3 (A+D+B):
  - KG unchanged (134K triples, 100K co_liked) — preserves CF signal
  - RotatE trained with balanced sampling on original KG
  - n_kg increased dynamically to ensure every user gets exactly 100 candidates
  - Improved user profile (IDF + time decay + full history)

Pipeline:
  1. Train RotatE on original KG (balanced sampling)
  2. Generate improved multi-recall with candidate compensation
  3. Label candidates → compute features → run ranker ablation
  4. Compare with baseline
"""
import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


RESULTS_DIR = "results/kg_recall_experiments/rotate_original_kg"
BACKUP_DIR = "backup/e2e_temp_exp4"

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

EMB_FILES = [
    "data/kg/transe_entity_emb.npy",
    "data/kg/transe_relation_emb.npy",
    "data/kg/transe_relation2id.json",
]


def backup_files():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for f in INTERMEDIATE_FILES + EMB_FILES:
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


def generate_recall_with_compensation(
    emb_path="data/kg/transe_entity_emb.npy",
    cf_scores_path=None,
    train_path="data/processed/train.csv",
    output_path="results/multi_recall_scores.csv",
    n_cf=70,
    n_kg_base=50,
    n_total=100,
    use_idf_weighting=True,
    use_time_decay=True,
):
    """
    Multi-recall with candidate compensation.
    Dynamically increases n_kg per user to ensure exactly n_total candidates after dedup.
    """
    from models.multi_recall import load_transe_for_recall, _auto_detect_recall_scores

    if cf_scores_path is None:
        cf_scores_path, auto_score_col = _auto_detect_recall_scores()
    else:
        auto_score_col = None

    print(f"Multi-recall (compensated): CF top-{n_cf} + KG top-{n_kg_base}+ -> {n_total}")

    # Load CF scores
    cf_df = pd.read_csv(cf_scores_path)
    if auto_score_col and auto_score_col in cf_df.columns and "cf_score" not in cf_df.columns:
        cf_df = cf_df.rename(columns={auto_score_col: "cf_score"})
    cf_by_user = {}
    for uid, group in cf_df.groupby("user_id"):
        sorted_g = group.sort_values("cf_score", ascending=False)
        cf_by_user[uid] = list(zip(sorted_g["movie_id"], sorted_g["cf_score"]))

    # Load embeddings
    emb_normed, movie2idx = load_transe_for_recall(emb_path=emb_path)
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

    # IDF weights
    movie_idf = {}
    if use_idf_weighting:
        n_users = len(user_train_items)
        movie_user_count = defaultdict(int)
        for uid, items in user_train_items.items():
            for mid in items:
                movie_user_count[mid] += 1
        for mid, count in movie_user_count.items():
            movie_idf[mid] = np.log(n_users / (count + 1))

    all_users = sorted(cf_by_user.keys())
    records = []
    source_stats = {"cf_only": 0, "kg_only": 0, "both": 0}
    compensation_count = 0

    for uid in tqdm(all_users, desc="Multi-recall (compensated)"):
        exclude = user_train_items.get(uid, set())
        history = user_history.get(uid, [])

        # Build user profile
        valid_movies = [mid for mid in history if mid in movie2idx]
        if not valid_movies:
            cf_cands = cf_by_user.get(uid, [])[:n_total]
            for mid, cf_score in cf_cands:
                records.append({"user_id": uid, "movie_id": mid,
                                "cf_score": cf_score, "kg_recall_score": 0.0})
                source_stats["cf_only"] += 1
            continue

        valid_indices = [movie2idx[mid] for mid in valid_movies]
        weights = np.ones(len(valid_movies))

        if use_idf_weighting:
            for i, mid in enumerate(valid_movies):
                weights[i] *= movie_idf.get(mid, 1.0)

        if use_time_decay:
            n = len(valid_movies)
            if n > 1:
                positions = np.linspace(0, 1, n)
                decay = np.exp(-2.0 * (1.0 - positions))
                weights *= decay

        weights = weights / weights.sum()
        user_profile = (emb_normed[valid_indices] * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile /= norm

        # CF candidates
        cf_cands = cf_by_user.get(uid, [])[:n_cf]
        cf_set = {mid for mid, _ in cf_cands}

        # KG candidates — compute all scores, then pick enough to fill n_total
        candidate_mids = [mid for mid in all_kg_movies if mid not in exclude and mid in movie2idx]
        if candidate_mids:
            candidate_indices = [movie2idx[mid] for mid in candidate_mids]
            candidate_embs = emb_normed[candidate_indices]
            scores = candidate_embs @ user_profile

            # Sort all KG candidates by score
            sorted_indices = np.argsort(scores)[::-1]

            # Take enough KG candidates to fill the gap
            n_cf_actual = len(cf_cands)
            # Start with n_kg_base, increase if too much overlap
            kg_cands = []
            for i in sorted_indices:
                if len(kg_cands) >= n_kg_base:
                    # Check if we have enough unique after merge
                    kg_unique = sum(1 for mid, _ in kg_cands if mid not in cf_set)
                    total_unique = n_cf_actual + kg_unique
                    if total_unique >= n_total:
                        break
                mid = candidate_mids[i]
                kg_cands.append((mid, float(scores[i])))
                if len(kg_cands) >= 200:  # hard cap
                    break

            if len(kg_cands) > n_kg_base:
                compensation_count += 1
        else:
            kg_cands = []

        kg_dict = {mid: score for mid, score in kg_cands}

        # Merge
        merged = {}
        for mid, cf_score in cf_cands:
            kg_score = kg_dict.get(mid, 0.0)
            merged[mid] = {"cf_score": cf_score, "kg_recall_score": kg_score,
                           "source": "both" if mid in kg_dict else "cf"}

        for mid, kg_score in kg_cands:
            if mid not in merged:
                merged[mid] = {"cf_score": 0.0, "kg_recall_score": kg_score, "source": "kg"}

        # Sort and take top n_total
        cf_items = [(mid, d) for mid, d in merged.items() if d["source"] != "kg"]
        kg_items = [(mid, d) for mid, d in merged.items() if d["source"] == "kg"]
        cf_items.sort(key=lambda x: x[1]["cf_score"], reverse=True)
        kg_items.sort(key=lambda x: x[1]["kg_recall_score"], reverse=True)
        final = (cf_items + kg_items)[:n_total]

        for mid, d in final:
            records.append({
                "user_id": uid, "movie_id": mid,
                "cf_score": d["cf_score"], "kg_recall_score": d["kg_recall_score"],
            })
            src = d["source"]
            if src == "cf":
                source_stats["cf_only"] += 1
            elif src == "kg":
                source_stats["kg_only"] += 1
            else:
                source_stats["both"] += 1

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_path, index=False)

    total = sum(source_stats.values())
    per_user = result_df.groupby("user_id").size()
    print(f"\n  Output: {len(result_df)} candidates, {result_df.user_id.nunique()} users")
    print(f"  Per user: mean={per_user.mean():.1f}, min={per_user.min()}, max={per_user.max()}")
    print(f"  Users compensated (needed extra KG): {compensation_count}")
    print(f"  Source: CF={source_stats['cf_only']/total*100:.1f}%, "
          f"KG={source_stats['kg_only']/total*100:.1f}%, "
          f"Both={source_stats['both']/total*100:.1f}%")
    print(f"  Unique movies: {result_df.movie_id.nunique()}")
    return result_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =========================================
    # Step 1: Backup
    # =========================================
    print("\n[1/8] Backing up current files...")
    backup_files()

    try:
        # =========================================
        # Step 2: Train RotatE on ORIGINAL KG (no KG rebuild!)
        # =========================================
        print("\n[2/8] Training RotatE on original KG (balanced, 300 epochs)...")
        from kg.rotate import train_rotate
        train_rotate(balanced=True, epochs=300, dim=128, gamma=12.0, lr=0.001)

        # =========================================
        # Step 3: Generate compensated multi-recall
        # =========================================
        print("\n[3/8] Generating multi-recall with candidate compensation...")
        generate_recall_with_compensation(
            output_path="results/multi_recall_scores.csv",
        )

        # =========================================
        # Step 4: Label candidates
        # =========================================
        print("\n[4/8] Labeling recall candidates...")
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
        # Step 5-7: Compute features
        # =========================================
        print("\n[5/8] Computing content similarity...")
        from kg.content_similarity import main as content_sim_main
        content_sim_main()

        print("\n[6/8] Computing KG hand-crafted features...")
        from kg.kg_features import main as kg_feat_main
        kg_feat_main()

        print("\n[7/8] Computing KG embedding features...")
        from kg.kg_embedding_features import main as kg_emb_main
        kg_emb_main()

        # =========================================
        # Step 8: Run ranker ablation
        # =========================================
        print("\n[8/8] Running ranker ablation...")
        from ranker.ranker import run_ablation_matched
        run_ablation_matched(cf_scores_path=scores_path, do_hp_search=False)

        # =========================================
        # Save results
        # =========================================
        print("\nSaving results...")
        for f in ["results/ablation_results.json", "results/feature_importance.json",
                   "results/ablation_per_user.pkl", "results/multi_recall_scores.csv"]:
            if os.path.exists(f):
                dest = os.path.join(RESULTS_DIR, os.path.basename(f))
                shutil.copy2(f, dest)

        # Also evaluate recall quality
        from run_kg_recall_experiments import evaluate_recall_quality
        recall_eval = evaluate_recall_quality("results/multi_recall_scores.csv")
        with open(os.path.join(RESULTS_DIR, "recall_eval.json"), "w") as f:
            json.dump(recall_eval, f, indent=2)

    finally:
        # =========================================
        # Restore
        # =========================================
        print("\nRestoring original files...")
        restore_files()

    # =========================================
    # Compare
    # =========================================
    print("\n" + "=" * 80)
    print("  COMPARISON: Baseline vs RotatE-OriginalKG")
    print("=" * 80)

    baseline_path = "results/ablation_results.json"
    new_path = os.path.join(RESULTS_DIR, "ablation_results.json")

    if os.path.exists(baseline_path) and os.path.exists(new_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        with open(new_path) as f:
            new = json.load(f)

        key_variants = [
            "Recall-only",
            "V2 (CF+Content) [LambdaMART]",
            "V3 (CF+Content+KG) [LambdaMART]",
            "V4 (CF+Content+KG+Emb) [LambdaMART]",
            "V2 (CF+Content) [Pointwise]",
            "V3 (CF+Content+KG) [Pointwise]",
            "V4 (CF+Content+KG+Emb) [Pointwise]",
        ]

        print(f"\n{'Variant':<42} {'Base NDCG':>10} {'New NDCG':>10} {'Δ':>8} | "
              f"{'Base R@10':>10} {'New R@10':>10} {'Δ':>8}")
        print("-" * 110)

        for v in key_variants:
            b = baseline.get(v, {})
            a = new.get(v, {})
            if not b or not a:
                continue
            bn, an = b.get('NDCG@10', 0), a.get('NDCG@10', 0)
            br, ar = b.get('Recall@10', 0), a.get('Recall@10', 0)
            nd = (an - bn) / max(bn, 1e-9) * 100
            rd = (ar - br) / max(br, 1e-9) * 100
            print(f"{v:<42} {bn:>10.4f} {an:>10.4f} {nd:>+7.1f}% | "
                  f"{br:>10.4f} {ar:>10.4f} {rd:>+7.1f}%")

        # KG lift comparison
        print("\n  KG Lift (V3 vs V2):")
        for method in ["Pointwise", "LambdaMART"]:
            v3 = f"V3 (CF+Content+KG) [{method}]"
            v2 = f"V2 (CF+Content) [{method}]"
            if v3 in baseline and v2 in baseline and v3 in new and v2 in new:
                b_lift = baseline[v3]["Recall@10"] - baseline[v2]["Recall@10"]
                n_lift = new[v3]["Recall@10"] - new[v2]["Recall@10"]
                print(f"    {method}: Base={b_lift:+.4f}, New={n_lift:+.4f} "
                      f"({n_lift/max(b_lift,1e-9)*100:.0f}% of baseline)")

    # Recall quality comparison
    recall_path = os.path.join(RESULTS_DIR, "recall_eval.json")
    if os.path.exists(recall_path):
        with open(recall_path) as f:
            r = json.load(f)
        print(f"\n  Recall Quality:")
        print(f"    KG-only hit rate: {r['kg_only_hit_rate']:.4f}")
        print(f"    KG score discrimination: {r.get('kg_score_discrimination', 0):.6f}")
        print(f"    Total positives: {r['total_positives_captured']}")
        print(f"    Movies covered: {r['total_movies']}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
