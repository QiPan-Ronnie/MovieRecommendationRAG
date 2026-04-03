"""
KG Recall improvement experiments.

Variants:
  - baseline:  Original (TransE, co_liked=100K, max_history=20, uniform sampling)
  - A:         co_liked=30K + balanced relation sampling
  - A+D:       A + improved user profile (no history limit, IDF weighting, time decay)
  - A+D+B:     A+D + RotatE instead of TransE

Each variant produces its own multi_recall_scores and evaluation.
"""
import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def evaluate_recall_quality(recall_path, test_path="data/processed/test.csv", k=10):
    """Evaluate recall candidate quality."""
    mr = pd.read_csv(recall_path)
    test = pd.read_csv(test_path)
    test_set = set(zip(test.user_id, test.movie_id))

    mr["is_positive"] = mr.apply(lambda r: (r.user_id, r.movie_id) in test_set, axis=1)

    # Per-source hit rate
    cf_only = mr[(mr.cf_score > 0) & (mr.kg_recall_score == 0)]
    kg_only = mr[(mr.cf_score == 0) & (mr.kg_recall_score > 0)]
    both = mr[(mr.cf_score > 0) & (mr.kg_recall_score > 0)]

    results = {
        "total_candidates": len(mr),
        "total_movies": int(mr.movie_id.nunique()),
        "overall_hit_rate": float(mr.is_positive.mean()),
        "cf_only_count": len(cf_only),
        "cf_only_hit_rate": float(cf_only.is_positive.mean()) if len(cf_only) > 0 else 0,
        "kg_only_count": len(kg_only),
        "kg_only_hit_rate": float(kg_only.is_positive.mean()) if len(kg_only) > 0 else 0,
        "both_count": len(both),
        "both_hit_rate": float(both.is_positive.mean()) if len(both) > 0 else 0,
        "total_positives_captured": int(mr.is_positive.sum()),
        "kg_unique_positives": int(mr[(mr.cf_score == 0) & mr.is_positive].shape[0]),
    }

    # KG score discrimination
    kg_items = mr[mr.kg_recall_score > 0]
    if len(kg_items) > 0:
        pos_kg = kg_items[kg_items.is_positive].kg_recall_score
        neg_kg = kg_items[~kg_items.is_positive].kg_recall_score
        results["kg_score_pos_mean"] = float(pos_kg.mean()) if len(pos_kg) > 0 else 0
        results["kg_score_neg_mean"] = float(neg_kg.mean()) if len(neg_kg) > 0 else 0
        results["kg_score_discrimination"] = results["kg_score_pos_mean"] - results["kg_score_neg_mean"]
        results["kg_score_std"] = float(kg_items.kg_recall_score.std())

    # Per-user recall@K (on recall candidates)
    from evaluation.metrics import evaluate_all
    predictions = {}
    for uid, group in mr.groupby("user_id"):
        # Sort by combined score
        group = group.copy()
        group["rank_score"] = group["cf_score"] + group["kg_recall_score"]
        group = group.sort_values("rank_score", ascending=False)
        predictions[uid] = group["movie_id"].tolist()

    ground_truth = defaultdict(set)
    for _, row in test.iterrows():
        ground_truth[row["user_id"]].add(row["movie_id"])

    total_items = mr.movie_id.nunique()
    metrics, _ = evaluate_all(predictions, ground_truth, k=k, total_items=total_items, ks=[5, 10, 20])
    results["Recall@10"] = metrics.get("Recall@10", 0)
    results["NDCG@10"] = metrics.get("NDCG@10", 0)
    results["Hit@10"] = metrics.get("Hit@10", 0)
    results["Recall@20"] = metrics.get("Recall@20", 0)

    return results


def build_kg_with_coliked_limit(max_coliked=30000):
    """Rebuild KG with reduced co_liked edges."""
    from kg.build_kg import load_tmdb_metadata, build_triples, build_decade_triples
    from kg.build_kg import build_collaborative_triples, build_networkx_graph, print_kg_stats

    out_dir = "data/kg"
    metadata = load_tmdb_metadata()
    print(f"Loaded metadata for {len(metadata)} movies")

    triples, entity_types = build_triples(metadata)
    print(f"  Metadata triples: {len(triples)}")

    decade_triples, decade_entities = build_decade_triples(metadata)
    triples.extend(decade_triples)
    entity_types.update(decade_entities)
    print(f"  + Decade triples: {len(decade_triples)}")

    collab_triples = build_collaborative_triples(max_edges=max_coliked)
    triples.extend(collab_triples)

    triples_df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    triples_df.to_csv(os.path.join(out_dir, "triples.csv"), index=False)

    # Rebuild entity mapping (same entities, just fewer co_liked edges)
    # Need to include all entities that appear in triples
    all_entities_in_triples = set()
    for h, r, t in triples:
        all_entities_in_triples.add(h)
        all_entities_in_triples.add(t)

    # Merge with existing entity_types
    for e in all_entities_in_triples:
        if e not in entity_types:
            if e.startswith("movie_"):
                entity_types[e] = "movie"

    entities = sorted(entity_types.keys())
    entity2id = {e: i for i, e in enumerate(entities)}
    entity_df = pd.DataFrame([
        {"entity": e, "entity_id": entity2id[e], "type": entity_types[e]}
        for e in entities
    ])
    entity_df.to_csv(os.path.join(out_dir, "entity2id.csv"), index=False)

    G = build_networkx_graph(triples)
    import pickle
    with open(os.path.join(out_dir, "kg_graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    print_kg_stats(triples, entity_types, G)
    return triples_df


def generate_multi_recall_improved(
    emb_path="data/kg/transe_entity_emb.npy",
    cf_scores_path=None,
    train_path="data/processed/train.csv",
    output_path="results/multi_recall_scores.csv",
    n_cf=70,
    n_kg=50,
    n_total=100,
    max_history=None,
    use_idf_weighting=False,
    use_time_decay=False,
):
    """
    Improved multi-route recall with better user profile construction.

    Improvements over original:
    - max_history=None: use all history (not just last 20)
    - use_idf_weighting: weight rare movies higher in user profile
    - use_time_decay: exponential decay for older interactions
    """
    from models.multi_recall import load_transe_for_recall, _auto_detect_recall_scores

    if cf_scores_path is None:
        cf_scores_path, auto_score_col = _auto_detect_recall_scores()
    else:
        auto_score_col = None

    print(f"Multi-route recall (improved): CF top-{n_cf} + KG top-{n_kg} -> {n_total}")
    print(f"  max_history={max_history}, idf={use_idf_weighting}, time_decay={use_time_decay}")

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

    # Build user histories with timestamps
    train_df = pd.read_csv(train_path)
    has_timestamps = "timestamp" in train_df.columns
    if has_timestamps:
        train_df = train_df.sort_values("timestamp")

    user_history = defaultdict(list)
    user_timestamps = defaultdict(list)
    user_train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        uid, mid = row["user_id"], row["movie_id"]
        user_history[uid].append(mid)
        if has_timestamps:
            user_timestamps[uid].append(row["timestamp"])
        user_train_items[uid].add(mid)

    # Compute IDF weights for movies
    movie_idf = {}
    if use_idf_weighting:
        n_users = len(user_train_items)
        movie_user_count = defaultdict(int)
        for uid, items in user_train_items.items():
            for mid in items:
                movie_user_count[mid] += 1
        for mid, count in movie_user_count.items():
            movie_idf[mid] = np.log(n_users / (count + 1))
        print(f"  IDF range: [{min(movie_idf.values()):.2f}, {max(movie_idf.values()):.2f}]")

    # Limit history if specified
    if max_history is not None:
        for uid in user_history:
            user_history[uid] = user_history[uid][-max_history:]
            if has_timestamps:
                user_timestamps[uid] = user_timestamps[uid][-max_history:]

    all_users = sorted(cf_by_user.keys())
    records = []
    source_stats = {"cf_only": 0, "kg_only": 0, "both": 0}

    for uid in tqdm(all_users, desc="Multi-recall (improved)"):
        exclude = user_train_items.get(uid, set())
        history = user_history.get(uid, [])

        # --- Build user profile with improvements ---
        valid_movies = [mid for mid in history if mid in movie2idx]
        if not valid_movies:
            # CF-only fallback
            cf_cands = cf_by_user.get(uid, [])[:n_total]
            for mid, cf_score in cf_cands:
                records.append({"user_id": uid, "movie_id": mid, "cf_score": cf_score, "kg_recall_score": 0.0})
                source_stats["cf_only"] += 1
            continue

        valid_indices = [movie2idx[mid] for mid in valid_movies]
        weights = np.ones(len(valid_movies))

        # IDF weighting
        if use_idf_weighting:
            for i, mid in enumerate(valid_movies):
                weights[i] *= movie_idf.get(mid, 1.0)

        # Time decay: use position as proxy (history is sorted by time)
        if use_time_decay:
            n = len(valid_movies)
            if n > 1:
                positions = np.linspace(0, 1, n)  # 0=oldest, 1=newest
                decay = np.exp(-2.0 * (1.0 - positions))  # newest~1.0, oldest~0.13
                weights *= decay

        # Weighted mean
        weights = weights / weights.sum()
        user_profile = (emb_normed[valid_indices] * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile /= norm

        # KG recall
        candidate_mids = [mid for mid in all_kg_movies if mid not in exclude and mid in movie2idx]
        if candidate_mids:
            candidate_indices = [movie2idx[mid] for mid in candidate_mids]
            candidate_embs = emb_normed[candidate_indices]
            scores = candidate_embs @ user_profile
            top_indices = np.argsort(scores)[::-1][:n_kg]
            kg_cands = [(candidate_mids[i], float(scores[i])) for i in top_indices]
        else:
            kg_cands = []

        kg_dict = {mid: score for mid, score in kg_cands}

        # CF candidates
        cf_cands = cf_by_user.get(uid, [])[:n_cf]
        cf_dict = {mid: score for mid, score in cf_cands}

        # Merge
        merged = {}
        for mid, cf_score in cf_cands:
            kg_score = kg_dict.get(mid, 0.0)
            merged[mid] = {"cf_score": cf_score, "kg_recall_score": kg_score,
                           "source": "both" if mid in kg_dict else "cf"}

        for mid, kg_score in kg_cands:
            if mid not in merged:
                merged[mid] = {"cf_score": 0.0, "kg_recall_score": kg_score, "source": "kg"}

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
    print(f"\n  Output: {len(result_df)} candidates, {result_df.user_id.nunique()} users")
    print(f"  Source: CF={source_stats['cf_only']/total*100:.1f}%, "
          f"KG={source_stats['kg_only']/total*100:.1f}%, "
          f"Both={source_stats['both']/total*100:.1f}%")
    print(f"  Unique movies: {result_df.movie_id.nunique()}")
    return result_df


def restore_baseline_kg():
    """Restore original KG files from backup."""
    backup_dir = "backup/kg_baseline"
    target_dir = "data/kg"
    for f in ["triples.csv", "entity2id.csv", "transe_entity_emb.npy",
              "transe_relation_emb.npy", "transe_relation2id.json"]:
        src = os.path.join(backup_dir, f)
        dst = os.path.join(target_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print("Restored baseline KG files from backup.")


def run_experiment(variant_name, description, build_fn, train_fn, recall_fn):
    """Run a single experiment variant."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {variant_name}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    output_dir = f"results/kg_recall_experiments/{variant_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build KG (if needed)
    if build_fn:
        build_fn()

    # Step 2: Train embeddings
    if train_fn:
        train_fn()

    # Step 3: Generate recall candidates
    recall_path = os.path.join(output_dir, "multi_recall_scores.csv")
    recall_fn(output_path=recall_path)

    # Step 4: Evaluate
    results = evaluate_recall_quality(recall_path)

    # Save results
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n--- {variant_name} Results ---")
    print(f"  Overall hit rate:     {results['overall_hit_rate']:.4f}")
    print(f"  CF-only hit rate:     {results['cf_only_hit_rate']:.4f}")
    print(f"  KG-only hit rate:     {results['kg_only_hit_rate']:.4f}")
    print(f"  KG score discrim:     {results.get('kg_score_discrimination', 0):.6f}")
    print(f"  KG score std:         {results.get('kg_score_std', 0):.6f}")
    print(f"  KG unique positives:  {results['kg_unique_positives']}")
    print(f"  Total positives:      {results['total_positives_captured']}")
    print(f"  Recall@10:            {results['Recall@10']:.4f}")
    print(f"  NDCG@10:              {results['NDCG@10']:.4f}")
    print(f"  Hit@10:               {results['Hit@10']:.4f}")
    print(f"  Movies covered:       {results['total_movies']}")

    return results


def main():
    all_results = {}

    # =============================================
    # Load already-completed results
    # =============================================
    for name, dirname in [("baseline", "baseline"), ("A", "A_balanced_coliked30k")]:
        result_path = f"results/kg_recall_experiments/{dirname}/eval_results.json"
        if os.path.exists(result_path):
            with open(result_path) as f:
                all_results[name] = json.load(f)
            print(f"  Loaded existing results for {name}")

    # =============================================
    # Rebuild KG with co_liked=30K if not already done
    # =============================================
    triples_df = pd.read_csv("data/kg/triples.csv")
    coliked_count = (triples_df.relation == "co_liked").sum()
    if coliked_count > 50000:
        print("Rebuilding KG with co_liked=30K...")
        build_kg_with_coliked_limit(max_coliked=30000)
        from kg.transe import train_transe
        train_transe(balanced=True, epochs=300)

    # =============================================
    # Variant A+D: A + improved user profile
    # =============================================
    def recall_ad(output_path):
        generate_multi_recall_improved(
            output_path=output_path,
            max_history=None,
            use_idf_weighting=True,
            use_time_decay=True,
        )

    all_results["A+D"] = run_experiment(
        "AD_balanced_improved_profile",
        "A + no history limit + IDF weighting + time decay",
        None, None, recall_ad  # reuse KG/embeddings from A
    )

    # =============================================
    # Variant A+D+B: A+D + RotatE
    # =============================================
    from kg.rotate import train_rotate

    def train_adb():
        train_rotate(balanced=True, epochs=300, dim=128, gamma=12.0, lr=0.001)

    def recall_adb(output_path):
        generate_multi_recall_improved(
            output_path=output_path,
            max_history=None,
            use_idf_weighting=True,
            use_time_decay=True,
        )

    all_results["A+D+B"] = run_experiment(
        "ADB_rotate_balanced_improved",
        "RotatE + co_liked=30K + balanced + improved profile",
        None, train_adb, recall_adb  # KG same as A, just different embeddings
    )

    # =============================================
    # Restore baseline KG
    # =============================================
    restore_baseline_kg()

    # =============================================
    # Summary
    # =============================================
    print("\n" + "="*60)
    print("  SUMMARY: KG Recall Experiments")
    print("="*60)
    print(f"\n{'Variant':<30} {'Hit Rate':>10} {'KG Hit':>10} {'KG Discrim':>12} "
          f"{'KG Pos':>8} {'Recall@10':>10} {'NDCG@10':>10} {'Movies':>8}")
    print("-" * 110)
    for name, r in all_results.items():
        print(f"{name:<30} {r['overall_hit_rate']:>10.4f} {r['kg_only_hit_rate']:>10.4f} "
              f"{r.get('kg_score_discrimination', 0):>12.6f} "
              f"{r['kg_unique_positives']:>8} {r['Recall@10']:>10.4f} "
              f"{r['NDCG@10']:>10.4f} {r['total_movies']:>8}")

    # Save summary
    summary_path = "results/kg_recall_experiments/summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
