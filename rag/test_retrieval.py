"""
Test retrieval quality: pick sample (user, movie) pairs and inspect results.

Usage:
    python -m rag.test_retrieval
    python -m rag.test_retrieval --alpha 0.3
    python -m rag.test_retrieval --num-samples 5
"""
import argparse
import sys
sys.path.insert(0, ".")

from rag.retriever import HybridRetriever, build_query
from rag.pipeline import get_top_k_recommendations, get_user_history, get_movie_info


def test_retrieval(alpha: float = 0.6, num_samples: int = 5, retrieval_k: int = 8):
    recs = get_top_k_recommendations(k=3)
    user_history = get_user_history()
    movie_info = get_movie_info()

    retriever = HybridRetriever(corpus_dir="data/rag", alpha=alpha)

    # Pick first N users, first recommended movie each
    users = sorted(recs["user_id"].unique())[:num_samples]

    for uid in users:
        user_recs = recs[recs["user_id"] == uid].iloc[0]
        mid = user_recs["movie_id"]
        minfo = movie_info.get(mid, {"title": f"Movie {mid}", "genres": ""})

        history_mids = user_history.get(uid, [])
        history_titles = [movie_info[m]["title"]
                          for m in history_mids if m in movie_info]

        query = build_query(
            candidate_title=minfo["title"],
            candidate_genres=minfo.get("genres", ""),
            history_titles=history_titles,
        )

        results = retriever.retrieve(
            query=query, top_k=retrieval_k,
            candidate_movie_id=mid,
            history_movie_ids=history_mids[:5],
        )

        print("=" * 80)
        print(f"User {uid} | Candidate: {minfo['title']}")
        print(f"History: {', '.join(history_titles[:5])}")
        print(f"Query: {query[:120]}...")
        print(f"\nTop-{retrieval_k} evidence (alpha={alpha}):")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']:.3f} "
                  f"(dense={r['dense_score']:.3f} bm25={r['bm25_score']:.3f}) "
                  f"[{r['source']}] mid={r['movie_id']}")
            print(f"      {r['text'][:100]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--retrieval-k", type=int, default=8)
    args = parser.parse_args()
    test_retrieval(alpha=args.alpha, num_samples=args.num_samples,
                   retrieval_k=args.retrieval_k)
