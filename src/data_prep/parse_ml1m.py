"""
Parse MovieLens 1M raw .dat files into clean CSV format.
Implements: positive threshold (rating >= 4), 3-way time-based split,
per-split negative sampling with no future leakage.
"""
import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw/ml-1m"
OUT_DIR = "data/processed"


def parse_ratings():
    """Parse ratings.dat -> DataFrame."""
    ratings = pd.read_csv(
        os.path.join(RAW_DIR, "ratings.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    print(f"Raw ratings: {len(ratings)} rows, "
          f"{ratings['user_id'].nunique()} users, "
          f"{ratings['movie_id'].nunique()} movies")
    return ratings


def parse_movies():
    """Parse movies.dat -> DataFrame."""
    movies = pd.read_csv(
        os.path.join(RAW_DIR, "movies.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)$", "", regex=True
    )
    print(f"Movies: {len(movies)} rows")
    return movies


def parse_users():
    """Parse users.dat -> DataFrame."""
    users = pd.read_csv(
        os.path.join(RAW_DIR, "users.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    print(f"Users: {len(users)} rows")
    return users


def filter_positive_interactions(ratings, min_rating=4):
    """Keep only ratings >= min_rating as positive implicit feedback."""
    positive = ratings[ratings["rating"] >= min_rating].copy()
    print(f"After positive threshold (rating >= {min_rating}): "
          f"{len(positive)} ratings, "
          f"{positive['user_id'].nunique()} users, "
          f"{positive['movie_id'].nunique()} movies")
    return positive


def filter_min_interactions(ratings, min_user=10, min_item=5):
    """
    Iteratively filter users and items below minimum interaction thresholds.
    Iterates until convergence since removing items may cause users to drop
    below threshold and vice versa.
    """
    prev_len = 0
    while len(ratings) != prev_len:
        prev_len = len(ratings)
        # Filter users
        user_counts = ratings["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        ratings = ratings[ratings["user_id"].isin(valid_users)]
        # Filter items
        item_counts = ratings["movie_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        ratings = ratings[ratings["movie_id"].isin(valid_items)]

    ratings = ratings.copy()
    print(f"After min interaction filter (user>={min_user}, item>={min_item}): "
          f"{len(ratings)} ratings, "
          f"{ratings['user_id'].nunique()} users, "
          f"{ratings['movie_id'].nunique()} movies")
    return ratings


def split_train_val_test(ratings, train_ratio=0.7, val_ratio=0.1):
    """
    Per-user time-based split into train / validation / test.
    For each user, interactions are sorted by timestamp:
      - first 70% -> train
      - next 10%  -> validation
      - last 20%  -> test
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])

    train_list, val_list, test_list = [], [], []

    for user_id, group in ratings.groupby("user_id"):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = max(1, int(n * val_ratio))
        # Ensure test has at least 1 item
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        train_list.append(group.iloc[:n_train])
        val_list.append(group.iloc[n_train:n_train + n_val])
        test_list.append(group.iloc[n_train + n_val:])

    train = pd.concat(train_list, ignore_index=True)
    val = pd.concat(val_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"  Users: train={train['user_id'].nunique()}, "
          f"val={val['user_id'].nunique()}, "
          f"test={test['user_id'].nunique()}")
    return train, val, test


def generate_negative_samples(positive_df, all_movie_ids, user_history,
                              neg_ratio=4, seed=42):
    """
    For each positive interaction in positive_df, sample neg_ratio negative
    movies that the user has NOT interacted with.

    Args:
        positive_df: DataFrame with positive interactions for this split.
        all_movie_ids: All valid movie IDs in the dataset.
        user_history: dict[user_id] -> set(movie_id) of ALL interactions
                      up to and including this split (to avoid sampling
                      a known item as negative).
        neg_ratio: Number of negatives per positive.
        seed: Random seed.

    Returns:
        DataFrame with columns [user_id, movie_id, label].
    """
    rng = np.random.RandomState(seed)
    all_movies = set(all_movie_ids)

    neg_samples = []
    for _, row in positive_df.iterrows():
        uid = row["user_id"]
        candidates = list(all_movies - user_history.get(uid, set()))
        n_neg = min(neg_ratio, len(candidates))
        if n_neg > 0:
            sampled = rng.choice(candidates, size=n_neg, replace=False).tolist()
            for mid in sampled:
                neg_samples.append({"user_id": uid, "movie_id": mid, "label": 0})

    neg_df = pd.DataFrame(neg_samples)

    # Positive labels
    pos_df = positive_df[["user_id", "movie_id"]].copy()
    pos_df["label"] = 1

    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    print(f"  Generated: {len(pos_df)} pos + {len(neg_df)} neg = {len(combined)} total")
    return combined


def build_user_history(dfs):
    """Build cumulative user history from a list of DataFrames."""
    history = {}
    for df in dfs:
        for _, row in df.iterrows():
            uid = row["user_id"]
            if uid not in history:
                history[uid] = set()
            history[uid].add(row["movie_id"])
    return history


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Parse raw data
    ratings = parse_ratings()
    movies = parse_movies()
    users = parse_users()

    movies.to_csv(os.path.join(OUT_DIR, "movies.csv"), index=False)
    users.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)

    # 2. Filter to positive interactions (rating >= 4)
    positive = filter_positive_interactions(ratings, min_rating=4)

    # 3. Filter minimum interactions
    clean = filter_min_interactions(positive, min_user=10, min_item=5)
    clean.to_csv(os.path.join(OUT_DIR, "clean_ratings.csv"), index=False)

    # 4. Three-way time-based split
    train, val, test = split_train_val_test(clean, train_ratio=0.7, val_ratio=0.1)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    # 5. Negative sampling (per split, no future leakage)
    all_movie_ids = clean["movie_id"].unique()

    # Train negatives: history = train only
    print("\nNegative sampling for train...")
    train_history = build_user_history([train])
    train_with_neg = generate_negative_samples(
        train, all_movie_ids, train_history, neg_ratio=4, seed=42
    )
    train_with_neg.to_csv(os.path.join(OUT_DIR, "train_with_neg.csv"), index=False)

    # Validation negatives: history = train + val
    print("Negative sampling for validation...")
    val_history = build_user_history([train, val])
    val_with_neg = generate_negative_samples(
        val, all_movie_ids, val_history, neg_ratio=4, seed=43
    )
    val_with_neg.to_csv(os.path.join(OUT_DIR, "val_with_neg.csv"), index=False)

    # Test negatives: history = train + val + test
    print("Negative sampling for test...")
    test_history = build_user_history([train, val, test])
    test_with_neg = generate_negative_samples(
        test, all_movie_ids, test_history, neg_ratio=4, seed=44
    )
    test_with_neg.to_csv(os.path.join(OUT_DIR, "test_with_neg.csv"), index=False)

    # 6. Summary
    print("\n" + "=" * 60)
    print("  DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Positive threshold: rating >= 4")
    print(f"  Users: {clean['user_id'].nunique()}")
    print(f"  Movies: {clean['movie_id'].nunique()}")
    print(f"  Interactions: {len(clean)}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"  Train+neg: {len(train_with_neg)} | Val+neg: {len(val_with_neg)} | Test+neg: {len(test_with_neg)}")
    print(f"\n  Files saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
