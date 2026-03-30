"""
Parse MovieLens 1M raw .dat files into clean CSV format.
Implements: positive threshold (rating >= 4), 3-way time-based split.

Negative sampling is NOT done here — the ranker uses distribution-matched
candidates from the recall model's top-100 (see ranker/ranker.py).
"""
import pandas as pd
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

    # 5. Summary
    print("\n" + "=" * 60)
    print("  DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Positive threshold: rating >= 4")
    print(f"  Users: {clean['user_id'].nunique()}")
    print(f"  Movies: {clean['movie_id'].nunique()}")
    print(f"  Interactions: {len(clean)}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"\n  Files saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
