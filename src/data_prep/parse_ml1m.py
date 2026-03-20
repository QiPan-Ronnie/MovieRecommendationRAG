"""
Parse MovieLens 1M raw .dat files into clean CSV format.
"""
import pandas as pd
import os

RAW_DIR = "data/raw/ml-1m"
OUT_DIR = "data/processed"


def parse_ratings():
    """Parse ratings.dat -> ratings.csv"""
    ratings = pd.read_csv(
        os.path.join(RAW_DIR, "ratings.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    print(f"Ratings: {len(ratings)} rows, {ratings['user_id'].nunique()} users, {ratings['movie_id'].nunique()} movies")
    return ratings


def parse_movies():
    """Parse movies.dat -> movies.csv"""
    movies = pd.read_csv(
        os.path.join(RAW_DIR, "movies.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    # Extract year from title like "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    print(f"Movies: {len(movies)} rows")
    return movies


def parse_users():
    """Parse users.dat -> users.csv"""
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


def clean_and_filter(ratings, min_interactions=20):
    """Filter users with fewer than min_interactions ratings."""
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered = ratings[ratings["user_id"].isin(valid_users)].copy()
    print(f"After filtering (min {min_interactions} interactions): {len(filtered)} ratings, {filtered['user_id'].nunique()} users, {filtered['movie_id'].nunique()} movies")
    return filtered


def split_train_test(ratings, test_ratio=0.2):
    """
    Time-based split: for each user, last test_ratio% of interactions go to test.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])

    train_list = []
    test_list = []

    for user_id, group in ratings.groupby("user_id"):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])

    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    print(f"Train: {len(train)} ratings | Test: {len(test)} ratings")
    return train, test


def generate_negative_samples(train, test, all_movie_ids, neg_ratio=4, seed=42):
    """
    For each positive in test, sample neg_ratio negative movies
    (movies the user hasn't interacted with).
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    # Build user -> set of interacted movies (train + test)
    user_pos = {}
    for _, row in pd.concat([train, test]).iterrows():
        uid = row["user_id"]
        if uid not in user_pos:
            user_pos[uid] = set()
        user_pos[uid].add(row["movie_id"])

    all_movies = set(all_movie_ids)
    neg_samples = []

    for _, row in test.iterrows():
        uid = row["user_id"]
        candidates = list(all_movies - user_pos[uid])
        if len(candidates) < neg_ratio:
            sampled = candidates
        else:
            sampled = rng.choice(candidates, size=neg_ratio, replace=False).tolist()
        for mid in sampled:
            neg_samples.append({"user_id": uid, "movie_id": mid, "rating": 0, "label": 0})

    neg_df = pd.DataFrame(neg_samples)

    # Add label column to test
    test_pos = test.copy()
    test_pos["label"] = 1

    test_with_neg = pd.concat([test_pos[["user_id", "movie_id", "rating", "label"]], neg_df], ignore_index=True)
    print(f"Test with negatives: {len(test_with_neg)} rows ({len(test)} pos + {len(neg_df)} neg)")
    return test_with_neg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse raw data
    ratings = parse_ratings()
    movies = parse_movies()
    users = parse_users()

    # Save parsed data
    movies.to_csv(os.path.join(OUT_DIR, "movies.csv"), index=False)
    users.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)

    # Clean and filter
    clean_ratings = clean_and_filter(ratings, min_interactions=20)
    clean_ratings.to_csv(os.path.join(OUT_DIR, "clean_ratings.csv"), index=False)

    # Split
    train, test = split_train_test(clean_ratings, test_ratio=0.2)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    # Negative sampling
    all_movie_ids = clean_ratings["movie_id"].unique()
    test_with_neg = generate_negative_samples(train, test, all_movie_ids, neg_ratio=4)
    test_with_neg.to_csv(os.path.join(OUT_DIR, "test_with_neg.csv"), index=False)

    print("\nDone! Files saved to", OUT_DIR)


if __name__ == "__main__":
    main()
