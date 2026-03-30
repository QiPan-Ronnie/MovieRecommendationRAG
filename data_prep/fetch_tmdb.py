"""
Fetch movie metadata from TMDB API with checkpoint/resume support.

Usage:
    python data_prep/fetch_tmdb.py --api_key YOUR_TMDB_API_KEY

The script caches each movie's response as a JSON file in data/tmdb/cache/.
On re-run, it skips already-cached movies (checkpoint/resume).
"""
import os
import json
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm

CACHE_DIR = "data/tmdb/cache"
OUT_DIR = "data/tmdb"
FAILED_FILE = os.path.join(OUT_DIR, "failed_ids.txt")

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_CREDITS_URL = "https://api.themoviedb.org/3/movie/{}/credits"


def load_movies():
    """Load movies from processed CSV."""
    movies = pd.read_csv("data/processed/movies.csv")
    return movies


def search_movie_by_title(title, year, api_key):
    """Search TMDB by movie title and year."""
    params = {
        "api_key": api_key,
        "query": title,
        "language": "en-US",
    }
    if year and str(year) != "nan":
        params["year"] = int(float(year))

    resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if results:
        return results[0]["id"]
    return None


def get_movie_details(tmdb_id, api_key):
    """Get movie details from TMDB."""
    params = {"api_key": api_key, "language": "en-US"}
    resp = requests.get(TMDB_MOVIE_URL.format(tmdb_id), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_movie_credits(tmdb_id, api_key):
    """Get movie credits (cast + crew) from TMDB."""
    params = {"api_key": api_key, "language": "en-US"}
    resp = requests.get(TMDB_CREDITS_URL.format(tmdb_id), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_single_movie(movie_id, title, year, api_key):
    """
    Fetch metadata for a single movie.
    Returns dict with extracted fields, or None on failure.
    """
    # Search for TMDB ID
    tmdb_id = search_movie_by_title(title, year, api_key)
    if tmdb_id is None:
        return None

    # Get details
    details = get_movie_details(tmdb_id, api_key)
    credits = get_movie_credits(tmdb_id, api_key)

    # Extract fields
    genres = [g["name"] for g in details.get("genres", [])]
    overview = details.get("overview", "")
    tagline = details.get("tagline", "")
    keywords_data = details.get("keywords", {})

    # Get top-5 actors
    cast = credits.get("cast", [])
    actors = [c["name"] for c in cast[:5]]

    # Get director(s)
    crew = credits.get("crew", [])
    directors = [c["name"] for c in crew if c.get("job") == "Director"]

    result = {
        "movie_id": movie_id,
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "genres": genres,
        "actors": actors,
        "directors": directors,
        "overview": overview,
        "tagline": tagline,
        "release_date": details.get("release_date", ""),
        "vote_average": details.get("vote_average", 0),
        "vote_count": details.get("vote_count", 0),
        "popularity": details.get("popularity", 0),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="TMDB API key")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between requests (seconds)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    movies = load_movies()
    print(f"Total movies to fetch: {len(movies)}")

    # Load previously failed IDs
    failed_ids = set()
    if os.path.exists(FAILED_FILE):
        with open(FAILED_FILE) as f:
            failed_ids = set(int(line.strip()) for line in f if line.strip())

    success_count = 0
    skip_count = 0
    fail_count = 0
    new_failed = []

    for _, row in tqdm(movies.iterrows(), total=len(movies), desc="Fetching TMDB"):
        movie_id = row["movie_id"]
        cache_path = os.path.join(CACHE_DIR, f"{movie_id}.json")

        # Skip if already cached
        if os.path.exists(cache_path):
            skip_count += 1
            continue

        try:
            result = fetch_single_movie(
                movie_id=movie_id,
                title=row["clean_title"],
                year=row.get("year"),
                api_key=args.api_key,
            )

            if result is not None:
                with open(cache_path, "w") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                success_count += 1
            else:
                new_failed.append(movie_id)
                fail_count += 1

        except Exception as e:
            print(f"\nError for movie_id={movie_id} ({row['clean_title']}): {e}")
            new_failed.append(movie_id)
            fail_count += 1

        time.sleep(args.delay)

    # Save failed IDs
    all_failed = list(failed_ids) + new_failed
    with open(FAILED_FILE, "w") as f:
        for mid in all_failed:
            f.write(f"{mid}\n")

    print(f"\nDone! Success: {success_count}, Skipped (cached): {skip_count}, Failed: {fail_count}")
    print(f"Total cached: {len(os.listdir(CACHE_DIR))}/{len(movies)}")

    # Merge cached results into a single CSV
    merge_cache(movies)


def merge_cache(movies):
    """Merge all cached JSON files into a single metadata CSV."""
    records = []
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(CACHE_DIR, fname)) as f:
            data = json.load(f)
        records.append({
            "movie_id": data["movie_id"],
            "tmdb_id": data.get("tmdb_id"),
            "title": data.get("title"),
            "year": data.get("year"),
            "genres": "|".join(data.get("genres", [])),
            "actors": "|".join(data.get("actors", [])),
            "directors": "|".join(data.get("directors", [])),
            "overview": data.get("overview", ""),
            "tagline": data.get("tagline", ""),
            "release_date": data.get("release_date", ""),
            "vote_average": data.get("vote_average", 0),
            "vote_count": data.get("vote_count", 0),
            "popularity": data.get("popularity", 0),
        })

    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, "tmdb_metadata.csv")
    df.to_csv(out_path, index=False)
    print(f"Metadata saved to {out_path}: {len(df)} movies")
    print(f"Coverage: {len(df)}/{len(movies)} ({100*len(df)/len(movies):.1f}%)")


if __name__ == "__main__":
    main()
