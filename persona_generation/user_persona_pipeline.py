#!/usr/bin/env python3
"""
Main pipeline to:
1. Load filtered_tweets.jsonl and split into train/test sets (time-based)
2. Load clustering output (cluster_assignments, cluster_topics)
3. Rank top users per category
4. Generate personas for top users

Output: user_personas.json compatible with OASIS simulation format
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    CLUSTER_ASSIGNMENTS_PATH,
    CLUSTER_TOPICS_PATH,
    FILTERED_TWEETS_PATH,
    MIN_TEST_TWEETS,
    MIN_TRAIN_TWEETS,
    OUTPUT_DIR,
    OUTPUT_FORMAT,
    TEST_MONTH,
    TEST_YEAR,
    TOP_N,
    TRAIN_MONTH,
    TRAIN_YEAR,
    TWEET_POPULARITY_PATH,
    USE_TWITTER_API,
)
from data_split import (
    filter_active_users_in_both_periods,
    load_tweets_with_timestamps,
    save_split_info,
    save_tweets_to_csv,
    split_tweets_by_time,
)
from generate_personas import generate_personas_for_users, save_personas
from rank_users_by_category import rank_users_by_category, save_user_rankings

REPO_ROOT = _REPO_ROOT


def load_cluster_assignments(path: Path) -> Dict[str, int]:
    """Load cluster_assignments.jsonl. Returns tweet_id -> cluster_id."""
    result = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tweet_id = obj.get("tweet_id")
                cluster_id = obj.get("cluster_id")
                if tweet_id is not None and cluster_id is not None:
                    result[str(tweet_id)] = int(cluster_id)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return result


def load_cluster_topics(path: Path) -> Dict[int, str]:
    """Load cluster_topics.json. Returns cluster_id -> topic string."""
    result = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in (data or {}).items():
        try:
            result[int(k)] = str(v)
        except (ValueError, TypeError):
            continue
    return result


def load_train_tweets_with_popularity_and_clusters(
    train_csv_path: Path,
    popularity_csv_path: Path,
    tweet_to_cluster: Dict[str, int],
) -> Dict[int, List[Dict]]:
    """
    Load train tweets from CSV, join with popularity scores and cluster assignments.
    Returns clusters: Dict[cluster_id, List[tweet_dict]] where tweet_dict has
    id, text, user_id, user_name, popularity.
    """
    # Load popularity: tweet_id -> score
    popularity_map: Dict[str, float] = {}
    if popularity_csv_path.exists():
        with popularity_csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("tweet_id")
                score = row.get("popularity_score")
                if tid:
                    try:
                        popularity_map[str(tid)] = float(score) if score else 0.0
                    except (ValueError, TypeError):
                        popularity_map[str(tid)] = 0.0

    # Build clusters from train CSV
    clusters: Dict[int, List[Dict]] = {}

    with train_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dup") != "0":
                continue
            if row.get("is_root") != "1":
                continue

            tweet_id = row.get("id", "")
            if not tweet_id:
                continue

            cluster_id = tweet_to_cluster.get(str(tweet_id))
            if cluster_id is None or cluster_id < 0:
                continue  # Skip noise or unassigned

            popularity = popularity_map.get(str(tweet_id), 0.0)

            tweet_dict = {
                "id": tweet_id,
                "text": row.get("text", ""),
                "user_id": row.get("user_id", ""),
                "user_name": row.get("user_name", ""),
                "popularity": popularity,
            }

            clusters.setdefault(cluster_id, []).append(tweet_dict)

    return clusters


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("User Persona Generation Pipeline")
    print("=" * 60)

    # Step 1: Load and split tweets
    print("\n[Step 1] Loading tweets and splitting by time...")
    all_tweets = load_tweets_with_timestamps(FILTERED_TWEETS_PATH)
    print(f"Loaded {len(all_tweets)} tweets")

    train_tweets, test_tweets = split_tweets_by_time(
        all_tweets,
        train_year=TRAIN_YEAR,
        train_month=TRAIN_MONTH,
        test_year=TEST_YEAR,
        test_month=TEST_MONTH,
    )
    print(f"Initial split - Train: {len(train_tweets)}, Test: {len(test_tweets)}")

    train_filtered, test_filtered = filter_active_users_in_both_periods(
        train_tweets,
        test_tweets,
        min_train_tweets=MIN_TRAIN_TWEETS,
        min_test_tweets=MIN_TEST_TWEETS,
    )
    print(f"After filtering - Train: {len(train_filtered)}, Test: {len(test_filtered)}")

    train_file = OUTPUT_DIR / "train_tweets.csv"
    test_file = OUTPUT_DIR / "test_tweets.csv"
    save_tweets_to_csv(train_filtered, train_file)
    save_tweets_to_csv(test_filtered, test_file)
    save_split_info(
        train_filtered,
        test_filtered,
        OUTPUT_DIR / "split_info.json",
        train_months=TRAIN_MONTH,
        test_months=TEST_MONTH,
    )
    print(f"Saved train/test splits to {train_file} and {test_file}")

    # Step 2: Load clustering output
    print("\n[Step 2] Loading clustering output...")
    tweet_to_cluster = load_cluster_assignments(CLUSTER_ASSIGNMENTS_PATH)
    topics_dict = load_cluster_topics(CLUSTER_TOPICS_PATH)
    print(f"Loaded {len(tweet_to_cluster)} cluster assignments, {len(topics_dict)} topics")

    clusters = load_train_tweets_with_popularity_and_clusters(
        train_file, TWEET_POPULARITY_PATH, tweet_to_cluster
    )
    print(f"Built {len(clusters)} clusters from train tweets")

    if not clusters:
        print("Warning: No clusters with assigned tweets. Ensure clustering has been run.")
        print("See persona_generation/CLUSTERING_INPUT_SPEC.md for required output format.")

    # Step 3: Rank users by category
    print(f"\n[Step 3] Ranking top {TOP_N} users per category...")
    rankings = rank_users_by_category(clusters, topics_dict, top_n=TOP_N)
    rankings_file = OUTPUT_DIR / "user_rankings.json"
    save_user_rankings(rankings, rankings_file)
    print(f"Saved rankings to {rankings_file}")

    # Step 4: Generate personas
    print("\n[Step 4] Generating personas...")
    with rankings_file.open() as f:
        rankings_data = json.load(f)

    personas = generate_personas_for_users(
        rankings_data, use_twitter_api=USE_TWITTER_API
    )

    if OUTPUT_FORMAT == "json":
        personas_file = OUTPUT_DIR / "user_personas.json"
    else:
        personas_file = OUTPUT_DIR / "user_personas.csv"

    save_personas(personas, personas_file, format=OUTPUT_FORMAT)
    print(f"Generated {len(personas)} personas and saved to {personas_file}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Train tweets: {train_file}")
    print(f"  - Test tweets: {test_file}")
    print(f"  - Split info: {OUTPUT_DIR / 'split_info.json'}")
    print(f"  - User rankings: {rankings_file}")
    print(f"  - User personas: {personas_file}")


if __name__ == "__main__":
    main()
