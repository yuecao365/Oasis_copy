#!/usr/bin/env python3
"""
Main pipeline to:
1. Split tweets into train/test sets (time-based)
2. Cluster train tweets by topic
3. Rank top users per category
4. Generate personas for top users

Output: user_personas.json compatible with OASIS simulation format
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.cluster import KMeans

from data_split import (
    filter_active_users_in_both_periods,
    load_tweets_with_timestamps,
    save_split_info,
    save_tweets_to_csv,
    split_tweets_by_time,
)
from generate_personas import generate_personas_for_users, save_personas
from rank_users_by_category import (
    group_tweets_by_cluster,
    load_cluster_assignments_from_labels,
    rank_users_by_category,
    save_user_rankings,
)
from clustering.topic_generation import (
    PCA_STATS_FILE,
    embed_texts,
    load_pca_stats,
    load_tweets,
)

REPO_ROOT = _REPO_ROOT


def load_train_tweets_for_clustering(train_tweets_file: Path) -> List:
    """
    Load train tweets in format compatible with clustering.topic_generation.load_tweets().
    
    Returns list of TweetRow-like objects with: text, popularity, id, topic_id, user_name
    """
    import csv
    import math
    
    load_pca_stats()
    
    # Load PCA stats
    pca_weights = {
        "likes": 0.220255,
        "retweets": 0.214415,
        "replies": 0.201749,
        "quotes": 0.180303,
        "impressions": 0.183278,
    }
    pca_means = None
    pca_stds = None
    
    if PCA_STATS_FILE.exists():
        with PCA_STATS_FILE.open() as f:
            stats = json.load(f)
        pca_weights = stats.get("weights", pca_weights)
        pca_means = stats.get("means")
        pca_stds = stats.get("stds")
    
    tweets = []
    with train_tweets_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dup") != "0":
                continue
            if row.get("is_root") != "1":
                continue
            
            metrics = {
                k: float(row.get(k, 0) or 0)
                for k in ["likes", "retweets", "replies", "quotes", "impressions"]
            }
            logs = {k: math.log1p(v) for k, v in metrics.items()}
            
            if pca_means and pca_stds:
                popularity = sum(
                    pca_weights[k]
                    * (
                        (logs[k] - pca_means.get(k, 0))
                        / (pca_stds.get(k, 1) or 1)
                    )
                    for k in pca_weights.keys()
                )
            else:
                popularity = sum(
                    pca_weights[k] * logs[k] for k in pca_weights.keys()
                )
            
            # Create TweetRow-like dict
            tweet_dict = {
                "text": row["text"],
                "popularity": popularity,
                "id": row.get("id", ""),
                "topic_id": row.get("topic_id", ""),
                "user_name": row.get("user_name", ""),
                "user_id": row.get("user_id", ""),
            }
            tweets.append(tweet_dict)
    
    return tweets


def cluster_tweets(
    tweets: List[Dict],
    clustering_method: str = "hdbscan_400",
) -> tuple[np.ndarray, Dict[int, str]]:
    """
    Cluster tweets and generate topic labels.
    
    Args:
        tweets: List of tweet dicts
        clustering_method: One of "hdbscan_200", "hdbscan_400", "hdbscan_600", "kmeans_3", "kmeans_4", "kmeans_5", "kmeans_6"
        
    Returns:
        Tuple of (cluster_labels, topics_dict)
    """
    if not tweets:
        # Nothing to cluster
        return np.array([], dtype=int), {}

    # Embed tweets
    texts = [t["text"] for t in tweets]
    embeddings = embed_texts(texts)
    
    # Cluster
    if clustering_method.startswith("hdbscan"):
        min_size = int(clustering_method.split("_")[1])
        n_samples = len(tweets)

        # HDBSCAN cannot produce clusters larger than the dataset; for small datasets,
        # downshift min_cluster_size or fall back to KMeans so we still get clusters
        # (otherwise you'll get all -1 labels → 0 clusters → empty rankings/personas).
        if n_samples < min_size:
            # Prefer a smaller-but-reasonable min_cluster_size for the available data.
            # Keep at least 5 so clusters aren't trivial.
            new_min_size = max(5, min(n_samples // 2, 50))
            print(
                f"Note: only {n_samples} tweets available; "
                f"adjusting HDBSCAN min_cluster_size {min_size} → {new_min_size}"
            )
            min_size = new_min_size

        try:
            import hdbscan  # lazy import; allows running without hdbscan installed

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=None,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embeddings)
        except ImportError:
            labels = np.full((n_samples,), -1, dtype=int)

        # If HDBSCAN yields all noise, fall back to KMeans with a small, data-driven k.
        if (labels == -1).all():
            # Choose a conservative number of clusters. Bound by [2, 6] and <= n_samples.
            n_clusters = min(max(2, n_samples // 20), 6, n_samples)
            if n_clusters < 2:
                # Degenerate case (e.g., n_samples==1): no clustering possible.
                return labels.astype(int), {}
            print(
                "Note: HDBSCAN produced all noise labels; "
                f"falling back to KMeans (k={n_clusters})"
            )
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            labels = km.fit_predict(embeddings)
    elif clustering_method.startswith("kmeans"):
        n_samples = len(tweets)
        n_clusters = int(clustering_method.split("_")[1])
        if n_clusters > n_samples:
            print(
                f"Note: requested kmeans_{n_clusters} but only {n_samples} samples; "
                f"using kmeans_{n_samples} instead"
            )
            n_clusters = n_samples
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        labels = km.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    # Generate topics for clusters
    from clustering.topic_generation import TweetRow, build_prompt, maybe_call_openai
    
    clusters: Dict[int, List] = {}
    for label, tweet in zip(labels, tweets):
        if label == -1:
            continue  # noise
        clusters.setdefault(int(label), []).append(tweet)
    
    topics_dict = {}
    for cluster_id in sorted(clusters.keys()):
        cluster_tweets = clusters[cluster_id]
        # Sort by popularity and take top 20
        sorted_tweets = sorted(
            cluster_tweets, key=lambda t: t.get("popularity", 0), reverse=True
        )[:20]
        
        # Build prompt and get topic
        # Convert to TweetRow-like objects for build_prompt
        tweet_rows = [
            TweetRow(
                text=t["text"],
                popularity=t.get("popularity", 0),
                id=t["id"],
                topic_id=t.get("topic_id", ""),
                user_name=t.get("user_name", ""),
            )
            for t in sorted_tweets
        ]
        
        record = build_prompt(cluster_id, tweet_rows)
        topic = maybe_call_openai(record["prompt"]) or f"Topic {cluster_id}"
        topics_dict[cluster_id] = topic
    
    return labels, topics_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate user personas from tweet data"
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=6,
        help="Number of months for training period (default: 6)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=1,
        help="Number of months for test period (default: 1)",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="hdbscan_400",
        choices=[
            "hdbscan_200",
            "hdbscan_400",
            "hdbscan_600",
            "kmeans_3",
            "kmeans_4",
            "kmeans_5",
            "kmeans_6",
        ],
        help="Clustering method (default: hdbscan_400)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top users per category (default: 10)",
    )
    parser.add_argument(
        "--min-train-tweets",
        type=int,
        default=1,
        help="Minimum tweets in train period (default: 1)",
    )
    parser.add_argument(
        "--min-test-tweets",
        type=int,
        default=1,
        help="Minimum tweets in test period (default: 1)",
    )
    parser.add_argument(
        "--use-twitter-api",
        action="store_true",
        help="Use Twitter API to fetch user profiles (requires TWITTER_API_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: tweet_processing/)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Output format: json (Reddit) or csv (Twitter) (default: json)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "tweet_processing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("User Persona Generation Pipeline")
    print("=" * 60)
    
    # Step 1: Load and split tweets
    print("\n[Step 1] Loading tweets and splitting by time...")
    all_tweets = load_tweets_with_timestamps()
    print(f"Loaded {len(all_tweets)} tweets")
    
    # Use Nov 2025 for training, Dec 2025 for testing
    train_tweets, test_tweets = split_tweets_by_time(
        all_tweets,
        train_year=2025,
        train_month=11,
        test_year=2025,
        test_month=12,
    )
    print(f"Initial split - Train: {len(train_tweets)}, Test: {len(test_tweets)}")
    
    # Filter to users active in both periods
    train_filtered, test_filtered = filter_active_users_in_both_periods(
        train_tweets,
        test_tweets,
        min_train_tweets=args.min_train_tweets,
        min_test_tweets=args.min_test_tweets,
    )
    print(
        f"After filtering - Train: {len(train_filtered)}, Test: {len(test_filtered)}"
    )
    
    # Save train/test splits
    train_file = output_dir / "train_tweets.csv"
    test_file = output_dir / "test_tweets.csv"
    save_tweets_to_csv(train_filtered, train_file)
    save_tweets_to_csv(test_filtered, test_file)
    save_split_info(
        train_filtered,
        test_filtered,
        output_dir / "split_info.json",
        train_months=1,  # Nov 2025 = 1 month
        test_months=1,    # Dec 2025 = 1 month
    )
    print(f"Saved train/test splits to {train_file} and {test_file}")
    
    # Step 2: Cluster train tweets
    print(f"\n[Step 2] Clustering train tweets using {args.clustering_method}...")
    train_tweets_for_clustering = load_train_tweets_for_clustering(train_file)
    print(f"Loaded {len(train_tweets_for_clustering)} train tweets for clustering")
    
    cluster_labels, topics_dict = cluster_tweets(
        train_tweets_for_clustering, clustering_method=args.clustering_method
    )
    print(f"Found {len(topics_dict)} clusters")
    
    # Step 3: Rank users by category
    print(f"\n[Step 3] Ranking top {args.top_n} users per category...")
    cluster_assignments = load_cluster_assignments_from_labels(
        train_tweets_for_clustering, cluster_labels
    )
    
    # Group tweets by cluster
    clusters = group_tweets_by_cluster(
        train_tweets_for_clustering, cluster_assignments
    )
    
    rankings = rank_users_by_category(clusters, topics_dict, top_n=args.top_n)
    rankings_file = output_dir / "user_rankings.json"
    save_user_rankings(rankings, rankings_file)
    print(f"Saved rankings to {rankings_file}")
    
    # Step 4: Generate personas
    print("\n[Step 4] Generating personas...")
    with rankings_file.open() as f:
        rankings_data = json.load(f)
    
    personas = generate_personas_for_users(
        rankings_data, use_twitter_api=args.use_twitter_api
    )
    
    if args.output_format == "json":
        personas_file = output_dir / "user_personas.json"
    else:
        personas_file = output_dir / "user_personas.csv"
    
    save_personas(personas, personas_file, format=args.output_format)
    print(f"Generated {len(personas)} personas and saved to {personas_file}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Train tweets: {train_file}")
    print(f"  - Test tweets: {test_file}")
    print(f"  - Split info: {output_dir / 'split_info.json'}")
    print(f"  - User rankings: {rankings_file}")
    print(f"  - User personas: {personas_file}")


if __name__ == "__main__":
    main()
