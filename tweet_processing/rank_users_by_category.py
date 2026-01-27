#!/usr/bin/env python3
"""
Rank users by tweet count within each cluster/category.
"""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class UserRanking:
    """User ranking information for a cluster."""
    user_name: str
    user_id: str
    tweet_count: int
    tweet_ids: List[str]
    sample_tweets: List[str]
    total_popularity: float = 0.0


@dataclass
class ClusterRanking:
    """Ranking information for a cluster."""
    cluster_id: int
    topic: str
    top_users: List[UserRanking]


def group_tweets_by_cluster(
    tweets: List[Dict],
    cluster_assignments: Dict[int, int],
) -> Dict[int, List[Dict]]:
    """
    Group tweets by cluster ID using cluster assignments.
    
    Args:
        tweets: List of tweet dicts
        cluster_assignments: Dict mapping tweet index to cluster_id
        
    Returns:
        Dict mapping cluster_id to list of tweet dicts
    """
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    
    for idx, tweet in enumerate(tweets):
        cluster_id = cluster_assignments.get(idx)
        if cluster_id is not None and cluster_id != -1:  # -1 is noise in HDBSCAN
            clusters[cluster_id].append(tweet)
    
    return clusters


def rank_users_by_category(
    clusters: Dict[int, List[Dict]],
    topics: Dict[int, str] = None,
    top_n: int = 10,
    max_sample_tweets: int = 5,
) -> List[ClusterRanking]:
    """
    Rank users by tweet count within each cluster.
    
    Args:
        clusters: Dict mapping cluster_id to list of tweets
        topics: Optional dict mapping cluster_id to topic name
        top_n: Number of top users to return per cluster
        max_sample_tweets: Maximum number of sample tweets to include
        
    Returns:
        List of ClusterRanking objects
    """
    rankings = []
    
    for cluster_id, tweets in clusters.items():
        # Count tweets per user
        user_data: Dict[str, Dict] = defaultdict(lambda: {
            "tweet_count": 0,
            "tweet_ids": [],
            "tweets": [],
            "total_popularity": 0.0,
        })
        
        for tweet in tweets:
            user_id = tweet["user_id"]
            user_data[user_id]["tweet_count"] += 1
            user_data[user_id]["tweet_ids"].append(tweet["id"])
            user_data[user_id]["tweets"].append(tweet)
            user_data[user_id]["total_popularity"] += tweet.get("popularity", 0.0)
        
        # Sort users by tweet count (descending)
        sorted_users = sorted(
            user_data.items(),
            key=lambda x: x[1]["tweet_count"],
            reverse=True
        )
        
        # Take top N users
        top_users = []
        for user_id, data in sorted_users[:top_n]:
            # Get sample tweets (most popular ones)
            sample_tweets = sorted(
                data["tweets"],
                key=lambda t: t.get("popularity", 0),
                reverse=True
            )[:max_sample_tweets]
            
            user_ranking = UserRanking(
                user_name=data["tweets"][0]["user_name"] if data["tweets"] else "",
                user_id=user_id,
                tweet_count=data["tweet_count"],
                tweet_ids=data["tweet_ids"],
                sample_tweets=[t["text"] for t in sample_tweets],
                total_popularity=data["total_popularity"],
            )
            top_users.append(user_ranking)
        
        topic = topics.get(cluster_id, "") if topics else ""
        rankings.append(
            ClusterRanking(
                cluster_id=cluster_id,
                topic=topic,
                top_users=top_users,
            )
        )
    
    return rankings


def save_user_rankings(rankings: List[ClusterRanking], output_path: Path):
    """Save user rankings to JSON file."""
    output_data = []
    for ranking in rankings:
        output_data.append({
            "cluster_id": ranking.cluster_id,
            "topic": ranking.topic,
            "top_users": [
                {
                    "user_name": user.user_name,
                    "user_id": user.user_id,
                    "tweet_count": user.tweet_count,
                    "tweet_ids": user.tweet_ids,
                    "sample_tweets": user.sample_tweets,
                    "total_popularity": user.total_popularity,
                }
                for user in ranking.top_users
            ],
        })
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def load_cluster_assignments_from_labels(
    tweets: List,
    cluster_labels: List[int],
) -> Dict[int, int]:
    """
    Create mapping from tweet index to cluster_id.
    
    Args:
        tweets: List of tweets
        cluster_labels: List of cluster labels (same length as tweets)
        
    Returns:
        Dict mapping tweet index to cluster_id
    """
    assignments = {}
    for idx, label in enumerate(cluster_labels):
        if label != -1:  # Skip noise
            assignments[idx] = int(label)
    return assignments


if __name__ == "__main__":
    # Example usage
    tweets_file = REPO_ROOT / "tweet_processing" / "train_tweets.csv"
    output_file = REPO_ROOT / "tweet_processing" / "user_rankings.json"
    
    if not tweets_file.exists():
        print(f"Tweets file not found: {tweets_file}")
        print("Run data_split.py first to create train_tweets.csv")
    else:
        clusters = load_tweets_with_clusters(tweets_file)
        rankings = rank_users_by_category(clusters, top_n=10)
        save_user_rankings(rankings, output_file)
        print(f"Saved rankings for {len(rankings)} clusters to {output_file}")
