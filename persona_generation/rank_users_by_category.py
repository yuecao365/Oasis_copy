#!/usr/bin/env python3
"""
Rank users by category/cluster with support for:
- "count"   : in-cluster tweet count (legacy behavior)
- "mixture" : user's topical purity (share of their tweets in this cluster)
- "blend"   : alpha * mixture + (1 - alpha) * normalized in-cluster count
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

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
    mixture_weight: float = 0.0   # share of this user's tweets that fall into this cluster
    score: float = 0.0            # score used for ranking (count/mixture/blend)


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
    rank_mode: str = "count",
    alpha: float = 0.6,
    min_weight: float = 0.0,
) -> List[ClusterRanking]:
    """
    Rank users within each cluster, with configurable strategy.

    Args:
        clusters: Dict mapping cluster_id to list of tweets
        topics: Optional dict mapping cluster_id to topic name
        top_n: Number of top users to return per cluster
        max_sample_tweets: Maximum number of sample tweets to include
        rank_mode: "count" (default), "mixture", or "blend"
                   - "count": sort by in-cluster tweet_count (current behavior)
                   - "mixture": sort by user_cluster_mixture weight for that cluster
                   - "blend": score = alpha * mixture + (1 - alpha) * normalized_count
        alpha: blending factor for "blend" mode
        min_weight: ignore users with mixture weight < min_weight (useful to reduce noise)

    Returns:
        List of ClusterRanking objects
    """
    rankings: List[ClusterRanking] = []
    # Pre-compute per-user mixture across clusters (normalized across user's total)
    user_cluster_mix = compute_user_cluster_mixture(clusters)

    for cluster_id, tweets in clusters.items():
        # Count tweets per user within this cluster
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

        # Prepare normalization for "blend": normalized in-cluster tweet_count ∈ [0,1]
        max_count = max((d["tweet_count"] for d in user_data.values()), default=0)

        # Compute score per user according to rank_mode
        scored_users: List[Tuple[str, Dict, float, float]] = []  # (user_id, data, mixture_weight, score)
        for uid, data in user_data.items():
            mixture_w = float(user_cluster_mix.get(uid, {}).get(cluster_id, 0.0))

            if rank_mode == "count":
                score = float(data["tweet_count"])
            elif rank_mode == "mixture":
                if mixture_w < min_weight:
                    continue
                score = mixture_w
            elif rank_mode == "blend":
                norm_count = (data["tweet_count"] / max_count) if max_count > 0 else 0.0
                if mixture_w < min_weight:
                    continue
                score = alpha * mixture_w + (1.0 - alpha) * norm_count
            else:
                raise ValueError(f"Unknown rank_mode: {rank_mode}")

            scored_users.append((uid, data, mixture_w, score))

        # Sort by score desc, then by tweet_count, then popularity as tiebreakers
        scored_users.sort(
            key=lambda x: (x[3], x[1]["tweet_count"], x[1]["total_popularity"]),
            reverse=True
        )

        # Take top N users and build payload
        top_users: List[UserRanking] = []
        for uid, data, mixture_w, score in scored_users[:top_n]:
            # Get sample tweets (most popular ones)
            sample_tweets = sorted(
                data["tweets"],
                key=lambda t: t.get("popularity", 0),
                reverse=True
            )[:max_sample_tweets]

            user_ranking = UserRanking(
                user_name=data["tweets"][0]["user_name"] if data["tweets"] else "",
                user_id=uid,
                tweet_count=data["tweet_count"],
                tweet_ids=data["tweet_ids"],
                sample_tweets=[t["text"] for t in sample_tweets],
                total_popularity=data["total_popularity"],
                mixture_weight=mixture_w,
                score=score,
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
                    "mixture_weight": user.mixture_weight,
                    "score": user.score,
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
    assignments: Dict[int, int] = {}
    for idx, label in enumerate(cluster_labels):
        if label != -1:  # Skip noise
            assignments[idx] = int(label)
    return assignments


def load_tweets_with_clusters(tweets_csv_path: Path) -> Dict[int, List[Dict]]:
    """
    Load tweets from CSV and group them by topic_id (used as cluster_id).

    Expected columns in train_tweets.csv:
      dup,id,topic_id,reply_to_id,reply_to_user,likes,retweets,replies,quotes,
      impressions,user_name,user_id,is_root,text,timestamp
    """
    clusters: Dict[int, List[Dict]] = defaultdict(list)

    with tweets_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use topic_id as the cluster label
            try:
                cluster_id = int(row.get("topic_id", -1))
            except ValueError:
                cluster_id = -1

            if cluster_id == -1:
                # Skip rows without a valid topic/cluster
                continue

            # Build a popularity proxy: likes + retweets + replies + quotes
            def to_int(x):
                try:
                    return int(float(x))
                except Exception:
                    return 0

            likes = to_int(row.get("likes", 0))
            retweets = to_int(row.get("retweets", 0))
            replies = to_int(row.get("replies", 0))
            quotes = to_int(row.get("quotes", 0))
            impressions = to_int(row.get("impressions", 0))

            popularity = likes + retweets + replies + quotes
            if popularity == 0 and impressions > 0:
                # Optional: if no engagements, use impressions as a weak signal
                popularity = impressions

            tweet = {
                "id": str(row.get("id", "")).strip(),
                "user_id": str(row.get("user_id", "")).strip(),
                "user_name": str(row.get("user_name", "")).strip(),
                "text": row.get("text", "") or "",
                "popularity": float(popularity),
                # topic name not provided in CSV; keep empty string
                "topic": "",
            }
            clusters[cluster_id].append(tweet)

    return clusters


# ---- Mixture utilities ------------------------------------------------------ #

def compute_user_cluster_mixture(clusters) -> dict:
    """
    Returns: Dict[user_id, Dict[cluster_id, weight]]
    Weight is normalized frequency across user's train tweets.
    """
    user_cluster_counts = defaultdict(Counter)
    for cid, tweets in clusters.items():
        for t in tweets:
            uid = t.get("user_id")
            if not uid:
                continue
            user_cluster_counts[uid][cid] += 1

    user_cluster_mix = {}
    for uid, ctr in user_cluster_counts.items():
        total = sum(ctr.values())
        if total == 0:
            continue
        mix = {cid: c / total for cid, c in ctr.items()}
        user_cluster_mix[uid] = dict(sorted(mix.items(), key=lambda x: x[1], reverse=True))
    return user_cluster_mix


def save_user_cluster_mix(user_cluster_mix: Dict[str, Dict[int, float]], output_path: Path):
    """
    Persist user->cluster mixture weights for QA/dashboards.
    """
    # Ensure int keys for clusters are serialized correctly
    serializable = {
        str(uid): {int(cid): float(w) for cid, w in mix.items()}
        for uid, mix in user_cluster_mix.items()
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


# ---- CLI example ------------------------------------------------------------ #

if __name__ == "__main__":
    # Example usage
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUTS_DIR = BASE_DIR / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tweets_file = OUTPUTS_DIR / "train_tweets.csv"
    output_file = OUTPUTS_DIR / "user_rankings.json"
    mixture_file = OUTPUTS_DIR / "user_cluster_mix.json"

    if not tweets_file.exists():
        print(f"Tweets file not found: {tweets_file}")
        print("Run data_split.py first to create train_tweets.csv")
    else:
        clusters = load_tweets_with_clusters(tweets_file)
        # Choose your strategy here:
        #   - rank_mode="count"   : legacy behavior
        #   - rank_mode="mixture" : topical purity per user
        #   - rank_mode="blend"   : alpha*mixture + (1-alpha)*normalized_count
        rankings = rank_users_by_category(
            clusters,
            top_n=10,
            rank_mode="mixture",
            alpha=0.6,
            min_weight=0.0,
        )
        # Save mixture for inspection
        mix = compute_user_cluster_mixture(clusters)
        save_user_cluster_mix(mix, mixture_file)

        save_user_rankings(rankings, output_file)
        print(f"Saved rankings for {len(rankings)} clusters to {output_file}")
        print(f"Saved user->cluster mixture to {mixture_file}")