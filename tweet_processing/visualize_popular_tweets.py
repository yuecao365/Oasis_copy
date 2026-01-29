#!/usr/bin/env python3
"""
Visualize the most popular tweets for each cluster/category.
Creates an HTML visualization showing top tweets ranked by popularity score.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
RANKINGS_FILE = REPO_ROOT / "tweet_processing" / "user_rankings.json"
TRAIN_TWEETS_FILE = REPO_ROOT / "tweet_processing" / "train_tweets.csv"
PCA_STATS_FILE = REPO_ROOT / "tweet_processing" / "pca_stats.json"
OUTPUT_DIR = REPO_ROOT / "tweet_processing" / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)
HTML_FILE = OUTPUT_DIR / "popular_tweets_by_cluster.html"


def load_pca_stats():
    """Load PCA statistics for popularity calculation."""
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
    
    return pca_weights, pca_means, pca_stds


def calculate_popularity(row: Dict, pca_weights: Dict, pca_means: Dict = None, pca_stds: Dict = None) -> float:
    """Calculate popularity score for a tweet."""
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
    
    return popularity


def load_tweets_with_clusters() -> Dict[int, List[Dict]]:
    """Load tweets and group them by cluster."""
    # Load rankings to get cluster assignments
    with RANKINGS_FILE.open() as f:
        rankings = json.load(f)
    
    # Build mapping: tweet_id -> cluster_id
    tweet_to_cluster = {}
    cluster_topics = {}
    
    for cluster_data in rankings:
        cluster_id = cluster_data["cluster_id"]
        topic = cluster_data.get("topic", f"Topic {cluster_id}")
        cluster_topics[cluster_id] = topic
        
        for user_data in cluster_data.get("top_users", []):
            for tweet_id in user_data.get("tweet_ids", []):
                tweet_to_cluster[tweet_id] = cluster_id
    
    # Load tweets and calculate popularity
    pca_weights, pca_means, pca_stds = load_pca_stats()
    
    tweets_by_cluster = defaultdict(list)
    
    with TRAIN_TWEETS_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = row.get("id", "")
            if not tweet_id:
                continue
            
            # Skip duplicates and non-root tweets
            if row.get("dup") != "0":
                continue
            if row.get("is_root") != "1":
                continue
            
            cluster_id = tweet_to_cluster.get(tweet_id)
            if cluster_id is None:
                continue  # Tweet not in any cluster
            
            popularity = calculate_popularity(row, pca_weights, pca_means, pca_stds)
            
            tweet_data = {
                "id": tweet_id,
                "text": row.get("text", ""),
                "user_name": row.get("user_name", ""),
                "user_id": row.get("user_id", ""),
                "popularity": popularity,
                "likes": int(row.get("likes", 0) or 0),
                "retweets": int(row.get("retweets", 0) or 0),
                "replies": int(row.get("replies", 0) or 0),
                "quotes": int(row.get("quotes", 0) or 0),
                "impressions": int(row.get("impressions", 0) or 0),
            }
            
            tweets_by_cluster[cluster_id].append(tweet_data)
    
    # Sort tweets by popularity within each cluster
    for cluster_id in tweets_by_cluster:
        tweets_by_cluster[cluster_id].sort(key=lambda x: x["popularity"], reverse=True)
    
    return tweets_by_cluster, cluster_topics


def generate_html(tweets_by_cluster: Dict[int, List[Dict]], cluster_topics: Dict[int, str], top_n: int = 10):
    """Generate HTML visualization."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Most Popular Tweets by Cluster</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .cluster-section {{
            margin-bottom: 50px;
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            border: 2px solid #e0e0e0;
        }}
        
        .cluster-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        .cluster-title {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .cluster-id {{
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        
        .tweets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        
        .tweet-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .tweet-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        
        .tweet-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        
        .tweet-rank {{
            background: #764ba2;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .tweet-popularity {{
            background: #4CAF50;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        
        .tweet-text {{
            font-size: 1em;
            line-height: 1.6;
            color: #333;
            margin-bottom: 15px;
            word-wrap: break-word;
        }}
        
        .tweet-author {{
            color: #667eea;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .tweet-metrics {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            font-size: 0.85em;
            color: #666;
        }}
        
        .metric-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #333;
        }}
        
        .stats-summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Most Popular Tweets by Cluster</h1>
        <p class="subtitle">Top {top_n} tweets ranked by popularity score for each category</p>
        
        <div class="stats-summary">
"""
    
    total_clusters = len(tweets_by_cluster)
    total_tweets = sum(len(tweets) for tweets in tweets_by_cluster.values())
    avg_tweets_per_cluster = total_tweets / total_clusters if total_clusters > 0 else 0
    
    html += f"""
            <div class="stat-item">
                <div class="stat-number">{total_clusters}</div>
                <div class="stat-label">Clusters</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{total_tweets}</div>
                <div class="stat-label">Total Tweets</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{avg_tweets_per_cluster:.0f}</div>
                <div class="stat-label">Avg Tweets/Cluster</div>
            </div>
        </div>
"""
    
    # Sort clusters by cluster_id
    for cluster_id in sorted(tweets_by_cluster.keys()):
        tweets = tweets_by_cluster[cluster_id][:top_n]
        topic = cluster_topics.get(cluster_id, f"Topic {cluster_id}")
        
        html += f"""
        <div class="cluster-section">
            <div class="cluster-header">
                <div class="cluster-title">{topic}</div>
                <div class="cluster-id">Cluster {cluster_id}</div>
            </div>
            <div class="tweets-grid">
"""
        
        for rank, tweet in enumerate(tweets, 1):
            text = tweet["text"]
            user_name = tweet["user_name"]
            popularity = tweet["popularity"]
            likes = tweet["likes"]
            retweets = tweet["retweets"]
            replies = tweet["replies"]
            quotes = tweet["quotes"]
            impressions = tweet["impressions"]
            
            html += f"""
                <div class="tweet-card">
                    <div class="tweet-header">
                        <div class="tweet-rank">#{rank}</div>
                        <div class="tweet-popularity">Popularity: {popularity:.2f}</div>
                    </div>
                    <div class="tweet-author">@{user_name}</div>
                    <div class="tweet-text">{text}</div>
                    <div class="tweet-metrics">
                        <div class="metric-item">
                            <span>❤️</span>
                            <span class="metric-value">{likes}</span>
                            <span>likes</span>
                        </div>
                        <div class="metric-item">
                            <span>🔄</span>
                            <span class="metric-value">{retweets}</span>
                            <span>retweets</span>
                        </div>
                        <div class="metric-item">
                            <span>💬</span>
                            <span class="metric-value">{replies}</span>
                            <span>replies</span>
                        </div>
                        <div class="metric-item">
                            <span>🔁</span>
                            <span class="metric-value">{quotes}</span>
                            <span>quotes</span>
                        </div>
                        <div class="metric-item">
                            <span>👁️</span>
                            <span class="metric-value">{impressions}</span>
                            <span>views</span>
                        </div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with HTML_FILE.open('w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML visualization saved to: {HTML_FILE}")
    print(f"   Open in browser: file://{HTML_FILE.absolute()}")


def main():
    """Main function."""
    print("Loading tweets and cluster assignments...")
    tweets_by_cluster, cluster_topics = load_tweets_with_clusters()
    
    print(f"Found {len(tweets_by_cluster)} clusters")
    for cluster_id in sorted(tweets_by_cluster.keys()):
        count = len(tweets_by_cluster[cluster_id])
        topic = cluster_topics.get(cluster_id, f"Topic {cluster_id}")
        print(f"  Cluster {cluster_id} ({topic}): {count} tweets")
    
    print(f"\nGenerating HTML visualization...")
    generate_html(tweets_by_cluster, cluster_topics, top_n=10)
    
    print(f"\n✅ Visualization generated successfully!")
    print(f"\nOpen the file in your browser:")
    print(f"   {HTML_FILE.absolute()}")


if __name__ == "__main__":
    main()
