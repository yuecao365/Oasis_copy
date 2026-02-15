"""Hyperparameters and paths for the persona generation pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths (relative to repo root)
FILTERED_TWEETS_PATH = REPO_ROOT / "data" / "processed" / "filtered_tweets.jsonl"
TWEET_POPULARITY_PATH = REPO_ROOT / "data" / "processed" / "tweet_popularity_scores.csv"
CLUSTER_ASSIGNMENTS_PATH = REPO_ROOT / "data" / "processed" / "cluster_assignments.jsonl"
CLUSTER_TOPICS_PATH = REPO_ROOT / "data" / "processed" / "cluster_topics.json"
OUTPUT_DIR = REPO_ROOT / "persona_generation" / "outputs"
VISUALIZATIONS_DIR = REPO_ROOT / "persona_generation" / "visualizations"

# Train/test split parameters (must match date range in filtered_tweets.jsonl)
TRAIN_YEAR = 2025
TRAIN_MONTH = 7  # July
TEST_YEAR = 2025
TEST_MONTH = 8   # August
MIN_TRAIN_TWEETS = 1
MIN_TEST_TWEETS = 1

DRY_RUN = True   # If True, skip all OpenAI and Twitter API calls

# Ranking and persona
TOP_N = 10  # Number of top users per cluster
OUTPUT_FORMAT = "csv"  # "csv" (Twitter/OASIS simulation) or "json" (Reddit-style)
USE_TWITTER_API = False  # Fetch user profiles from Twitter API (requires TWITTER_API_KEY)
