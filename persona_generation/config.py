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
TRAIN_START_DATE = "2025-05-01T00:00:00+00:00"
TRAIN_END_DATE = "2025-08-01T00:00:00+00:00"
TEST_START_DATE = "2025-08-01T00:00:00+00:00"
TEST_END_DATE = "2025-09-01T00:00:00+00:00"
MIN_TRAIN_TWEETS = 1
MIN_TEST_TWEETS = 1

DRY_RUN = False   # If True, skip all OpenAI and Twitter API calls

# Ranking and persona
TOP_N = 10  # Number of top users per cluster
OUTPUT_FORMAT = "csv"  # "csv" (Twitter/OASIS simulation) or "json" (Reddit-style)
USE_TWITTER_API = False  # Fetch user profiles from Twitter API (requires TWITTER_API_KEY)

# -----------------------------------------------------------------------------
# Ranking strategy configuration for user persona pipeline
# -----------------------------------------------------------------------------
# RANK_MODE controls how users are ranked within each cluster/category.
#   - "count"   : based on in-cluster tweet count (legacy behavior)
#   - "mixture" : based on user's topical purity (share of their tweets in cluster)
#   - "blend"   : alpha * mixture + (1 - alpha) * normalized in-cluster count
#
# BLEND_ALPHA is only used when RANK_MODE == "blend". Typical range: 0.4–0.8.
# MIN_MIX_WEIGHT filters out users whose mixture weight in a cluster is tiny.
#   Example: 0.03 drops users with <3% of their tweets in that cluster.
# -----------------------------------------------------------------------------

# User ranking strategy (choose one: "count", "mixture", "blend")
RANK_MODE: str = "mixture"

# Blend factor for "blend" mode (ignored for other modes)
BLEND_ALPHA: float = 0.6

# Optional lower bound on mixture weight to reduce noise
MIN_MIX_WEIGHT: float = 0.0

# Follow-graph density control for generated personas
MAX_FOLLOWEES: int = 4
