# User Persona Generation Pipeline

This directory contains modules for ranking top users by tweet category and generating personas for simulation.

## Overview

The pipeline:
1. **Splits tweets** into train/test sets using time-based splitting (first 6 months train, next 1-3 months test)
2. **Clusters train tweets** by topic using HDBSCAN or KMeans
3. **Ranks top N users** per category based on tweet count
4. **Generates personas** for top users using LLM inference

## Modules

### `data_split.py`
Time-based train/test splitting:
- Extracts timestamps from Snowflake tweet IDs
- Splits chronologically (first N months → train, next M months → test)
- Filters to only users active in BOTH periods

### `rank_users_by_category.py`
Ranks users within each cluster:
- Groups tweets by cluster ID
- Counts tweets per user per cluster
- Returns top N users per cluster

### `generate_personas.py`
Generates personas using hybrid approach:
- Optional Twitter API for location/created_at hints
- LLM inference for demographics (age, gender, MBTI, country, profession)
- LLM generation for detailed persona

### `user_persona_pipeline.py`
Main orchestration script that runs the full pipeline.

## Usage

### Basic Usage

```bash
# Set OpenAI API key (required)
export OPENAI_API_KEY="your-key-here"

# Optional: Set Twitter API key for location hints
export TWITTER_API_KEY="your-bearer-token"

# Run pipeline
python user_persona_pipeline.py
```

### Advanced Options

```bash
python user_persona_pipeline.py \
    --train-months 6 \
    --test-months 1 \
    --clustering-method hdbscan_400 \
    --top-n 10 \
    --min-train-tweets 1 \
    --min-test-tweets 1 \
    --use-twitter-api \
    --output-format json \
    --output-dir ./output
```

### Arguments

- `--train-months`: Number of months for training period (default: 6)
- `--test-months`: Number of months for test period (default: 1)
- `--clustering-method`: One of `hdbscan_200`, `hdbscan_400`, `hdbscan_600`, `kmeans_3`, `kmeans_4`, `kmeans_5`, `kmeans_6` (default: `hdbscan_400`)
- `--top-n`: Number of top users per category (default: 10)
- `--min-train-tweets`: Minimum tweets in train period (default: 1)
- `--min-test-tweets`: Minimum tweets in test period (default: 1)
- `--use-twitter-api`: Use Twitter API to fetch user profiles (requires `TWITTER_API_KEY`)
- `--output-format`: `json` (Reddit format) or `csv` (Twitter format) (default: `json`)
- `--output-dir`: Output directory (default: `tweet_processing/`)

## Output Files

- `train_tweets.csv`: Training tweets (first N months)
- `test_tweets.csv`: Test tweets (next M months)
- `split_info.json`: Split metadata (date ranges, user counts)
- `user_rankings.json`: Top users per cluster
- `user_personas.json` or `user_personas.csv`: Generated personas

## Output Format

### JSON Format (Reddit)
```json
[
  {
    "username": "generated_username",
    "realname": "Generated Name",
    "bio": "Bio text",
    "persona": "Detailed persona description...",
    "age": 28,
    "gender": "male",
    "mbti": "ENTP",
    "country": "US",
    "profession": "Information Technology",
    "category": "VeChain Hayabusa Upgrade",
    "cluster_id": 0,
    "original_user_name": "username1",
    "original_user_id": "123456",
    "tweet_count": 45
  }
]
```

### CSV Format (Twitter)
```csv
user_id,name,username,user_char,description
0,Generated Name,generated_username,"Detailed persona description...","Bio text"
```

## Requirements

- Python 3.8+
- `hdbscan`
- `scikit-learn`
- `sentence-transformers`
- `openai`
- `numpy`
- `pandas` (optional, for easier data manipulation)
- `requests` (optional, if using Twitter API)

## Notes

- **Time-based split**: Uses chronological splitting (not random) to ensure users are active in test period
- **User filtering**: Only includes users active in BOTH train and test periods
- **Demographics**: Twitter API doesn't provide age/gender/MBTI, so LLM inference is required
- **Cost**: LLM calls for demographics inference and persona generation (consider batching/caching)
