# User Persona Generation Pipeline

This directory contains modules for ranking top users by tweet category and generating personas for OASIS simulation. The pipeline loads cleaned tweets and clustering output, then ranks users and generates LLM-based personas.

## Overview

The pipeline:
1. **Loads** `filtered_tweets.jsonl` and splits into train/test sets by time
2. **Loads** clustering output (`cluster_assignments.jsonl`, `cluster_topics.json`) from the clustering stage
3. **Ranks** top N users per cluster by tweet count (and popularity)
4. **Generates** personas for ranked users using LLM inference

Clustering is **not** performed here; it is done in the `clustering/` directory. See `CLUSTERING_INPUT_SPEC.md` for the expected clustering output format.

---

## Required Inputs

All default paths are defined in `config.py` and are relative to the repo root.

| Input | Default Path | Description |
|-------|--------------|-------------|
| `filtered_tweets.jsonl` | `data/processed/filtered_tweets.jsonl` | Cleaned tweets from the cleaning stage |
| `tweet_popularity_scores.csv` | `data/processed/tweet_popularity_scores.csv` | PCA-based popularity scores from the PCA stage |
| `cluster_assignments.jsonl` | `data/processed/cluster_assignments.jsonl` | Tweet → cluster assignments from clustering |
| `cluster_topics.json` | `data/processed/cluster_topics.json` | Cluster ID → topic description from clustering |

---


## Outputs and Locations

By default, outputs are written to `persona_generation/outputs/` (configurable via `--output-dir` or `config.OUTPUT_DIR`).

| Output | Location | Description |
|--------|----------|-------------|
| `train_tweets.csv` | `{OUTPUT_DIR}/train_tweets.csv` | Training tweets (time-split, filtered) |
| `test_tweets.csv` | `{OUTPUT_DIR}/test_tweets.csv` | Test tweets |
| `split_info.json` | `{OUTPUT_DIR}/split_info.json` | Split metadata (date ranges, user counts) |
| `user_rankings.json` | `{OUTPUT_DIR}/user_rankings.json` | Top users per cluster with sample tweets |
| `user_personas.csv` or `user_personas.json` | `{OUTPUT_DIR}/user_personas.csv` | Generated personas (OASIS Twitter simulation format) |

---

## Usage

### Usage

All parameters are set in `config.py`. Edit that file to change paths, split boundaries, or other settings.

```bash
# From repo root
cd persona_generation

# Set OpenAI API key (required)
export OPENAI_API_KEY="your-key-here"

# Optional: Twitter API for profile hints (set USE_TWITTER_API=True in config.py)
export TWITTER_API_KEY="your-bearer-token"

# Run pipeline
python user_persona_pipeline.py
```

---

## Visualizing Results

### HTML Dashboards (no extra dependencies)

**Rankings and personas dashboard**

```bash
cd persona_generation
python visualizations/visualize_rankings_html.py
```

- **Output**: `persona_generation/visualizations/rankings_dashboard.html` (or `{OUTPUT_DIR}/visualizations/` depending on config)
- Shows: top users by category, sample personas, category distribution
- Open in a browser: `file:///path/to/rankings_dashboard.html`

**Popular tweets by cluster**

```bash
python visualizations/visualize_popular_tweets.py
```

- **Output**: `persona_generation/outputs/visualizations/popular_tweets_by_cluster.html`
- Shows: most popular tweets per cluster, ranked by popularity score

### Matplotlib/Seaborn (requires `matplotlib`, `seaborn`)

```bash
python visualizations/visualize_rankings.py
```

- **Output**: PNG files in `persona_generation/outputs/visualizations/`
  - `top_users_by_category.png`
  - `category_distribution.png`
  - `user_rankings_heatmap.png`
  - `persona_examples.png`
  - `summary_statistics.png`

---

## Output Formats

### JSON (Reddit/OASIS)

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
    "category": "Primary topic",
    "all_categories": ["Topic A", "Topic B"],
    "cluster_id": 0,
    "original_user_name": "username1",
    "original_user_id": "123456",
    "tweet_count": 45
  }
]
```

### CSV (Twitter / OASIS simulation)

Default format. Compatible with `oasis.social_agent.agents_generator.generate_agents`.

```csv
user_id,name,username,following_agentid_list,previous_tweets,user_char,description
0,CryptoGnomie,CryptoGnomie,[],[],"Detailed persona description...","Bio text"
```

- `following_agentid_list`: Empty `[]` (no follow graph from pipeline)
- `previous_tweets`: Sample tweets from the user's training data (up to 15 per user), seeded so agents have posts to interact with
- Set `simulation/config.yaml` → `data.csv_path` to `persona_generation/outputs/user_personas.csv` to run Twitter simulation

---

## Config

All parameters are in `config.py`:

- **Paths**: `FILTERED_TWEETS_PATH`, `TWEET_POPULARITY_PATH`, `CLUSTER_ASSIGNMENTS_PATH`, `CLUSTER_TOPICS_PATH`, `OUTPUT_DIR`
- **Split**: `TRAIN_YEAR`, `TRAIN_MONTH`, `TEST_YEAR`, `TEST_MONTH`, `MIN_TRAIN_TWEETS`, `MIN_TEST_TWEETS`
- **Ranking**: `TOP_N`
- **Persona**: `OUTPUT_FORMAT` (`"csv"` default for Twitter sim, or `"json"` for Reddit-style), `USE_TWITTER_API` (bool)

---

## Requirements

- Python 3.8+
- `openai` (LLM persona generation)
- `requests` (optional, for Twitter API)

Clustering dependencies (`hdbscan`, `sentence-transformers`, `scikit-learn`) are **not** required by persona generation; clustering runs separately.

---

## Pipeline Order

1. **Cleaning** → `filtered_tweets.jsonl`
2. **PCA** → `tweet_popularity_scores.csv`
3. **Clustering** → `cluster_assignments.jsonl`, `cluster_topics.json` (see `CLUSTERING_INPUT_SPEC.md`)
4. **Persona generation** (this pipeline) → `user_rankings.json`, `user_personas.csv`
5. **Simulation** → OASIS Twitter simulation: set `simulation/config.yaml` `data.csv_path` to `persona_generation/outputs/user_personas.csv`
