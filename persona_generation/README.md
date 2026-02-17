# Persona Generation Pipeline

This repository produces OASIS‑compatible user personas from Twitter/X data by:

1. **Splitting** tweets into train/test time windows  
2. **Joining** training tweets with cluster assignments & topics  
3. **Ranking** users per cluster/category  
4. **Generating** personas (LLM‑based or DRY‑RUN placeholders)

> ✅ **Single entrypoint:** Run **`user_persona_pipeline.py`** only.  
> The other scripts are internal modules used by the pipeline.

---

## Quick Start

### 0.5 Package Installation
If you use LLM persona generation or demographics inference (ie DRY_RUN=False), install:
```
pip install openai
```
If you call Twitter API (ie USE_TWITTER_API=True), also install:
```
pip install requests
```

### 1. Environment Variables

```bash
# Required ONLY if DRY_RUN=False (LLM mode)
export OPENAI_API_KEY="sk-..."

# Optional: To fetch profile hints from Twitter API v2 if USE_TWITTER_API=True
export TWITTER_API_KEY="Bearer <YOUR_TWITTER_V2_BEARER_TOKEN>"

# Optional overrides
export OPENAI_MODEL="gpt-4o-mini"
export GLOBAL_SEED="20260215"
export NOISE_EPS="0.05"
```

### 2. Provide Inputs at Paths Configured in config.py
See inputs & Data Contracts below for file formats

### 3. Run the Pipeline (only script you should run)
```
python user_persona_pipeline.py
```

### 4. Find Outputs
All artifacts are written to OUTPUT_DIR (defined in config.py), including:
- `train_tweets.csv`, `test_tweets.csv`, `split_info.json`
- `user_cluster_mix.json`, `user_rankings.json`
- Final personas: `user_personas.json` or `user_personas.csv` (based on `OUTPUT_FORMAT`)

## Repository Overview (Key Files)
- `user_persona_pipeline.py` — Main pipeline (entrypoint).
- `config.py` — Central configuration (paths, split windows, ranking strategy, output format).
- `data_split.py` — Utilities to load filtered tweets, split by time, filter active users, and write CSVs.
- `rank_users_by_category.py` — Ranking logic (count / mixture / blend) and user‑cluster mixture utilities.
- `generate_personas.py` — Demographics inference (LLM or DRY‑RUN) and persona generation (LLM or DRY‑RUN).

**Do not run the internal modules directly; use user_persona_pipeline.py only.**

## Inputs & Data Contracts
All paths are configured in `config.py`

### 1) Filtered Tweets — `FILTERED_TWEETS_PATH` (JSONL)

Format: JSON Lines; one tweet per line.
Fields used:
- `id` (str)
- `author.id` (str), `author.userName` (str)
- `clean_text` (str) or `text` (str)
- `createdAt` (str, e.g., "Tue Jul 22 07:01:42 +0000 2025")
- `isReply` (bool)
- Engagements (ints): `likeCount`, `retweetCount`, `replyCount`, `quoteCount`, `viewCount`

Example line:
```
{
  "id": "1891234567890123456",
  "author": {"id": "123456", "userName": "someuser"},
  "clean_text": "Exploring LLMs and data viz!",
  "createdAt": "Tue Jul 22 07:01:42 +0000 2025",
  "isReply": false,
  "likeCount": 10,
  "retweetCount": 2,
  "replyCount": 1,
  "quoteCount": 0,
  "viewCount": 150
}
```

### 2. Cluster Assignments — CLUSTER_ASSIGNMENTS_PATH (JSONL)

Format: JSON Lines, maps tweet_id → cluster_id
Example line:
```
{"tweet_id": "1891234567890123456", "cluster_id": 7}
```

### 3. Cluster Topics — CLUSTER_TOPICS_PATH (JSON)

Format: JSON object; maps cluster_id → topic string
Keys may be strings in JSON; the pipeline casts them to int.
```
{
  "7": "AI & Data Science",
  "12": "Sustainability"
}
```

### 4. Popularity Scores — TWEET_POPULARITY_PATH (CSV)

Used to enrich training tweets.
Columns:

- `tweet_id` (str)
- `popularity_score` (float)
```
tweet_id,popularity_score
1891234567890123456,18.5
```

## Configuration (config.py)
The pipeline reads these values at runtime.

### Paths (relative to repo root)

- `FILTERED_TWEETS_PATH`: Path
- `TWEET_POPULARITY_PATH`: Path
- `CLUSTER_ASSIGNMENTS_PATH`: Path
- `CLUSTER_TOPICS_PATH`: Path
- `OUTPUT_DIR`: Path — where all artifacts are written
- `VISUALIZATIONS_DIR`: Path — reserved for charts/plots (if added later)

### Train/Test Split

- `TRAIN_YEAR`: int (e.g., 2025)
- `TRAIN_MONTH`: int (1–12)
- `TEST_YEAR`: int
- `TEST_MONTH`: int
- `MIN_TRAIN_TWEETS`: int — per‑user minimum in train
- `MIN_TEST_TWEETS`: int — per‑user minimum in test


**Important**: These months must match the date coverage of filtered_tweets.jsonl.

### Run Toggles

- `DRY_RUN`: bool — if True, **no OpenAI/Twitter API calls**; demographics/personas are synthetic placeholders.
- `USE_TWITTER_API`: bool — if True, fetch optional profile hints (location, created_at, bio). Requires `TWITTER_API_KEY`.

### Ranking & Output

- `TOP_N`: int — number of top users per cluster
- `RANK_MODE`: Literal["count", "mixture", "blend"] — user ranking strategy
- 1BLEND_ALPHA1: float — used only when `RANK_MODE`="blend" (recommended range: 0.4–0.8)
- `MIN_MIX_WEIGHT`: float — drop users with mixture weight < `MIN_MIX_WEIGHT` (noise filter)
`OUTPUT_FORMAT`: Literal["json", "csv"] — final personas format

### Environment Variables (read in generate_personas.py)

- `OPENAI_MODEL` (default: "gpt-4o-mini")
- `OPENAI_API_KEY` (required if `DRY_RUN=False`)
- `TWITTER_API_KEY` (optional; Twitter API v2 bearer token)
- `GLOBAL_SEED` (default: `20260215`) — base seed for deterministic per‑user RNG
- `NOISE_EPS` (default: `0.05`) — small noise in prior‑based sampling (DRY_RUN)

Suggested type hints (optional) for extra safety in config.py:
```
from typing import Literal
RANK_MODE: Literal["count", "mixture", "blend"] = "mixture"
OUTPUT_FORMAT: Literal["csv", "json"] = "csv"
```

# Pipeline Reference: Splits, Schema, Ranking, Personas & Outputs

This document summarizes the **core operational details** of the pipeline: split logic and fallbacks, train/test CSV schema, ranking strategies (and how scores are computed), the persona generation flow, what output artifacts contain, key notes/assumptions, and troubleshooting.

---

## 🔢 Split Logic & Fallbacks

### Explicit Month Windows (Primary Mode)
- The pipeline splits **by month** using configuration in `config.py`:
  - **Train window:** `(TRAIN_YEAR, TRAIN_MONTH)` (UTC)
  - **Test window:** `(TEST_YEAR, TEST_MONTH)` (UTC)
- Boundaries are **inclusive start** and **exclusive end**, and timestamps are treated as **UTC**.
- After splitting, the pipeline **keeps only users active in both periods**:
  - A user must have at least `MIN_TRAIN_TWEETS` tweets in the train window **and** at least `MIN_TEST_TWEETS` tweets in the test window.

### Internal Utilities (Alternate Behaviors)
The lower-level split utilities also support:
- **Rolling split:** use the **most recent N months** for train and the **preceding M months** for test (30-day month approximation).
- **80/20 time fallback:** if the chosen test window is too small (e.g., very few tweets or falls outside data range), the utility falls back to:
  - **Last 80%** of the timeline for **train**
  - **Immediate previous 20%** for **test**

> The entrypoint script (`user_persona_pipeline.py`) uses the **explicit month windows** for reproducibility. The rolling and 80/20 behaviors are internal safeguards in the split utility.

---

## 🗃️ Train/Test CSV Schema

**Files:** `train_tweets.csv`, `test_tweets.csv`  
**Columns (in order):**
dup, id, topic_id, reply_to_id, reply_to_user, likes, retweets, replies, quotes,
impressions, user_name, user_id, is_root, text, timestamp

### Column Notes
- `dup` — `"0"` if the tweet is retained; `"1"` indicates duplicates (the split writer emits `"0"` for kept rows).
- `id` — tweet ID (string).
- `topic_id` — **not** assigned at split time; clustering later provides this mapping. Remains empty/blank until joined with cluster assignments.
- `reply_to_id`, `reply_to_user` — not used by this pipeline stage (placeholders for compatibility).
- `likes`, `retweets`, `replies`, `quotes`, `impressions` — integer engagement metrics (0 if missing).
- `user_name`, `user_id` — author handle and ID.
- `is_root` — `"1"` if **not** a reply (root tweet), `"0"` otherwise.
- `text` — canonical tweet text (prefer `clean_text` if available).
- `timestamp` — ISO 8601 in **UTC**.

---

## 🧮 Ranking Strategies (Per Cluster)

After joining **training tweets** with **cluster assignments** and **popularity**:

Let:
- `c_user` = user’s **in-cluster tweet count**
- `c_max` = **max** in-cluster count among all users in the **same** cluster
- `mixture_weight` = user’s topical purity for the cluster  
  = `(user’s train tweets in this cluster) / (user’s total train tweets across all clusters)`

### Modes
1. **`count`**  
   Score = `c_user`  
   (Ranks by in-cluster activity; legacy behavior)

2. **`mixture`**  
   Score = `mixture_weight`  
   - Users with `mixture_weight < MIN_MIX_WEIGHT` are **excluded**.  
   (Emphasizes how focused a user is on the cluster)

3. **`blend`**  
   Score = `BLEND_ALPHA * mixture_weight + (1 - BLEND_ALPHA) * (c_user / c_max)`  
   - Users with `mixture_weight < MIN_MIX_WEIGHT` are **excluded**.  
   (Balances topical focus with raw activity)

### Sorting / Tie-breakers
- Sort by **score** (desc), then **tweet_count** (desc), then **total_popularity** (desc).
- `TOP_N` users per cluster are retained in the output.

**Diagnostics written:**  
- `user_cluster_mix.json`: `user_id -> {cluster_id: mixture_weight}`  
- `user_rankings.json`: per-cluster top users, including `mixture_weight`, `score`, `tweet_count`, `sample_tweets`, and `total_popularity`.

---

## 🧠 Persona Generation Flow

**Input:** `user_rankings.json` (top users per cluster).

### Steps
1. **Primary Category Selection**
   - If a user appears in multiple categories, choose the **primary** category by **highest tweet_count**; break ties by **best rank** in that category.

2. **Tweet Aggregation**
   - Combine the user’s sample tweets from **all** their categories.
   - **Deduplicate** while preserving order; **limit to 15** tweets total.

3. **Demographics Inference**
   - **DRY_RUN = True**  
     - Return **synthetic demographics** sampled from mild priors: `age ∈ [20, 60]`, gender ∈ `{male, female, other}`, MBTI ∈ 16 types, country from a small set, profession from a broad CTE-like list (with slight tilt to IT/STEAM by default).
   - **DRY_RUN = False**  
     - Use the **OpenAI model** to infer demographics **from user tweets** (and optional hints from the Twitter profile if `USE_TWITTER_API=True` and credentials are available).

4. **Persona Text Generation**
   - **DRY_RUN = True**  
     - Generate **placeholder bio and persona** text.
   - **DRY_RUN = False**  
     - Use the **OpenAI model** to generate a **bio** (1–2 sentences) and a **persona** (2–3 sentences).
   - **Names/usernames** are **never hallucinated**. The original username is kept for both `username` and `realname`.

### Determinism
- A per-user deterministic RNG is derived from `sha256(f"{GLOBAL_SEED}-{user_id}")`.
- In LLM mode, a seed may be used when available; tweet samples are shuffled using the per-user RNG for reproducibility.

### Outputs
- **JSON** mode (Reddit-style array): includes `username`, `realname`, `bio`, `persona`, demographics, primary `category`, `all_categories`, `cluster_id`, original names/IDs, `tweet_count`, `sample_tweets`.
- **CSV** mode (OASIS Twitter Simulation):  
  Columns:  
  `user_id, name, username, following_agentid_list, previous_tweets, user_char, description`
  - `user_id` — sequential index in output (0..N-1)
  - `following_agentid_list` — deterministic per-user subset of other agent IDs
  - `previous_tweets` — JSON array **string** of sample tweets
  - `user_char` — persona text
  - `description` — bio text

---

## 🗂️ Output Artifacts (What They Contain)

All artifacts are written to `OUTPUT_DIR`:

### Splits
- **`train_tweets.csv` / `test_tweets.csv`**  
  See schema above. These are filtered to **active** users in both windows. Only **root tweets** and **non-duplicate** rows are retained for downstream use.
- **`split_info.json`**  
  Summary metadata, e.g.:
  ```json
  {
    "train_months": 7,
    "test_months": 8,
    "train_period": {"start": "...", "end": "...", "tweet_count": 1234, "user_count": 99},
    "test_period":  {"start": "...", "end": "...", "tweet_count": 234,  "user_count": 55},
    "active_users_count": 42,
    "active_user_ids": ["123456", "7890", "..."]
  }

## Ranking / QA
- `user_cluster_mix.json` — user_id -> {cluster_id: mixture_weight} (topical purity map).
- `user_rankings.json` — array of
```
{
  "cluster_id": 7,
  "topic": "AI & Data Science",
  "top_users": [
    {
      "user_name": "someuser",
      "user_id": "123456",
      "tweet_count": 42,
      "tweet_ids": ["..."],
      "sample_tweets": ["...", "..."],
      "total_popularity": 185.0,
      "mixture_weight": 0.78,
      "score": 0.89
    }
  ]
}
```

### Personas (Final)

user_personas.json (Reddit-style) or user_personas.csv (OASIS Twitter Simulation), ready for downstream simulation/consumption