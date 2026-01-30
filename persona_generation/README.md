# persona_generation – User rankings and personas

Split tweets in time, cluster them, rank users by category, and generate personas for simulation.

## What it does

1. **Split** – Time-based train/test split (e.g. Nov vs Dec), filter to users active in both.
2. **Cluster** – Cluster train tweets (HDBSCAN/KMeans), optionally label topics via OpenAI.
3. **Rank** – Top users per cluster by tweet count.
4. **Personas** – Generate persona descriptions (and optionally demographics) for ranked users.

## Main files

- **`data_split.py`** – Train/test split and user filtering.
- **`user_persona_pipeline.py`** – Orchestrates split → cluster → rank → personas.
- **`rank_users_by_category.py`** – Rankings per cluster.
- **`generate_personas.py`** – Persona generation (LLM, optional Twitter API).
- **`README_personas.md`** – Detailed pipeline and persona options.

## Usage

From repo root. Uses `tweet_processing/tweet_Nov+Dec.txt` and `pca_stats.json`. Set `OPENAI_API_KEY` for topics and personas.

```bash
python persona_generation/user_persona_pipeline.py
```

Outputs include `train_tweets.csv`, `test_tweets.csv`, `user_rankings.json`, `user_personas.json`. Personas can be turned into agent CSVs for **simulation** (see `README_personas.md`).
