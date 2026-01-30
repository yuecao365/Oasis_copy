# clustering – Topic clustering

Cluster tweets by embedding similarity and label clusters with topics (optional OpenAI).

## What it does

- Loads preprocessed, non-duplicate root tweets (e.g. from `tweet_processing/tweet_Nov+Dec.txt`)
- Computes PCA-weighted popularity, embeds text (Sentence-BERT), clusters (HDBSCAN or KMeans)
- Optionally calls OpenAI to name each cluster from top tweets

## Main files

- **`topic_generation.py`** – Load tweets, embed, build prompts. Used by `run_cluster_topics`.
- **`run_cluster_topics.py`** – Runs clustering with several min_cluster_size / n_clusters settings, writes `tweet_processing/cluster_topics.txt`.
- **`cluster_topics.txt`** – Output summary of cluster topics.

## Usage

From repo root. Requires `tweet_processing/tweet_Nov+Dec.txt` (from **cleaning**) and optionally `tweet_processing/pca_stats.json` (from **pca_cleaning**). Set `OPENAI_API_KEY` if you want LLM-generated topic labels.

```bash
python clustering/run_cluster_topics.py
```

Output feeds into **persona_generation** (user rankings, personas).
