# Clustering Output Format (Contract for Clustering Directory)

The persona generation pipeline expects clustering to produce the following outputs. The clustering directory should implement this contract.

## Inputs for Clustering (from earlier pipeline stages)

- **filtered_tweets.jsonl** – from cleaning (`user_filtering_and_replychain.py`)
- **tweet_popularity_scores.csv** – from PCA (`tweet_popularity_pca.py`)
- **reply_chain.jsonl** (optional) – for conversation structure

---

## Required Outputs

### `cluster_assignments.jsonl`

One JSON object per line. Each line assigns a tweet to a cluster:

```json
{"tweet_id": "1947552617514561676", "cluster_id": 0}
```

| Field | Type | Description |
|-------|------|-------------|
| `tweet_id` | string | Must match `id` in `filtered_tweets.jsonl` |
| `cluster_id` | int | Cluster label. Use `-1` for noise (excluded from ranking) |

Only tweets with `cluster_id >= 0` will be used for persona generation ranking.

---

### `cluster_topics.json`

JSON object mapping cluster IDs (as strings) to topic descriptions:

```json
{
  "0": "Cryptocurrency and blockchain discussions",
  "1": "Tech and product updates"
}
```

| Keys | Type | Description |
|------|------|-------------|
| Cluster ID | string | e.g. `"0"`, `"1"` |
| Value | string | Human-readable topic description |

---

## Notes

- Clustering should use `filtered_tweets.jsonl` and `tweet_popularity_scores.csv` as inputs.
- Use top *k* tweets by popularity per cluster to generate topic descriptions (e.g. via LLM).
- Persona generation reads these files from `data/processed/` (or paths in `config.py`).
