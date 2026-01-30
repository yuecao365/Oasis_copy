# pca_cleaning – PCA weights for engagement

Compute PCA weights for engagement metrics (likes, retweets, replies, quotes, impressions) used in popularity scoring.

## What it does

- Reads non-duplicate rows from `tweet_processing/tweet_Nov+Dec.txt` (from **cleaning**)
- Log(1 + count) transform, z-score, first principal component
- Writes `tweet_processing/pca_stats.json` (weights, means, stds)

## Main files

- **`PCA`** – Script. Input/output paths are configurable inside.
- **`pca_stats.json`** – Output weights and stats.
- **`tweet_Nov+Dec.txt`** – Optional copy of preprocessed tweets for local runs; otherwise produced by **cleaning**.

## Usage

From repo root. Requires `tweet_processing/tweet_Nov+Dec.txt`.

```bash
python pca_cleaning/PCA
```

`pca_stats.json` is used by **clustering** / **topic_generation** for popularity-based ranking.
