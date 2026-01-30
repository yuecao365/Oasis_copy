# cleaning – Tweet preprocessing

Preprocess raw Twitter JSON dumps into a unified CSV used by later steps (PCA, clustering, topic generation).

## What it does

- Lowercases text, normalizes URLs → `<URL>`, @mentions → `@user`, numbers → `<num>`
- Marks duplicates via Jaccard similarity on token sets
- Outputs CSV with `dup`, `id`, `topic_id`, `reply_to_id`, `reply_to_user`, engagement metrics, `user_name`, `user_id`, `is_root`, `text`

## Main file

- **`preprocessing`** – Script. Reads from `dataset/twitter/twitter/2025-11-*` and `2025-12-*` JSON; writes to `tweet_processing/tweet_Nov+Dec.txt` (configurable inside the script).

## Usage

Run from repo root. Adjust `DATA_ROOT`, `OUTPUT_PATH`, and input paths inside `preprocessing` for your layout.

```bash
python cleaning/preprocessing
# or redirect: python cleaning/preprocessing > cleaning/cleaned.txt
```

Output is used by **pca_cleaning** and **clustering** / **persona_generation**.
