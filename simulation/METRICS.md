# Simulation-vs-Ground-Truth Metrics

This document defines how to compare OASIS simulation outputs against ground truth in `persona_generation/outputs/test_tweets.csv`.

## Data Sources

- Simulation DB (SQLite): `data/simu_db/.../*.db`
- Test split: `persona_generation/outputs/test_tweets.csv`
- Personas used in simulation: `persona_generation/outputs/user_personas.csv`
- Filtered tweets (for reply-derived ground-truth graph): `data/processed/filtered_tweets.jsonl`

All metrics should be computed on the same persona user set (matched by username).

## Paper-Aligned Metrics (OASIS-inspired)

The OASIS paper reports cascade-shape metrics and normalized RMSE for dynamics, plus polarization/herding analyses.
For this Twitter pipeline, we operationalize the same spirit as follows:

1. Dynamics error (NRMSE over time)
- Build cumulative time curves for:
  - posts
  - interactions (likes + shares + replies/comments)
- Compare simulation vs test with:
  - RMSE
  - NRMSE = RMSE / (max(test_curve) - min(test_curve))

2. Network dynamics
- Compare simulation follow graph vs test reply-interaction graph:
  - edge precision/recall/F1
  - density
  - reciprocity

## Core Fit Metrics

1. User Activity Fit
- Per-user post counts:
  - test: root tweets (`is_root == 1`)
  - sim: original posts (`original_post_id IS NULL`, `created_at > 0`)
- Metrics:
  - Spearman correlation
  - Jensen-Shannon divergence between normalized user activity distributions
  - Top-K active user overlap (K=10, 20)

2. Engagement Fit
- Test side:
  - mean likes, retweets, replies, quotes per root tweet
- Simulation side:
  - mean `num_likes`, `num_shares`, comment-count per original post
- Also report total volumes and per-post means.

3. Temporal Rhythm Fit
- Hour-of-day distributions:
  - test from tweet timestamps
  - sim from timestep clock (`created_at * timestep_hours`)
- Metric:
  - Jensen-Shannon divergence over 24-bin hour histograms

4. Graph-Structure Fit
- Simulation graph: `follow(follower_id -> followee_id)`
- Ground truth graph: `replier -> replied-to` from test-period tweets
- Metrics:
  - nodes, edges
  - density
  - reciprocity
  - edge precision/recall/F1 (on overlapping user set)

## Notes / Limitations

- This pipeline does not directly expose full retweet/reply trees in test CSV, so cascade-depth/max-breadth is approximated through available interaction/time metrics.
- If simulation actions exclude comments/reposts, some engagement metrics may be sparse.
- For robust results, run multiple seeds and report mean ± CI.

## Script

Use:

```bash
python simulation/compare_sim_to_test.py \
  --db data/simu_db/yaml_gpt/False_Business_0.db \
  --test_csv persona_generation/outputs/test_tweets.csv \
  --personas_csv persona_generation/outputs/user_personas.csv \
  --filtered_jsonl data/processed/filtered_tweets.jsonl
```

Outputs:
- `simulation/metrics_report.json`
- `simulation/metrics_report.md`
