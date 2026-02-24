# Simulation-vs-Ground-Truth Metrics

This document defines the current metrics computed by
`simulation/compare_sim_to_test.py` against
`persona_generation/outputs/test_tweets.csv`.

## Data Sources

- Simulation DB (SQLite): `data/simu_db/.../*.db`
- Test split: `persona_generation/outputs/test_tweets.csv`
- Personas: `persona_generation/outputs/user_personas.csv`
- Split metadata: `persona_generation/outputs/split_info.json` (used for weekday alignment)

All metrics are evaluated on the persona user set (username-matched).

## Current Report Sections

The comparison report has these top-level sections:

- `coverage`
- `user_activity_fit`
- `temporal_fit`
- `engagement_fit`

Graph-structure metrics are not included in the current report.

## Metric Definitions

1. Coverage
- `persona_user_count`: number of persona users considered.
- `gt_active_users`: users with at least one root post in test data.
- `sim_active_users`: users with at least one simulated root post.

2. User Activity Fit
- Per-user post counts:
  - GT: root posts (`is_root == 1`)
  - Sim: original posts (`original_post_id IS NULL`, `created_at > 0`)
- Metrics:
  - `spearman_posts_per_user` (higher is better)
  - `jsd_posts_per_user` (lower is better)
  - `top10_overlap`, `top20_overlap` (higher is better)

3. Temporal Fit (current weekly-aligned method)
- `day_of_week_jsd`: JSD between GT weekday post distribution and simulated weekday distribution.
- `cumulative_posts_daily_rmse`: RMSE between cumulative daily post curves.
- `cumulative_posts_daily_nrmse`: normalized RMSE (lower is better).

4. Engagement Fit
- GT means per root post:
  - likes, retweets, replies, quotes
- Sim means per original post:
  - likes (`num_likes`), shares/reposts (`num_shares`), replies/comments
- Also includes totals:
  - repost/quote totals
  - total posts in GT vs Sim

## Default Timestep Assumption

`compare_sim_to_test.py` now defaults to:

- `--timestep_hours 24.0`

So one simulation step is treated as one day unless overridden.

## Script Usage

```bash
python simulation/compare_sim_to_test.py \
  --db data/simu_db/yaml_gpt/sweep_high_posts.db \
  --sim_config simulation/config.yaml
```

For non-daily simulations, pass `--timestep_hours` explicitly.

## Outputs

- JSON report (default): `simulation/metrics_report.json`
- Markdown report (default): `simulation/metrics_report.md`
- Archived per-run reports: `simulation/metrics_history/...`
