# Simulation-vs-Ground-Truth Metrics

This document defines the metrics computed by
`simulation/compare_sim_to_test.py` against
`persona_generation/outputs/test_tweets.csv`.

## Data Sources

- Simulation DB (SQLite): `data/simu_db/.../*.db`
- Test split: `persona_generation/outputs/test_tweets.csv`
- Personas: `persona_generation/outputs/user_personas.csv`
- Split metadata: `persona_generation/outputs/split_info.json` (used for weekday alignment)

All metrics are evaluated on the persona user set (username-matched).

## Report Sections

The comparison report has five top-level sections:

- `coverage`
- `user_activity_fit`
- `temporal_fit`
- `engagement_fit`
- `content_fit` ← **primary focus**

---

## Metric Definitions

### 1. Coverage
- `persona_user_count`: number of persona users considered.
- `gt_active_users`: users with at least one root post in test data.
- `sim_active_users`: users with at least one simulated root post.

### 2. User Activity Fit
Per-user post counts compared between GT (root posts, `is_root == 1`) and Sim (original posts, `original_post_id IS NULL`, `created_at > 0`).

- `spearman_posts_per_user`: rank correlation of per-user post counts (higher is better).
- `top10_overlap`: fraction of top-10 most-active users shared between GT and Sim (higher is better).

### 3. Temporal Fit
- `day_of_week_jsd`: JSD between GT and Sim weekday post distributions (lower is better).
- `cumulative_posts_daily_rmse`: RMSE between cumulative daily post curves.
- `cumulative_posts_daily_nrmse`: normalized RMSE (lower is better).

### 4. Engagement Fit
- `gt_total_posts` / `sim_total_posts`: raw post counts for reference.
- `likes_top_share_curve_rmse`: RMSE between GT and Sim "top X% posts capture Y% of likes" curves at cutoffs `[1%, 5%, 10%, 20%, 50%]` (lower is better).
- `spearman_total_engagement_per_user`: rank correlation of total engagement (likes + replies + reposts + quotes) per user (higher is better).

### 5. Content Fit

This section measures the *quality and nature of content* produced by the simulation, not just action counts. It has three subsections.

#### 5a. Engagement Source (`engagement_source`)

Classifies each engagement action (like, comment, repost) as either **organic** (actor follows the post author) or **recsys-driven** (actor does not follow the post author), using the follow table as a post-hoc heuristic.

- `total_engagements_classified`: total like/comment/repost actions analyzed.
- `organic_engagements`: count of actions where the actor follows the post author.
- `recsys_engagements`: count of actions where the actor does not follow the post author.
- `pct_organic_engagements`: fraction of engagements that are organic.
- `pct_recsys_engagements`: fraction of engagements driven by the recsys.
- `organic_vs_recsys_ratio`: organic / recsys (> 1 means organic-dominated feed).

#### 5b. Semantic Similarity (`semantic_similarity`)

Measures how topically close the sim posts are to GT posts using `paraphrase-MiniLM-L6-v2` embeddings (GPU-accelerated when available). Requires `sentence-transformers`.

- `sim_post_count` / `gt_post_count`: corpus sizes.
- `sim_to_gt_centroid_sim`: cosine similarity between the mean sim embedding and the mean GT embedding. Values near 1.0 indicate the sim corpus covers similar topics to GT.
- `sim_to_gt_semantic_max_match`: for each sim post, the cosine similarity to its nearest GT post, averaged over all sim posts. Higher is better; near 1.0 means every sim post has a close GT analogue.

#### 5c. Intra-User Repetition (`repetition`)

Measures how repetitive individual agents are in their posting. High similarity within a user's posts indicates the agent is producing near-duplicate content. Requires `sentence-transformers`.

Only users with ≥ 2 posts are included (`eligible_users`).

- `mean_intra_user_similarity`: average of per-user mean pairwise cosine similarity across all eligible users. Lower is more diverse.
- `high_repetition_user_rate`: fraction of users whose mean pairwise similarity exceeds **0.85**. High values indicate many agents are stuck in repetitive loops.
- `near_duplicate_rate`: fraction of same-user post pairs with cosine similarity > **0.92**. A direct measure of near-duplicate content production.

---

## Default Timestep Assumption

`compare_sim_to_test.py` defaults to:

- `--timestep_hours 24.0`

So one simulation step is treated as one day unless overridden.

---

## Removed Metrics

The following metrics were removed to reduce noise and shift focus to content quality:

| Removed | Reason |
|---------|--------|
| `jsd_posts_per_user` | Redundant with Spearman; less interpretable |
| `top20_overlap` | Marginal over top10 |
| `percentile_rank_jsd` (all) | Too verbose; top-share RMSE is sufficient |
| Engagement top-share for retweets/replies/quotes | Likes alone is a representative proxy |
| `jsd_total_engagement_per_user` | Redundant with Spearman |
| `spearman_engagement_per_post_per_user` | Secondary metric; noisy for sparse users |
| `topk_overlap_total/per_post_per_user` | Covered by activity top10_overlap |
| `gt/sim_top10/20pct_share_total_engagement` | Concentration diagnostics; not a fit metric |

---

## Script Usage

```bash
python simulation/compare_sim_to_test.py \
  --db data/simu_db/yaml_gpt/sweep_high_posts.db \
  --sim_config simulation/config.yaml
```

For non-daily simulations, pass `--timestep_hours` explicitly.

**Note:** Semantic similarity and repetition metrics require `sentence-transformers` to be installed. If not available, those subsections will contain `{"error": "sentence-transformers not installed"}` instead of values.

---

## Outputs

- JSON report (default): `simulation/metrics_report.json`
- Markdown report (default): `simulation/metrics_report.md`
- Archived per-run reports: `simulation/metrics_history/...`
