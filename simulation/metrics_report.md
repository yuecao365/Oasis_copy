# Simulation vs Test Metrics

- DB: `data/simu_db/yaml_gpt/sweep_high_posts.db`
- Test CSV: `persona_generation/outputs/test_tweets.csv`
- Personas CSV: `persona_generation/outputs/user_personas.csv`

## Coverage
- Persona users: 156
- GT active users: 132
- Sim active users: 83

## User Activity Fit
- Spearman (posts/user): 0.1961
- JSD (posts/user): 0.3957
- Top-10 overlap: 0.4000
- Top-20 overlap: 0.4500

## Temporal Fit
- Day-of-week JSD: 0.0146
- Cumulative daily posts RMSE: 346.5153
- Cumulative daily posts NRMSE: 0.4321

## Engagement Fit
- GT total posts: 827
- Sim total posts: 261

### User-level
- Components: likes, replies, reposts, quotes
- Top-K values: 10, 20
- Spearman (total engagement/user): 0.0909
- JSD (total engagement/user): 0.7654
- Spearman (engagement-per-post per user): 0.0600
- JSD (engagement-per-post per user): 0.8399
- Top-10 overlap (total engagement/user): 0.1000
- Top-20 overlap (total engagement/user): 0.1000
- Top-10 overlap (engagement-per-post/user): 0.0000
- Top-20 overlap (engagement-per-post/user): 0.1000
- GT/Sim top-10% share of total engagement: 0.7541 / 1.0000
- GT/Sim top-20% share of total engagement: 0.8995 / 1.0000

### Post-level
- Top-share curve RMSE (likes): 0.3687
- Top-share curve RMSE (retweets vs reposts): 0.4493
- Top-share curve RMSE (replies): 0.2217
- Top-share curve RMSE (quotes): 0.7983
- Percentile-rank JSD (likes): 0.7495
- Percentile-rank JSD (retweets vs reposts): 0.7357
- Percentile-rank JSD (replies): 0.9529
- Percentile-rank JSD (quotes): 1.0000
