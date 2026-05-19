# Simulation vs Test Metrics

- DB: `data\simu_db\evelyn_baseline_run.db`
- Test CSV: `persona_generation\outputs\test_tweets.csv`
- Personas CSV: `persona_generation\outputs\user_personas_evelyn.csv`

## Coverage
- Persona users: 143
- GT active users: 57
- Sim active users: 111

## User Activity Fit
- Spearman (posts/user): 0.0529
- JSD (posts/user): 0.5971
- Top-10 overlap: 0.2000
- Top-20 overlap: 0.3000

## Temporal Fit
- Day-of-week JSD: 0.0155
- Cumulative daily posts RMSE: 35.3099
- Cumulative daily posts NRMSE: 0.0937

## Engagement Fit
- GT total posts: 384
- Sim total posts: 447

### User-level
- Components: likes, replies, reposts, quotes
- Top-K values: 10, 20
- Spearman (total engagement/user): 0.1830
- JSD (total engagement/user): 0.5429
- Spearman (engagement-per-post per user): 0.1223
- JSD (engagement-per-post per user): 0.6364
- Top-10 overlap (total engagement/user): 0.4000
- Top-20 overlap (total engagement/user): 0.3500
- Top-10 overlap (engagement-per-post/user): 0.3000
- Top-20 overlap (engagement-per-post/user): 0.2500
- GT/Sim top-10% share of total engagement: 0.8443 / 0.5589
- GT/Sim top-20% share of total engagement: 0.9848 / 0.7725

### Post-level
- Top-share curve RMSE (likes): 0.1645
- Top-share curve RMSE (retweets vs reposts): 0.4969
- Top-share curve RMSE (replies): 0.1976
- Top-share curve RMSE (quotes): 0.8399
- Percentile-rank JSD (likes): 0.5901
- Percentile-rank JSD (retweets vs reposts): 0.9768
- Percentile-rank JSD (replies): 0.8054
- Percentile-rank JSD (quotes): 1.0000


## Official Post Engagement
- Official accounts: ['vechainofficial']
- GT official root posts: 23
- Sim official posts injected: 23
- GT total community engagement: 9588.0
- Sim total community engagement: 24.0
- GT mean engagement/post: 416.8700
- Sim mean engagement/post: 1.0400
- Spearman (engagement/post): -0.4101

## Community Health
- Persona users: 143
- Community posts: 447
- Total community engagement: 501
- Avg engagement per community post: 1.1208
- Reply rate (posts with ≥1 comment): 0.2081
- Unanswered post rate (0 engagement): 0.6555
- User participation rate: 1.0000
- Peripheral user activation rate: 1.0000
- Follow growth per timestep: 0.0000
