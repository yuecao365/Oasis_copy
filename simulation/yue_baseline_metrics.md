# Simulation vs Test Metrics

- DB: `data\simu_db\yue_run.db`
- Test CSV: `persona_generation\outputs\test_tweets.csv`
- Personas CSV: `persona_generation\outputs\user_personas_yue.csv`

## Coverage
- Persona users: 40
- GT active users: 23
- Sim active users: 0

## User Activity Fit
- Spearman (posts/user): 0.0000
- JSD (posts/user): 0.4902
- Top-10 overlap: 0.0000
- Top-20 overlap: 0.0000

## Temporal Fit
- Day-of-week JSD: 0.0224
- Cumulative daily posts RMSE: 134.0487
- Cumulative daily posts NRMSE: 0.5905

## Engagement Fit
- GT total posts: 228
- Sim total posts: 0

### User-level
- Components: likes, replies, reposts, quotes
- Top-K values: 10, 20
- Spearman (total engagement/user): 0.0000
- JSD (total engagement/user): 0.5333
- Spearman (engagement-per-post per user): 0.0000
- JSD (engagement-per-post per user): 0.5049
- Top-10 overlap (total engagement/user): 0.0000
- Top-20 overlap (total engagement/user): 0.0000
- Top-10 overlap (engagement-per-post/user): 0.0000
- Top-20 overlap (engagement-per-post/user): 0.0000
- GT/Sim top-10% share of total engagement: 0.7617 / 0.0000
- GT/Sim top-20% share of total engagement: 0.9015 / 0.0000

### Post-level
- Top-share curve RMSE (likes): 0.7159
- Top-share curve RMSE (retweets vs reposts): 0.8127
- Top-share curve RMSE (replies): 0.7553
- Top-share curve RMSE (quotes): 0.8429
- Percentile-rank JSD (likes): 0.0725
- Percentile-rank JSD (retweets vs reposts): 0.3821
- Percentile-rank JSD (replies): 0.3683
- Percentile-rank JSD (quotes): 0.6583


## Official Post Engagement
- Official accounts: ['vechainofficial']
- GT official root posts: 23
- Sim official posts injected: 23
- GT total community engagement: 9588.0
- Sim total community engagement: 0.0
- GT mean engagement/post: 416.8700
- Sim mean engagement/post: 0.0000
- Spearman (engagement/post): 0.0000

## Community Health
- Persona users: 40
- Community posts: 0
- Total community engagement: 0
- Avg engagement per community post: 0.0000
- Reply rate (posts with ≥1 comment): 0.0000
- Unanswered post rate (0 engagement): 0.0000
- User participation rate: 1.0000
- Peripheral user activation rate: 1.0000
- Follow growth per timestep: 0.0000
