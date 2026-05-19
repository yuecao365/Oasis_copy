# Simulation vs Test Metrics

- DB: `data\simu_db\evelyn_facilitated_run.db`
- Test CSV: `persona_generation\outputs\test_tweets.csv`
- Personas CSV: `persona_generation\outputs\user_personas_evelyn.csv`

## Coverage
- Persona users: 143
- GT active users: 57
- Sim active users: 107

## User Activity Fit
- Spearman (posts/user): 0.1474
- JSD (posts/user): 0.5616
- Top-10 overlap: 0.2000
- Top-20 overlap: 0.3500

## Temporal Fit
- Day-of-week JSD: 0.0181
- Cumulative daily posts RMSE: 56.5939
- Cumulative daily posts NRMSE: 0.1501

## Engagement Fit
- GT total posts: 384
- Sim total posts: 467

### User-level
- Components: likes, replies, reposts, quotes
- Top-K values: 10, 20
- Spearman (total engagement/user): 0.2438
- JSD (total engagement/user): 0.5250
- Spearman (engagement-per-post per user): 0.1512
- JSD (engagement-per-post per user): 0.6227
- Top-10 overlap (total engagement/user): 0.3000
- Top-20 overlap (total engagement/user): 0.4000
- Top-10 overlap (engagement-per-post/user): 0.1000
- Top-20 overlap (engagement-per-post/user): 0.3000
- GT/Sim top-10% share of total engagement: 0.8443 / 0.4603
- GT/Sim top-20% share of total engagement: 0.9848 / 0.6712

### Post-level
- Top-share curve RMSE (likes): 0.1167
- Top-share curve RMSE (retweets vs reposts): 0.4969
- Top-share curve RMSE (replies): 0.1250
- Top-share curve RMSE (quotes): 0.8399
- Percentile-rank JSD (likes): 0.5759
- Percentile-rank JSD (retweets vs reposts): 0.9776
- Percentile-rank JSD (replies): 0.7866
- Percentile-rank JSD (quotes): 1.0000


## Official Post Engagement
- Official accounts: ['vechainofficial']
- GT official root posts: 23
- Sim official posts injected: 23
- GT total community engagement: 9588.0
- Sim total community engagement: 24.0
- GT mean engagement/post: 416.8700
- Sim mean engagement/post: 1.0400
- Spearman (engagement/post): -0.3366

## Community Health
- Persona users: 143
- Community posts: 467
- Total community engagement: 517
- Avg engagement per community post: 1.1071
- Reply rate (posts with ≥1 comment): 0.2570
- Unanswered post rate (0 engagement): 0.5996
- User participation rate: 1.0000
- Peripheral user activation rate: 1.0000
- Follow growth per timestep: 0.0000
