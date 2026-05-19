# Simulation vs Test Metrics

- DB: `data\simu_db\evelyn_run.db`
- Test CSV: `persona_generation\outputs\test_tweets.csv`
- Personas CSV: `persona_generation\outputs\user_personas_evelyn.csv`

## Coverage
- Persona users: 143
- GT active users: 57
- Sim active users: 54

## User Activity Fit
- Spearman (posts/user): -0.1829
- JSD (posts/user): 0.8556
- Top-10 overlap: 0.1000
- Top-20 overlap: 0.0500

## Temporal Fit
- Day-of-week JSD: 0.0098
- Cumulative daily posts RMSE: 139.6250
- Cumulative daily posts NRMSE: 0.3704

## Engagement Fit
- GT total posts: 384
- Sim total posts: 347

### User-level
- Components: likes, replies, reposts, quotes
- Top-K values: 10, 20
- Spearman (total engagement/user): -0.1369
- JSD (total engagement/user): 0.8881
- Spearman (engagement-per-post per user): -0.1308
- JSD (engagement-per-post per user): 0.8621
- Top-10 overlap (total engagement/user): 0.0000
- Top-20 overlap (total engagement/user): 0.1000
- Top-10 overlap (engagement-per-post/user): 0.0000
- Top-20 overlap (engagement-per-post/user): 0.0000
- GT/Sim top-10% share of total engagement: 0.8443 / 0.6521
- GT/Sim top-20% share of total engagement: 0.9848 / 0.8881

### Post-level
- Top-share curve RMSE (likes): 0.2180
- Top-share curve RMSE (retweets vs reposts): 0.3532
- Top-share curve RMSE (replies): 0.0229
- Top-share curve RMSE (quotes): 0.8399
- Percentile-rank JSD (likes): 0.6084
- Percentile-rank JSD (retweets vs reposts): 0.9481
- Percentile-rank JSD (replies): 0.5348
- Percentile-rank JSD (quotes): 1.0000


## Official Post Engagement
- Error: official_user table missing or empty — run simulation with official_accounts configured

## Community Health
- Persona users: 143
- Community posts: 347
- Total community engagement: 2153
- Avg engagement per community post: 6.2046
- Reply rate (posts with ≥1 comment): 0.6571
- Unanswered post rate (0 engagement): 0.3372
- User participation rate: 1.0000
- Peripheral user activation rate: 1.0000
- Follow growth per timestep: 0.0000
