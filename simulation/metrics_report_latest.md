# Simulation vs Test Metrics

- DB: `data/simu_db/yaml_gpt/sweep_high_posts.db`
- Test CSV: `persona_generation/outputs/test_tweets.csv`
- Personas CSV: `persona_generation/outputs/user_personas.csv`

## Coverage
- Persona users: 60
- GT active users: 51
- Sim active users: 41

## User Activity Fit
- Spearman (posts/user): -0.1275
- JSD (posts/user): 0.4515
- Top-10 overlap: 0.2000
- Top-20 overlap: 0.3500

## Temporal Fit
- Day-of-week JSD: 0.0228
- Cumulative daily posts RMSE: 71.1938
- Cumulative daily posts NRMSE: 0.2770

## Engagement Fit
- GT mean likes/retweets/replies/quotes per post: 1.592 / 0.412 / 0.386 / 0.022
- Sim mean likes/shares/replies per post: 0.228 / 0.055 / 0.069
- Sim mean reposts/quotes per post: 0.055 / 0.000
- GT total retweets/quotes: 112 / 6
- Sim total reposts/quotes: 8 / 0
