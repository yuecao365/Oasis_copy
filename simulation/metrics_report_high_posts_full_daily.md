# Simulation vs Test Metrics

- DB: `data/simu_db/yaml_gpt/sweep_high_posts.db`
- Test CSV: `persona_generation/outputs/test_tweets.csv`
- Personas CSV: `persona_generation/outputs/user_personas.csv`

## Coverage
- Persona users: 60
- GT active users: 51
- Sim active users: 48

## User Activity Fit
- Spearman (posts/user): 0.1501
- JSD (posts/user): 0.3006
- Top-10 overlap: 0.3000
- Top-20 overlap: 0.4500

## Temporal Fit
- Day-of-week JSD: 0.0291
- Cumulative daily posts RMSE: 70.9784
- Cumulative daily posts NRMSE: 0.2762

## Engagement Fit
- GT mean likes/retweets/replies/quotes per post: 1.592 / 0.412 / 0.386 / 0.022
- Sim mean likes/shares/replies per post: 0.182 / 0.027 / 0.081
- Sim mean reposts/quotes per post: 0.027 / 0.000
- GT total retweets/quotes: 112 / 6
- Sim total reposts/quotes: 4 / 0
