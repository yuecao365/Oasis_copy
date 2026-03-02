# Simulation vs Test Metrics

- DB: `data/simu_db/yaml_gpt/False_Business_0.db`
- Test CSV: `persona_generation/outputs/test_tweets.csv`
- Personas CSV: `persona_generation/outputs/user_personas.csv`

## Coverage
- Persona users: 60
- GT active users: 51
- Sim active users: 41

## User Activity Fit
- Spearman (posts/user): -0.1891
- JSD (posts/user): 0.4457
- Top-10 overlap: 0.1000
- Top-20 overlap: 0.2500

## Temporal Fit
- Hour-of-day JSD: 0.9133
- Cumulative posts RMSE: 104.5875
- Cumulative posts NRMSE: 0.4101

## Engagement Fit
- GT mean likes/retweets/replies/quotes per post: 1.592 / 0.412 / 0.386 / 0.022
- Sim mean likes/shares/replies per post: 0.161 / 0.011 / 0.011
- Sim mean reposts/quotes per post: 0.011 / 0.000
- GT total retweets/quotes: 112 / 6
- Sim total reposts/quotes: 1 / 0
