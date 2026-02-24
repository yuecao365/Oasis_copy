# Simulation vs Test Metrics

- DB: `data/simu_db/yaml_gpt/smoke_twhin.db`
- Test CSV: `persona_generation/outputs/test_tweets.csv`
- Personas CSV: `persona_generation/outputs/user_personas.csv`

## Coverage
- Persona users: 60
- GT active users: 51
- Sim active users: 0

## User Activity Fit
- Spearman (posts/user): 0.0000
- JSD (posts/user): 0.2264
- Top-10 overlap: 0.0000
- Top-20 overlap: 0.0000

## Temporal Fit
- Hour-of-day JSD: 0.0596
- Cumulative posts RMSE: 161.5226
- Cumulative posts NRMSE: 0.6334

## Engagement Fit
- GT mean likes/retweets/replies/quotes per post: 1.592 / 0.412 / 0.386 / 0.022
- Sim mean likes/shares/replies per post: 0.000 / 0.000 / 0.000
- Sim mean reposts/quotes per post: 0.000 / 0.000
- GT total retweets/quotes: 112 / 6
- Sim total reposts/quotes: 0 / 0

## Graph Fit
- Sim edges: 600
- GT reply edges: 1
- Sim density / GT reply density: 0.16949 / 0.00028
- Sim reciprocity / GT reply reciprocity: 0.98333 / 0.00000
- Edge precision / recall / F1: 0.00000 / 0.00000 / 0.00000
