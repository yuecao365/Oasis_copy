# Simulation vs Test Metrics

- DB: `data/simu_db/yaml_gpt/False_Business_0.db`
- Test CSV: `persona_generation/outputs/test_tweets.csv`
- Personas CSV: `persona_generation/outputs/user_personas.csv`

## Coverage
- Persona users: 60
- GT active users: 54
- Sim active users: 6

## User Activity Fit
- Spearman (posts/user): -0.1261
- JSD (posts/user): 0.8503
- Top-10 overlap: 0.1000
- Top-20 overlap: 0.0500

## Temporal Fit
- Hour-of-day JSD: 0.8817
- Cumulative posts RMSE: 145.8725
- Cumulative posts NRMSE: 0.5882

## Engagement Fit
- GT mean likes/retweets/replies/quotes per post: 4.839 / 0.939 / 0.755 / 0.019
- Sim mean likes/shares/replies per post: 0.000 / 1.091 / 1.727
- Sim mean reposts/quotes per post: 0.909 / 0.182
- GT total retweets/quotes: 245 / 5
- Sim total reposts/quotes: 10 / 2

## Graph Fit
- Sim edges: 601
- GT reply edges: 2
- Sim density / GT reply density: 0.16977 / 0.00056
- Sim reciprocity / GT reply reciprocity: 0.98170 / 0.00000
- Edge precision / recall / F1: 0.00166 / 0.50000 / 0.00332
