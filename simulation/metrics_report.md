# Simulation vs Test Metrics

- DB: `data\simu_db\yaml_gpt\sweep_high_posts.db`
- Test CSV: `persona_generation\outputs\test_tweets.csv`
- Personas CSV: `persona_generation\outputs\user_personas.csv`

## Coverage
- Persona users: 67
- GT active users: 56
- Sim active users: 53

## User Activity Fit
- Spearman (posts/user): 0.5207
- Top-10 overlap: 0.7000

## Temporal Fit
- Day-of-week JSD: 0.0221
- Cumulative daily posts RMSE: 45.1572
- Cumulative daily posts NRMSE: 0.1792

## Engagement Fit
- GT total posts: 267
- Sim total posts: 197
- Likes top-share curve RMSE: 0.1036
- Spearman (total engagement/user): 0.2923

## Content Fit

### Engagement Source (Recsys vs Organic)
- Total engagements classified: 170
- Organic engagements: 85 (0.5000 of total)
- Recsys-driven engagements: 85 (0.5000 of total)
- Organic / Recsys ratio: 1.0000

### Semantic Similarity (Sim vs GT)
- Sim posts embedded: 197
- GT posts embedded: 267
- Centroid similarity: 0.9615
- Mean max-match similarity: 0.6788

### Intra-User Repetition
- Eligible users (≥2 posts): 38
- Mean intra-user similarity: 0.6456
- High-repetition user rate (>0.85): 0.2632
- Near-duplicate pair rate (>0.92): 0.1455
