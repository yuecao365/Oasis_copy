# Pipeline

End-to-end pipeline from raw tweet data to OASIS simulation: clean tweets, score popularity, cluster by topic, generate user personas, and run simulations.

## Overview

1. **cleaning** – Clean the tweet dataset and build reply chains. Filter out users that show bot-like behavior.
2. **PCA** – Perform PCA on the cleaned dataset and use results to compute popularity scores for tweets.
3. **clustering** – Cluster tweets by topic. Use the top *k* tweets by popularity score for each cluster to generate a description of each topic.
4. **persona_generation** – Rank top *n* users per topic (by tweet count or popularity score) and create LLM-generated personas for ranked users from their tweets and interests, for use in simulation with OASIS.
5. **simulation** – Simulate the users we generated personas for in OASIS.

## Directory layout

| Step | Directory | Purpose |
|------|-----------|---------|
| 1 | `cleaning/` | Clean tweets, build reply chains, filter bot-like users |
| 2 | `pca/` | PCA on cleaned data; popularity scores |
| 3 | `clustering/` | Cluster by topic; topic descriptions from top *k* tweets |
| 4 | `persona_generation/` | Rank users per topic; LLM personas for simulation |
| 5 | `simulation/` | OASIS simulation with generated personas |

Run each step from the **repo root**. Outputs from one step feed into the next (e.g. cleaned data → PCA → clustering → persona_generation → simulation config).