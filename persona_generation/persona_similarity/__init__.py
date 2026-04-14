"""
Persona similarity and population-level comparison metrics.

- Semantic distribution: embedding overlap between real and simulated (or persona-derived) text.
- Activity spread depth: concentration of volume among top users (HHI, entropy, Gini).
- Repost dynamics: p_repost distribution divergence; weekly retweet curves from CSV.
"""

from .behavior_similarity import cosine_similarity_actions, pairwise_action_cosine_matrix
from .activity_spread import (
    activity_spread_depth,
    compare_activity_spread,
    effective_users_from_counts,
    top_k_volume_share,
)
from .io import (
    action_probs_from_row,
    corpus_from_persona_rows,
    load_personas_csv,
    load_personas_json,
    load_train_tweet_texts,
    per_user_tweet_counts_from_train_csv,
    persona_text_bundle,
)
from .repost_dynamics import (
    compare_persona_repost_distributions,
    compare_weekly_retweet_series,
    summarize_repost_propensity,
    weekly_retweet_dynamics_from_train_csv,
)
from .semantic_distribution import semantic_distribution_report

__all__ = [
    "action_probs_from_row",
    "cosine_similarity_actions",
    "activity_spread_depth",
    "compare_activity_spread",
    "compare_persona_repost_distributions",
    "compare_weekly_retweet_series",
    "corpus_from_persona_rows",
    "effective_users_from_counts",
    "load_personas_csv",
    "load_personas_json",
    "load_train_tweet_texts",
    "pairwise_action_cosine_matrix",
    "per_user_tweet_counts_from_train_csv",
    "persona_text_bundle",
    "semantic_distribution_report",
    "summarize_repost_propensity",
    "top_k_volume_share",
    "weekly_retweet_dynamics_from_train_csv",
]
