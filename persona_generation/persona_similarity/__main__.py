#!/usr/bin/env python3
"""Example: run from repo root with `python -m persona_generation.persona_similarity`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# persona_generation/ on path (same pattern as user_persona_pipeline.py)
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from persona_similarity import (  # noqa: E402
    activity_spread_depth,
    compare_persona_repost_distributions,
    corpus_from_persona_rows,
    load_personas_csv,
    load_train_tweet_texts,
    per_user_tweet_counts_from_train_csv,
    semantic_distribution_report,
    summarize_repost_propensity,
    weekly_retweet_dynamics_from_train_csv,
)


def main() -> None:
    out_dir = _SCRIPT_DIR / "outputs"
    train_csv = out_dir / "train_tweets.csv"
    personas_csv = out_dir / "user_personas.csv"

    report: dict = {}

    counts = per_user_tweet_counts_from_train_csv(train_csv)
    report["train_activity_spread"] = activity_spread_depth(counts, top_k=10)

    personas = load_personas_csv(personas_csv)
    report["persona_p_repost_summary"] = summarize_repost_propensity(personas)

    # Same file twice → JS divergence ~0 (sanity check)
    report["persona_repost_self_compare"] = compare_persona_repost_distributions(personas, personas)

    report["weekly_retweet_train"] = weekly_retweet_dynamics_from_train_csv(train_csv)

    real_texts = load_train_tweet_texts(train_csv, max_rows=400)
    sim_texts = corpus_from_persona_rows(personas[: min(80, len(personas))])
    report["semantic_real_vs_persona_corpus"] = semantic_distribution_report(real_texts, sim_texts)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
