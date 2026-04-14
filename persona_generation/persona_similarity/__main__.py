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

from persona_similarity.visualize_similarity_report import build_default_report  # noqa: E402


def main() -> None:
    out_dir = _SCRIPT_DIR / "outputs"
    train_csv = out_dir / "train_tweets.csv"
    personas_csv = out_dir / "user_personas.csv"

    report = build_default_report(
        train_csv=train_csv,
        personas_csv=personas_csv,
        max_train_tweets_semantic=400,
        max_personas_semantic=80,
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
