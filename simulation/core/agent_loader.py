"""
agent_loader.py — builds a list of Agent instances from the persona CSV.

Replaces the OASIS agents_generator.py without any CAMEL dependency.
All logic for reading action probabilities, previous tweets, and activity
profiles is consolidated here so the simulation loop stays clean.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .agent import Agent

loader_log = logging.getLogger("social.loader")

# ── Action probability columns in the persona CSV ──────────────────────────
_ACTION_COLS: dict[str, str] = {
    "p_create_post": "create_post",
    "p_create_comment": "create_comment",
    "p_do_nothing": "do_nothing",
    "p_repost": "repost",
    "p_quote_post": "quote_post",
    "p_like_post": "like_post",
    "p_follow": "follow",
}

_DEFAULT_ACTION_PROBS: dict[str, float] = {
    "like_post": 0.60,
    "create_post": 0.10,
    "create_comment": 0.20,
    "repost": 0.10,
}


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in probs.values())
    if total <= 0:
        return {"do_nothing": 1.0}
    return {k: max(0.0, v) / total for k, v in probs.items()}


def _parse_previous_tweets(raw: Any) -> list[str]:
    if not raw or (isinstance(raw, float)):   # NaN
        return []
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if str(t).strip()]
    except Exception:
        pass
    return []


def _build_activity_profiles(
    train_csv_path: str | Path,
    usernames: set[str],
    granularity: str,
    timestep_hours: float,
    default_prob: float,
) -> dict[str, list[float]]:
    """Compute per-user activation probability vectors from training data.

    Args:
        train_csv_path: Path to the training tweet CSV.
        usernames:      Set of usernames to compute profiles for.
        granularity:    "weekly" (7 bins) or "hourly" (24 bins).
        timestep_hours: Hours represented by one simulation step.
        default_prob:   Fallback probability for users with sparse data.

    Returns:
        Dict mapping username → list of floats (one per time bin).
    """
    try:
        df = pd.read_csv(train_csv_path)
    except Exception as e:
        loader_log.warning("Failed to load training CSV %s: %s", train_csv_path, e)
        return {}

    username_col = next(
        (c for c in ["user_name", "username", "user"] if c in df.columns), None
    )
    if username_col is None or "timestamp" not in df.columns:
        loader_log.warning("Training CSV missing username/timestamp columns; skipping activity profiles.")
        return {}

    work = df[[username_col, "timestamp"]].copy()
    work[username_col] = work[username_col].astype(str).str.strip().str.lower()
    work = work[work[username_col].isin(usernames)]
    if work.empty:
        return {}

    work["dt"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["dt"])
    if work.empty:
        return {}

    if granularity == "hourly":
        work["bin"] = work["dt"].dt.hour
        num_bins = 24
        scalar = max(timestep_hours, 1e-9)
        all_range = pd.date_range(
            start=work["dt"].min().floor("h"),
            end=work["dt"].max().ceil("h"),
            freq="h",
        )
        occ_map = all_range.hour.value_counts().to_dict()
    else:  # weekly
        work["bin"] = work["dt"].dt.weekday
        num_bins = 7
        scalar = max(timestep_hours / 24.0, 1e-9)
        all_range = pd.date_range(
            start=work["dt"].min().floor("d"),
            end=work["dt"].max().ceil("d"),
            freq="d",
        )
        occ_map = all_range.weekday.value_counts().to_dict()

    grouped = work.groupby([username_col, "bin"]).size()
    profiles: dict[str, list[float]] = {}
    for uname in usernames:
        vec: list[float] = []
        for i in range(num_bins):
            count = int(grouped.get((uname, i), 0))
            occ = int(occ_map.get(i, 1))
            rate = count / max(occ, 1)
            vec.append(max(0.0, min(1.0, rate * scalar)))
        profiles[uname] = vec

    return profiles


def load_agents(
    personas_csv: str | Path,
    train_csv: str | Path | None = None,
    default_action_probs: dict[str, float] | None = None,
    activity_granularity: str = "weekly",
    timestep_hours: float = 24.0,
    default_activity_prob: float = 0.10,
) -> list[Agent]:
    """Build Agent instances from the persona CSV.

    Args:
        personas_csv:          Path to user_personas.csv.
        train_csv:             Path to train_tweets.csv used for activity
                               profiles.  Pass None to use default_activity_prob
                               for all agents.
        default_action_probs:  Global action probability distribution used
                               when a persona has no per-action columns.
        activity_granularity:  "weekly" or "hourly".
        timestep_hours:        Hours per simulation step (for scaling rates).
        default_activity_prob: Fallback activation probability when training
                               data is absent or sparse.

    Returns:
        List of Agent instances ordered by agent_id (0-based).
    """
    personas_csv = Path(personas_csv)
    df = pd.read_csv(personas_csv)

    # Normalise column names to lowercase stripped strings.
    df.columns = [c.strip().lower() for c in df.columns]

    global_probs = _normalize(default_action_probs or _DEFAULT_ACTION_PROBS)

    # Determine which action-probability columns are present.
    present_action_cols = {col: name for col, name in _ACTION_COLS.items()
                           if col in df.columns}

    usernames = set(
        df["username"].astype(str).str.strip().str.lower().tolist()
    )

    # Build activity profiles from training data.
    activity_profiles: dict[str, list[float]] = {}
    if train_csv is not None:
        activity_profiles = _build_activity_profiles(
            train_csv_path=train_csv,
            usernames=usernames,
            granularity=activity_granularity,
            timestep_hours=timestep_hours,
            default_prob=default_activity_prob,
        )

    agents: list[Agent] = []
    for agent_id, (_, row) in enumerate(df.iterrows()):
        username = str(row.get("username", "")).strip().lower()
        name = str(row.get("name", row.get("full_name", username))).strip()
        bio = str(row.get("description", row.get("bio", ""))).strip()
        if bio == "nan":
            bio = ""

        # Per-agent action probabilities (from CSV columns if present).
        if present_action_cols:
            raw: dict[str, float] = {}
            for col, action_name in present_action_cols.items():
                val = row.get(col)
                if pd.isna(val):
                    continue
                try:
                    raw[action_name] = float(val)
                except Exception:
                    pass
            action_probs = _normalize(raw) if raw else global_probs
        else:
            action_probs = global_probs

        # Previous tweets for content generation.
        previous_tweets = _parse_previous_tweets(row.get("previous_tweets"))

        # Activity profile for temporal activation.
        profile = activity_profiles.get(username)
        if profile is None:
            # Uniform fallback.
            bins = 7 if activity_granularity == "weekly" else 24
            profile = [default_activity_prob] * bins

        # Preserve all remaining CSV columns as extra_info.
        skip = {"username", "name", "full_name", "description", "bio",
                "previous_tweets"} | set(present_action_cols.keys())
        extra = {k: row[k] for k in df.columns if k not in skip
                 and not pd.isna(row.get(k, float("nan")))}

        agents.append(Agent(
            agent_id=agent_id,
            username=username,
            name=name,
            bio=bio,
            previous_tweets=previous_tweets,
            action_probabilities=action_probs,
            activity_profile=profile,
            extra_info=extra,
        ))

    loader_log.info("Loaded %d agents from %s", len(agents), personas_csv)
    return agents
