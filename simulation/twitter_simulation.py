"""
LLM-driven Twitter simulation. Run from repo root:

  python simulation/twitter_simulation.py --config_path simulation/config.yaml

Requires DASHSCOPE_API_KEY (or the env var named in inference.api_key_env) to be set.
Paths in the config are relative to the repo root.

Pipeline:
  persona_generation/outputs/user_personas.csv  (agent profiles)
  persona_generation/outputs/split_info.json     (test-period timing)
  persona_generation/outputs/train_tweets.csv    (activity profiles)
     → LLM-driven simulation
     → data/simu_db/.../*.db                     (SQLite output)
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import math
import os
import random
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from colorama import Back
from yaml import safe_load

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SIMULATION_DIR = os.path.dirname(os.path.abspath(__file__))

# Add simulation/ to path so we can import llm_client
if _SIMULATION_DIR not in sys.path:
    sys.path.insert(0, _SIMULATION_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import llm_client as _llm  # noqa: E402  (after sys.path setup)

_DEFAULT_CSV = os.path.join(
    _REPO_ROOT, "persona_generation", "outputs", "user_personas.csv"
)
_DEFAULT_DB = ":memory:"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
social_log = logging.getLogger(name="social")
social_log.propagate = False
social_log.setLevel("DEBUG")
_fh = logging.FileHandler("social.log")
_fh.setLevel("DEBUG")
_fh.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setLevel("INFO")
_sh.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(_sh)

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LLM-driven Twitter simulation.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to YAML config (default: simulation/config.yaml).",
    required=False,
    default="",
)


# ---------------------------------------------------------------------------
# Agent dataclass
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    agent_id: int
    username: str
    name: str
    bio: str
    user_char: str
    category_description: str
    previous_tweets: list[str] = field(default_factory=list)
    following_ids: list[int] = field(default_factory=list)
    action_probs: dict[str, float] = field(default_factory=dict)
    activity_hourly: list[float] | None = None
    activity_weekly: list[float] | None = None


# ---------------------------------------------------------------------------
# SimDB — owns the SQLite connection for simulation output
# ---------------------------------------------------------------------------
class SimDB:
    """
    Minimal SQLite backend that produces the schema expected by
    simulation/compare_sim_to_test.py.

    Tables: user, follow, post, comment, like
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS user (
        user_id         INTEGER PRIMARY KEY,
        agent_id        INTEGER,
        user_name       TEXT,
        name            TEXT,
        bio             TEXT,
        created_at      REAL    DEFAULT 0,
        num_followings  INTEGER DEFAULT 0,
        num_followers   INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS follow (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        follower_id INTEGER,
        followee_id INTEGER,
        created_at  REAL    DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS post (
        post_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id          INTEGER,
        content          TEXT,
        created_at       REAL    DEFAULT 0,
        num_likes        INTEGER DEFAULT 0,
        num_shares       INTEGER DEFAULT 0,
        num_dislikes     INTEGER DEFAULT 0,
        original_post_id INTEGER DEFAULT NULL,
        quote_content    TEXT    DEFAULT NULL
    );
    CREATE TABLE IF NOT EXISTS comment (
        comment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id     INTEGER,
        user_id     INTEGER,
        content     TEXT,
        created_at  REAL    DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS "like" (
        like_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id     INTEGER,
        user_id     INTEGER,
        created_at  REAL    DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS observation (
        obs_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestep        REAL,
        agent_id        INTEGER,
        username        TEXT,
        feed_total      INTEGER DEFAULT 0,
        feed_following  INTEGER DEFAULT 0,
        feed_rec        INTEGER DEFAULT 0,
        action_taken    TEXT,
        action_success  INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS official_user (
        user_id  INTEGER PRIMARY KEY,
        username TEXT
    );
    """

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        if db_path == ":memory:":
            self._con = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row

    def initialize_schema(self) -> None:
        cur = self._con.cursor()
        cur.executescript(self._SCHEMA)
        self._con.commit()

    # ------------------------------------------------------------------
    # Bulk initialization (called once before the simulation loop)
    # ------------------------------------------------------------------
    def insert_users(self, agents: list[Agent], start_time: float = 0.0) -> None:
        rows = [
            (a.agent_id, a.agent_id, a.username, a.name, a.bio, start_time, 0, 0)
            for a in agents
        ]
        self._con.executemany(
            "INSERT OR IGNORE INTO user "
            "(user_id, agent_id, user_name, name, bio, created_at, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._con.commit()

    def insert_initial_follows(
        self,
        follows: list[tuple[int, int]],
        start_time: float = 0.0,
    ) -> None:
        if not follows:
            return
        self._con.executemany(
            "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)",
            [(f, t, start_time) for f, t in follows],
        )
        # Update counters
        from collections import Counter
        following_count = Counter(f for f, _ in follows)
        follower_count = Counter(t for _, t in follows)
        for uid, cnt in following_count.items():
            self._con.execute(
                "UPDATE user SET num_followings = num_followings + ? WHERE user_id = ?",
                (cnt, uid),
            )
        for uid, cnt in follower_count.items():
            self._con.execute(
                "UPDATE user SET num_followers = num_followers + ? WHERE user_id = ?",
                (cnt, uid),
            )
        self._con.commit()

    def insert_official_user(self, user_id: int, username: str) -> None:
        self._con.execute(
            "INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, created_at) "
            "VALUES (?, -1, ?, ?, '', 0.0)",
            (user_id, username, username),
        )
        self._con.execute(
            "INSERT OR IGNORE INTO official_user (user_id, username) VALUES (?, ?)",
            (user_id, username.lower()),
        )
        self._con.commit()

    def insert_initial_posts(
        self,
        posts: list[tuple[int, str]],
        start_time: float = 0.0,
    ) -> None:
        """Seed previous_tweets into the post table with created_at=0 (filtered out by compare script)."""
        if not posts:
            return
        self._con.executemany(
            "INSERT INTO post (user_id, content, created_at, num_likes, num_shares) "
            "VALUES (?, ?, ?, 0, 0)",
            [(uid, content, start_time) for uid, content in posts],
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------
    def get_feed(
        self,
        agent_id: int,
        limit: int = 8,
        pinned_post_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return up to `limit` feed posts for this agent:
          - Pinned posts (official account posts for this timestep) always first
          - Posts from users they follow (ordered by num_likes desc)
          - Plus random recent posts for discovery
        Only returns original posts (original_post_id IS NULL) created during simulation (created_at > 0).
        """
        cur = self._con.cursor()

        # Pinned posts (VeChain official posts injected this timestep)
        pinned: list[dict[str, Any]] = []
        if pinned_post_ids:
            placeholders = ",".join("?" for _ in pinned_post_ids)
            cur.execute(
                f"SELECT post_id, user_id, content, created_at, num_likes "
                f"FROM post WHERE post_id IN ({placeholders})",
                pinned_post_ids,
            )
            pinned = [{**dict(r), "source": "official"} for r in cur.fetchall()]

        seen: set[int] = {p["post_id"] for p in pinned if p["post_id"] is not None}
        merged: list[dict[str, Any]] = list(pinned)

        if len(merged) < limit:
            # Posts from followed users
            cur.execute(
                """
                SELECT post_id, user_id, content, created_at, num_likes
                FROM post
                WHERE user_id IN (SELECT followee_id FROM follow WHERE follower_id = ?)
                  AND original_post_id IS NULL
                  AND CAST(created_at AS REAL) > 0
                ORDER BY num_likes DESC, created_at DESC
                LIMIT ?
                """,
                (agent_id, limit),
            )
            followed_posts = [{**dict(r), "source": "following"} for r in cur.fetchall()]

            # Random posts for discovery (recommendation system)
            cur.execute(
                """
                SELECT post_id, user_id, content, created_at, num_likes
                FROM post
                WHERE original_post_id IS NULL
                  AND CAST(created_at AS REAL) > 0
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (limit,),
            )
            random_posts = [{**dict(r), "source": "recommendation"} for r in cur.fetchall()]

            # Merge: following posts first, then fill with recommendation posts.
            # Deduplicate by post_id; exclude agent's own posts.
            for p in followed_posts + random_posts:
                pid = p["post_id"]
                if pid not in seen and pid is not None and p["user_id"] != agent_id:
                    seen.add(pid)
                    merged.append(p)
                if len(merged) >= limit:
                    break

        return merged

    # ------------------------------------------------------------------
    # Actions (called during the simulation loop)
    # ------------------------------------------------------------------
    def create_post(self, user_id: int, content: str, timestep: int) -> int:
        """Insert a new post and return its post_id."""
        cur = self._con.cursor()
        cur.execute(
            "INSERT INTO post (user_id, content, created_at, num_likes, num_shares) "
            "VALUES (?, ?, ?, 0, 0)",
            (user_id, content, float(timestep)),
        )
        self._con.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def create_comment(
        self, user_id: int, post_id: int, content: str, timestep: int
    ) -> bool:
        """Add a comment to an existing post. Returns False if post not found."""
        cur = self._con.cursor()
        cur.execute("SELECT 1 FROM post WHERE post_id = ?", (post_id,))
        if not cur.fetchone():
            return False
        cur.execute(
            "INSERT INTO comment (post_id, user_id, content, created_at) VALUES (?, ?, ?, ?)",
            (post_id, user_id, content, float(timestep)),
        )
        self._con.commit()
        return True

    def repost(self, user_id: int, post_id: int, timestep: int) -> bool:
        """Repost an existing post. Guards against duplicates. Returns False on failure."""
        cur = self._con.cursor()
        cur.execute("SELECT content FROM post WHERE post_id = ? AND original_post_id IS NULL", (post_id,))
        row = cur.fetchone()
        if not row:
            return False
        # Guard: has this user already reposted this post?
        cur.execute(
            "SELECT 1 FROM post WHERE user_id = ? AND original_post_id = ? AND quote_content IS NULL",
            (user_id, post_id),
        )
        if cur.fetchone():
            return False
        original_content = row["content"]
        cur.execute(
            "INSERT INTO post (user_id, content, created_at, num_likes, num_shares, original_post_id) "
            "VALUES (?, ?, ?, 0, 0, ?)",
            (user_id, original_content, float(timestep), post_id),
        )
        cur.execute("UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?", (post_id,))
        self._con.commit()
        return True

    def quote_post(
        self, user_id: int, post_id: int, quote_content: str, timestep: int
    ) -> bool:
        """Quote-tweet an existing post with commentary. Returns False on failure."""
        cur = self._con.cursor()
        cur.execute("SELECT content FROM post WHERE post_id = ? AND original_post_id IS NULL", (post_id,))
        row = cur.fetchone()
        if not row:
            return False
        original_content = row["content"]
        cur.execute(
            "INSERT INTO post (user_id, content, created_at, num_likes, num_shares, original_post_id, quote_content) "
            "VALUES (?, ?, ?, 0, 0, ?, ?)",
            (user_id, original_content, float(timestep), post_id, quote_content),
        )
        cur.execute("UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?", (post_id,))
        self._con.commit()
        return True

    def like_post(self, user_id: int, post_id: int, timestep: int = 0) -> bool:
        """Like a post. Guards against duplicate likes. Returns False if already liked or not found."""
        cur = self._con.cursor()
        cur.execute("SELECT 1 FROM post WHERE post_id = ?", (post_id,))
        if not cur.fetchone():
            return False
        cur.execute(
            'SELECT 1 FROM "like" WHERE post_id = ? AND user_id = ?',
            (post_id, user_id),
        )
        if cur.fetchone():
            return False  # Already liked
        cur.execute(
            'INSERT INTO "like" (post_id, user_id, created_at) VALUES (?, ?, ?)',
            (post_id, user_id, float(timestep)),
        )
        cur.execute("UPDATE post SET num_likes = num_likes + 1 WHERE post_id = ?", (post_id,))
        self._con.commit()
        return True

    def follow(self, follower_id: int, followee_id: int, timestep: int) -> bool:
        """Follow a user. Guards against duplicates. Returns False if already following or self-follow."""
        if follower_id == followee_id:
            return False
        cur = self._con.cursor()
        cur.execute(
            "SELECT 1 FROM follow WHERE follower_id = ? AND followee_id = ?",
            (follower_id, followee_id),
        )
        if cur.fetchone():
            return False
        cur.execute(
            "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)",
            (follower_id, followee_id, float(timestep)),
        )
        cur.execute(
            "UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?",
            (follower_id,),
        )
        cur.execute(
            "UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?",
            (followee_id,),
        )
        self._con.commit()
        return True

    def log_observation(
        self,
        timestep: int,
        agent_id: int,
        username: str,
        feed_total: int,
        feed_following: int,
        feed_rec: int,
        action_taken: str,
        action_success: bool,
    ) -> None:
        self._con.execute(
            "INSERT INTO observation "
            "(timestep, agent_id, username, feed_total, feed_following, feed_rec, action_taken, action_success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (float(timestep), agent_id, username, feed_total, feed_following, feed_rec, action_taken, int(action_success)),
        )
        self._con.commit()

    def close(self) -> None:
        self._con.close()


# ---------------------------------------------------------------------------
# Utility functions (preserved from original twitter_simulation.py)
# ---------------------------------------------------------------------------

def _resolve(path: str, base: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def _normalize_action_probabilities(
    action_probabilities: dict[str, float] | None,
    available_actions: list[str] | None,
) -> dict[str, float]:
    if action_probabilities:
        probs = {
            str(k): max(0.0, float(v))
            for k, v in action_probabilities.items()
        }
    else:
        actions = list(available_actions or [])
        if not actions:
            actions = ["create_post", "create_comment", "repost", "like_post"]
        probs = {str(a): 1.0 for a in actions}

    total = sum(probs.values())
    if total <= 0:
        return {"do_nothing": 1.0}
    return {k: v / total for k, v in probs.items()}


def _extract_action_probabilities_by_username(
    csv_path: str,
    available_actions: list[str] | None,
) -> dict[str, dict[str, float]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    if "username" not in df.columns:
        return {}

    action_columns = {
        "p_create_post": "create_post",
        "p_create_comment": "create_comment",
        "p_do_nothing": "do_nothing",
        "p_repost": "repost",
        "p_quote_post": "quote_post",
        "p_like_post": "like_post",
        "p_follow": "follow",
    }
    present_cols = [c for c in action_columns if c in df.columns]
    if not present_cols:
        return {}

    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        username = str(row.get("username", "")).strip().lower()
        if not username:
            continue

        raw_probs: dict[str, float] = {}
        for col in present_cols:
            raw_val = row.get(col)
            if pd.isna(raw_val):
                continue
            try:
                raw_probs[action_columns[col]] = float(raw_val)
            except Exception:
                continue

        if not raw_probs:
            continue

        out[username] = _normalize_action_probabilities(
            action_probabilities=raw_probs,
            available_actions=available_actions,
        )

    return out


def _extract_previous_tweets_by_username(csv_path: str) -> dict[str, list[str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    if "username" not in df.columns or "previous_tweets" not in df.columns:
        return {}

    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        username = str(row.get("username", "")).strip().lower()
        if not username:
            continue
        raw = row.get("previous_tweets", "[]")
        tweets: list[str] = []
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                tweets = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        out[username] = tweets
    return out


def _build_activity_profiles(
    train_csv_path: str,
    usernames: set[str],
    granularity: str,
    timestep_hours: float,
    default_prob: float,
) -> dict[str, list[float]]:
    try:
        df = pd.read_csv(train_csv_path)
    except Exception as e:
        social_log.warning(f"Failed to load activity source csv {train_csv_path}: {e}")
        return {}

    username_col = None
    for c in ["user_name", "username", "user"]:
        if c in df.columns:
            username_col = c
            break
    if username_col is None or "timestamp" not in df.columns:
        social_log.warning(
            "Activity profiling requires username and timestamp columns; skipping."
        )
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

    start_dt = work["dt"].min()
    end_dt = work["dt"].max()

    profiles: dict[str, list[float]] = {}
    if granularity == "hourly":
        all_hours = pd.date_range(start=start_dt.floor("h"), end=end_dt.ceil("h"), freq="h")
        hour_occ = all_hours.hour.value_counts().to_dict()
        work["bin"] = work["dt"].dt.hour
        num_bins = 24
        scalar = max(timestep_hours, 1e-9)
    else:
        all_days = pd.date_range(start=start_dt.floor("d"), end=end_dt.ceil("d"), freq="d")
        weekday_occ = all_days.weekday.value_counts().to_dict()
        work["bin"] = work["dt"].dt.weekday
        hour_occ = weekday_occ
        num_bins = 7
        scalar = max(timestep_hours / 24.0, 1e-9)

    grouped = work.groupby([username_col, "bin"]).size()
    for uname in usernames:
        vec = [float(default_prob)] * num_bins
        for i in range(num_bins):
            count = int(grouped.get((uname, i), 0))
            occ = int(hour_occ.get(i, 1))
            rate = count / max(occ, 1)
            vec[i] = max(0.0, min(1.0, rate * scalar))
        profiles[uname] = vec

    return profiles


def _estimate_target_posts_per_step(
    train_csv_path: str,
    usernames: set[str],
    timestep_hours: float,
) -> int | None:
    try:
        df = pd.read_csv(train_csv_path)
    except Exception as e:
        social_log.warning(f"Failed to load post-rate source csv {train_csv_path}: {e}")
        return None

    username_col = None
    for c in ["user_name", "username", "user"]:
        if c in df.columns:
            username_col = c
            break
    if username_col is None or "timestamp" not in df.columns:
        return None

    work = df[[username_col, "timestamp"]].copy()
    work[username_col] = work[username_col].astype(str).str.strip().str.lower()
    work = work[work[username_col].isin(usernames)]
    if work.empty:
        return None

    work["dt"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["dt"])
    if work.empty:
        return None

    total_posts = len(work)
    duration_hours = max(
        (work["dt"].max() - work["dt"].min()).total_seconds() / 3600.0,
        1.0,
    )
    posts_per_hour = total_posts / duration_hours
    estimate = int(round(posts_per_hour * max(timestep_hours, 1e-9)))
    return max(1, estimate)


def _choose_threshold(
    agent: Agent,
    simulation_dt: datetime,
    activity_granularity: str,
) -> float:
    if activity_granularity == "hourly":
        vec = agent.activity_hourly or [0.1] * 24
        idx = int(simulation_dt.hour) % 24
    else:
        vec = agent.activity_weekly
        if not vec:
            return 0.1
        idx = int(simulation_dt.weekday()) % 7
    try:
        return float(vec[idx])
    except Exception:
        return 0.1


def _sample_num_actions_per_agent(
    max_actions_per_active_agent: int,
    extra_action_probability: float,
) -> int:
    max_actions = max(1, int(max_actions_per_active_agent))
    p = max(0.0, min(1.0, float(extra_action_probability)))
    actions = 1
    while actions < max_actions and random.random() < p:
        actions += 1
    return actions


# ---------------------------------------------------------------------------
# Official account post loading
# ---------------------------------------------------------------------------

def _load_official_posts(
    csv_path: str,
    official_usernames: set[str],
    test_start_dt: datetime,
    timestep_hours: float,
) -> dict[int, list[tuple[str, str]]]:
    """Read real official account root posts from the test CSV and bin them by timestep.

    Returns {timestep: [(username, text), ...]} where timestep >= 1.
    Posts before the simulation start are dropped; posts after the last timestep
    are clamped to timestep 1 (so they always appear).
    """
    out: dict[int, list[tuple[str, str]]] = {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        social_log.warning("_load_official_posts: failed to read %s — %s", csv_path, e)
        return out

    start = test_start_dt.replace(tzinfo=None)

    for _, row in df.iterrows():
        uname = str(row.get("user_name", "")).strip().lower()
        if uname not in official_usernames:
            continue
        if str(row.get("is_root", "0")) != "1":
            continue
        text = str(
            row.get("text") or row.get("full_text") or row.get("content") or ""
        ).strip()
        if not text:
            continue
        ts_raw = str(row.get("timestamp", ""))
        try:
            ts_dt = datetime.fromisoformat(ts_raw).replace(tzinfo=None)
        except Exception:
            continue
        hours_offset = max(0.0, (ts_dt - start).total_seconds() / 3600.0)
        timestep = max(1, int(math.ceil(hours_offset / max(timestep_hours, 1e-9))))
        out.setdefault(timestep, []).append((uname, text))

    total = sum(len(v) for v in out.values())
    social_log.info(
        "_load_official_posts: %d posts from %s across %d timesteps",
        total, sorted(official_usernames), len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agents(
    csv_path: str,
    available_actions: list[str] | None,
    global_action_probs: dict[str, float],
) -> list[Agent]:
    """Load agents from the persona CSV. Returns a list ordered by user_id."""
    df = pd.read_csv(csv_path)
    action_probs_by_username = _extract_action_probabilities_by_username(
        csv_path=csv_path,
        available_actions=available_actions,
    )
    prev_tweets_by_username = _extract_previous_tweets_by_username(csv_path)

    agents: list[Agent] = []
    for idx, row in df.iterrows():
        agent_id = int(row.get("user_id", idx))
        username = str(row.get("username", f"user_{agent_id}")).strip()
        uname_lower = username.lower()

        # Parse following list
        following_ids: list[int] = []
        raw_following = row.get("following_agentid_list", "[]")
        try:
            parsed = ast.literal_eval(raw_following) if isinstance(raw_following, str) else raw_following
            if isinstance(parsed, list):
                following_ids = [int(x) for x in parsed if str(x).strip()]
        except Exception:
            pass

        # Action probabilities (per-agent from CSV, fall back to global)
        action_probs = action_probs_by_username.get(uname_lower) or global_action_probs

        # Previous tweets
        previous_tweets = prev_tweets_by_username.get(uname_lower, [])

        agent = Agent(
            agent_id=agent_id,
            username=username,
            name=str(row.get("name", username)).strip(),
            bio=str(row.get("description", "")).strip(),
            user_char=str(row.get("user_char", "")).strip(),
            category_description=str(row.get("category_description", "")).strip(),
            previous_tweets=previous_tweets,
            following_ids=following_ids,
            action_probs=action_probs,
        )
        agents.append(agent)

    social_log.info("Loaded %d agents from %s", len(agents), csv_path)
    return agents


# ---------------------------------------------------------------------------
# Action application (DB writes after LLM decisions)
# ---------------------------------------------------------------------------

def _apply_decision(
    db: SimDB,
    agent: Agent,
    decision: dict[str, Any],
    timestep: int,
    remaining_post_budget: int | None,
    blocked_actions: set[str],
) -> tuple[str, bool, int | None]:
    """
    Apply one LLM decision to the database.

    Returns (action_taken, success, updated_remaining_post_budget).
    """
    action = decision.get("action", "do_nothing")
    content = decision.get("content", "")
    target_post_id = decision.get("target_post_id")
    target_user_id = decision.get("target_user_id")

    if action in blocked_actions:
        action = "do_nothing"

    success = False
    if action == "create_post":
        if remaining_post_budget is not None and remaining_post_budget <= 0:
            action = "do_nothing"
        else:
            if content:
                db.create_post(agent.agent_id, content, timestep)
                success = True
                if remaining_post_budget is not None:
                    remaining_post_budget -= 1

    elif action == "create_comment":
        if target_post_id is not None and content:
            success = db.create_comment(agent.agent_id, int(target_post_id), content, timestep)

    elif action == "repost":
        if target_post_id is not None:
            success = db.repost(agent.agent_id, int(target_post_id), timestep)

    elif action == "like_post":
        if target_post_id is not None:
            success = db.like_post(agent.agent_id, int(target_post_id), timestep)

    elif action == "quote_post":
        if target_post_id is not None and content:
            success = db.quote_post(agent.agent_id, int(target_post_id), content, timestep)

    elif action == "follow":
        if target_user_id is not None:
            success = db.follow(agent.agent_id, int(target_user_id), timestep)

    # do_nothing: no DB write needed
    return action, success, remaining_post_budget


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

async def running(
    db_path: str | None = _DEFAULT_DB,
    csv_path: str | None = _DEFAULT_CSV,
    num_timesteps: int = 3,
    timestep_hours: float = 24.0,
    agent_activation_multiplier: float = 1.0,
    min_activation_threshold: float = 0.0,
    auto_num_timesteps_from_split: bool = False,
    split_info_path: str = "persona_generation/outputs/split_info.json",
    train_activity_csv_path: str = "persona_generation/outputs/train_tweets.csv",
    activity_granularity: str = "weekly",
    default_activity_probability: float = 0.1,
    target_posts_per_step: int | None = None,
    auto_target_posts_per_step_from_train: bool = True,
    max_actions_per_active_agent: int = 1,
    extra_action_probability: float = 0.0,
    action_probabilities: dict[str, float] | None = None,
    available_actions: list[str] | None = None,
    inference_configs: dict[str, Any] | None = None,
    # Official account settings
    official_accounts: list[str] | None = None,
    official_posts_csv_path: str = "persona_generation/outputs/test_tweets.csv",
    # Legacy OASIS params accepted but ignored:
    recsys_type: str = "random",
    clock_factor: int = 60,
    clock_step_units: int = 1,
    action_selection_mode: str = "llm",
    **_kwargs: Any,
) -> None:
    db_path = _DEFAULT_DB if db_path is None else db_path
    csv_path = _DEFAULT_CSV if csv_path is None else csv_path

    if db_path != ":memory:":
        db_path = _resolve(db_path, _REPO_ROOT)
    csv_path = _resolve(csv_path, _REPO_ROOT)

    # Clean up old DB
    if db_path != ":memory:" and os.path.exists(db_path):
        os.remove(db_path)

    # --- LLM client setup ---
    cfg = inference_configs or {}
    model_name: str = str(cfg.get("model_name", "qwen3.5-plus"))
    base_url: str = str(cfg.get("base_url", "https://coding.dashscope.aliyuncs.com/v1"))
    api_key_env: str = str(cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    llm_timeout: float = float(cfg.get("llm_timeout", 60.0))
    llm_max_retries: int = int(cfg.get("llm_max_retries", 2))
    llm_concurrency: int = int(cfg.get("llm_concurrency", 20))
    max_feed_posts: int = int(cfg.get("max_feed_posts", 8))

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        social_log.warning(
            "Environment variable %s is not set. LLM calls will likely fail.", api_key_env
        )

    client = _llm.get_client(base_url=base_url, api_key=api_key, timeout=llm_timeout, max_retries=llm_max_retries)

    # --- Global action probabilities ---
    normalized_action_probs = _normalize_action_probabilities(
        action_probabilities=action_probabilities,
        available_actions=available_actions,
    )

    # --- Load agents ---
    agents = load_agents(
        csv_path=csv_path,
        available_actions=available_actions,
        global_action_probs=normalized_action_probs,
    )
    if not agents:
        social_log.error("No agents loaded from %s — aborting.", csv_path)
        return

    # --- Initialize DB ---
    db = SimDB(db_path)
    db.initialize_schema()
    db.insert_users(agents, start_time=0.0)

    # Insert initial follows
    initial_follows: list[tuple[int, int]] = [
        (a.agent_id, fid)
        for a in agents
        for fid in a.following_ids
        if 0 <= fid < len(agents)
    ]
    db.insert_initial_follows(initial_follows, start_time=0.0)

    # Seed previous tweets as initial posts (created_at=0, filtered by compare script)
    initial_posts: list[tuple[int, str]] = [
        (a.agent_id, tweet)
        for a in agents
        for tweet in a.previous_tweets[:5]
        if tweet.strip()
    ]
    db.insert_initial_posts(initial_posts, start_time=0.0)

    # --- Official accounts setup ---
    # Official accounts (e.g. vechainofficial) are NOT simulated agents; their real
    # posts from the test split are injected at the correct timestep so community
    # agents can react to them. Official post IDs are pinned to every agent's feed.
    official_usernames: set[str] = {u.lower() for u in (official_accounts or [])}
    official_user_ids: dict[str, int] = {}   # username -> DB user_id
    official_posts_by_timestep: dict[int, list[tuple[str, str]]] = {}

    if official_usernames:
        max_agent_id = max(a.agent_id for a in agents) if agents else -1
        for i, uname in enumerate(sorted(official_usernames)):
            uid = max_agent_id + 1 + i
            db.insert_official_user(uid, uname)
            official_user_ids[uname] = uid
            social_log.info("Registered official account @%s as DB user_id=%d", uname, uid)

        # Posts will be loaded after simulation_start_dt is resolved below.
        pass

    # --- Activity profiles ---
    usernames = {a.username.lower() for a in agents}
    activity_profiles = _build_activity_profiles(
        train_csv_path=_resolve(train_activity_csv_path, _REPO_ROOT),
        usernames=usernames,
        granularity=activity_granularity,
        timestep_hours=timestep_hours,
        default_prob=default_activity_probability,
    )
    for agent in agents:
        profile = activity_profiles.get(agent.username.lower())
        if profile:
            if activity_granularity == "hourly":
                agent.activity_hourly = profile
            else:
                agent.activity_weekly = profile

    # --- Post-rate estimation ---
    if auto_target_posts_per_step_from_train and target_posts_per_step is None:
        target_posts_per_step = _estimate_target_posts_per_step(
            train_csv_path=_resolve(train_activity_csv_path, _REPO_ROOT),
            usernames=usernames,
            timestep_hours=timestep_hours,
        )
        if target_posts_per_step is not None:
            social_log.info("Auto post budget: target_posts_per_step=%d", target_posts_per_step)

    # --- Derive num_timesteps from split info ---
    simulation_start_dt: datetime | None = None
    if auto_num_timesteps_from_split:
        try:
            split_path = _resolve(split_info_path, _REPO_ROOT)
            with open(split_path, "r", encoding="utf-8") as f:
                split_info = safe_load(f)
            test_period = (split_info or {}).get("test_period", {})
            test_start_raw = test_period.get("start")
            test_end_raw = test_period.get("end")
            if test_start_raw and test_end_raw:
                test_start = datetime.fromisoformat(str(test_start_raw))
                test_end = datetime.fromisoformat(str(test_end_raw))
                simulation_start_dt = test_start
                duration_hours = max(1.0, (test_end - test_start).total_seconds() / 3600.0)
                num_timesteps = max(1, int(math.ceil(duration_hours / max(timestep_hours, 1e-9))))
                social_log.info(
                    "Auto timesteps: test_duration_hours=%.2f timestep_hours=%.3f num_timesteps=%d",
                    duration_hours, timestep_hours, num_timesteps,
                )
        except Exception as e:
            social_log.warning("Failed to auto-set num_timesteps from split info: %s", e)

    if simulation_start_dt is None:
        simulation_start_dt = datetime.now()

    # Now that simulation_start_dt is resolved, reload official posts with correct start time.
    if official_usernames:
        official_posts_by_timestep = _load_official_posts(
            csv_path=_resolve(official_posts_csv_path, _REPO_ROOT),
            official_usernames=official_usernames,
            test_start_dt=simulation_start_dt,
            timestep_hours=timestep_hours,
        )

    social_log.info(
        "Starting simulation: %d agents, %d timesteps, model=%s",
        len(agents), num_timesteps, model_name,
    )

    # --- Concurrency semaphore for LLM calls ---
    sem = asyncio.Semaphore(llm_concurrency)

    async def _call_llm_for_agent(agent: Agent, feed: list[dict]) -> dict[str, Any]:
        async with sem:
            return await _llm.decide_action(
                username=agent.username,
                name=agent.name,
                bio=agent.bio,
                user_char=agent.user_char,
                category_description=agent.category_description,
                previous_tweets=agent.previous_tweets,
                feed_posts=feed,
                available_actions=list(available_actions or normalized_action_probs.keys()),
                action_probs=agent.action_probs,
                model=model_name,
                client=client,
            )

    # ---------------------------------------------------------------------------
    # Main simulation loop
    # ---------------------------------------------------------------------------
    for timestep in range(1, num_timesteps + 1):
        db_file = Path(db_path).name if db_path != ":memory:" else ":memory:"
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}/{num_timesteps}" + Back.RESET)

        simulation_dt = simulation_start_dt + timedelta(hours=timestep_hours * timestep)

        # Determine active agents
        active_agents: list[Agent] = []
        for agent in agents:
            base_threshold = _choose_threshold(
                agent=agent,
                simulation_dt=simulation_dt,
                activity_granularity=activity_granularity,
            )
            threshold = max(
                min_activation_threshold,
                min(1.0, float(base_threshold) * float(agent_activation_multiplier)),
            )
            if random.random() < threshold:
                active_agents.append(agent)

        social_log.info(
            "Timestep %d/%d: %d/%d agents active (sim_dt=%s)",
            timestep, num_timesteps, len(active_agents), len(agents),
            simulation_dt.strftime("%Y-%m-%d %H:%M"),
        )

        if not active_agents:
            continue

        # --- Inject official account posts for this timestep ---
        # These are real posts from e.g. @vechainofficial, replayed from the test split.
        # Their post_ids are pinned into every community agent's feed so agents react to them.
        current_pinned_post_ids: list[int] = []
        for uname, text in official_posts_by_timestep.get(timestep, []):
            uid = official_user_ids.get(uname)
            if uid is not None:
                pid = db.create_post(uid, text, timestep)
                current_pinned_post_ids.append(pid)
                social_log.info(
                    "t=%d Injected official post from @%s (post_id=%d): %s…",
                    timestep, uname, pid, text[:60],
                )

        # Gather feeds and LLM decisions concurrently for all active agents
        # (each agent may take multiple actions)
        remaining_post_budget: int | None = target_posts_per_step

        # Build one flat list of (agent, action_index) tasks to run
        # Each active agent does _sample_num_actions_per_agent() actions
        agent_action_counts = {
            a.agent_id: _sample_num_actions_per_agent(
                max_actions_per_active_agent=max_actions_per_active_agent,
                extra_action_probability=extra_action_probability,
            )
            for a in active_agents
        }

        # Run all LLM calls concurrently (one per action slot).
        # Each task returns a list of (decision, feed_stats) pairs.
        async def _agent_action_task(
            agent: Agent,
            pinned_post_ids: list[int],
        ) -> list[tuple[dict[str, Any], dict[str, int]]]:
            n = agent_action_counts[agent.agent_id]
            results = []
            for _ in range(n):
                feed = db.get_feed(agent.agent_id, limit=max_feed_posts, pinned_post_ids=pinned_post_ids)
                n_following = sum(1 for p in feed if p.get("source") == "following")
                n_rec = sum(1 for p in feed if p.get("source") == "recommendation")
                n_official = sum(1 for p in feed if p.get("source") == "official")
                social_log.info(
                    "t=%d @%s | obs: feed=%d (following=%d, rec=%d, official=%d)",
                    timestep, agent.username, len(feed), n_following, n_rec, n_official,
                )
                decision = await _call_llm_for_agent(agent, feed)
                feed_stats = {
                    "feed_total": len(feed),
                    "feed_following": n_following,
                    "feed_rec": n_rec,
                }
                results.append((decision, feed_stats))
            return results

        all_results: list[list[tuple[dict[str, Any], dict[str, int]]]] = await asyncio.gather(
            *[_agent_action_task(a, current_pinned_post_ids) for a in active_agents],
            return_exceptions=False,
        )

        # Apply decisions to DB sequentially (safe single-threaded writes)
        for agent, action_results in zip(active_agents, all_results):
            for decision, feed_stats in action_results:
                blocked: set[str] = set()
                if remaining_post_budget is not None and remaining_post_budget <= 0:
                    blocked.add("create_post")
                action_taken, success, remaining_post_budget = _apply_decision(
                    db=db,
                    agent=agent,
                    decision=decision,
                    timestep=timestep,
                    remaining_post_budget=remaining_post_budget,
                    blocked_actions=blocked,
                )
                social_log.debug(
                    "t=%d @%s → %s (success=%s)",
                    timestep, agent.username, action_taken, success,
                )
                db.log_observation(
                    timestep=timestep,
                    agent_id=agent.agent_id,
                    username=agent.username,
                    feed_total=feed_stats["feed_total"],
                    feed_following=feed_stats["feed_following"],
                    feed_rec=feed_stats["feed_rec"],
                    action_taken=action_taken,
                    action_success=success,
                )

    db.close()
    social_log.info("Simulation finished. DB: %s", db_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _default_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "config.yaml")


if __name__ == "__main__":
    args = parser.parse_args()
    config_path = args.config_path or _default_config_path()
    config_path = _resolve(config_path, _REPO_ROOT)

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data", {})
        simulation_params = cfg.get("simulation", {})
        inference_configs = cfg.get("inference", {})

        asyncio.run(
            running(
                **data_params,
                **simulation_params,
                inference_configs=inference_configs,
            )
        )
    else:
        social_log.warning("Config not found at %s; running with defaults.", config_path)
        asyncio.run(running())

    social_log.info("Done.")
