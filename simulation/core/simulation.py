"""
simulation.py — clean synchronous simulation loop.

Replaces twitter_simulation.py + OASIS OasisEnv with a straightforward
for-loop that:
  1. Registers agents in the platform DB.
  2. Each timestep: refreshes the rec table, activates agents, executes actions.
  3. Agents track their own state (liked/followed/reposted) to avoid duplicate spam.

Entry point: run_simulation(config_path) or __main__ via CLI.

Usage (from repo root, with venv active):
    python -m simulation.core.simulation
    python -m simulation.core.simulation --config simulation/config.yaml
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from colorama import Back, init as colorama_init
from yaml import safe_load

from .agent import Agent
from .agent_loader import load_agents, _normalize
from .platform import Platform

colorama_init(autoreset=True)

# ── Logging ─────────────────────────────────────────────────────────────────
os.makedirs("./log", exist_ok=True)
_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
social_log = logging.getLogger("social")
social_log.setLevel(logging.DEBUG)
_fh = logging.FileHandler("social.log")
_fh.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(_fh)
social_log.addHandler(_sh)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve(path: str, root: Path) -> str:
    p = Path(path)
    return str(p) if p.is_absolute() else str(root / p)


def _sample_num_actions(max_actions: int, extra_prob: float) -> int:
    """Geometric expansion: at least 1, up to max_actions."""
    n = 1
    while n < max_actions and random.random() < extra_prob:
        n += 1
    return n


def _bin_index(dt: datetime, granularity: str) -> int:
    return dt.weekday() if granularity == "weekly" else dt.hour


def _derive_num_timesteps(split_info_path: str) -> int | None:
    try:
        info = json.loads(Path(split_info_path).read_text(encoding="utf-8"))
        start = datetime.fromisoformat(info["test_period"]["start"])
        end = datetime.fromisoformat(info["test_period"]["end"])
        days = max(1, (end - start).days)
        return days
    except Exception as e:
        social_log.warning("Could not derive num_timesteps from split_info: %s", e)
        return None


def _estimate_target_posts(train_csv: str, usernames: set[str],
                            timestep_hours: float) -> int | None:
    try:
        import pandas as pd
        df = pd.read_csv(train_csv)
        ucol = next((c for c in ["user_name", "username"] if c in df.columns), None)
        if ucol is None or "timestamp" not in df.columns:
            return None
        df[ucol] = df[ucol].astype(str).str.strip().str.lower()
        df = df[df[ucol].isin(usernames)]
        if df.empty:
            return None
        df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["dt"])
        total_hours = (df["dt"].max() - df["dt"].min()).total_seconds() / 3600
        if total_hours <= 0:
            return None
        rate_per_step = len(df) / max(total_hours / timestep_hours, 1)
        return max(1, int(round(rate_per_step)))
    except Exception as e:
        social_log.warning("Could not estimate target_posts_per_step: %s", e)
        return None


# ── Action execution ─────────────────────────────────────────────────────────

def _pick_target_post(feed: list[dict], agent: Agent,
                      exclude_own: bool = True) -> dict | None:
    """Pick a post from the feed the agent can interact with."""
    candidates = [
        p for p in feed
        if not (exclude_own and p["user_id"] == agent.user_id)
    ]
    return random.choice(candidates) if candidates else None


def _pick_unfollowed_user(feed: list[dict], agent: Agent,
                          all_user_ids: list[int]) -> int | None:
    """Pick a user from the feed (or all users) that the agent doesn't follow."""
    candidates = [
        p["user_id"] for p in feed
        if agent.can_follow(p["user_id"])
    ]
    if not candidates:
        # Fall back to any user in the platform.
        candidates = [uid for uid in all_user_ids if agent.can_follow(uid)]
    return random.choice(candidates) if candidates else None


def _execute_action(action: str, agent: Agent, platform: Platform,
                    feed: list[dict], all_user_ids: list[int],
                    post_budget: list[int | None]) -> dict:
    """Dispatch a single action for an agent and update agent state."""
    uid = agent.user_id
    result: dict[str, Any] = {"success": False, "error": "no-op"}

    if action == "create_post":
        content = agent.generate_post_content()
        result = platform.create_post(uid, content)
        if result.get("success"):
            agent.record_post()
            if post_budget[0] is not None:
                post_budget[0] -= 1

    elif action == "like_post":
        post = _pick_target_post(
            [p for p in feed if agent.can_like(p["post_id"])],
            agent,
        )
        if post:
            result = platform.like_post(uid, post["post_id"])
            if result.get("success"):
                agent.record_like(post["post_id"])
        else:
            result = {"success": False, "error": "no unliked posts in feed"}

    elif action == "create_comment":
        post = _pick_target_post(feed, agent)
        if post:
            content = agent.generate_comment_content(post.get("content", ""))
            result = platform.create_comment(uid, post["post_id"], content)
            if result.get("success"):
                agent.record_comment(post["post_id"])
        else:
            result = {"success": False, "error": "no posts to comment on"}

    elif action == "repost":
        post = _pick_target_post(
            [p for p in feed if agent.can_repost(p["post_id"])],
            agent,
        )
        if post:
            result = platform.repost(uid, post["post_id"])
            if result.get("success"):
                agent.record_repost(post["post_id"])
        else:
            result = {"success": False, "error": "no repostable posts in feed"}

    elif action == "quote_post":
        post = _pick_target_post(feed, agent)
        if post:
            content = agent.generate_quote_content(post.get("content", ""))
            result = platform.quote_post(uid, post["post_id"], content)
        else:
            result = {"success": False, "error": "no posts to quote"}

    elif action == "follow":
        target_uid = _pick_unfollowed_user(feed, agent, all_user_ids)
        if target_uid is not None:
            result = platform.follow(uid, target_uid)
            if result.get("success"):
                agent.record_follow(target_uid)
        else:
            result = {"success": False, "error": "no unfollowed users available"}

    elif action == "do_nothing":
        result = platform.do_nothing(uid)

    social_log.info("Agent %d [%s]: %s", agent.agent_id, action, result)
    return result


# ── Main simulation ──────────────────────────────────────────────────────────

def run_simulation(config_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = safe_load(Path(config_path).read_text(encoding="utf-8")) or {}

    data_cfg = cfg.get("data", {})
    sim_cfg = cfg.get("simulation", {})

    db_path = _resolve(data_cfg.get("db_path", "data/simu_db/simulation.db"), repo_root)
    personas_csv = _resolve(
        data_cfg.get("csv_path", "persona_generation/outputs/user_personas.csv"),
        repo_root,
    )
    train_csv = _resolve(
        sim_cfg.get("train_activity_csv_path",
                    "persona_generation/outputs/train_tweets.csv"),
        repo_root,
    )
    split_info_path = _resolve(
        sim_cfg.get("split_info_path",
                    "persona_generation/outputs/split_info.json"),
        repo_root,
    )

    num_timesteps: int = sim_cfg.get("num_timesteps", 7)
    if sim_cfg.get("auto_num_timesteps_from_split"):
        derived = _derive_num_timesteps(split_info_path)
        if derived:
            num_timesteps = derived

    timestep_hours: float = float(sim_cfg.get("timestep_hours", 24.0))
    activity_granularity: str = sim_cfg.get("activity_granularity", "weekly")
    default_activity_prob: float = float(sim_cfg.get("default_activity_probability", 0.10))
    agent_activation_multiplier: float = float(sim_cfg.get("agent_activation_multiplier", 2.0))
    min_activation_threshold: float = float(sim_cfg.get("min_activation_threshold", 0.10))
    max_actions: int = int(sim_cfg.get("max_actions_per_active_agent", 1))
    extra_action_prob: float = float(sim_cfg.get("extra_action_probability", 0.5))
    recsys_type: str = sim_cfg.get("recsys_type", "twhin-bert")
    max_rec_post_len: int = int(sim_cfg.get("max_rec_post_len", 2))
    refresh_rec_post_count: int = int(sim_cfg.get("refresh_rec_post_count", 1))
    following_post_count: int = int(sim_cfg.get("following_post_count", 3))
    clock_step_units: int = int(sim_cfg.get("clock_step_units", 1))

    global_action_probs = _normalize(
        sim_cfg.get("action_probabilities") or {
            "like_post": 0.60,
            "create_post": 0.10,
            "create_comment": 0.20,
            "repost": 0.10,
        }
    )

    target_posts_per_step: int | None = sim_cfg.get("target_posts_per_step")

    # ── Load agents ──────────────────────────────────────────────────────
    social_log.info("Loading agents from %s", personas_csv)
    agents = load_agents(
        personas_csv=personas_csv,
        train_csv=train_csv,
        default_action_probs=global_action_probs,
        activity_granularity=activity_granularity,
        timestep_hours=timestep_hours,
        default_activity_prob=default_activity_prob,
    )

    # Merge global action probabilities as fallback for agents without per-agent probs.
    for agent in agents:
        if not agent.action_probabilities:
            agent.action_probabilities = global_action_probs

    usernames = {a.username for a in agents}

    if sim_cfg.get("auto_target_posts_per_step_from_train") and target_posts_per_step is None:
        target_posts_per_step = _estimate_target_posts(train_csv, usernames, timestep_hours)
        if target_posts_per_step:
            social_log.info("Auto target_posts_per_step = %d", target_posts_per_step)

    # ── Initialise platform ──────────────────────────────────────────────
    # Always start fresh so runs are comparable.
    if Path(db_path).exists():
        Path(db_path).unlink()
        social_log.info("Deleted existing DB for fresh run: %s", db_path)
    social_log.info("Opening DB at %s", db_path)
    platform = Platform(
        db_path=db_path,
        recsys_type=recsys_type,
        max_rec_post_len=max_rec_post_len,
        refresh_rec_post_count=refresh_rec_post_count,
        following_post_count=following_post_count,
    )

    # ── Sign up agents ───────────────────────────────────────────────────
    social_log.info("Signing up %d agents...", len(agents))
    for agent in agents:
        platform.sign_up(agent.agent_id, agent.username, agent.name, agent.bio)
        agent.user_id = agent.agent_id  # user_id == agent_id by convention

    all_user_ids = [a.user_id for a in agents if a.user_id >= 0]

    # ── Seed agent state from DB (handles resumed/reused DBs) ────────────
    # Build lookup so we can find agents by user_id quickly.
    agent_by_uid = {a.user_id: a for a in agents if a.user_id >= 0}
    platform._cur.execute("SELECT follower_id, followee_id FROM follow")
    for follower_id, followee_id in platform._cur.fetchall():
        if follower_id in agent_by_uid:
            agent_by_uid[follower_id].followed_user_ids.add(followee_id)
    platform._cur.execute("SELECT user_id, post_id FROM 'like'")
    for user_id, post_id in platform._cur.fetchall():
        if user_id in agent_by_uid:
            agent_by_uid[user_id].liked_post_ids.add(post_id)
    platform._cur.execute(
        "SELECT user_id, original_post_id FROM post "
        "WHERE original_post_id IS NOT NULL AND "
        "(quote_content IS NULL OR TRIM(quote_content) = '')"
    )
    for user_id, orig_id in platform._cur.fetchall():
        if user_id in agent_by_uid:
            agent_by_uid[user_id].reposted_post_ids.add(orig_id)
    social_log.info("Agent state seeded from existing DB.")

    # ── Derive simulation start datetime for temporal activation ─────────
    sim_start_dt = datetime.now()
    try:
        info = json.loads(Path(split_info_path).read_text(encoding="utf-8"))
        sim_start_dt = datetime.fromisoformat(info["test_period"]["start"])
    except Exception:
        pass

    # ── Simulation loop ──────────────────────────────────────────────────
    social_log.info(
        "Starting simulation: %d timesteps × %.1fh each, recsys=%s",
        num_timesteps, timestep_hours, recsys_type,
    )

    for timestep in range(1, num_timesteps + 1):
        platform.time_step = float(timestep * clock_step_units)
        db_name = Path(db_path).name
        print(Back.GREEN + f"DB:{db_name}  step:{timestep}/{num_timesteps}" + Back.RESET)

        # 1. Refresh recommendations.
        platform.update_rec_table()

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            social_log.info("[GPU] Step %d: VRAM allocated %.2f GB", timestep, vram_gb)

        # 2. Determine which agents activate this step.
        step_dt = sim_start_dt + timedelta(hours=timestep_hours * timestep)
        bin_idx = _bin_index(step_dt, activity_granularity)

        active_agents = [
            a for a in agents
            if a.should_activate(bin_idx, agent_activation_multiplier,
                                 min_activation_threshold)
        ]
        social_log.info("Step %d: %d/%d agents active",
                        timestep, len(active_agents), len(agents))

        # 3. Execute actions.
        remaining_posts = [target_posts_per_step]  # mutable via list

        for agent in active_agents:
            feed_result = platform.get_feed(agent.user_id)
            feed = feed_result.get("posts", [])

            n_actions = _sample_num_actions(max_actions, extra_action_prob)
            for _ in range(n_actions):
                blocked: set[str] = set()
                if remaining_posts[0] is not None and remaining_posts[0] <= 0:
                    blocked.add("create_post")
                action = agent.sample_action(blocked=blocked)
                _execute_action(action, agent, platform, feed,
                                all_user_ids, remaining_posts)

    # ── Cleanup ──────────────────────────────────────────────────────────
    platform.close()
    social_log.info("Simulation finished.")


# ── CLI entry point ──────────────────────────────────────────────────────────

def _default_config() -> str:
    return str(Path(__file__).resolve().parents[1] / "config.yaml")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OASIS-replacement simulation.")
    parser.add_argument(
        "--config", default=_default_config(),
        help="Path to config.yaml (default: simulation/config.yaml)",
    )
    args = parser.parse_args()
    run_simulation(args.config)
