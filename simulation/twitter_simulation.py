# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E402
"""
Oasis Twitter simulation. Run from repo root.

  python simulation/twitter_simulation.py --config_path simulation/config.yaml

Edit simulation/config.yaml to set csv_path and db_path to your own data.
Paths in the config are relative to the repo root.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import math
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from colorama import Back
from yaml import safe_load

# Repo root = parent of simulation (Oasis_copy root). Used for data/config paths.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Oasis package from submodule at oasis/ (camel-ai/oasis). Add submodule root so
# "import oasis" resolves to oasis/oasis/.
_OASIS_SUBMODULE = os.path.join(_REPO_ROOT, "oasis")
if os.path.isdir(_OASIS_SUBMODULE) and _OASIS_SUBMODULE not in sys.path:
    sys.path.insert(0, _OASIS_SUBMODULE)
elif _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

social_log = logging.getLogger(name="social")
social_log.propagate = False
social_log.setLevel("DEBUG")

file_handler = logging.FileHandler("social.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
)
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
)
social_log.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Oasis Twitter simulation.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to YAML config (default: simulation/config.yaml).",
    required=False,
    default="",
)

if os.path.isdir(_OASIS_SUBMODULE):
    _DATA_PREFIX = os.path.join(_REPO_ROOT, "oasis", "data")
else:
    _DATA_PREFIX = os.path.join(_REPO_ROOT, "data")
_ALL_TOPICS_CSV = os.path.join(_DATA_PREFIX, "twitter_dataset", "all_topics.csv")
_DEFAULT_CSV = os.path.join(
    _DATA_PREFIX, "twitter_dataset", "anonymous_topic_200_1h", "False_Business_0.csv"
)
_DEFAULT_DB = ":memory:"


def _resolve(path: str, base: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def _normalize_action_probabilities(
    action_probabilities: dict[str, float] | None,
    available_actions: list[ActionType] | None,
) -> dict[str, float]:
    if action_probabilities:
        probs = {
            str(k): max(0.0, float(v))
            for k, v in action_probabilities.items()
        }
    else:
        actions = [
            a.value if isinstance(a, ActionType) else str(a)
            for a in (available_actions or [])
        ]
        if not actions:
            actions = ["create_post", "create_comment", "repost", "like_post"]
        probs = {a: 1.0 for a in actions}

    total = sum(probs.values())
    if total <= 0:
        return {"do_nothing": 1.0}
    return {k: v / total for k, v in probs.items()}


def _extract_action_probabilities_by_username(
    csv_path: str,
    available_actions: list[ActionType] | None,
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
    agent,
    simulation_dt: datetime,
    activity_granularity: str,
) -> float:
    other = agent.user_info.profile.get("other_info", {})
    if activity_granularity == "hourly":
        vec = other.get("activity_hourly") or other.get("active_threshold") or [0.1] * 24
        idx = int(simulation_dt.hour) % 24
    else:
        vec = other.get("activity_weekly")
        if not vec:
            hour_vec = other.get("active_threshold") or [0.1] * 24
            return float(hour_vec[int(simulation_dt.hour) % 24])
        idx = int(simulation_dt.weekday()) % 7
    try:
        return float(vec[idx])
    except Exception:
        return 0.1


def _make_post_content(agent) -> str:
    other = agent.user_info.profile.get("other_info", {})
    seeds = other.get("previous_tweets") or []
    if seeds:
        return random.choice(seeds)
    user_profile = str(other.get("user_profile", "")).strip()
    if user_profile:
        return f"Thought of the day: {user_profile[:180]}"
    return "Sharing a quick update."


def _make_comment_content(agent) -> str:
    other = agent.user_info.profile.get("other_info", {})
    seeds = other.get("previous_tweets") or []
    if seeds:
        return random.choice(seeds)[:180]
    return random.choice([
        "Interesting point, thanks for sharing.",
        "I see this differently, but appreciate the post.",
        "Good thread. Curious what others think.",
    ])


def _make_quote_content(agent) -> str:
    other = agent.user_info.profile.get("other_info", {})
    seeds = other.get("previous_tweets") or []
    if seeds:
        return f"My take: {random.choice(seeds)[:160]}"
    return "Adding context to this."


async def _try_action_on_feed_posts(agent, action_name: str, feed_posts: list[dict[str, Any]], *extra_args):
    if not feed_posts:
        return {"success": False, "error": "No feed posts"}
    shuffled = feed_posts[:]
    random.shuffle(shuffled)
    for post in shuffled:
        post_id = post.get("post_id")
        if post_id is None:
            continue
        try:
            result = await agent.perform_action_by_data(action_name, int(post_id), *extra_args)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        if isinstance(result, dict) and result.get("success"):
            return result
    return {"success": False, "error": f"No valid candidate for {action_name}"}


async def _try_follow_on_feed_posts(agent, feed_posts: list[dict[str, Any]]):
    if not feed_posts:
        return {"success": False, "error": "No feed posts"}

    own_user_id = None
    try:
        own_user_id = int(agent.social_agent_id)
    except Exception:
        own_user_id = None

    candidates = []
    for post in feed_posts:
        user_id = post.get("user_id")
        if user_id is None:
            continue
        try:
            uid = int(user_id)
        except Exception:
            continue
        if own_user_id is not None and uid == own_user_id:
            continue
        candidates.append(uid)

    if not candidates:
        return {"success": False, "error": "No valid follow candidate"}

    random.shuffle(candidates)
    seen = set()
    for uid in candidates:
        if uid in seen:
            continue
        seen.add(uid)
        try:
            result = await agent.perform_action_by_data("follow", uid)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        if isinstance(result, dict) and result.get("success"):
            return result

    return {"success": False, "error": "No valid follow candidate"}


async def _perform_manual_probabilistic_action(
    agent,
    action_probs: dict[str, float],
    blocked_actions: set[str] | None = None,
) -> dict[str, Any]:
    refresh_result = await agent.env.action.refresh()
    feed_posts = refresh_result.get("posts", []) if isinstance(refresh_result, dict) else []
    blocked = blocked_actions or set()

    candidates: list[tuple[str, float]] = []
    for action_name, prob in action_probs.items():
        if prob <= 0:
            continue
        if action_name in blocked:
            continue
        if action_name in {"like_post", "create_comment", "repost", "quote_post", "follow"} and not feed_posts:
            continue
        candidates.append((action_name, prob))

    if not candidates:
        result = await agent.perform_action_by_data("do_nothing")
        return {"action": "do_nothing", "result": result}

    total = sum(w for _, w in candidates)
    normalized = [(a, w / total) for a, w in candidates]
    action_name = random.choices(
        [a for a, _ in normalized],
        weights=[w for _, w in normalized],
        k=1,
    )[0]

    if action_name == "create_post":
        result = await agent.perform_action_by_data("create_post", _make_post_content(agent))
        return {"action": action_name, "result": result}
    if action_name == "create_comment":
        result = await _try_action_on_feed_posts(
            agent,
            "create_comment",
            feed_posts,
            _make_comment_content(agent),
        )
        return {"action": action_name, "result": result}
    if action_name == "repost":
        result = await _try_action_on_feed_posts(agent, "repost", feed_posts)
        return {"action": action_name, "result": result}
    if action_name == "quote_post":
        result = await _try_action_on_feed_posts(
            agent,
            "quote_post",
            feed_posts,
            _make_quote_content(agent),
        )
        return {"action": action_name, "result": result}
    if action_name == "like_post":
        result = await _try_action_on_feed_posts(agent, "like_post", feed_posts)
        return {"action": action_name, "result": result}
    if action_name == "follow":
        result = await _try_follow_on_feed_posts(agent, feed_posts)
        return {"action": action_name, "result": result}
    if action_name == "do_nothing":
        result = await agent.perform_action_by_data("do_nothing")
        return {"action": action_name, "result": result}

    result = await agent.perform_action_by_data("do_nothing")
    return {"action": "do_nothing", "result": result}


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


async def running(
    db_path: str | None = _DEFAULT_DB,
    csv_path: str | None = _DEFAULT_CSV,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    clock_step_units: int = 1,
    timestep_hours: float = 1.0,
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
    recsys_type: str = "twhin-bert",
    action_selection_mode: str = "manual_probabilistic",
    action_probabilities: dict[str, float] | None = None,
    inference_configs: dict[str, Any] | None = None,
    available_actions: list[ActionType] | None = None,
) -> None:
    db_path = _DEFAULT_DB if db_path is None else db_path
    csv_path = _DEFAULT_CSV if csv_path is None else csv_path
    if db_path != ":memory:":
        db_path = _resolve(db_path, _REPO_ROOT)
    csv_path = _resolve(csv_path, _REPO_ROOT)

    if db_path != ":memory:" and os.path.exists(db_path):
        os.remove(db_path)
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if recsys_type == "reddit":
        start_time = datetime.now()
    else:
        start_time = 0
    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    infra = Platform(
        db_path=db_path,
        channel=twitter_channel,
        sandbox_clock=clock,
        start_time=start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )
    twitter_task = asyncio.create_task(infra.running())
    model = None
    model_type = str((inference_configs or {}).get("model_type", ""))
    if model_type[:3] == "gpt":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType(model_type),
        )

    start_hour = 13.0  # Default when topic metadata unavailable
    try:
        all_topic_df = pd.read_csv(_ALL_TOPICS_CSV)
        if "False" in csv_path or "True" in csv_path:
            if "-" not in csv_path:
                topic_name = Path(csv_path).stem
            else:
                topic_name = Path(csv_path).stem.split("-")[0]
            row = all_topic_df[all_topic_df["topic_name"] == topic_name]
            start_time_str = row["start_time"].item().split(" ")[1]
            h, m = start_time_str.split(":")
            start_hour = int(h) + float(m) / 60
    except Exception:
        social_log.info("No real-world topic data; using start_hour = 13 (1 PM).")

    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        channel=twitter_channel,
        start_time=start_time,
        model=model,
        recsys_type=recsys_type,
        available_actions=available_actions,
        twitter=infra,
    )

    prev_tweets_by_username = _extract_previous_tweets_by_username(csv_path)
    action_probs_by_username = _extract_action_probabilities_by_username(
        csv_path=csv_path,
        available_actions=available_actions,
    )
    usernames = {
        str(agent.user_info.name).strip().lower()
        for _node_id, agent in agent_graph.get_agents()
    }
    activity_profiles = _build_activity_profiles(
        train_csv_path=_resolve(train_activity_csv_path, _REPO_ROOT),
        usernames=usernames,
        granularity=activity_granularity,
        timestep_hours=timestep_hours,
        default_prob=default_activity_probability,
    )
    for _node_id, agent in agent_graph.get_agents():
        uname = str(agent.user_info.name).strip().lower()
        other = agent.user_info.profile.setdefault("other_info", {})
        other["previous_tweets"] = prev_tweets_by_username.get(uname, [])
        if uname in action_probs_by_username:
            other["action_probabilities"] = action_probs_by_username[uname]
        if uname in activity_profiles:
            if activity_granularity == "hourly":
                other["activity_hourly"] = activity_profiles[uname]
            else:
                other["activity_weekly"] = activity_profiles[uname]

    normalized_action_probs = _normalize_action_probabilities(
        action_probabilities=action_probabilities,
        available_actions=available_actions,
    )
    if auto_target_posts_per_step_from_train and target_posts_per_step is None:
        target_posts_per_step = _estimate_target_posts_per_step(
            train_csv_path=_resolve(train_activity_csv_path, _REPO_ROOT),
            usernames=usernames,
            timestep_hours=timestep_hours,
        )
        if target_posts_per_step is not None:
            social_log.info(
                "Auto post budget enabled. target_posts_per_step=%d",
                target_posts_per_step,
            )

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
                    "Auto timesteps enabled. test_duration_hours=%.2f timestep_hours=%.3f num_timesteps=%d",
                    duration_hours,
                    timestep_hours,
                    num_timesteps,
                )
        except Exception as e:
            social_log.warning(f"Failed to auto-set num_timesteps from split info: {e}")

    if simulation_start_dt is None:
        simulation_start_dt = datetime.now()

    for timestep in range(1, num_timesteps + 1):
        clock.time_step = timestep * clock_step_units
        db_file = Path(db_path).name
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        await infra.update_rec_table()

        simulation_dt = simulation_start_dt + timedelta(hours=timestep_hours * timestep)
        tasks = []
        for _node_id, agent in agent_graph.get_agents():
            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()
                base_threshold = _choose_threshold(
                    agent=agent,
                    simulation_dt=simulation_dt,
                    activity_granularity=activity_granularity,
                )
                threshold = max(
                    min_activation_threshold,
                    min(1.0, float(base_threshold) * float(agent_activation_multiplier)),
                )
                if agent_ac_prob < threshold:
                    if action_selection_mode == "manual_probabilistic":
                        tasks.append(agent)
                    else:
                        tasks.append(agent.perform_action_by_llm())
            else:
                await agent.perform_action_by_hci()

        if action_selection_mode == "manual_probabilistic":
            remaining_post_budget = target_posts_per_step
            for agent in tasks:
                num_actions = _sample_num_actions_per_agent(
                    max_actions_per_active_agent=max_actions_per_active_agent,
                    extra_action_probability=extra_action_probability,
                )
                for _ in range(num_actions):
                    blocked_actions: set[str] = set()
                    if remaining_post_budget is not None and remaining_post_budget <= 0:
                        blocked_actions.add("create_post")
                    per_agent_probs = (
                        agent.user_info.profile.get("other_info", {}).get("action_probabilities")
                        or normalized_action_probs
                    )
                    action_out = await _perform_manual_probabilistic_action(
                        agent=agent,
                        action_probs=per_agent_probs,
                        blocked_actions=blocked_actions,
                    )
                    selected_action = str(action_out.get("action", ""))
                    result = action_out.get("result")
                    if (
                        selected_action == "create_post"
                        and remaining_post_budget is not None
                        and isinstance(result, dict)
                        and result.get("success")
                    ):
                        remaining_post_budget -= 1
        else:
            await asyncio.gather(*tasks)

    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await twitter_task


def _default_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "config.yaml")


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)

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
        asyncio.run(running())

    social_log.info("Simulation finished.")
