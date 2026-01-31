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
import asyncio
import logging
import os
import random
import sys
from datetime import datetime
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


async def running(
    db_path: str | None = _DEFAULT_DB,
    csv_path: str | None = _DEFAULT_CSV,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
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
    if inference_configs["model_type"][:3] == "gpt":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType(inference_configs["model_type"]),
        )

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
        start_hour = 13.0

    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        channel=twitter_channel,
        start_time=start_time,
        model=model,
        recsys_type=recsys_type,
        available_actions=available_actions,
        twitter=infra,
    )

    for timestep in range(1, num_timesteps + 1):
        clock.time_step = timestep * 3
        db_file = Path(db_path).name
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        await infra.update_rec_table()

        simulation_time_hour = start_hour + 0.05 * timestep
        tasks = []
        for _node_id, agent in agent_graph.get_agents():
            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()
                threshold = agent.user_info.profile["other_info"]["active_threshold"][
                    int(simulation_time_hour % 24)
                ]
                if agent_ac_prob < threshold:
                    tasks.append(agent.perform_action_by_llm())
            else:
                await agent.perform_action_by_hci()

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
