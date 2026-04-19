"""
recsys.py — self-contained recommendation system for the simulation core.

Contains all four recsys variants (random, reddit, twitter/personalized,
twhin-bert) plus the run_recsys dispatcher. No dependency on the oasis package.
"""
from __future__ import annotations

import heapq
import logging
import os
import random
import time
from ast import literal_eval
from datetime import datetime
from math import log
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from .process_recsys_posts import (
    _apply_gpu_optimizations,
    _DEFAULT_BATCH_SIZE,
    generate_post_vector,
    generate_post_vector_openai,
)

recsys_log = logging.getLogger("social.recsys")
rec_log = logging.getLogger("social.rec")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Module-level model cache ──────────────────────────────────────────────────
twhin_tokenizer = None
twhin_model = None

# ── TWHIN state (accumulated across steps within one run) ─────────────────────
user_previous_post_all: dict = {}
user_previous_post: dict = {}
user_profiles: list = []
t_items: dict = {}
u_items: dict = {}
date_score: list = []


# ── TWHIN model loading ───────────────────────────────────────────────────────

def _resolve_twhin_model_source() -> tuple[str, bool]:
    """Resolve a local TWHIN model directory; fallback to HuggingFace Hub."""
    explicit = os.getenv("TWHIN_MODEL_PATH")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    cwd = Path.cwd()
    candidates.extend([
        cwd / ".cache" / "hf" / "Twitter" / "twhin-bert-base",
        cwd / "Twitter" / "twhin-bert-base",
    ])
    here = Path(__file__).resolve()
    for p in here.parents:
        candidates.append(p / ".cache" / "hf" / "Twitter" / "twhin-bert-base")
    for cand in candidates:
        try:
            if cand.is_dir() and (cand / "config.json").exists():
                return str(cand), True
        except OSError:
            continue
    return "Twitter/twhin-bert-base", False


def _get_twhin_tokenizer():
    global twhin_tokenizer
    if twhin_tokenizer is None:
        from transformers import AutoTokenizer
        source, local_only = _resolve_twhin_model_source()
        rec_log.info("Loading TWHIN tokenizer from source=%s local_files_only=%s",
                     source, local_only)
        twhin_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=source,
            model_max_length=512,
            local_files_only=local_only,
        )
    return twhin_tokenizer


def _get_twhin_model():
    global twhin_model
    if twhin_model is None:
        from transformers import AutoModel
        source, local_only = _resolve_twhin_model_source()
        rec_log.info("Loading TWHIN model from source=%s local_files_only=%s",
                     source, local_only)
        twhin_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=source,
            local_files_only=local_only,
        ).to(_device)
        twhin_model = _apply_gpu_optimizations(twhin_model)
    return twhin_model


# ── Helper functions ──────────────────────────────────────────────────────────

def _calculate_hot_score(num_likes: int, num_dislikes: int,
                         created_at: datetime) -> float:
    s = num_likes - num_dislikes
    order = log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0
    epoch = datetime(1970, 1, 1)
    td = created_at - epoch
    epoch_seconds = td.days * 86400 + td.seconds + float(td.microseconds) / 1e6
    seconds = epoch_seconds - 1134028003
    return round(sign * order + seconds / 45000, 7)


def _coarse_filtering(input_list: list, scale: int) -> tuple:
    if len(input_list) <= scale:
        return (input_list, range(len(input_list)))
    sampled_indices = random.sample(range(len(input_list)), scale)
    sampled_elements = [input_list[idx] for idx in sampled_indices]
    return (sampled_elements, sampled_indices)


def _get_like_post_ids(user_id: int, trace_table: list) -> list:
    ids = [
        literal_eval(t["info"])["post_id"]
        for t in trace_table
        if t["user_id"] == user_id and t["action"] == "like_post"
    ]
    if 0 < len(ids) < 5:
        ids += [ids[-1]] * (5 - len(ids))
    elif len(ids) > 5:
        ids = ids[-5:]
    else:
        ids = [0]
    return ids


def _get_trace_contents(user_id: int, action: str, post_table: list,
                        trace_table: list) -> list:
    trace_ids = {
        literal_eval(t["info"]).get("post_id")
        for t in trace_table
        if t["user_id"] == user_id and t["action"] == action
    }
    return [p["content"] for p in post_table if p["post_id"] in trace_ids]


# ── Recsys variants ───────────────────────────────────────────────────────────

def rec_sys_random(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    post_ids = [p["post_id"] for p in post_table]
    if len(post_ids) <= max_rec_post_len:
        return [post_ids] * len(rec_matrix)
    return [random.sample(post_ids, max_rec_post_len) for _ in rec_matrix]


def rec_sys_reddit(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    post_ids = [p["post_id"] for p in post_table]
    if len(post_ids) <= max_rec_post_len:
        return [post_ids] * len(rec_matrix)
    scores = []
    for post in post_table:
        try:
            dt = datetime.strptime(post["created_at"], "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            try:
                dt = datetime.strptime(post["created_at"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = datetime.now()
        scores.append((_calculate_hot_score(post["num_likes"],
                                            post["num_dislikes"], dt),
                       post["post_id"]))
    top_ids = [pid for _, pid in heapq.nlargest(max_rec_post_len, scores,
                                                 key=lambda x: x[0])]
    return [top_ids] * len(rec_matrix)


def rec_sys_personalized_with_trace(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
) -> List[List]:
    """Personalized recsys using sentence-transformers + interaction trace."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2",
                                device=str(_device),
                                cache_folder="./models")
    post_ids = [p["post_id"] for p in post_table]
    if len(post_ids) <= max_rec_post_len:
        return [post_ids] * len(rec_matrix)

    new_rec_matrix = []
    for user in user_table:
        uid = user["user_id"]
        bio = user.get("bio") or ""
        available = [(p["post_id"], p["content"])
                     for p in post_table if p["user_id"] != uid]
        like_contents = _get_trace_contents(uid, "like_post", post_table, trace_table)
        dislike_contents = _get_trace_contents(uid, "unlike_post", post_table, trace_table)

        post_scores = []
        for pid, content in available:
            u_emb = model.encode(bio)
            p_emb = model.encode(content)
            sim = float(np.dot(u_emb, p_emb) /
                        (np.linalg.norm(u_emb) * np.linalg.norm(p_emb) + 1e-9))
            post_scores.append((pid, sim))

        post_scores.sort(key=lambda x: x[1], reverse=True)
        rec_ids = [pid for pid, _ in post_scores[:max_rec_post_len]]
        new_rec_matrix.append(rec_ids)
    return new_rec_matrix


def rec_sys_personalized_twh(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    latest_post_count: int,
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
    current_time: int,
    use_openai_embedding: bool = False,
) -> List[List]:
    global twhin_model, twhin_tokenizer
    global date_score, t_items, u_items, user_previous_post
    global user_previous_post_all, user_profiles

    tokenizer = _get_twhin_tokenizer()
    model = _get_twhin_model()

    # Build uid→follower_count map (reset when user count changes).
    if not u_items or len(u_items) != len(user_table):
        u_items = {u["user_id"]: u["num_followers"] for u in user_table}

    # Build per-user post history maps (reset when user count changes).
    if not user_previous_post_all or len(user_previous_post_all) != len(user_table):
        user_previous_post_all = {i: [] for i in range(len(user_table))}
        user_previous_post = {i: "" for i in range(len(user_table))}

    # Build user profile list (reset when user count changes).
    if not user_profiles or len(user_profiles) != len(user_table):
        user_profiles.clear()
        for u in user_table:
            user_profiles.append(u["bio"] or "This user does not have profile")

    # Incrementally add new posts to t_items / date_score.
    if len(t_items) < len(post_table):
        for post in post_table[-latest_post_count:]:
            uid = post["user_id"]
            # Guard against posts with invalid user_ids (e.g. from failed sign-ups).
            if uid not in range(len(user_table)):
                continue
            t_items[post["post_id"]] = post["content"] or ""
            user_previous_post_all[uid].append(post["content"] or "")
            user_previous_post[uid] = post["content"] or ""
            date_score.append(
                np.log(max(
                    (271.8 - (current_time - int(post["created_at"]))) / 100,
                    1e-6
                ))
            )

    date_score_np = np.array(date_score)

    tids = [p["post_id"] for p in post_table]
    if len(post_table) <= max_rec_post_len:
        return [tids] * len(rec_matrix)

    # Update user profiles with their most recent post content.
    for idx in user_previous_post:
        try:
            recent = user_previous_post[idx]
            if not recent:
                continue
            update = f" # Recent post:{recent}"
            if "# Recent post:" not in user_profiles[idx]:
                user_profiles[idx] += update
            elif update not in user_profiles[idx]:
                user_profiles[idx] = (
                    user_profiles[idx].split("# Recent post:")[0] + update
                )
        except Exception:
            pass

    # Coarse filter to 4000 posts for memory efficiency.
    filtered_texts, filter_indices = _coarse_filtering(
        list(t_items.values()), 4000
    )
    corpus = user_profiles + filtered_texts

    t0 = time.time()
    if use_openai_embedding:
        all_vectors = generate_post_vector_openai(corpus, batch_size=1000)
    else:
        all_vectors = generate_post_vector(model, tokenizer, corpus,
                                           batch_size=_DEFAULT_BATCH_SIZE)
    rec_log.info("twhin model cost time: %s", time.time() - t0)

    user_vectors = all_vectors[:len(user_profiles)]
    post_vectors = all_vectors[len(user_profiles):]

    t1 = time.time()
    sims = cosine_similarity(user_vectors, post_vectors)
    rec_log.info("get cosine_similarity time: %s", time.time() - t1)

    scores = date_score_np[list(filter_indices)]
    sims = sims * scores
    sims_t = torch.tensor(sims)

    k = min(max_rec_post_len, sims_t.shape[1])
    _, indices = torch.topk(sims_t, k, dim=1, largest=True, sorted=True)
    filter_idx_t = torch.tensor(list(filter_indices))
    indices = filter_idx_t[indices]

    post_id_list = list(t_items.keys())
    new_rec_matrix = []
    for row in indices.cpu().numpy():
        new_rec_matrix.append([post_id_list[i] for i in row])
    return new_rec_matrix


# ── Dispatcher ────────────────────────────────────────────────────────────────

def run_recsys(
    recsys_type: str,
    user_table: list,
    post_table: list,
    trace_table: list,
    rec_matrix: list,
    max_rec_post_len: int,
    time_step: int,
    use_openai_embedding: bool = False,
) -> list:
    rtype = recsys_type.lower().strip()

    if rtype == "random":
        return rec_sys_random(post_table, rec_matrix, max_rec_post_len)

    if rtype == "reddit":
        return rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)

    if rtype == "twitter":
        return rec_sys_personalized_with_trace(
            user_table, post_table, trace_table, rec_matrix, max_rec_post_len
        )

    if rtype == "twhin-bert":
        if len(post_table) < 2:
            recsys_log.info("Not enough posts for TWHIN-BERT; returning unchanged matrix.")
            return rec_matrix
        latest_time = post_table[-1]["created_at"]
        second_latest = post_table[-2]["created_at"]
        latest_count = sum(
            1 for p in post_table
            if p["created_at"] in (latest_time, second_latest)
        )
        return rec_sys_personalized_twh(
            user_table, post_table, latest_count,
            trace_table, rec_matrix, max_rec_post_len,
            time_step,
            use_openai_embedding=use_openai_embedding,
        )

    raise ValueError(f"Unknown recsys_type: {recsys_type!r}")
