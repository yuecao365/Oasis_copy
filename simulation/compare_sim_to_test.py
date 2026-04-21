#!/usr/bin/env python3
"""
Compare OASIS Twitter simulation output against ground-truth test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from yaml import safe_load

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _rankdata(vals: List[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(vals))
    ranks = [0.0] * len(vals)
    j = 0
    while j < len(pairs):
        k = j
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        avg_rank = (j + 1 + k) / 2.0
        for t in range(j, k):
            ranks[pairs[t][1]] = avg_rank
        j = k
    return ranks


def spearman(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _normalize(vals: Iterable[float]) -> List[float]:
    arr = [max(0.0, float(v)) for v in vals]
    s = sum(arr)
    if s == 0.0:
        n = len(arr)
        return [1.0 / n] * n if n > 0 else []
    return [v / s for v in arr]


def js_divergence(p: List[float], q: List[float]) -> float:
    p = _normalize(p)
    q = _normalize(q)
    if len(p) != len(q):
        raise ValueError("JS divergence vectors must be same length")
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def _kl(a: List[float], b: List[float]) -> float:
        out = 0.0
        for x, y in zip(a, b):
            if x <= 0:
                continue
            out += x * math.log(x / max(y, 1e-12), 2)
        return out

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def rmse(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def nrmse(a: List[float], b: List[float]) -> float:
    e = rmse(a, b)
    if not b:
        return 0.0
    rng = max(b) - min(b)
    if rng == 0:
        return 0.0
    return e / rng


def topk_overlap(a_scores: Dict[str, float], b_scores: Dict[str, float], k: int) -> float:
    a_top = {u for u, _ in sorted(a_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}
    b_top = {u for u, _ in sorted(b_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}
    if k <= 0:
        return 0.0
    return len(a_top & b_top) / float(k)


def topk_overlap_positive(a_scores: Dict[str, float], b_scores: Dict[str, float], k: int) -> float:
    """
    Top-K overlap on strictly-positive scores only.
    Avoids tie artifacts when one side is all zeros.
    """
    a_pos = {u: v for u, v in a_scores.items() if float(v) > 0.0}
    b_pos = {u: v for u, v in b_scores.items() if float(v) > 0.0}
    return topk_overlap(a_pos, b_pos, k)


def graph_density(nodes: int, edges: int, directed: bool = True) -> float:
    if nodes <= 1:
        return 0.0
    denom = nodes * (nodes - 1) if directed else nodes * (nodes - 1) / 2.0
    return edges / denom


def graph_reciprocity(edge_set: set[Tuple[str, str]]) -> float:
    if not edge_set:
        return 0.0
    mutual = 0
    for a, b in edge_set:
        if (b, a) in edge_set:
            mutual += 1
    return mutual / len(edge_set)


def top_share_curve(values: List[float], cutoffs: List[float]) -> List[float]:
    arr = [max(0.0, float(v)) for v in values]
    n = len(arr)
    if n == 0:
        return [0.0] * len(cutoffs)
    arr_sorted = sorted(arr, reverse=True)
    total = sum(arr_sorted)
    if total <= 0.0:
        return [0.0] * len(cutoffs)
    out = []
    for c in cutoffs:
        frac = max(0.0, min(1.0, float(c)))
        k = max(1, int(math.ceil(frac * n)))
        out.append(sum(arr_sorted[:k]) / total)
    return out


def percentile_rank_histogram(values: List[float], bins: int = 10) -> List[int]:
    vals = [max(0.0, float(v)) for v in values]
    n = len(vals)
    if bins <= 0:
        return []
    if n == 0:
        return [0] * bins

    ranks = _rankdata(vals)
    if n == 1:
        percentiles = [0.0]
    else:
        percentiles = [(r - 1.0) / (n - 1.0) for r in ranks]

    hist = [0] * bins
    for p in percentiles:
        idx = int(math.floor(p * bins))
        idx = max(0, min(bins - 1, idx))
        hist[idx] += 1
    return hist


def load_persona_mapping(personas_csv: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with personas_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            aid = _safe_int(row.get("user_id"), -1)
            uname = (row.get("username") or "").strip().lower()
            if aid >= 0 and uname:
                out[aid] = uname
    return out


def load_test(test_csv: Path, persona_usernames: set[str]) -> Dict:
    per_user_posts = Counter()
    per_user_total_engagement = Counter()
    gt_weekday = [0] * 7
    gt_posts_ts: List[datetime] = []
    likes = []
    retweets = []
    replies = []
    quotes = []
    post_texts: List[str] = []
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uname = (row.get("user_name") or "").strip().lower()
            if uname not in persona_usernames:
                continue
            is_root = row.get("is_root") == "1"
            if is_root:
                per_user_posts[uname] += 1
                ts = _parse_ts(row.get("timestamp", ""))
                if ts is not None:
                    gt_weekday[ts.weekday()] += 1
                    gt_posts_ts.append(ts)
                nlikes = _safe_int(row.get("likes"), 0)
                nretweets = _safe_int(row.get("retweets"), 0)
                nreplies = _safe_int(row.get("replies"), 0)
                nquotes = _safe_int(row.get("quotes"), 0)
                likes.append(nlikes)
                retweets.append(nretweets)
                replies.append(nreplies)
                quotes.append(nquotes)
                per_user_total_engagement[uname] += (nlikes + nretweets + nreplies + nquotes)
                text = (
                    row.get("full_text") or row.get("text") or row.get("content") or ""
                ).strip()
                if text:
                    post_texts.append(text)
    return {
        "per_user_posts": dict(per_user_posts),
        "eng_total_per_user": dict(per_user_total_engagement),
        "weekday_hist": gt_weekday,
        "post_timestamps": gt_posts_ts,
        "likes": likes,
        "retweets": retweets,
        "replies": replies,
        "quotes": quotes,
        "post_texts": post_texts,
    }


def load_sim(db_path: Path, agent_to_username: Dict[int, str], timestep_hours: float) -> Dict:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    sim_posts_by_user = Counter()
    sim_total_engagement_by_user = Counter()
    sim_hour = [0] * 24
    sim_post_steps: List[float] = []
    sim_eng_like = []
    sim_eng_share = []
    sim_eng_reposts = []
    sim_eng_quotes = []
    sim_post_texts: List[str] = []
    sim_user_post_texts: Dict[str, List[str]] = {}

    # Original posts created by simulation (exclude initialization created_at == 0).
    cur.execute(
        """
        SELECT p.post_id, p.user_id, p.created_at, p.num_likes, p.num_shares, p.content
        FROM post p
        WHERE p.original_post_id IS NULL
          AND CAST(p.created_at AS REAL) > 0
        """
    )
    sim_posts = cur.fetchall()
    post_ids = []
    post_user = {}
    post_like = {}
    for post_id, user_id, created_at, nlikes, nshares, content in sim_posts:
        uname = agent_to_username.get(_safe_int(user_id, -1))
        if not uname:
            continue
        pid = _safe_int(post_id, -1)
        sim_posts_by_user[uname] += 1
        step = _safe_float(created_at, 0.0)
        sim_post_steps.append(step)
        hour = int((step * timestep_hours) % 24)
        sim_hour[hour] += 1
        sim_eng_like.append(_safe_int(nlikes, 0))
        sim_eng_share.append(_safe_int(nshares, 0))
        post_ids.append(pid)
        post_user[pid] = uname
        post_like[pid] = _safe_int(nlikes, 0)
        text = (content or "").strip()
        if text:
            sim_post_texts.append(text)
            sim_user_post_texts.setdefault(uname, []).append(text)

    # Comment counts per post.
    sim_eng_replies = []
    repost_count = {}
    quote_count = {}
    if post_ids:
        placeholders = ",".join("?" for _ in post_ids)
        cur.execute(
            f"SELECT post_id, COUNT(*) FROM comment WHERE post_id IN ({placeholders}) GROUP BY post_id",
            post_ids,
        )
        comment_count = {pid: cnt for pid, cnt in cur.fetchall()}

        # Repost and quote counts per original post.
        # Repost rows: child post with original_post_id set and quote_content NULL/empty.
        # Quote rows: child post with original_post_id set and non-empty quote_content.
        cur.execute(
            f"""
            SELECT original_post_id,
                   SUM(CASE WHEN quote_content IS NULL OR TRIM(quote_content) = '' THEN 1 ELSE 0 END) AS reposts,
                   SUM(CASE WHEN quote_content IS NOT NULL AND TRIM(quote_content) <> '' THEN 1 ELSE 0 END) AS quotes
            FROM post
            WHERE original_post_id IN ({placeholders})
            GROUP BY original_post_id
            """,
            post_ids,
        )
        for opid, rcount, qcount in cur.fetchall():
            repost_count[_safe_int(opid, -1)] = _safe_int(rcount, 0)
            quote_count[_safe_int(opid, -1)] = _safe_int(qcount, 0)

        for pid in post_ids:
            nreplies = _safe_int(comment_count.get(pid, 0), 0)
            nreposts = _safe_int(repost_count.get(pid, 0), 0)
            nquotes = _safe_int(quote_count.get(pid, 0), 0)
            sim_eng_replies.append(nreplies)
            sim_eng_reposts.append(nreposts)
            sim_eng_quotes.append(nquotes)
            uname = post_user.get(pid)
            if uname:
                sim_total_engagement_by_user[uname] += (
                    _safe_int(post_like.get(pid, 0), 0) + nreplies + nreposts + nquotes
                )

    # Follow graph in simulation (map to usernames).
    cur.execute(
        """
        SELECT follower_id, followee_id
        FROM follow
        """
    )
    sim_edges = set()
    for follower_id, followee_id in cur.fetchall():
        a = agent_to_username.get(_safe_int(follower_id, -1))
        b = agent_to_username.get(_safe_int(followee_id, -1))
        if a and b and a != b:
            sim_edges.add((a, b))

    con.close()
    return {
        "per_user_posts": dict(sim_posts_by_user),
        "eng_total_per_user": dict(sim_total_engagement_by_user),
        "hour_hist": sim_hour,
        "post_steps": sim_post_steps,
        "likes": sim_eng_like,
        "shares": sim_eng_share,
        "replies": sim_eng_replies,
        "reposts": sim_eng_reposts,
        "quotes": sim_eng_quotes,
        "follow_edges": sim_edges,
        "post_texts": sim_post_texts,
        "user_post_texts": sim_user_post_texts,
    }


def cumulative_curve_from_datetimes(ts_list: List[datetime], bins: int = 24) -> List[float]:
    if not ts_list:
        return [0.0] * bins
    tmin = min(ts_list)
    tmax = max(ts_list)
    span = (tmax - tmin).total_seconds()
    if span <= 0:
        return [float(len(ts_list))] * bins
    hist = [0] * bins
    for t in ts_list:
        pos = int(((t - tmin).total_seconds() / span) * (bins - 1))
        pos = max(0, min(bins - 1, pos))
        hist[pos] += 1
    out = []
    run = 0
    for h in hist:
        run += h
        out.append(float(run))
    return out


def cumulative_curve_from_steps(steps: List[float], bins: int = 24) -> List[float]:
    if not steps:
        return [0.0] * bins
    smin = min(steps)
    smax = max(steps)
    span = smax - smin
    if span <= 0:
        return [float(len(steps))] * bins
    hist = [0] * bins
    for s in steps:
        pos = int(((s - smin) / span) * (bins - 1))
        pos = max(0, min(bins - 1, pos))
        hist[pos] += 1
    out = []
    run = 0
    for h in hist:
        run += h
        out.append(float(run))
    return out


def cumulative_curve_from_day_offsets(day_offsets: List[int], bins: int) -> List[float]:
    if bins <= 0:
        return []
    hist = [0] * bins
    for d in day_offsets:
        pos = max(0, min(bins - 1, int(d)))
        hist[pos] += 1
    out = []
    run = 0
    for h in hist:
        run += h
        out.append(float(run))
    return out


def day_offsets_from_datetimes(ts_list: List[datetime]) -> List[int]:
    if not ts_list:
        return []
    tmin = min(ts_list)
    return [max(0, int((t - tmin).total_seconds() // 86400)) for t in ts_list]


def day_offsets_from_steps(steps: List[float], timestep_hours: float) -> List[int]:
    if not steps:
        return []
    smin = min(steps)
    return [
        max(0, int(math.floor(((s - smin) * timestep_hours) / 24.0)))
        for s in steps
    ]


def weekday_hist_from_steps(
    steps: List[float],
    timestep_hours: float,
    simulation_start: datetime | None,
) -> List[int]:
    hist = [0] * 7
    if not steps or simulation_start is None:
        return hist
    for s in steps:
        dt = simulation_start + timedelta(hours=float(s) * timestep_hours)
        hist[dt.weekday()] += 1
    return hist


def _compute_engagement_source(db_path: Path, agent_to_username: Dict[int, str]) -> Dict:
    """
    Classify each engagement action (like, comment, repost) as organic or recsys-driven.

    Heuristic: if the acting user follows the post author at the time of analysis,
    the engagement is considered organic (from their following list); otherwise it is
    attributed to the recommendation system.
    """
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Build follow set: (follower_id, followee_id)
    cur.execute("SELECT follower_id, followee_id FROM follow")
    follow_set: set[Tuple[int, int]] = {
        (_safe_int(r, -1), _safe_int(f, -1)) for r, f in cur.fetchall()
    }

    # Build post_author map: post_id -> user_id
    cur.execute("SELECT post_id, user_id FROM post WHERE original_post_id IS NULL AND CAST(created_at AS REAL) > 0")
    post_author: Dict[int, int] = {_safe_int(pid, -1): _safe_int(uid, -1) for pid, uid in cur.fetchall()}

    # Collect engagements from likes, comments, and reposts directly.
    # (actor_id, post_author_id) pairs — trace table not required.
    engagement_pairs: List[Tuple[int, int]] = []

    # Likes
    cur.execute('SELECT user_id, post_id FROM "like" WHERE CAST(created_at AS REAL) > 0')
    for uid, pid in cur.fetchall():
        author = post_author.get(_safe_int(pid, -1), -1)
        if author >= 0:
            engagement_pairs.append((_safe_int(uid, -1), author))

    # Comments
    cur.execute("SELECT user_id, post_id FROM comment WHERE CAST(created_at AS REAL) > 0")
    for uid, pid in cur.fetchall():
        author = post_author.get(_safe_int(pid, -1), -1)
        if author >= 0:
            engagement_pairs.append((_safe_int(uid, -1), author))

    # Reposts (original_post_id IS NOT NULL means it's a repost/quote)
    cur.execute("SELECT user_id, original_post_id FROM post WHERE original_post_id IS NOT NULL AND CAST(created_at AS REAL) > 0")
    for uid, orig_pid in cur.fetchall():
        author = post_author.get(_safe_int(orig_pid, -1), -1)
        if author >= 0:
            engagement_pairs.append((_safe_int(uid, -1), author))

    # Fall back to trace table if it exists (legacy OASIS DBs)
    try:
        cur.execute(
            "SELECT user_id, action, info FROM trace "
            "WHERE action IN ('like_post', 'create_comment', 'repost', 'quote_post')"
        )
        for user_id, action, info_str in cur.fetchall():
            actor_id = _safe_int(user_id, -1)
            try:
                info = json.loads(info_str) if info_str else {}
            except Exception:
                continue
            pid = _safe_int(info.get("post_id") or info.get("postId") or -1, -1)
            author = post_author.get(pid, -1)
            if actor_id >= 0 and author >= 0:
                engagement_pairs.append((actor_id, author))
    except Exception:
        pass

    con.close()

    organic = 0
    recsys = 0
    for actor_id, author_id in engagement_pairs:
        if actor_id < 0 or author_id < 0 or actor_id == author_id:
            continue
        if (actor_id, author_id) in follow_set:
            organic += 1
        else:
            recsys += 1

    total = organic + recsys
    return {
        "total_engagements_classified": total,
        "organic_engagements": organic,
        "recsys_engagements": recsys,
        "pct_organic_engagements": round(organic / total, 4) if total > 0 else None,
        "pct_recsys_engagements": round(recsys / total, 4) if total > 0 else None,
        "organic_vs_recsys_ratio": round(organic / recsys, 4) if recsys > 0 else None,
    }


def _compute_semantic_similarity(gt_texts: List[str], sim_texts: List[str]) -> Dict:
    """
    Measure how semantically close the sim corpus is to the GT corpus.
    Uses paraphrase-MiniLM-L6-v2 (GPU-accelerated if available).
    """
    if not _HAS_ST:
        return {"error": "sentence-transformers not installed"}
    if not gt_texts or not sim_texts:
        return {"error": "empty text corpus"}

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

    gt_emb = model.encode(gt_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    sim_emb = model.encode(sim_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    gt_emb = np.array(gt_emb, dtype=np.float32)
    sim_emb = np.array(sim_emb, dtype=np.float32)

    # Centroid similarity
    gt_centroid = gt_emb.mean(axis=0)
    sim_centroid = sim_emb.mean(axis=0)
    centroid_sim = float(np.dot(gt_centroid, sim_centroid) / (
        np.linalg.norm(gt_centroid) * np.linalg.norm(sim_centroid) + 1e-9
    ))

    # Max-match: for each sim post, cosine sim to its nearest GT post
    # sim_emb is already normalized, gt_emb is already normalized → dot product = cosine sim
    chunk = 512  # process in chunks to avoid OOM
    max_sims = []
    for i in range(0, len(sim_emb), chunk):
        block = sim_emb[i:i + chunk]  # (chunk, d)
        sims = block @ gt_emb.T       # (chunk, n_gt)
        max_sims.extend(sims.max(axis=1).tolist())
    mean_max_match = float(np.mean(max_sims))

    return {
        "sim_to_gt_centroid_sim": round(centroid_sim, 4),
        "sim_to_gt_semantic_max_match": round(mean_max_match, 4),
        "sim_post_count": len(sim_texts),
        "gt_post_count": len(gt_texts),
    }


def _compute_repetition(user_post_texts: Dict[str, List[str]]) -> Dict:
    """
    Measure intra-user content repetition in the simulation.
    High similarity = agent is producing near-duplicate posts.
    """
    if not _HAS_ST:
        return {"error": "sentence-transformers not installed"}

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

    # Only consider users with ≥2 posts
    eligible = {u: texts for u, texts in user_post_texts.items() if len(texts) >= 2}
    if not eligible:
        return {"mean_intra_user_similarity": None, "high_repetition_user_rate": None, "near_duplicate_rate": None}

    user_mean_sims = []
    total_pairs = 0
    near_dup_pairs = 0

    for uname, texts in eligible.items():
        emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        emb = np.array(emb, dtype=np.float32)
        sim_matrix = emb @ emb.T  # (n, n) cosine sims
        n = len(texts)
        # Upper triangle only (exclude diagonal)
        pair_sims = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        user_mean_sims.append(float(np.mean(pair_sims)))
        for s in pair_sims:
            total_pairs += 1
            if s > 0.92:
                near_dup_pairs += 1

    mean_intra = float(np.mean(user_mean_sims))
    high_rep_rate = float(sum(1 for s in user_mean_sims if s > 0.85) / len(user_mean_sims))
    near_dup_rate = float(near_dup_pairs / total_pairs) if total_pairs > 0 else 0.0

    return {
        "eligible_users": len(eligible),
        "mean_intra_user_similarity": round(mean_intra, 4),
        "high_repetition_user_rate": round(high_rep_rate, 4),
        "near_duplicate_rate": round(near_dup_rate, 4),
    }


def _slug(v: object) -> str:
    s = str(v)
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return s or "na"


def _build_archive_stem(db_path: Path, timestep_hours: float, sim_config_path: Path) -> str:
    db_stem = _slug(db_path.stem)
    parts = [f"db-{db_stem}", f"ts-{_slug(timestep_hours)}"]

    try:
        cfg = safe_load(sim_config_path.read_text(encoding="utf-8")) or {}
        sim = cfg.get("simulation", {}) or {}
        wanted = {
            "gran": sim.get("activity_granularity"),
            "sel": sim.get("action_selection_mode"),
            "mult": sim.get("agent_activation_multiplier"),
            "min": sim.get("min_activation_threshold"),
            "clock": sim.get("clock_factor"),
            "recsys": sim.get("recsys_type"),
            "posts": sim.get("target_posts_per_step"),
            "posts_auto": sim.get("auto_target_posts_per_step_from_train"),
        }
        for k, v in wanted.items():
            parts.append(f"{k}-{_slug(v)}")
    except Exception:
        parts.append("cfg-unavailable")

    return "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OASIS simulation DB vs test split.")
    parser.add_argument("--db", required=True, type=Path, help="Path to simulation SQLite DB.")
    parser.add_argument(
        "--test_csv",
        default=Path("persona_generation/outputs/test_tweets.csv"),
        type=Path,
    )
    parser.add_argument(
        "--personas_csv",
        default=Path("persona_generation/outputs/user_personas.csv"),
        type=Path,
    )
    parser.add_argument(
        "--filtered_jsonl",
        default=Path("data/processed/filtered_tweets.jsonl"),
        type=Path,
    )
    parser.add_argument(
        "--timestep_hours",
        type=float,
        default=24.0,
        help="How many real hours one simulation timestep represents.",
    )
    parser.add_argument(
        "--split_info",
        default=Path("persona_generation/outputs/split_info.json"),
        type=Path,
        help="Split info JSON to recover simulation start date for weekday metrics.",
    )
    parser.add_argument(
        "--output_json",
        default=Path("simulation/metrics_report.json"),
        type=Path,
    )
    parser.add_argument(
        "--output_md",
        default=Path("simulation/metrics_report.md"),
        type=Path,
    )
    parser.add_argument(
        "--archive_dir",
        default=Path("simulation/metrics_history"),
        type=Path,
        help="Directory to store per-run archived reports with hyperparameter-tagged filenames.",
    )
    parser.add_argument(
        "--sim_config",
        default=Path("simulation/config.yaml"),
        type=Path,
        help="Simulation config path used to build hyperparameter tags in archive filenames.",
    )
    parser.add_argument(
        "--run_tag",
        default="",
        type=str,
        help="Optional suffix tag for archive filenames (e.g., experiment id).",
    )
    args = parser.parse_args()

    agent_to_username = load_persona_mapping(args.personas_csv)
    persona_usernames = set(agent_to_username.values())

    gt = load_test(args.test_csv, persona_usernames)
    sim = load_sim(args.db, agent_to_username, args.timestep_hours)

    sim_start = None
    if args.split_info.exists():
        try:
            split_info = json.loads(args.split_info.read_text(encoding="utf-8"))
            test_start_raw = ((split_info or {}).get("test_period", {}) or {}).get("start")
            if test_start_raw:
                sim_start = datetime.fromisoformat(str(test_start_raw))
        except Exception:
            sim_start = None
    if sim_start is None and gt["post_timestamps"]:
        sim_start = min(gt["post_timestamps"])

    users = sorted(persona_usernames)
    gt_vec = [float(gt["per_user_posts"].get(u, 0.0)) for u in users]
    sim_vec = [float(sim["per_user_posts"].get(u, 0.0)) for u in users]

    activity_spearman = spearman(sim_vec, gt_vec)
    top10_overlap_val = topk_overlap(sim["per_user_posts"], gt["per_user_posts"], 10)

    sim_weekday_hist = weekday_hist_from_steps(sim["post_steps"], args.timestep_hours, sim_start)
    weekday_jsd = js_divergence(sim_weekday_hist, gt["weekday_hist"])

    sim_day_offsets = day_offsets_from_steps(sim["post_steps"], args.timestep_hours)
    gt_day_offsets = day_offsets_from_datetimes(gt["post_timestamps"])
    day_bins = max(
        (max(sim_day_offsets) + 1) if sim_day_offsets else 0,
        (max(gt_day_offsets) + 1) if gt_day_offsets else 0,
        1,
    )
    sim_curve = cumulative_curve_from_day_offsets(sim_day_offsets, bins=day_bins)
    gt_curve = cumulative_curve_from_day_offsets(gt_day_offsets, bins=day_bins)
    curve_rmse = rmse(sim_curve, gt_curve)
    curve_nrmse = nrmse(sim_curve, gt_curve)

    # Post-level engagement: likes top-share curve only.
    engagement_cutoffs = [0.01, 0.05, 0.10, 0.20, 0.50]
    gt_likes_curve = top_share_curve(gt["likes"], engagement_cutoffs)
    sim_likes_curve = top_share_curve(sim["likes"], engagement_cutoffs)
    likes_top_share_rmse = rmse(gt_likes_curve, sim_likes_curve)

    # User-level engagement: Spearman correlation only.
    gt_eng_total_map = {u: float(gt["eng_total_per_user"].get(u, 0.0)) for u in users}
    sim_eng_total_map = {u: float(sim["eng_total_per_user"].get(u, 0.0)) for u in users}
    gt_eng_total_vec = [gt_eng_total_map[u] for u in users]
    sim_eng_total_vec = [sim_eng_total_map[u] for u in users]
    eng_spearman = spearman(sim_eng_total_vec, gt_eng_total_vec)

    # Content-fit metrics.
    print("Computing engagement source (recsys vs organic)...")
    engagement_source = _compute_engagement_source(args.db, agent_to_username)
    print("Computing semantic similarity (sim vs GT)...")
    semantic = _compute_semantic_similarity(gt["post_texts"], sim["post_texts"])
    print("Computing intra-user repetition...")
    repetition = _compute_repetition(sim["user_post_texts"])

    report = {
        "inputs": {
            "db": str(args.db),
            "test_csv": str(args.test_csv),
            "personas_csv": str(args.personas_csv),
            "filtered_jsonl": str(args.filtered_jsonl),
            "timestep_hours": args.timestep_hours,
            "split_info": str(args.split_info),
        },
        "coverage": {
            "persona_user_count": len(users),
            "gt_active_users": len([u for u in users if gt["per_user_posts"].get(u, 0) > 0]),
            "sim_active_users": len([u for u in users if sim["per_user_posts"].get(u, 0) > 0]),
        },
        "user_activity_fit": {
            "spearman_posts_per_user": activity_spearman,
            "top10_overlap": top10_overlap_val,
        },
        "temporal_fit": {
            "day_of_week_jsd": weekday_jsd,
            "cumulative_posts_daily_rmse": curve_rmse,
            "cumulative_posts_daily_nrmse": curve_nrmse,
        },
        "engagement_fit": {
            "gt_total_posts": len(gt["post_timestamps"]),
            "sim_total_posts": len(sim["post_steps"]),
            "likes_top_share_curve_rmse": likes_top_share_rmse,
            "spearman_total_engagement_per_user": eng_spearman,
        },
        "content_fit": {
            "engagement_source": engagement_source,
            "semantic_similarity": semantic,
            "repetition": repetition,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _fmt(v, fmt=".4f"):
        if v is None:
            return "N/A"
        try:
            return format(float(v), fmt)
        except Exception:
            return str(v)

    es = report["content_fit"]["engagement_source"]
    sem = report["content_fit"]["semantic_similarity"]
    rep = report["content_fit"]["repetition"]

    md_lines = [
        "# Simulation vs Test Metrics",
        "",
        f"- DB: `{args.db}`",
        f"- Test CSV: `{args.test_csv}`",
        f"- Personas CSV: `{args.personas_csv}`",
        "",
        "## Coverage",
        f"- Persona users: {report['coverage']['persona_user_count']}",
        f"- GT active users: {report['coverage']['gt_active_users']}",
        f"- Sim active users: {report['coverage']['sim_active_users']}",
        "",
        "## User Activity Fit",
        f"- Spearman (posts/user): {_fmt(activity_spearman)}",
        f"- Top-10 overlap: {_fmt(top10_overlap_val)}",
        "",
        "## Temporal Fit",
        f"- Day-of-week JSD: {_fmt(weekday_jsd)}",
        f"- Cumulative daily posts RMSE: {_fmt(curve_rmse)}",
        f"- Cumulative daily posts NRMSE: {_fmt(curve_nrmse)}",
        "",
        "## Engagement Fit",
        f"- GT total posts: {report['engagement_fit']['gt_total_posts']}",
        f"- Sim total posts: {report['engagement_fit']['sim_total_posts']}",
        f"- Likes top-share curve RMSE: {_fmt(likes_top_share_rmse)}",
        f"- Spearman (total engagement/user): {_fmt(eng_spearman)}",
        "",
        "## Content Fit",
        "",
        "### Engagement Source (Recsys vs Organic)",
        f"- Total engagements classified: {es.get('total_engagements_classified', 'N/A')}",
        f"- Organic engagements: {es.get('organic_engagements', 'N/A')} "
        f"({_fmt(es.get('pct_organic_engagements'))} of total)",
        f"- Recsys-driven engagements: {es.get('recsys_engagements', 'N/A')} "
        f"({_fmt(es.get('pct_recsys_engagements'))} of total)",
        f"- Organic / Recsys ratio: {_fmt(es.get('organic_vs_recsys_ratio'))}",
        "",
        "### Semantic Similarity (Sim vs GT)",
    ]
    if "error" in sem:
        md_lines.append(f"- Error: {sem['error']}")
    else:
        md_lines += [
            f"- Sim posts embedded: {sem.get('sim_post_count', 'N/A')}",
            f"- GT posts embedded: {sem.get('gt_post_count', 'N/A')}",
            f"- Centroid similarity: {_fmt(sem.get('sim_to_gt_centroid_sim'))}",
            f"- Mean max-match similarity: {_fmt(sem.get('sim_to_gt_semantic_max_match'))}",
        ]
    md_lines += [
        "",
        "### Intra-User Repetition",
    ]
    if "error" in rep:
        md_lines.append(f"- Error: {rep['error']}")
    else:
        md_lines += [
            f"- Eligible users (≥2 posts): {rep.get('eligible_users', 'N/A')}",
            f"- Mean intra-user similarity: {_fmt(rep.get('mean_intra_user_similarity'))}",
            f"- High-repetition user rate (>0.85): {_fmt(rep.get('high_repetition_user_rate'))}",
            f"- Near-duplicate pair rate (>0.92): {_fmt(rep.get('near_duplicate_rate'))}",
        ]
    md_lines.append("")
    args.output_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Also write a historical, hyperparameter-labeled artifact for comparison over time.
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    archive_stem = _build_archive_stem(args.db, args.timestep_hours, args.sim_config)
    if args.run_tag:
        archive_stem = f"{archive_stem}__tag-{_slug(args.run_tag)}"
    archive_json = args.archive_dir / f"{archive_stem}.json"
    archive_md = args.archive_dir / f"{archive_stem}.md"
    archive_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    archive_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {archive_json}")
    print(f"Wrote {archive_md}")


if __name__ == "__main__":
    main()
