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
    return {
        "per_user_posts": dict(per_user_posts),
        "eng_total_per_user": dict(per_user_total_engagement),
        "weekday_hist": gt_weekday,
        "post_timestamps": gt_posts_ts,
        "likes": likes,
        "retweets": retweets,
        "replies": replies,
        "quotes": quotes,
    }


def _load_synthetic_ids(cur) -> List[int]:
    """Return user_ids of synthetic agents; empty list if the table doesn't exist (e.g. baseline DB)."""
    try:
        cur.execute("SELECT user_id FROM synthetic_agent")
        return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


def load_sim(db_path: Path, agent_to_username: Dict[int, str], timestep_hours: float) -> Dict:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Synthetic agent IDs are excluded from all engagement counts so their
    # actions don't inflate fidelity metrics.
    synthetic_ids = _load_synthetic_ids(cur)
    synth_ph = ",".join("?" for _ in synthetic_ids) if synthetic_ids else None

    sim_posts_by_user = Counter()
    sim_total_engagement_by_user = Counter()
    sim_hour = [0] * 24
    sim_post_steps: List[float] = []
    sim_eng_like = []
    sim_eng_share = []
    sim_eng_reposts = []
    sim_eng_quotes = []

    # Original posts by persona users created during simulation (created_at == 0 are seed rows).
    cur.execute(
        """
        SELECT p.post_id, p.user_id, p.created_at, p.num_shares
        FROM post p
        WHERE p.original_post_id IS NULL
          AND CAST(p.created_at AS REAL) > 0
        """
    )
    sim_posts = cur.fetchall()
    post_ids = []
    post_user = {}
    for post_id, user_id, created_at, nshares in sim_posts:
        uname = agent_to_username.get(_safe_int(user_id, -1))
        if not uname:
            continue
        pid = _safe_int(post_id, -1)
        sim_posts_by_user[uname] += 1
        step = _safe_float(created_at, 0.0)
        sim_post_steps.append(step)
        hour = int((step * timestep_hours) % 24)
        sim_hour[hour] += 1
        sim_eng_share.append(_safe_int(nshares, 0))
        post_ids.append(pid)
        post_user[pid] = uname

    # Engagement counts per persona post — all excluding synthetic agents.
    sim_eng_replies = []
    repost_count = {}
    quote_count = {}
    like_count: Dict[int, int] = {}
    if post_ids:
        pid_ph = ",".join("?" for _ in post_ids)

        # Likes from the like table (not num_likes, which includes synthetic agent likes).
        if synth_ph:
            cur.execute(
                f'SELECT post_id, COUNT(*) FROM "like" '
                f'WHERE post_id IN ({pid_ph}) AND user_id NOT IN ({synth_ph}) GROUP BY post_id',
                post_ids + synthetic_ids,
            )
        else:
            cur.execute(
                f'SELECT post_id, COUNT(*) FROM "like" WHERE post_id IN ({pid_ph}) GROUP BY post_id',
                post_ids,
            )
        like_count = {pid: cnt for pid, cnt in cur.fetchall()}

        # Comments excluding synthetic agents.
        if synth_ph:
            cur.execute(
                f"SELECT post_id, COUNT(*) FROM comment "
                f"WHERE post_id IN ({pid_ph}) AND user_id NOT IN ({synth_ph}) GROUP BY post_id",
                post_ids + synthetic_ids,
            )
        else:
            cur.execute(
                f"SELECT post_id, COUNT(*) FROM comment WHERE post_id IN ({pid_ph}) GROUP BY post_id",
                post_ids,
            )
        comment_count = {pid: cnt for pid, cnt in cur.fetchall()}

        # Reposts and quotes excluding synthetic agents.
        if synth_ph:
            cur.execute(
                f"""
                SELECT original_post_id,
                       SUM(CASE WHEN quote_content IS NULL OR TRIM(quote_content) = '' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN quote_content IS NOT NULL AND TRIM(quote_content) <> '' THEN 1 ELSE 0 END)
                FROM post
                WHERE original_post_id IN ({pid_ph}) AND user_id NOT IN ({synth_ph})
                GROUP BY original_post_id
                """,
                post_ids + synthetic_ids,
            )
        else:
            cur.execute(
                f"""
                SELECT original_post_id,
                       SUM(CASE WHEN quote_content IS NULL OR TRIM(quote_content) = '' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN quote_content IS NOT NULL AND TRIM(quote_content) <> '' THEN 1 ELSE 0 END)
                FROM post
                WHERE original_post_id IN ({pid_ph})
                GROUP BY original_post_id
                """,
                post_ids,
            )
        for opid, rcount, qcount in cur.fetchall():
            repost_count[_safe_int(opid, -1)] = _safe_int(rcount, 0)
            quote_count[_safe_int(opid, -1)] = _safe_int(qcount, 0)

        for pid in post_ids:
            nlikes = _safe_int(like_count.get(pid, 0), 0)
            nreplies = _safe_int(comment_count.get(pid, 0), 0)
            nreposts = _safe_int(repost_count.get(pid, 0), 0)
            nquotes = _safe_int(quote_count.get(pid, 0), 0)
            sim_eng_like.append(nlikes)
            sim_eng_replies.append(nreplies)
            sim_eng_reposts.append(nreposts)
            sim_eng_quotes.append(nquotes)
            uname = post_user.get(pid)
            if uname:
                sim_total_engagement_by_user[uname] += nlikes + nreplies + nreposts + nquotes

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


def _compute_official_post_engagement(
    test_csv: Path,
    db_path: Path,
    official_accounts: List[str],
) -> Dict:
    """
    Compare community engagement with official account posts in GT vs simulation.

    GT engagement: likes + retweets + replies + quotes on each official root post
      (from the test CSV columns).
    Sim engagement: likes + comments + reposts/quotes recorded in the DB against
      each official post (identified via the official_user table).

    Returns per-post vectors and summary statistics.
    """
    official_lower = {u.lower() for u in official_accounts}
    if not official_lower:
        return {"error": "no official accounts specified"}

    # --- Ground-truth engagement per official post ---
    gt_engagements: List[float] = []
    gt_posts: List[Dict] = []
    try:
        with test_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                uname = (row.get("user_name") or "").strip().lower()
                if uname not in official_lower:
                    continue
                if row.get("is_root") != "1":
                    continue
                eng = (
                    _safe_int(row.get("likes"), 0)
                    + _safe_int(row.get("retweets"), 0)
                    + _safe_int(row.get("replies"), 0)
                    + _safe_int(row.get("quotes"), 0)
                )
                gt_engagements.append(float(eng))
                gt_posts.append({
                    "user": uname,
                    "timestamp": row.get("timestamp", ""),
                    "gt_engagement": eng,
                    "text_snippet": (row.get("text") or "")[:60],
                })
    except Exception as e:
        return {"error": f"failed to load GT: {e}"}

    if not gt_posts:
        return {"error": "no official root posts found in test CSV"}

    # --- Simulation engagement per official post ---
    sim_engagements: List[float] = []
    sim_total_by_post: Dict[int, int] = {}
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()

        # Get official user IDs from the dedicated table
        cur.execute("SELECT user_id FROM official_user")
        official_db_user_ids = [r[0] for r in cur.fetchall()]

        if not official_db_user_ids:
            con.close()
            return {
                "error": "official_user table empty — run simulation with official_accounts configured",
                "gt_post_count": len(gt_posts),
                "gt_total_engagement": sum(gt_engagements),
            }

        uid_placeholders = ",".join("?" for _ in official_db_user_ids)

        # Get all official posts (created during simulation, not seed)
        cur.execute(
            f"SELECT post_id FROM post "
            f"WHERE user_id IN ({uid_placeholders}) AND CAST(created_at AS REAL) > 0",
            official_db_user_ids,
        )
        official_post_ids = [r[0] for r in cur.fetchall()]

        if official_post_ids:
            pid_placeholders = ",".join("?" for _ in official_post_ids)

            # Exclude synthetic agents from community engagement counts.
            synthetic_ids = _load_synthetic_ids(cur)
            synth_ph = ",".join("?" for _ in synthetic_ids) if synthetic_ids else None

            # Likes (excluding synthetic agents)
            if synth_ph:
                cur.execute(
                    f'SELECT post_id, COUNT(*) FROM "like" '
                    f'WHERE post_id IN ({pid_placeholders}) AND user_id NOT IN ({synth_ph}) GROUP BY post_id',
                    official_post_ids + synthetic_ids,
                )
            else:
                cur.execute(
                    f'SELECT post_id, COUNT(*) FROM "like" WHERE post_id IN ({pid_placeholders}) GROUP BY post_id',
                    official_post_ids,
                )
            likes_by_pid = {r[0]: r[1] for r in cur.fetchall()}

            # Comments (excluding synthetic agents)
            if synth_ph:
                cur.execute(
                    f"SELECT post_id, COUNT(*) FROM comment "
                    f"WHERE post_id IN ({pid_placeholders}) AND user_id NOT IN ({synth_ph}) GROUP BY post_id",
                    official_post_ids + synthetic_ids,
                )
            else:
                cur.execute(
                    f"SELECT post_id, COUNT(*) FROM comment WHERE post_id IN ({pid_placeholders}) GROUP BY post_id",
                    official_post_ids,
                )
            comments_by_pid = {r[0]: r[1] for r in cur.fetchall()}

            # Reposts + quotes (excluding synthetic agents)
            if synth_ph:
                cur.execute(
                    f"SELECT original_post_id, COUNT(*) FROM post "
                    f"WHERE original_post_id IN ({pid_placeholders}) AND user_id NOT IN ({synth_ph}) GROUP BY original_post_id",
                    official_post_ids + synthetic_ids,
                )
            else:
                cur.execute(
                    f"SELECT original_post_id, COUNT(*) FROM post "
                    f"WHERE original_post_id IN ({pid_placeholders}) GROUP BY original_post_id",
                    official_post_ids,
                )
            shares_by_pid = {r[0]: r[1] for r in cur.fetchall()}

            for pid in official_post_ids:
                total = (
                    _safe_int(likes_by_pid.get(pid, 0), 0)
                    + _safe_int(comments_by_pid.get(pid, 0), 0)
                    + _safe_int(shares_by_pid.get(pid, 0), 0)
                )
                sim_total_by_post[pid] = total
                sim_engagements.append(float(total))

        con.close()
    except Exception as e:
        return {"error": f"failed to query sim DB: {e}", "gt_post_count": len(gt_posts)}

    gt_total = sum(gt_engagements)
    sim_total = sum(sim_engagements)
    gt_per_post_mean = gt_total / len(gt_posts) if gt_posts else 0.0
    sim_per_post_mean = sim_total / len(sim_engagements) if sim_engagements else 0.0

    # Rank correlation — only meaningful when lengths match (both vectors represent posts)
    # GT has one entry per official root post; sim has one entry per injected official post.
    # Lengths may differ if some posts fall outside the simulated window.
    rho: float | None = None
    if len(gt_engagements) >= 2 and len(sim_engagements) >= 2:
        # Pad shorter vector with zeros for comparison
        n = max(len(gt_engagements), len(sim_engagements))
        gt_pad = gt_engagements + [0.0] * (n - len(gt_engagements))
        sim_pad = sim_engagements + [0.0] * (n - len(sim_engagements))
        rho = round(spearman(sim_pad, gt_pad), 4)

    return {
        "gt_official_post_count": len(gt_posts),
        "sim_official_post_count": len(sim_engagements),
        "gt_total_engagement": gt_total,
        "sim_total_engagement": sim_total,
        "gt_mean_engagement_per_post": round(gt_per_post_mean, 2),
        "sim_mean_engagement_per_post": round(sim_per_post_mean, 2),
        "spearman_engagement_per_post": rho,
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


def _compute_community_health(
    db_path: Path,
    persona_usernames: set,
    official_accounts: List[str],
) -> Dict:
    """
    Absolute community-health metrics — no ground-truth comparison needed.
    Measures the intrinsic quality of engagement within the simulated community.
    Designed to show improvement when synthetic engagement agents are added.

    All metrics are scoped to persona users (from the persona CSV) excluding
    official accounts.  Engagement on persona posts may come from anyone
    (including synthetic agents).
    """
    official_lower = {a.lower() for a in official_accounts if a}
    persona_lower = {u.lower() for u in persona_usernames}

    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()

        # Map username → user_id for everyone in DB
        cur.execute("SELECT user_id, user_name FROM user")
        all_db_users = {row[1].lower(): row[0] for row in cur.fetchall()}

        # Official user IDs (excluded from community metrics)
        cur.execute("SELECT user_id FROM official_user")
        official_ids = {r[0] for r in cur.fetchall()}

        # Persona user IDs: in persona CSV and not official
        persona_user_ids = [
            uid for uname, uid in all_db_users.items()
            if uname in persona_lower and uid not in official_ids
        ]
        total_persona_users = len(persona_user_ids)

        if total_persona_users == 0:
            con.close()
            return {"error": "no persona users found in DB"}

        uid_ph = ",".join("?" for _ in persona_user_ids)

        # Community posts: original posts by persona users created during simulation
        cur.execute(
            f"SELECT post_id FROM post "
            f"WHERE user_id IN ({uid_ph}) "
            f"AND original_post_id IS NULL "
            f"AND CAST(created_at AS REAL) > 0",
            persona_user_ids,
        )
        community_post_ids = [r[0] for r in cur.fetchall()]
        total_community_posts = len(community_post_ids)

        # Synthetic agents are excluded from all engagement counts.
        synthetic_ids = _load_synthetic_ids(cur)
        synth_ph = ",".join("?" for _ in synthetic_ids) if synthetic_ids else None

        # Per-post engagement tallies — actions by synthetic agents excluded
        posts_with_reply = 0
        posts_with_zero_engagement = 0
        total_likes = 0
        total_comments = 0
        total_reposts = 0

        if community_post_ids:
            pid_ph = ",".join("?" for _ in community_post_ids)

            if synth_ph:
                cur.execute(
                    f'SELECT post_id, COUNT(*) FROM "like" '
                    f'WHERE post_id IN ({pid_ph}) AND user_id NOT IN ({synth_ph}) GROUP BY post_id',
                    community_post_ids + synthetic_ids,
                )
            else:
                cur.execute(
                    f'SELECT post_id, COUNT(*) FROM "like" WHERE post_id IN ({pid_ph}) GROUP BY post_id',
                    community_post_ids,
                )
            likes_by_pid = {r[0]: r[1] for r in cur.fetchall()}

            if synth_ph:
                cur.execute(
                    f"SELECT post_id, COUNT(*) FROM comment "
                    f"WHERE post_id IN ({pid_ph}) AND user_id NOT IN ({synth_ph}) GROUP BY post_id",
                    community_post_ids + synthetic_ids,
                )
            else:
                cur.execute(
                    f"SELECT post_id, COUNT(*) FROM comment WHERE post_id IN ({pid_ph}) GROUP BY post_id",
                    community_post_ids,
                )
            comments_by_pid = {r[0]: r[1] for r in cur.fetchall()}

            if synth_ph:
                cur.execute(
                    f"SELECT original_post_id, COUNT(*) FROM post "
                    f"WHERE original_post_id IN ({pid_ph}) AND user_id NOT IN ({synth_ph}) GROUP BY original_post_id",
                    community_post_ids + synthetic_ids,
                )
            else:
                cur.execute(
                    f"SELECT original_post_id, COUNT(*) FROM post "
                    f"WHERE original_post_id IN ({pid_ph}) GROUP BY original_post_id",
                    community_post_ids,
                )
            reposts_by_pid = {r[0]: r[1] for r in cur.fetchall()}

            for pid in community_post_ids:
                l = likes_by_pid.get(pid, 0)
                c = comments_by_pid.get(pid, 0)
                r = reposts_by_pid.get(pid, 0)
                total_likes += l
                total_comments += c
                total_reposts += r
                if c > 0:
                    posts_with_reply += 1
                if l == 0 and c == 0 and r == 0:
                    posts_with_zero_engagement += 1

        reply_rate = posts_with_reply / total_community_posts if total_community_posts else 0.0
        unanswered_rate = posts_with_zero_engagement / total_community_posts if total_community_posts else 0.0
        total_community_engagement = total_likes + total_comments + total_reposts
        avg_engagement_per_post = total_community_engagement / total_community_posts if total_community_posts else 0.0

        # User participation rate: persona users (not synthetic) who took any action
        active_ids: set = set()

        cur.execute(
            f"SELECT DISTINCT user_id FROM post WHERE user_id IN ({uid_ph}) AND CAST(created_at AS REAL) > 0",
            persona_user_ids,
        )
        active_ids.update(r[0] for r in cur.fetchall())

        cur.execute(
            f'SELECT DISTINCT user_id FROM "like" WHERE user_id IN ({uid_ph})',
            persona_user_ids,
        )
        active_ids.update(r[0] for r in cur.fetchall())

        cur.execute(
            f"SELECT DISTINCT user_id FROM comment WHERE user_id IN ({uid_ph})",
            persona_user_ids,
        )
        active_ids.update(r[0] for r in cur.fetchall())

        cur.execute(
            f"SELECT DISTINCT follower_id FROM follow WHERE follower_id IN ({uid_ph})",
            persona_user_ids,
        )
        active_ids.update(r[0] for r in cur.fetchall())

        user_participation_rate = len(active_ids) / total_persona_users

        # Peripheral user activation: bottom-50% of persona users by post count
        cur.execute(
            f"SELECT user_id, COUNT(*) FROM post "
            f"WHERE user_id IN ({uid_ph}) AND original_post_id IS NULL AND CAST(created_at AS REAL) > 0 "
            f"GROUP BY user_id",
            persona_user_ids,
        )
        posts_by_uid = {r[0]: r[1] for r in cur.fetchall()}
        ranked = sorted(persona_user_ids, key=lambda uid: posts_by_uid.get(uid, 0))
        n_peripheral = max(1, len(ranked) // 2)
        peripheral_ids = set(ranked[:n_peripheral])
        peripheral_activation_rate = len(peripheral_ids & active_ids) / len(peripheral_ids)

        # Follow growth per timestep
        cur.execute("SELECT COUNT(*) FROM follow WHERE CAST(created_at AS REAL) > 0")
        total_new_follows = cur.fetchone()[0]
        cur.execute("SELECT MAX(timestep) FROM observation")
        row = cur.fetchone()
        max_timestep = (row[0] or 0) if row else 0
        follow_growth_per_timestep = total_new_follows / max_timestep if max_timestep > 0 else 0.0

        con.close()

        return {
            "total_persona_users": total_persona_users,
            "total_community_posts": total_community_posts,
            "total_community_engagement": total_community_engagement,
            "avg_engagement_per_community_post": round(avg_engagement_per_post, 4),
            "reply_rate": round(reply_rate, 4),
            "unanswered_post_rate": round(unanswered_rate, 4),
            "user_participation_rate": round(user_participation_rate, 4),
            "peripheral_user_activation_rate": round(peripheral_activation_rate, 4),
            "follow_growth_per_timestep": round(follow_growth_per_timestep, 4),
        }

    except Exception as e:
        return {"error": f"community health compute failed: {e}"}


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
    parser.add_argument(
        "--official_accounts",
        nargs="*",
        default=["vechainofficial"],
        help="Usernames of official accounts whose post engagement is measured separately "
             "(default: ['vechainofficial']). Pass empty string to disable.",
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
    activity_jsd = js_divergence(sim_vec, gt_vec)
    top10_overlap = topk_overlap(sim["per_user_posts"], gt["per_user_posts"], 10)
    top20_overlap = topk_overlap(sim["per_user_posts"], gt["per_user_posts"], 20)

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

    engagement_cutoffs = [0.01, 0.05, 0.10, 0.20, 0.50]
    engagement_pairs = {
        "likes": (gt["likes"], sim["likes"]),
        "retweets_reposts": (gt["retweets"], sim["reposts"]),
        "replies": (gt["replies"], sim["replies"]),
        "quotes": (gt["quotes"], sim["quotes"]),
    }
    engagement_concentration = {}
    engagement_percentile_jsd = {}
    for metric_name, (gt_vals, sim_vals) in engagement_pairs.items():
        gt_curve_top = top_share_curve(gt_vals, engagement_cutoffs)
        sim_curve_top = top_share_curve(sim_vals, engagement_cutoffs)
        engagement_concentration[metric_name] = {
            "cutoffs": engagement_cutoffs,
            "gt_top_share_curve": gt_curve_top,
            "sim_top_share_curve": sim_curve_top,
            "top_share_curve_rmse": rmse(gt_curve_top, sim_curve_top),
        }

        gt_hist = percentile_rank_histogram(gt_vals, bins=10)
        sim_hist = percentile_rank_histogram(sim_vals, bins=10)
        engagement_percentile_jsd[metric_name] = {
            "bins": 10,
            "gt_percentile_hist": gt_hist,
            "sim_percentile_hist": sim_hist,
            "percentile_hist_jsd": js_divergence(gt_hist, sim_hist),
        }

    # User-level engagement ranking fit (additive to existing engagement diagnostics).
    # Components used: likes + replies + reposts + quotes.
    gt_eng_total_map = {u: float(gt["eng_total_per_user"].get(u, 0.0)) for u in users}
    sim_eng_total_map = {u: float(sim["eng_total_per_user"].get(u, 0.0)) for u in users}
    gt_eng_avg_map = {
        u: (
            gt_eng_total_map[u] / float(gt["per_user_posts"].get(u, 0))
            if gt["per_user_posts"].get(u, 0) > 0 else 0.0
        )
        for u in users
    }
    sim_eng_avg_map = {
        u: (
            sim_eng_total_map[u] / float(sim["per_user_posts"].get(u, 0))
            if sim["per_user_posts"].get(u, 0) > 0 else 0.0
        )
        for u in users
    }

    gt_eng_total_vec = [gt_eng_total_map[u] for u in users]
    sim_eng_total_vec = [sim_eng_total_map[u] for u in users]
    gt_eng_avg_vec = [gt_eng_avg_map[u] for u in users]
    sim_eng_avg_vec = [sim_eng_avg_map[u] for u in users]

    gt_top_shares = top_share_curve(gt_eng_total_vec, [0.10, 0.20])
    sim_top_shares = top_share_curve(sim_eng_total_vec, [0.10, 0.20])
    engagement_user_level = {
        "scope": "user",
        "components": ["likes", "replies", "reposts", "quotes"],
        "topk_values": [10, 20],
        "gt_total_engagement": float(sum(gt_eng_total_vec)),
        "sim_total_engagement": float(sum(sim_eng_total_vec)),
        "spearman_total_engagement_per_user": spearman(sim_eng_total_vec, gt_eng_total_vec),
        "jsd_total_engagement_per_user": js_divergence(sim_eng_total_vec, gt_eng_total_vec),
        "spearman_engagement_per_post_per_user": spearman(sim_eng_avg_vec, gt_eng_avg_vec),
        "jsd_engagement_per_post_per_user": js_divergence(sim_eng_avg_vec, gt_eng_avg_vec),
        "topk_overlap_total_engagement_per_user": {
            "top10": topk_overlap_positive(sim_eng_total_map, gt_eng_total_map, 10),
            "top20": topk_overlap_positive(sim_eng_total_map, gt_eng_total_map, 20),
        },
        "topk_overlap_engagement_per_post_per_user": {
            "top10": topk_overlap_positive(sim_eng_avg_map, gt_eng_avg_map, 10),
            "top20": topk_overlap_positive(sim_eng_avg_map, gt_eng_avg_map, 20),
        },
        "gt_top10pct_share_total_engagement": gt_top_shares[0] if gt_top_shares else 0.0,
        "sim_top10pct_share_total_engagement": sim_top_shares[0] if sim_top_shares else 0.0,
        "gt_top20pct_share_total_engagement": gt_top_shares[1] if len(gt_top_shares) > 1 else 0.0,
        "sim_top20pct_share_total_engagement": sim_top_shares[1] if len(sim_top_shares) > 1 else 0.0,
    }

    # Official account post engagement.
    official_accounts_list = [a for a in (args.official_accounts or []) if a]
    print(f"Computing official post engagement ({official_accounts_list})...")
    official_engagement = _compute_official_post_engagement(
        test_csv=args.test_csv,
        db_path=args.db,
        official_accounts=official_accounts_list,
    )

    print("Computing community health metrics...")
    community_health = _compute_community_health(
        db_path=args.db,
        persona_usernames=persona_usernames,
        official_accounts=official_accounts_list,
    )

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
            "jsd_posts_per_user": activity_jsd,
            "top10_overlap": top10_overlap,
            "top20_overlap": top20_overlap,
        },
        "temporal_fit": {
            "day_of_week_jsd": weekday_jsd,
            "cumulative_posts_daily_rmse": curve_rmse,
            "cumulative_posts_daily_nrmse": curve_nrmse,
        },
        "engagement_fit": {
            "gt_total_posts": len(gt["post_timestamps"]),
            "sim_total_posts": len(sim["post_steps"]),
            "post_level": {
                "top_share_curve": engagement_concentration,
                "percentile_rank_jsd": engagement_percentile_jsd,
            },
            "user_level": engagement_user_level,
        },
        "official_post_engagement": official_engagement,
        "community_health": community_health,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

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
        f"- Spearman (posts/user): {activity_spearman:.4f}",
        f"- JSD (posts/user): {activity_jsd:.4f}",
        f"- Top-10 overlap: {top10_overlap:.4f}",
        f"- Top-20 overlap: {top20_overlap:.4f}",
        "",
        "## Temporal Fit",
        f"- Day-of-week JSD: {weekday_jsd:.4f}",
        f"- Cumulative daily posts RMSE: {curve_rmse:.4f}",
        f"- Cumulative daily posts NRMSE: {curve_nrmse:.4f}",
        "",
        "## Engagement Fit",
        f"- GT total posts: {report['engagement_fit']['gt_total_posts']}",
        f"- Sim total posts: {report['engagement_fit']['sim_total_posts']}",
        "",
        "### User-level",
        f"- Components: {', '.join(report['engagement_fit']['user_level']['components'])}",
        f"- Top-K values: {', '.join(str(k) for k in report['engagement_fit']['user_level']['topk_values'])}",
        f"- Spearman (total engagement/user): "
        f"{report['engagement_fit']['user_level']['spearman_total_engagement_per_user']:.4f}",
        f"- JSD (total engagement/user): "
        f"{report['engagement_fit']['user_level']['jsd_total_engagement_per_user']:.4f}",
        f"- Spearman (engagement-per-post per user): "
        f"{report['engagement_fit']['user_level']['spearman_engagement_per_post_per_user']:.4f}",
        f"- JSD (engagement-per-post per user): "
        f"{report['engagement_fit']['user_level']['jsd_engagement_per_post_per_user']:.4f}",
        f"- Top-10 overlap (total engagement/user): "
        f"{report['engagement_fit']['user_level']['topk_overlap_total_engagement_per_user']['top10']:.4f}",
        f"- Top-20 overlap (total engagement/user): "
        f"{report['engagement_fit']['user_level']['topk_overlap_total_engagement_per_user']['top20']:.4f}",
        f"- Top-10 overlap (engagement-per-post/user): "
        f"{report['engagement_fit']['user_level']['topk_overlap_engagement_per_post_per_user']['top10']:.4f}",
        f"- Top-20 overlap (engagement-per-post/user): "
        f"{report['engagement_fit']['user_level']['topk_overlap_engagement_per_post_per_user']['top20']:.4f}",
        f"- GT/Sim top-10% share of total engagement: "
        f"{report['engagement_fit']['user_level']['gt_top10pct_share_total_engagement']:.4f} / "
        f"{report['engagement_fit']['user_level']['sim_top10pct_share_total_engagement']:.4f}",
        f"- GT/Sim top-20% share of total engagement: "
        f"{report['engagement_fit']['user_level']['gt_top20pct_share_total_engagement']:.4f} / "
        f"{report['engagement_fit']['user_level']['sim_top20pct_share_total_engagement']:.4f}",
        "",
        "### Post-level",
        f"- Top-share curve RMSE (likes): "
        f"{report['engagement_fit']['post_level']['top_share_curve']['likes']['top_share_curve_rmse']:.4f}",
        f"- Top-share curve RMSE (retweets vs reposts): "
        f"{report['engagement_fit']['post_level']['top_share_curve']['retweets_reposts']['top_share_curve_rmse']:.4f}",
        f"- Top-share curve RMSE (replies): "
        f"{report['engagement_fit']['post_level']['top_share_curve']['replies']['top_share_curve_rmse']:.4f}",
        f"- Top-share curve RMSE (quotes): "
        f"{report['engagement_fit']['post_level']['top_share_curve']['quotes']['top_share_curve_rmse']:.4f}",
        f"- Percentile-rank JSD (likes): "
        f"{report['engagement_fit']['post_level']['percentile_rank_jsd']['likes']['percentile_hist_jsd']:.4f}",
        f"- Percentile-rank JSD (retweets vs reposts): "
        f"{report['engagement_fit']['post_level']['percentile_rank_jsd']['retweets_reposts']['percentile_hist_jsd']:.4f}",
        f"- Percentile-rank JSD (replies): "
        f"{report['engagement_fit']['post_level']['percentile_rank_jsd']['replies']['percentile_hist_jsd']:.4f}",
        f"- Percentile-rank JSD (quotes): "
        f"{report['engagement_fit']['post_level']['percentile_rank_jsd']['quotes']['percentile_hist_jsd']:.4f}",
        "",
    ]

    def _fmtv(v):
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    oe = report.get("official_post_engagement", {})
    md_lines += ["", "## Official Post Engagement"]
    if "error" in oe:
        md_lines.append(f"- Error: {oe['error']}")
    else:
        md_lines += [
            f"- Official accounts: {official_accounts_list}",
            f"- GT official root posts: {oe.get('gt_official_post_count', 'N/A')}",
            f"- Sim official posts injected: {oe.get('sim_official_post_count', 'N/A')}",
            f"- GT total community engagement: {oe.get('gt_total_engagement', 'N/A')}",
            f"- Sim total community engagement: {oe.get('sim_total_engagement', 'N/A')}",
            f"- GT mean engagement/post: {_fmtv(oe.get('gt_mean_engagement_per_post'))}",
            f"- Sim mean engagement/post: {_fmtv(oe.get('sim_mean_engagement_per_post'))}",
            f"- Spearman (engagement/post): {_fmtv(oe.get('spearman_engagement_per_post'))}",
        ]
    md_lines.append("")

    ch = report.get("community_health", {})
    md_lines += ["## Community Health"]
    if "error" in ch:
        md_lines.append(f"- Error: {ch['error']}")
    else:
        md_lines += [
            f"- Persona users: {ch.get('total_persona_users', 'N/A')}",
            f"- Community posts: {ch.get('total_community_posts', 'N/A')}",
            f"- Total community engagement: {ch.get('total_community_engagement', 'N/A')}",
            f"- Avg engagement per community post: {_fmtv(ch.get('avg_engagement_per_community_post'))}",
            f"- Reply rate (posts with ≥1 comment): {_fmtv(ch.get('reply_rate'))}",
            f"- Unanswered post rate (0 engagement): {_fmtv(ch.get('unanswered_post_rate'))}",
            f"- User participation rate: {_fmtv(ch.get('user_participation_rate'))}",
            f"- Peripheral user activation rate: {_fmtv(ch.get('peripheral_user_activation_rate'))}",
            f"- Follow growth per timestep: {_fmtv(ch.get('follow_growth_per_timestep'))}",
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
