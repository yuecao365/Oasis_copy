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
                likes.append(_safe_int(row.get("likes"), 0))
                retweets.append(_safe_int(row.get("retweets"), 0))
                replies.append(_safe_int(row.get("replies"), 0))
                quotes.append(_safe_int(row.get("quotes"), 0))
    return {
        "per_user_posts": dict(per_user_posts),
        "weekday_hist": gt_weekday,
        "post_timestamps": gt_posts_ts,
        "likes": likes,
        "retweets": retweets,
        "replies": replies,
        "quotes": quotes,
    }


def load_sim(db_path: Path, agent_to_username: Dict[int, str], timestep_hours: float) -> Dict:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    sim_posts_by_user = Counter()
    sim_hour = [0] * 24
    sim_post_steps: List[float] = []
    sim_eng_like = []
    sim_eng_share = []
    sim_eng_reposts = []
    sim_eng_quotes = []

    # Original posts created by simulation (exclude initialization created_at == 0).
    cur.execute(
        """
        SELECT p.post_id, p.user_id, p.created_at, p.num_likes, p.num_shares
        FROM post p
        WHERE p.original_post_id IS NULL
          AND CAST(p.created_at AS REAL) > 0
        """
    )
    sim_posts = cur.fetchall()
    post_ids = []
    for post_id, user_id, created_at, nlikes, nshares in sim_posts:
        uname = agent_to_username.get(_safe_int(user_id, -1))
        if not uname:
            continue
        sim_posts_by_user[uname] += 1
        step = _safe_float(created_at, 0.0)
        sim_post_steps.append(step)
        hour = int((step * timestep_hours) % 24)
        sim_hour[hour] += 1
        sim_eng_like.append(_safe_int(nlikes, 0))
        sim_eng_share.append(_safe_int(nshares, 0))
        post_ids.append(_safe_int(post_id, -1))

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
            sim_eng_replies.append(_safe_int(comment_count.get(pid, 0), 0))
            sim_eng_reposts.append(_safe_int(repost_count.get(pid, 0), 0))
            sim_eng_quotes.append(_safe_int(quote_count.get(pid, 0), 0))

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
            "gt_mean_likes_per_post": (sum(gt["likes"]) / len(gt["likes"])) if gt["likes"] else 0.0,
            "gt_mean_retweets_per_post": (sum(gt["retweets"]) / len(gt["retweets"])) if gt["retweets"] else 0.0,
            "gt_mean_replies_per_post": (sum(gt["replies"]) / len(gt["replies"])) if gt["replies"] else 0.0,
            "gt_mean_quotes_per_post": (sum(gt["quotes"]) / len(gt["quotes"])) if gt["quotes"] else 0.0,
            "gt_total_retweets": sum(gt["retweets"]) if gt["retweets"] else 0,
            "gt_total_quotes": sum(gt["quotes"]) if gt["quotes"] else 0,
            "sim_mean_likes_per_post": (sum(sim["likes"]) / len(sim["likes"])) if sim["likes"] else 0.0,
            "sim_mean_shares_per_post": (sum(sim["shares"]) / len(sim["shares"])) if sim["shares"] else 0.0,
            "sim_mean_replies_per_post": (sum(sim["replies"]) / len(sim["replies"])) if sim["replies"] else 0.0,
            "sim_mean_reposts_per_post": (sum(sim["reposts"]) / len(sim["reposts"])) if sim["reposts"] else 0.0,
            "sim_mean_quotes_per_post": (sum(sim["quotes"]) / len(sim["quotes"])) if sim["quotes"] else 0.0,
            "sim_total_reposts": sum(sim["reposts"]) if sim["reposts"] else 0,
            "sim_total_quotes": sum(sim["quotes"]) if sim["quotes"] else 0,
            "gt_total_posts": len(gt["post_timestamps"]),
            "sim_total_posts": len(sim["post_steps"]),
        },
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
        f"- GT mean likes/retweets/replies/quotes per post: "
        f"{report['engagement_fit']['gt_mean_likes_per_post']:.3f} / "
        f"{report['engagement_fit']['gt_mean_retweets_per_post']:.3f} / "
        f"{report['engagement_fit']['gt_mean_replies_per_post']:.3f} / "
        f"{report['engagement_fit']['gt_mean_quotes_per_post']:.3f}",
        f"- Sim mean likes/shares/replies per post: "
        f"{report['engagement_fit']['sim_mean_likes_per_post']:.3f} / "
        f"{report['engagement_fit']['sim_mean_shares_per_post']:.3f} / "
        f"{report['engagement_fit']['sim_mean_replies_per_post']:.3f}",
        f"- Sim mean reposts/quotes per post: "
        f"{report['engagement_fit']['sim_mean_reposts_per_post']:.3f} / "
        f"{report['engagement_fit']['sim_mean_quotes_per_post']:.3f}",
        f"- GT total retweets/quotes: "
        f"{report['engagement_fit']['gt_total_retweets']} / "
        f"{report['engagement_fit']['gt_total_quotes']}",
        f"- Sim total reposts/quotes: "
        f"{report['engagement_fit']['sim_total_reposts']} / "
        f"{report['engagement_fit']['sim_total_quotes']}",
        "",
    ]
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
