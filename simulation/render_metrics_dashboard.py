#!/usr/bin/env python3
"""
Render a slide-ready HTML dashboard from simulation metrics outputs.

No plotting libraries required. Uses inline HTML/CSS/SVG so it works in minimal envs.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from pathlib import Path


def _mean(vals: list[float]) -> float:
    return (sum(vals) / len(vals)) if vals else 0.0


def _load_persona_usernames(personas_csv: Path) -> set[str]:
    users: set[str] = set()
    with personas_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            u = (row.get("username") or "").strip().lower()
            if u:
                users.add(u)
    return users


def _compute_engagement_density(
    test_csv: Path,
    personas_csv: Path,
    db_path: Path,
) -> dict:
    persona_users = _load_persona_usernames(personas_csv)

    gt_likes: list[float] = []
    gt_replies: list[float] = []
    gt_retweets: list[float] = []
    gt_quotes: list[float] = []

    with test_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uname = (row.get("user_name") or "").strip().lower()
            if uname not in persona_users:
                continue
            if str(row.get("is_root", "")).strip() != "1":
                continue
            gt_likes.append(float(row.get("likes") or 0))
            gt_replies.append(float(row.get("replies") or 0))
            gt_retweets.append(float(row.get("retweets") or 0))
            gt_quotes.append(float(row.get("quotes") or 0))

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    cur.execute(
        """
        SELECT post_id, num_likes
        FROM post
        WHERE original_post_id IS NULL AND created_at > 0
        """
    )
    sim_posts = cur.fetchall()
    post_ids = [int(pid) for pid, _ in sim_posts]
    post_like = {int(pid): float(nlikes or 0) for pid, nlikes in sim_posts}

    comment_count = {pid: 0 for pid in post_ids}
    repost_count = {pid: 0 for pid in post_ids}
    quote_count = {pid: 0 for pid in post_ids}

    if post_ids:
        placeholders = ",".join(["?"] * len(post_ids))

        cur.execute(
            f"""
            SELECT post_id, COUNT(*)
            FROM comment
            WHERE post_id IN ({placeholders})
            GROUP BY post_id
            """,
            post_ids,
        )
        for pid, c in cur.fetchall():
            comment_count[int(pid)] = int(c or 0)

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
        for pid, rp, qt in cur.fetchall():
            repost_count[int(pid)] = int(rp or 0)
            quote_count[int(pid)] = int(qt or 0)

    con.close()

    sim_likes = [post_like[pid] for pid in post_ids]
    sim_replies = [float(comment_count[pid]) for pid in post_ids]
    sim_reposts = [float(repost_count[pid]) for pid in post_ids]
    sim_quotes = [float(quote_count[pid]) for pid in post_ids]

    return {
        "gt_root_posts": len(gt_likes),
        "sim_root_posts": len(sim_likes),
        "gt_mean_likes_per_post": _mean(gt_likes),
        "sim_mean_likes_per_post": _mean(sim_likes),
        "gt_mean_replies_per_post": _mean(gt_replies),
        "sim_mean_replies_per_post": _mean(sim_replies),
        "gt_total_retweets": float(sum(gt_retweets)),
        "sim_total_reposts": float(sum(sim_reposts)),
        "gt_total_quotes": float(sum(gt_quotes)),
        "sim_total_quotes": float(sum(sim_quotes)),
    }


def _bar_pair(title: str, gt: float, sim: float) -> str:
    maxv = max(gt, sim, 1e-9)
    gt_w = 100.0 * gt / maxv
    sim_w = 100.0 * sim / maxv
    return f"""
    <div class="pair">
      <div class="pair-title">{html.escape(title)}</div>
      <div class="bars">
        <div class="bar-row"><span class="lbl">GT</span><div class="bar gt" style="width:{gt_w:.2f}%"></div><span class="val">{gt:.3f}</span></div>
        <div class="bar-row"><span class="lbl">Sim</span><div class="bar sim" style="width:{sim_w:.2f}%"></div><span class="val">{sim:.3f}</span></div>
      </div>
    </div>
    """


def render_html(report: dict, density: dict, run_tag: str) -> str:
    cov = report["coverage"]
    act = report["user_activity_fit"]
    tmp = report["temporal_fit"]
    eng_ul = report["engagement_fit"]["user_level"]

    likes_pair = _bar_pair(
        "Likes per Post",
        density["gt_mean_likes_per_post"],
        density["sim_mean_likes_per_post"],
    )
    replies_pair = _bar_pair(
        "Replies per Post",
        density["gt_mean_replies_per_post"],
        density["sim_mean_replies_per_post"],
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simulation Metrics Dashboard</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 24px;
      color: #0f172a;
      background: #f8fafc;
    }}
    .slide {{
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 20px;
    }}
    h1, h2 {{ margin: 0 0 10px 0; line-height: 1.1; }}
    .subtitle {{ color: #475569; margin-bottom: 14px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 14px;
      background: #fff;
    }}
    .k {{ color: #334155; font-weight: 600; }}
    .v {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .pair {{ margin-top: 12px; }}
    .pair-title {{ font-weight: 700; margin-bottom: 6px; }}
    .bars {{ display: grid; gap: 8px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 42px 1fr 90px;
      align-items: center;
      gap: 8px;
    }}
    .lbl {{ font-weight: 700; color: #334155; }}
    .bar {{
      height: 14px;
      border-radius: 8px;
      min-width: 2px;
    }}
    .bar.gt {{ background: linear-gradient(90deg, #111827, #374151); }}
    .bar.sim {{ background: linear-gradient(90deg, #1d4ed8, #2563eb); }}
    .val {{ text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    ul {{ margin: 8px 0 0 20px; }}
    .small {{ color: #64748b; font-size: 13px; margin-top: 10px; }}
    @media (max-width: 1000px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <section class="slide">
    <h1>Simulation Performance: Coverage, Activity, Temporal</h1>
    <div class="subtitle">Run tag: <span class="v">{html.escape(run_tag)}</span></div>
    <div class="grid">
      <div class="card">
        <h2>Coverage</h2>
        <div><span class="k">Persona users:</span> <span class="v">{cov['persona_user_count']}</span></div>
        <div><span class="k">GT active users:</span> <span class="v">{cov['gt_active_users']}</span></div>
        <div><span class="k">Sim active users:</span> <span class="v">{cov['sim_active_users']}</span></div>
      </div>
      <div class="card">
        <h2>User Activity Fit</h2>
        <div><span class="k">Spearman (posts/user):</span> <span class="v">{act['spearman_posts_per_user']:.4f}</span></div>
        <div><span class="k">JSD (posts/user):</span> <span class="v">{act['jsd_posts_per_user']:.4f}</span></div>
        <div><span class="k">Top-20 overlap:</span> <span class="v">{act['top20_overlap']:.4f}</span></div>
      </div>
      <div class="card">
        <h2>Temporal Fit</h2>
        <div><span class="k">Day-of-week JSD:</span> <span class="v">{tmp['day_of_week_jsd']:.4f}</span></div>
        <div><span class="k">Cumulative daily NRMSE:</span> <span class="v">{tmp['cumulative_posts_daily_nrmse']:.4f}</span></div>
      </div>
    </div>
  </section>

  <section class="slide">
    <h1>User-Level Engagement Ranking Fit</h1>
    <div class="grid">
      <div class="card">
        <div><span class="k">Spearman total engagement/user:</span> <span class="v">{eng_ul['spearman_total_engagement_per_user']:.4f}</span></div>
        <div><span class="k">Spearman engagement-per-post/user:</span> <span class="v">{eng_ul['spearman_engagement_per_post_per_user']:.4f}</span></div>
        <div><span class="k">Top-20 overlap (total):</span> <span class="v">{eng_ul['topk_overlap_total_engagement_per_user']['top20']:.4f}</span></div>
        <div><span class="k">Top-20 overlap (per-post):</span> <span class="v">{eng_ul['topk_overlap_engagement_per_post_per_user']['top20']:.4f}</span></div>
      </div>
      <div class="card">
        <div><span class="k">GT top-10% share:</span> <span class="v">{eng_ul['gt_top10pct_share_total_engagement']:.4f}</span></div>
        <div><span class="k">Sim top-10% share:</span> <span class="v">{eng_ul['sim_top10pct_share_total_engagement']:.4f}</span></div>
        <div><span class="k">GT top-20% share:</span> <span class="v">{eng_ul['gt_top20pct_share_total_engagement']:.4f}</span></div>
        <div><span class="k">Sim top-20% share:</span> <span class="v">{eng_ul['sim_top20pct_share_total_engagement']:.4f}</span></div>
      </div>
      <div class="card">
        <h2>Interpretation</h2>
        <ul>
          <li>Run is now valid (non-zero sim activity).</li>
          <li>Ranking alignment exists but remains weak.</li>
          <li>Engagement is over-concentrated in simulation.</li>
        </ul>
      </div>
    </div>
  </section>

  <section class="slide">
    <h1>The Engagement Density Gap</h1>
    <div class="subtitle">Simulation vs Ground Truth (Persona-user subset)</div>
    {likes_pair}
    {replies_pair}
    <div class="small">
      Total Retweets/Reposts: GT {density['gt_total_retweets']:.0f} vs Sim {density['sim_total_reposts']:.0f}
      &nbsp;|&nbsp;
      Total Quotes: GT {density['gt_total_quotes']:.0f} vs Sim {density['sim_total_quotes']:.0f}
    </div>
  </section>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HTML dashboard from simulation metrics report.")
    parser.add_argument("--metrics_json", type=Path, default=Path("simulation/metrics_report.json"))
    parser.add_argument("--output_html", type=Path, default=Path("simulation/slide_metrics_dashboard.html"))
    parser.add_argument("--output_summary_json", type=Path, default=Path("simulation/slide_metrics_summary.json"))
    parser.add_argument("--run_tag", type=str, default="latest")
    args = parser.parse_args()

    report = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    inputs = report["inputs"]
    density = _compute_engagement_density(
        test_csv=Path(inputs["test_csv"]),
        personas_csv=Path(inputs["personas_csv"]),
        db_path=Path(inputs["db"]),
    )

    html_out = render_html(report=report, density=density, run_tag=args.run_tag)
    args.output_html.write_text(html_out, encoding="utf-8")

    summary = {
        "run_tag": args.run_tag,
        "metrics_report": str(args.metrics_json),
        "coverage": report["coverage"],
        "user_activity_fit": report["user_activity_fit"],
        "temporal_fit": report["temporal_fit"],
        "engagement_user_level": report["engagement_fit"]["user_level"],
        "engagement_density": density,
    }
    args.output_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_html}")
    print(f"Wrote {args.output_summary_json}")


if __name__ == "__main__":
    main()

