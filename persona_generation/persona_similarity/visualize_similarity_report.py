#!/usr/bin/env python3
"""
Build persona_similarity metrics and write a single self-contained HTML report.

Run from the persona_generation directory:
  python -m persona_similarity.visualize_similarity_report
  python -m persona_similarity.visualize_similarity_report --out visualizations/persona_similarity_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from persona_similarity import (  # noqa: E402
    activity_spread_depth,
    compare_persona_repost_distributions,
    corpus_from_persona_rows,
    load_personas_csv,
    load_train_tweet_texts,
    per_user_tweet_counts_from_train_csv,
    semantic_distribution_report,
    summarize_repost_propensity,
    weekly_retweet_dynamics_from_train_csv,
)


def build_default_report(
    *,
    train_csv: Path,
    personas_csv: Path,
    max_train_tweets_semantic: int = 400,
    max_personas_semantic: int = 80,
) -> Dict[str, Any]:
    """Collect the same metrics as the persona_similarity __main__ demo."""
    report: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_csv": str(train_csv.resolve()),
            "personas_csv": str(personas_csv.resolve()),
            "max_train_tweets_semantic": max_train_tweets_semantic,
            "max_personas_semantic": max_personas_semantic,
        }
    }

    counts = per_user_tweet_counts_from_train_csv(train_csv)
    report["train_activity_spread"] = activity_spread_depth(counts, top_k=10)

    personas = load_personas_csv(personas_csv)
    report["persona_p_repost_summary"] = summarize_repost_propensity(personas)
    report["persona_repost_self_compare"] = compare_persona_repost_distributions(personas, personas)
    report["weekly_retweet_train"] = weekly_retweet_dynamics_from_train_csv(train_csv)

    real_texts = load_train_tweet_texts(train_csv, max_rows=max_train_tweets_semantic)
    n = min(max_personas_semantic, len(personas))
    sim_texts = corpus_from_persona_rows(personas[:n]) if n else []
    report["semantic_real_vs_persona_corpus"] = semantic_distribution_report(real_texts, sim_texts)

    return report


def _json_for_html(obj: Any) -> str:
    """Safe JSON embed inside <script> (avoid breaking on </ sequences)."""
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")


def render_similarity_html(report: Dict[str, Any]) -> str:
    """Single-file HTML with Chart.js (CDN) and embedded report JSON."""
    meta = report.get("meta") or {}
    title = "Persona similarity report"
    payload = _json_for_html(report)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" crossorigin="anonymous"></script>
  <style>
    :root {{
      --bg: #f6f7f2;
      --panel: #fffdf7;
      --ink: #11212d;
      --muted: #5d6b75;
      --accent: #0d9488;
      --accent-2: #f59e0b;
      --line: #d8ddd2;
      --danger: #b91c1c;
      --ok: #047857;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1000px 400px at 10% -10%, #c5f5ef 0%, transparent 60%),
        radial-gradient(900px 300px at 100% 0%, #ffe4b0 0%, transparent 55%),
        var(--bg);
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px 22px;
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 1.75rem; }}
    .sub {{ color: var(--muted); font-size: 0.95rem; margin: 0 0 10px; }}
    .paths {{ font-size: 0.8rem; color: var(--muted); word-break: break-all; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(17, 33, 45, 0.06);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 1.15rem;
      color: var(--ink);
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
    }}
    .stat {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .stat .n {{ font-size: 1.25rem; font-weight: 700; color: var(--accent); }}
    .stat .lbl {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; }}
    table.metrics {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    table.metrics th, table.metrics td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
    }}
    table.metrics th {{ background: #ecfeff; color: #134e4a; }}
    .err {{
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: var(--danger);
      padding: 12px;
      border-radius: 10px;
      font-size: 0.9rem;
    }}
    .ok {{
      background: #ecfdf5;
      border: 1px solid #a7f3d0;
      color: #065f46;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 0.88rem;
      margin-bottom: 12px;
    }}
    .chart-wrap {{
      position: relative;
      height: 280px;
      margin-top: 8px;
    }}
    .chart-wrap.tall {{ height: 320px; }}
    footer {{
      margin-top: 24px;
      font-size: 0.8rem;
      color: var(--muted);
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>{html.escape(title)}</h1>
      <p class="sub">Activity spread, repost behavior, weekly retweet dynamics, and semantic overlap (train tweets vs persona text).</p>
      <p class="paths"><strong>Generated (UTC):</strong> {html.escape(str(meta.get("generated_at_utc", "")))}</p>
      <p class="paths"><strong>Train CSV:</strong> {html.escape(str(meta.get("train_csv", "")))}</p>
      <p class="paths"><strong>Personas CSV:</strong> {html.escape(str(meta.get("personas_csv", "")))}</p>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Activity spread (train)</h2>
        <div class="stats" id="activity-stats"></div>
        <table class="metrics" style="margin-top:14px" id="activity-table"></table>
      </div>
      <div class="card">
        <h2>Repost propensity (personas)</h2>
        <div id="repost-summary"></div>
        <div class="chart-wrap"><canvas id="chartRepostHist" aria-label="p_repost histogram"></canvas></div>
      </div>
    </div>

    <div class="grid">
      <div class="card" style="grid-column: 1 / -1;">
        <h2>Weekly retweet dynamics (train)</h2>
        <p class="sub" style="margin:0 0 8px;">Mean retweets per tweet by ISO week (UTC).</p>
        <div class="chart-wrap tall"><canvas id="chartWeekly" aria-label="Weekly mean retweets"></canvas></div>
      </div>
    </div>

    <div class="grid">
      <div class="card" style="grid-column: 1 / -1;">
        <h2>Semantic distribution (real tweets vs persona corpus)</h2>
        <div id="semantic-block"></div>
      </div>
    </div>

    <div class="card">
      <h2>Repost distribution check</h2>
      <p class="sub" style="margin:0;">Self-comparison of persona <code>p_repost</code> (expect Jensen–Shannon ≈ 0).</p>
      <table class="metrics" style="margin-top:12px" id="repost-js-table"></table>
    </div>

    <footer>persona_similarity · Oasis persona_generation</footer>
  </div>

  <script type="application/json" id="report-data">{payload}</script>
  <script>
(function() {{
  const report = JSON.parse(document.getElementById('report-data').textContent);

  function el(html) {{
    const t = document.createElement('template');
    t.innerHTML = html.trim();
    return t.content.firstChild;
  }}

  // --- Activity spread ---
  const act = report.train_activity_spread || {{}};
  const actStats = document.getElementById('activity-stats');
  const actTable = document.getElementById('activity-table');
  if (act.error) {{
    actStats.appendChild(el(`<div class="err">${{act.error}}</div>`));
  }} else {{
    const cards = [
      ['Users', act.user_count],
      ['Total tweets', act.total_tweets],
      ['Effective # users', act.effective_number_of_users != null ? act.effective_number_of_users.toFixed(2) : '—'],
      ['Norm. entropy', act.normalized_entropy_shares],
      ['Gini (counts)', act.gini_tweet_counts],
    ];
    cards.forEach(([lbl, val]) => {{
      const d = document.createElement('div');
      d.className = 'stat';
      d.innerHTML = `<div class="n">${{val ?? '—'}}</div><div class="lbl">${{lbl}}</div>`;
      actStats.appendChild(d);
    }});
    const tk = act.top_k_concentration || {{}};
    actTable.innerHTML = `
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>HHI (users)</td><td>${{act.herfindahl_users}}</td></tr>
        <tr><td>Top-${{tk.top_k ?? 'k'}} volume share</td><td>${{tk.volume_share_top_k ?? '—'}}</td></tr>
        <tr><td>Users in corpus</td><td>${{tk.user_count ?? '—'}}</td></tr>
      </tbody>`;
  }}

  // --- Repost summary + histogram ---
  const rs = report.persona_p_repost_summary || {{}};
  const repBox = document.getElementById('repost-summary');
  if (rs.error) {{
    repBox.innerHTML = `<div class="err">${{rs.error}}</div>`;
  }} else {{
    repBox.innerHTML = `<div class="ok">n=${{rs.count}} · mean p_repost=${{rs.mean_p_repost}} · std=${{rs.std_p_repost}}</div>`;
    const dens = rs.hist_density_20 || [];
    const labels = dens.map((_, i) => `${{(i / dens.length).toFixed(2)}}–${{((i+1) / dens.length).toFixed(2)}}`);
    new Chart(document.getElementById('chartRepostHist'), {{
      type: 'bar',
      data: {{
        labels,
        datasets: [{{
          label: 'Density',
          data: dens,
          backgroundColor: 'rgba(13, 148, 136, 0.55)',
          borderColor: 'rgba(13, 148, 136, 1)',
          borderWidth: 1
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'p_repost bin' }}, ticks: {{ maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 10 }} }},
          y: {{ title: {{ display: true, text: 'Density' }}, beginAtZero: true }}
        }}
      }}
    }});
  }}

  // --- Weekly retweets ---
  const wk = report.weekly_retweet_train || {{}};
  const series = wk.series || [];
  const weeklyCanvas = document.getElementById('chartWeekly');
  if (series.length) {{
    const wlabels = series.map(s => `${{s.year}}-W${{String(s.iso_week).padStart(2,'0')}}`);
    const wvals = series.map(s => s.mean_retweets_per_tweet);
    new Chart(weeklyCanvas, {{
      type: 'line',
      data: {{
        labels: wlabels,
        datasets: [{{
          label: 'Mean retweets / tweet',
          data: wvals,
          borderColor: 'rgba(245, 158, 11, 1)',
          backgroundColor: 'rgba(245, 158, 11, 0.15)',
          fill: true,
          tension: 0.25
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Week' }} }},
          y: {{ title: {{ display: true, text: 'Mean retweets per tweet' }}, beginAtZero: true }}
        }}
      }}
    }});
  }} else {{
    weeklyCanvas.parentElement.innerHTML = '<p class="sub">No weekly series (empty train CSV or missing timestamps).</p>';
  }}

  // --- Semantic ---
  const sem = report.semantic_real_vs_persona_corpus || {{}};
  const semDiv = document.getElementById('semantic-block');
  if (sem.error) {{
    semDiv.innerHTML = `<div class="err"><strong>Semantic metrics unavailable:</strong> ${{sem.error}}</div>
      <p class="sub" style="margin-top:10px;">Install <code>numpy</code> and <code>sentence-transformers</code> (and PyTorch) to enable embeddings.</p>`;
  }} else {{
    const rows = [
      ['Model', sem.model],
      ['Real texts', sem.real_post_count],
      ['Sim / persona texts', sem.sim_post_count],
      ['Centroid cosine (sim vs real)', sem.sim_to_real_centroid_cosine],
      ['Mean max-match', sem.sim_to_real_mean_max_match],
      ['Real random-pair cohesion', sem.real_mean_random_pair_cosine],
      ['Sim random-pair cohesion', sem.sim_mean_random_pair_cosine],
    ];
    let tb = '<table class="metrics"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
    rows.forEach(([k,v]) => {{ tb += `<tr><td>${{k}}</td><td>${{v ?? '—'}}</td></tr>`; }});
    tb += '</tbody></table>';
    semDiv.innerHTML = tb;
  }}

  // --- JS divergence table ---
  const rc = report.persona_repost_self_compare || {{}};
  const jst = document.getElementById('repost-js-table');
  if (rc.error) {{
    jst.innerHTML = `<tbody><tr><td>${{rc.error}}</td></tr></tbody>`;
  }} else {{
    jst.innerHTML = `<thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Jensen–Shannon (p_repost)</td><td>${{rc.jensen_shannon_p_repost}}</td></tr>
        <tr><td>Bins</td><td>${{rc.bins}}</td></tr>
      </tbody>`;
  }}
}})();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report for persona_similarity metrics.")
    parser.add_argument(
        "--train",
        type=Path,
        default=_SCRIPT_DIR / "outputs" / "train_tweets.csv",
        help="Train tweets CSV (default: persona_generation/outputs/train_tweets.csv)",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=_SCRIPT_DIR / "outputs" / "user_personas.csv",
        help="Personas CSV (default: persona_generation/outputs/user_personas.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT_DIR / "visualizations" / "persona_similarity_report.html",
        help="Output HTML path",
    )
    parser.add_argument("--max-train-semantic", type=int, default=400)
    parser.add_argument("--max-personas-semantic", type=int, default=80)
    args = parser.parse_args()

    report = build_default_report(
        train_csv=args.train,
        personas_csv=args.personas,
        max_train_tweets_semantic=args.max_train_semantic,
        max_personas_semantic=args.max_personas_semantic,
    )

    html_out = render_similarity_html(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
