#!/usr/bin/env python3
"""
Generate a two-tab HTML dashboard from a simulation SQLite database + persona CSV.

Tabs:
  Agents   — persona cards for every simulated agent
  Timeline — per-timestep activity view with observation panel

Usage (from repo root):
    python simulation/generate_timeline.py
    python simulation/generate_timeline.py --db data/simu_db/yue_run.db
    python simulation/generate_timeline.py --db data/simu_db/yue_run.db \\
        --csv persona_generation/outputs/user_personas_yue.csv \\
        --out simulation/simulation_timeline.html
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sqlite3
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_DB  = os.path.join(_REPO_ROOT, "data", "simu_db", "yue_run.db")
_DEFAULT_CSV = os.path.join(_REPO_ROOT, "persona_generation", "outputs", "user_personas_yue.csv")
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "simulation_timeline.html")


def _parse_list_field(raw: str) -> list:
    if not raw or raw.strip() in ("", "[]", "nan"):
        return []
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def load_personas(csv_path: str) -> list[dict]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tweets = _parse_list_field(row.get("previous_tweets", ""))
            row["previous_tweets_list"] = [str(t)[:200] for t in tweets[:5]]
            row["following_list"] = _parse_list_field(row.get("following_agentid_list", ""))
            rows.append(row)
    return rows


def load_data(db_path: str, csv_path: str = "") -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT user_id, user_name, name, bio, num_followings, num_followers FROM user ORDER BY user_id")
    users = [dict(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT p.post_id, p.user_id, u.user_name, p.content, p.created_at,
               p.num_likes, p.num_shares, p.original_post_id, p.quote_content
        FROM post p JOIN user u ON u.user_id = p.user_id
        WHERE CAST(p.created_at AS REAL) > 0
        ORDER BY p.created_at, p.post_id
    """)
    sim_posts = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT user_id, COUNT(*) as count FROM post WHERE CAST(created_at AS REAL) = 0 GROUP BY user_id")
    seed_counts = {str(r["user_id"]): r["count"] for r in cur.fetchall()}

    cur.execute("SELECT follower_id, followee_id FROM follow")
    follows = [dict(r) for r in cur.fetchall()]

    cur.execute("""SELECT c.post_id, c.user_id, u.user_name, c.content, c.created_at
        FROM comment c JOIN user u ON u.user_id = c.user_id WHERE CAST(c.created_at AS REAL) > 0""")
    comments = [dict(r) for r in cur.fetchall()]

    cur.execute('SELECT post_id, user_id, created_at FROM "like" WHERE CAST(created_at AS REAL) > 0')
    likes = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT follower_id, followee_id FROM follow WHERE CAST(created_at AS REAL) > 0")
    new_follows = [dict(r) for r in cur.fetchall()]

    observations: list[dict] = []
    try:
        cur.execute(
            "SELECT timestep, agent_id, username, feed_total, feed_following, "
            "feed_rec, action_taken, action_success FROM observation ORDER BY timestep, obs_id"
        )
        observations = [dict(r) for r in cur.fetchall()]
    except Exception:
        pass

    con.close()

    obs_timesteps = sorted(set(o["timestep"] for o in observations))
    all_timesteps = sorted(set(p["created_at"] for p in sim_posts) | set(obs_timesteps))

    personas = load_personas(csv_path or _DEFAULT_CSV)

    return {
        "users": users,
        "sim_posts": sim_posts,
        "seed_counts": seed_counts,
        "follows": follows,
        "comments": comments,
        "likes": likes,
        "new_follows": new_follows,
        "timesteps": all_timesteps,
        "observations": observations,
        "total_seed_posts": sum(seed_counts.values()),
        "personas": personas,
    }


def generate_html(data: dict, db_path: str, metrics: dict | None = None) -> str:
    if metrics is None:
        db_stem = Path(db_path).stem
        history_dir = Path(_REPO_ROOT) / "simulation" / "metrics_history"
        if history_dir.exists():
            matches = sorted(history_dir.glob(f"db-{db_stem}__*.json"))
            if matches:
                with open(matches[-1], encoding="utf-8") as _f:
                    metrics = json.load(_f)
    data_json = json.dumps(data)
    metrics_json = json.dumps(metrics or {})
    db_label = Path(db_path).name

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Simulation — {db_label}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {{
    --bg:#0d1117;--surface:#161b22;--surface2:#21262d;--border:#30363d;
    --text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;
    --yellow:#d29922;--purple:#bc8cff;--orange:#f78166;--pink:#f778ba;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}}

  /* ── Header ── */
  .header{{background:var(--surface);border-bottom:1px solid var(--border);padding:20px 40px;display:flex;align-items:center;gap:20px;}}
  .header h1{{font-size:20px;font-weight:600;}}
  .header .subtitle{{color:var(--muted);font-size:12px;margin-top:2px;}}
  .badge{{background:var(--accent);color:#0d1117;font-size:11px;font-weight:700;border-radius:12px;padding:2px 10px;}}

  /* ── Tabs ── */
  .tab-bar{{display:flex;gap:0;background:var(--surface);border-bottom:1px solid var(--border);padding:0 40px;}}
  .tab{{padding:12px 20px;font-size:13px;font-weight:500;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;transition:color .15s,border-color .15s;user-select:none;}}
  .tab:hover{{color:var(--text);}}
  .tab.active{{color:var(--accent);border-bottom-color:var(--accent);}}
  .tab-panel{{display:none;}}
  .tab-panel.active{{display:block;}}

  /* ── Stat bar ── */
  .stats{{display:flex;gap:0;border-bottom:1px solid var(--border);background:var(--surface);}}
  .stat{{flex:1;padding:14px 20px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:4px;}}
  .stat:last-child{{border-right:none;}}
  .stat .val{{font-size:24px;font-weight:700;color:var(--accent);}}
  .stat .lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;}}

  /* ── Agents tab ── */
  .agents-container{{padding:32px 40px;}}
  .agents-filter{{display:flex;gap:10px;margin-bottom:24px;flex-wrap:wrap;align-items:center;}}
  .filter-label{{font-size:12px;color:var(--muted);}}
  .filter-btn{{font-size:12px;padding:5px 12px;border-radius:20px;border:1px solid var(--border);background:var(--surface2);color:var(--muted);cursor:pointer;transition:all .15s;}}
  .filter-btn:hover,.filter-btn.active{{background:var(--accent);color:#0d1117;border-color:var(--accent);font-weight:600;}}
  .agents-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;}}
  .agent-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px;transition:border-color .15s;}}
  .agent-card:hover{{border-color:var(--accent);}}
  .agent-card-header{{display:flex;align-items:center;gap:12px;margin-bottom:12px;}}
  .agent-avatar{{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:700;flex-shrink:0;}}
  .agent-name{{font-size:14px;font-weight:600;}}
  .agent-handle{{font-size:12px;color:var(--muted);}}
  .category-badge{{display:inline-block;font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;background:rgba(88,166,255,.15);color:var(--accent);border:1px solid rgba(88,166,255,.2);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;}}
  .agent-bio{{font-size:12px;color:var(--muted);line-height:1.5;margin-bottom:10px;}}
  .agent-char{{font-size:11px;color:var(--text);line-height:1.5;background:var(--surface2);border-radius:6px;padding:8px 10px;margin-bottom:10px;border-left:2px solid var(--border);}}
  .agent-section-label{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:5px;}}
  .action-bar{{display:flex;height:8px;border-radius:4px;overflow:hidden;gap:1px;margin-bottom:8px;}}
  .action-bar-seg{{height:100%;}}
  .tweets-list{{display:flex;flex-direction:column;gap:5px;}}
  .tweet-item{{font-size:11px;color:var(--muted);line-height:1.4;padding:5px 8px;background:var(--surface2);border-radius:4px;border-left:2px solid var(--border);}}
  .agent-footer{{display:flex;gap:12px;margin-top:10px;font-size:11px;color:var(--muted);border-top:1px solid var(--border);padding-top:10px;}}
  .agent-footer span{{display:flex;align-items:center;gap:4px;}}

  /* ── Timeline tab ── */
  .tl-container{{display:grid;grid-template-columns:1fr 380px;gap:0;min-height:calc(100vh - 200px);}}
  .tl-left{{padding:32px 40px;border-right:1px solid var(--border);}}
  .tl-right{{padding:24px;}}
  .section-title{{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:20px;display:flex;align-items:center;gap:8px;}}
  .section-title::after{{content:'';flex:1;height:1px;background:var(--border);}}
  .timeline{{position:relative;padding-left:28px;}}
  .timeline::before{{content:'';position:absolute;left:8px;top:0;bottom:0;width:2px;background:var(--border);}}
  .tl-phase{{margin-bottom:40px;position:relative;}}
  .tl-dot{{position:absolute;left:-24px;top:4px;width:14px;height:14px;border-radius:50%;border:2px solid var(--bg);z-index:1;}}
  .tl-dot.seed{{background:var(--muted);}}
  .tl-dot.active{{background:var(--green);box-shadow:0 0 8px rgba(63,185,80,.5);}}
  .tl-phase-header{{display:flex;align-items:baseline;gap:12px;margin-bottom:12px;flex-wrap:wrap;}}
  .tl-phase-label{{font-size:15px;font-weight:600;}}
  .tl-phase-meta{{font-size:12px;color:var(--muted);}}
  .tl-phase-badge{{font-size:11px;font-weight:600;padding:2px 8px;border-radius:4px;background:var(--surface2);border:1px solid var(--border);}}
  .post-grid{{display:flex;flex-direction:column;gap:10px;}}
  .post-card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;transition:border-color .15s;}}
  .post-card:hover{{border-color:var(--accent);}}
  .post-header{{display:flex;align-items:center;gap:10px;margin-bottom:8px;}}
  .avatar{{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;}}
  .post-user{{font-size:13px;font-weight:600;}}
  .post-bio{{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:340px;}}
  .post-type-badge{{margin-left:auto;font-size:10px;font-weight:600;padding:2px 7px;border-radius:4px;white-space:nowrap;}}
  .type-post{{background:rgba(88,166,255,.15);color:var(--accent);border:1px solid rgba(88,166,255,.3);}}
  .type-repost{{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3);}}
  .type-quote{{background:rgba(188,140,255,.15);color:var(--purple);border:1px solid rgba(188,140,255,.3);}}
  .type-comment{{background:rgba(247,120,106,.15);color:var(--orange);border:1px solid rgba(247,120,106,.3);}}
  .post-content{{font-size:13px;line-height:1.55;color:var(--text);}}
  .post-footer{{display:flex;gap:16px;margin-top:10px;font-size:11px;color:var(--muted);}}
  .seed-summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}}
  .seed-card{{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:12px;}}
  .seed-card .sname{{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .seed-card .sbio{{color:var(--muted);font-size:11px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .seed-card .scnt{{color:var(--green);font-size:11px;margin-top:4px;font-weight:600;}}
  .see-more{{font-size:12px;color:var(--muted);text-align:center;padding:8px;cursor:pointer;border:1px dashed var(--border);border-radius:6px;margin-top:8px;}}
  .see-more:hover{{color:var(--accent);border-color:var(--accent);}}
  .tl-right h2{{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:14px;}}
  #network-svg{{width:100%;border-radius:8px;background:var(--surface);border:1px solid var(--border);}}
  .legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;}}
  .leg-item{{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted);}}
  .leg-dot{{width:8px;height:8px;border-radius:50%;}}
  .agent-list{{margin-top:20px;}}
  .agent-scroll{{max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;scrollbar-width:thin;scrollbar-color:var(--border) transparent;}}
  .agent-row{{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:6px;background:var(--surface);border:1px solid var(--border);font-size:12px;}}
  .agent-row .aname{{font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .agent-row .afol{{color:var(--muted);white-space:nowrap;}}
  .aactive{{width:6px;height:6px;border-radius:50%;flex-shrink:0;}}
  .aactive.yes{{background:var(--green);}} .aactive.no{{background:var(--border);}}
  .empty-ts{{color:var(--muted);font-size:13px;padding:12px;background:var(--surface2);border-radius:6px;border:1px solid var(--border);}}

  /* ── Observation panel ── */
  .obs-panel{{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin-bottom:14px;}}
  .obs-title{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:8px;}}
  .obs-meta{{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:4px;}}
  .obs-meta strong{{color:var(--text);}}
  .obs-legend{{display:flex;flex-wrap:wrap;gap:8px;font-size:11px;margin-top:4px;}}
  .obs-table{{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px;}}
  .obs-table th{{text-align:left;color:var(--muted);font-weight:600;padding:4px 6px;border-bottom:1px solid var(--border);}}
  .obs-table td{{padding:4px 6px;border-bottom:1px solid rgba(48,54,61,.5);vertical-align:middle;}}
  .obs-table tr:last-child td{{border-bottom:none;}}

  /* ── Tooltip ── */
  #tooltip{{position:fixed;background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-size:12px;pointer-events:none;z-index:100;max-width:240px;display:none;line-height:1.5;}}
  #tooltip .tt-name{{font-weight:600;color:var(--text);margin-bottom:2px;}}
  #tooltip .tt-bio{{color:var(--muted);}}
  #tooltip .tt-stats{{color:var(--accent);margin-top:4px;}}

  /* ── Metrics tab ── */
  .metrics-container{{padding:32px 40px;max-width:960px;}}
  .metrics-section{{margin-bottom:32px;}}
  .metrics-section-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border);}}
  .metrics-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;}}
  .metric-card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;}}
  .metric-val{{font-size:22px;font-weight:700;color:var(--accent);margin-bottom:4px;}}
  .metric-val.na{{color:var(--muted);font-size:16px;}}
  .metric-lbl{{font-size:11px;color:var(--muted);line-height:1.4;}}
  .metric-sub{{font-size:10px;color:var(--border);margin-top:4px;}}
  .metrics-none{{color:var(--muted);font-size:13px;padding:24px;background:var(--surface2);border-radius:8px;border:1px solid var(--border);}}
</style>
</head>
<body>
<div id="tooltip"><div class="tt-name" id="tt-name"></div><div class="tt-bio" id="tt-bio"></div><div class="tt-stats" id="tt-stats"></div></div>

<header class="header">
  <div>
    <h1>LLM Simulation Dashboard</h1>
    <div class="subtitle">qwen3.5-plus · DashScope · DB: {db_label}</div>
  </div>
  <span class="badge">RUN OUTPUT</span>
</header>

<div class="tab-bar">
  <div class="tab active" onclick="switchTab('agents')">Agents</div>
  <div class="tab" onclick="switchTab('timeline')">Timeline</div>
  <div class="tab" onclick="switchTab('metrics')">Metrics</div>
</div>

<div class="stats" id="stat-bar"></div>

<!-- ═══════════════════ METRICS TAB ═══════════════════ -->
<div class="tab-panel" id="panel-metrics">
  <div class="metrics-container" id="metrics-container"></div>
</div>

<!-- ═══════════════════ AGENTS TAB ═══════════════════ -->
<div class="tab-panel active" id="panel-agents">
  <div class="agents-container">
    <div class="agents-filter">
      <span class="filter-label">Filter by cluster:</span>
      <button class="filter-btn active" onclick="filterAgents('all', this)">All</button>
      <span id="cluster-btns"></span>
    </div>
    <div class="agents-grid" id="agents-grid"></div>
  </div>
</div>

<!-- ═══════════════════ TIMELINE TAB ═══════════════════ -->
<div class="tab-panel" id="panel-timeline">
  <div class="tl-container">
    <div class="tl-left">
      <div class="section-title">Agent Activity Timeline</div>
      <div class="timeline" id="timeline"></div>
    </div>
    <div class="tl-right">
      <h2>Follow Network</h2>
      <svg id="network-svg" viewBox="0 0 360 320"></svg>
      <div class="legend" id="legend"></div>
      <div class="agent-list">
        <h2 style="margin-bottom:10px">All Agents</h2>
        <div class="agent-scroll" id="agent-list"></div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
const METRICS = {metrics_json};

// ── Shared helpers ───────────────────────────────────────────────────────────
const PALETTE = ['#58a6ff','#3fb950','#bc8cff','#f78166','#f778ba','#39d353','#d29922','#79c0ff','#56d364','#e3b341'];
const ACTION_COLORS = {{
  create_post:'#58a6ff', create_comment:'#f78166', repost:'#3fb950',
  like_post:'#f778ba', quote_post:'#bc8cff', follow:'#39d353', do_nothing:'#30363d',
}};
const CLUSTER_COLORS = ['#58a6ff','#3fb950','#bc8cff','#f78166','#f778ba','#d29922','#39d353','#79c0ff'];

function avatarColor(uid) {{ return PALETTE[uid % PALETTE.length]; }}
function initials(name) {{ return (name||'?').slice(0,2).toUpperCase(); }}
function esc(s) {{ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t,i) => {{
    const n = ['agents','timeline','metrics'][i];
    t.classList.toggle('active', n === name);
  }});
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if (name === 'timeline' && !window._networkBuilt) buildNetwork();
}}

// ── Stat bar ─────────────────────────────────────────────────────────────────
const activeUserIds = new Set(DATA.sim_posts.map(p => p.user_id));
const userById = {{}};
DATA.users.forEach(u => {{ userById[u.user_id] = u; }});

const statDefs = [
  {{val: DATA.users.length,        lbl: 'Total Agents'}},
  {{val: DATA.timesteps.length,    lbl: 'Timesteps Run'}},
  {{val: DATA.sim_posts.length,    lbl: 'New Posts'}},
  {{val: DATA.total_seed_posts,    lbl: 'Seed Posts'}},
  {{val: DATA.follows.length,      lbl: 'Follow Edges'}},
  {{val: activeUserIds.size,       lbl: 'Active Agents'}},
  {{val: DATA.comments.length,     lbl: 'Comments'}},
  {{val: DATA.likes.length,        lbl: 'Likes'}},
];
const statBar = document.getElementById('stat-bar');
statDefs.forEach(s => {{
  statBar.innerHTML += `<div class="stat"><div class="val">${{s.val}}</div><div class="lbl">${{s.lbl}}</div></div>`;
}});

// ══════════════════════════════════════════════════════════════════════════════
// AGENTS TAB
// ══════════════════════════════════════════════════════════════════════════════
const personas = DATA.personas || [];

// Cluster filter buttons
const clusters = [...new Set(personas.map(p => p.category_label).filter(Boolean))];
const clusterColorMap = {{}};
clusters.forEach((c, i) => {{ clusterColorMap[c] = CLUSTER_COLORS[i % CLUSTER_COLORS.length]; }});

const clusterBtns = document.getElementById('cluster-btns');
clusters.forEach(c => {{
  const btn = document.createElement('button');
  btn.className = 'filter-btn';
  btn.textContent = c;
  btn.style.setProperty('--cbg', clusterColorMap[c]);
  btn.onclick = () => filterAgents(c, btn);
  clusterBtns.appendChild(btn);
}});

function filterAgents(cluster, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.agent-card').forEach(card => {{
    card.style.display = (cluster === 'all' || card.dataset.cluster === cluster) ? '' : 'none';
  }});
}}

function actionBar(persona) {{
  const fields = [
    ['create_post', persona.p_create_post],
    ['create_comment', persona.p_create_comment],
    ['repost', persona.p_repost],
    ['like_post', persona.p_like_post],
    ['quote_post', persona.p_quote_post],
    ['do_nothing', persona.p_do_nothing],
  ];
  const segs = fields
    .map(([a, v]) => {{
      const pct = parseFloat(v || 0) * 100;
      if (pct < 0.5) return '';
      return `<div class="action-bar-seg" title="${{a}}: ${{pct.toFixed(0)}}%" style="flex:${{pct}};background:${{ACTION_COLORS[a]||'#555'}}"></div>`;
    }}).join('');
  return `<div class="action-bar">${{segs}}</div>`;
}}

const agentsGrid = document.getElementById('agents-grid');
personas.forEach((p, idx) => {{
  const uid = parseInt(p.user_id) || idx;
  const color = avatarColor(uid);
  const clusterColor = clusterColorMap[p.category_label] || 'var(--accent)';
  const tweetsHtml = (p.previous_tweets_list || []).map(t =>
    `<div class="tweet-item">"${{esc(t)}}"</div>`).join('');
  const followCount = (p.following_list || []).length;

  const card = document.createElement('div');
  card.className = 'agent-card';
  card.dataset.cluster = p.category_label || '';
  card.innerHTML = `
    <div class="agent-card-header">
      <div class="agent-avatar" style="background:${{color}};color:#0d1117">${{initials(p.name || p.username)}}</div>
      <div style="min-width:0;flex:1">
        <div class="agent-name">${{esc(p.name || p.username)}}</div>
        <div class="agent-handle">@${{esc(p.username)}}</div>
        <span class="category-badge" style="background:${{clusterColor}}22;color:${{clusterColor}};border-color:${{clusterColor}}44">${{esc(p.category_label)}}</span>
      </div>
    </div>
    ${{p.description ? `<div class="agent-bio">${{esc(p.description)}}</div>` : ''}}
    ${{p.user_char ? `<div class="agent-char">${{esc((p.user_char||'').slice(0,220))}}${{(p.user_char||'').length > 220 ? '…' : ''}}</div>` : ''}}
    <div class="agent-section-label">Action tendencies</div>
    ${{actionBar(p)}}
    ${{tweetsHtml ? `<div class="agent-section-label" style="margin-top:8px">Sample tweets</div><div class="tweets-list">${{tweetsHtml}}</div>` : ''}}
    <div class="agent-footer">
      <span>👥 ${{followCount}} following</span>
      <span>${{activeUserIds.has(uid) ? '✓ Active in sim' : '· Inactive'}}</span>
    </div>`;
  agentsGrid.appendChild(card);
}});

if (personas.length === 0) {{
  agentsGrid.innerHTML = '<div class="empty-ts" style="grid-column:1/-1">No persona CSV loaded. Pass --csv when running generate_timeline.py.</div>';
}}

// ══════════════════════════════════════════════════════════════════════════════
// TIMELINE TAB
// ══════════════════════════════════════════════════════════════════════════════
function postType(p) {{
  if (p.original_post_id != null && p.quote_content) return 'quote';
  if (p.original_post_id != null) return 'repost';
  return 'post';
}}
const typeLabel = {{post:'Original Post',repost:'Repost',quote:'Quote',comment:'Comment'}};

function renderPostCard(p) {{
  const u = userById[p.user_id] || {{}};
  const t = postType(p);
  const raw = esc(p.content || '');
  const content = t === 'quote'
    ? `<div style="border-left:2px solid var(--border);padding-left:10px;color:var(--muted);font-size:12px;margin-bottom:6px">${{raw.slice(0,120)}}…</div>${{esc(p.quote_content||'')}}`
    : raw;
  return `<div class="post-card">
    <div class="post-header">
      <div class="avatar" style="background:${{avatarColor(p.user_id)}};color:#0d1117">${{initials(p.user_name)}}</div>
      <div style="flex:1;min-width:0">
        <div class="post-user">@${{esc(p.user_name)}}</div>
        <div class="post-bio">${{esc((u.bio||''))}}</div>
      </div>
      <span class="post-type-badge type-${{t}}">${{typeLabel[t]}}</span>
    </div>
    <div class="post-content">${{content}}</div>
    <div class="post-footer"><span>♥ ${{p.num_likes}}</span><span>🔁 ${{p.num_shares}}</span><span style="margin-left:auto">post #${{p.post_id}}</span></div>
  </div>`;
}}

function renderObsPanel(obs) {{
  if (!obs || obs.length === 0) return '';
  const actionCounts = {{}};
  let totalFeed = 0, totalFollowing = 0, totalRec = 0;
  obs.forEach(o => {{
    actionCounts[o.action_taken] = (actionCounts[o.action_taken] || 0) + 1;
    totalFeed += o.feed_total; totalFollowing += o.feed_following; totalRec += o.feed_rec;
  }});
  const n = obs.length;
  const avgFeed = (totalFeed / n).toFixed(1);
  const avgFollowing = (totalFollowing / n).toFixed(1);
  const avgRec = (totalRec / n).toFixed(1);
  const actionOrder = ['create_post','create_comment','repost','like_post','quote_post','follow','do_nothing'];
  const barSegs = actionOrder.filter(a => actionCounts[a]).map(a => {{
    const pct = ((actionCounts[a] / n) * 100).toFixed(1);
    return `<div title="${{a}}: ${{actionCounts[a]}} (${{pct}}%)" style="flex:${{actionCounts[a]}};background:${{ACTION_COLORS[a]||'#888'}};height:100%;border-radius:2px"></div>`;
  }}).join('');
  const agentRows = obs.map(o => {{
    const srcBar = o.feed_total > 0
      ? `<div style="display:flex;gap:2px;height:6px;width:60px;border-radius:2px;overflow:hidden;background:var(--border)"><div style="flex:${{o.feed_following}};background:var(--accent)"></div><div style="flex:${{o.feed_rec}};background:var(--purple)"></div></div>`
      : `<span style="color:var(--muted);font-size:10px">—</span>`;
    const dot = o.action_success ? `<span style="color:var(--green)">✓</span>` : `<span style="color:var(--muted)">·</span>`;
    return `<tr><td>@${{esc(o.username)}}</td><td><span style="color:${{ACTION_COLORS[o.action_taken]||'#888'}};font-size:11px">${{o.action_taken}}</span> ${{dot}}</td><td style="text-align:center">${{o.feed_total}}</td><td>${{srcBar}}</td></tr>`;
  }}).join('');
  return `<div class="obs-panel">
    <div class="obs-title">Observation Space</div>
    <div class="obs-meta">
      <span>${{n}} agent actions</span>
      <span>avg feed <strong>${{avgFeed}}</strong></span>
      <span style="color:var(--accent)">■ following ${{avgFollowing}}</span>
      <span style="color:var(--purple)">■ rec ${{avgRec}}</span>
    </div>
    <div style="display:flex;height:14px;gap:2px;border-radius:4px;overflow:hidden;margin:8px 0">${{barSegs}}</div>
    <div class="obs-legend">${{Object.entries(actionCounts).map(([a,c])=>`<span style="color:${{ACTION_COLORS[a]||'#888'}}">${{a}} ${{c}}</span>`).join('')}}</div>
    <details style="margin-top:10px"><summary style="font-size:11px;color:var(--muted);cursor:pointer;user-select:none">Per-agent breakdown</summary>
      <table class="obs-table"><thead><tr><th>Agent</th><th>Action</th><th>Feed</th><th>Following/Rec</th></tr></thead><tbody>${{agentRows}}</tbody></table>
    </details>
  </div>`;
}}

// Build timeline HTML
const tl = document.getElementById('timeline');
const seedUsers = DATA.users.filter(u => DATA.seed_counts[String(u.user_id)] > 0);
const seedHtml = seedUsers.slice(0,9).map(u =>
  `<div class="seed-card"><div class="sname">@${{esc(u.user_name)}}</div><div class="sbio">${{esc((u.bio||'').slice(0,50))}}</div><div class="scnt">📝 ${{DATA.seed_counts[String(u.user_id)]}} posts · ${{u.num_followers}} followers</div></div>`).join('');
const extraHtml = seedUsers.slice(9).map(u =>
  `<div class="seed-card"><div class="sname">@${{esc(u.user_name)}}</div><div class="sbio">${{esc((u.bio||'').slice(0,50))}}</div><div class="scnt">📝 ${{DATA.seed_counts[String(u.user_id)]}} posts · ${{u.num_followers}} followers</div></div>`).join('');

tl.innerHTML += `<div class="tl-phase">
  <div class="tl-dot seed"></div>
  <div class="tl-phase-header">
    <span class="tl-phase-label" style="color:var(--muted)">Seed Phase</span>
    <span class="tl-phase-meta">t = 0 · initialization</span>
    <span class="tl-phase-badge">${{DATA.users.length}} agents · ${{DATA.total_seed_posts}} posts · ${{DATA.follows.length}} follows</span>
  </div>
  <div class="seed-summary">${{seedHtml}}</div>
  ${{extraHtml ? `<div class="see-more" onclick="this.style.display='none';document.getElementById('seed-extra').style.display='grid'">+ ${{seedUsers.length-9}} more — click to expand</div><div id="seed-extra" class="seed-summary" style="display:none;margin-top:8px">${{extraHtml}}</div>` : ''}}
</div>`;

DATA.timesteps.forEach(ts => {{
  const posts    = DATA.sim_posts.filter(p => p.created_at === ts);
  const comments = DATA.comments.filter(c => c.created_at === ts);
  const likes    = DATA.likes.filter(l => l.created_at === ts);
  const obs      = (DATA.observations || []).filter(o => o.timestep === ts);
  const unique   = new Set(obs.map(o => o.agent_id));
  const stepHours = ts * 12;
  tl.innerHTML += `<div class="tl-phase">
    <div class="tl-dot active"></div>
    <div class="tl-phase-header">
      <span class="tl-phase-label">Timestep ${{ts}}</span>
      <span class="tl-phase-meta">+${{stepHours}}h from simulation start</span>
      <span class="tl-phase-badge" style="color:var(--green);border-color:rgba(63,185,80,.3)">${{unique.size}} active · ${{posts.length}} posts · ${{comments.length}} comments · ${{likes.length}} likes</span>
    </div>
    ${{renderObsPanel(obs)}}
    <div class="post-grid" style="margin-top:12px">${{posts.map(renderPostCard).join('')}}</div>
    ${{posts.length === 0 && obs.length === 0 ? '<div class="empty-ts">No activity this timestep.</div>' : ''}}
  </div>`;
}});
if (DATA.timesteps.length === 0) {{
  tl.innerHTML += `<div class="empty-ts" style="margin-top:8px">No simulation timesteps yet. Run the simulation first.</div>`;
}}

// Agent sidebar list
const agentList = document.getElementById('agent-list');
const sorted = [...DATA.users].sort((a,b) => {{
  const d = (activeUserIds.has(b.user_id)?1:0) - (activeUserIds.has(a.user_id)?1:0);
  return d || b.num_followers - a.num_followers;
}});
agentList.innerHTML = sorted.map(u => `
  <div class="agent-row">
    <div class="aactive ${{activeUserIds.has(u.user_id)?'yes':'no'}}"></div>
    <div class="avatar" style="width:24px;height:24px;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;border-radius:50%;background:${{avatarColor(u.user_id)}};color:#0d1117">${{initials(u.user_name)}}</div>
    <div class="aname">@${{esc(u.user_name)}}</div>
    <div class="afol">${{u.num_followers}} · ${{u.num_followings}}</div>
  </div>`).join('');

// ══════════════════════════════════════════════════════════════════════════════
// METRICS TAB
// ══════════════════════════════════════════════════════════════════════════════
(function() {{
  const mc = document.getElementById('metrics-container');
  if (!METRICS || Object.keys(METRICS).length === 0) {{
    mc.innerHTML = '<div class="metrics-none">No metrics available. Run compare_sim_to_test.py or use run.py to generate metrics.</div>';
    return;
  }}
  function fmt(v) {{
    if (v === null || v === undefined) return {{cls:'na', txt:'N/A'}};
    if (typeof v === 'string') return {{cls:'na', txt:v}};
    if (typeof v === 'number') return {{cls:'', txt: Number.isInteger(v) ? String(v) : v.toFixed(4)}};
    return {{cls:'na', txt:String(v)}};
  }}
  function card(label, val, sub) {{
    const {{cls, txt}} = fmt(val);
    return `<div class="metric-card"><div class="metric-val ${{cls}}">${{esc(txt)}}</div><div class="metric-lbl">${{esc(label)}}</div>${{sub ? `<div class="metric-sub">${{esc(sub)}}</div>` : ''}}</div>`;
  }}
  function section(title, cards) {{
    return `<div class="metrics-section"><div class="metrics-section-title">${{title}}</div><div class="metrics-grid">${{cards.join('')}}</div></div>`;
  }}
  let html = '';

  const cov = METRICS.coverage || {{}};
  html += section('Coverage', [
    card('Persona Users', cov.persona_user_count),
    card('GT Active Users', cov.gt_active_users),
    card('Sim Active Users', cov.sim_active_users),
  ]);

  const ua = METRICS.user_activity_fit || {{}};
  html += section('User Activity Fit', [
    card('Spearman Posts/User', ua.spearman_posts_per_user, 'rank correlation of per-user post counts'),
    card('Top-10 User Overlap', ua.top10_overlap, 'fraction of top-10 most active users shared'),
  ]);

  const tf = METRICS.temporal_fit || {{}};
  html += section('Temporal Fit', [
    card('Day-of-Week JSD', tf.day_of_week_jsd, 'JS divergence of weekday post distribution (lower = better)'),
    card('Cumulative Posts RMSE', tf.cumulative_posts_daily_rmse, 'daily cumulative post count RMSE vs ground truth'),
    card('Cumulative Posts NRMSE', tf.cumulative_posts_daily_nrmse, 'normalized RMSE (lower = better)'),
  ]);

  const ef = METRICS.engagement_fit || {{}};
  const efu = ef.user_level || {{}};
  const efp = ((ef.post_level || {{}}).top_share_curve || {{}});
  html += section('Engagement Fit', [
    card('GT Total Posts', ef.gt_total_posts),
    card('Sim Total Posts', ef.sim_total_posts),
    card('Spearman Engagement/User', efu.spearman_total_engagement_per_user, 'rank correlation of per-user total engagement'),
    card('JSD Engagement/User', efu.jsd_total_engagement_per_user, 'JS divergence of per-user engagement distribution (lower = better)'),
    card('Top-10 User Overlap', (efu.topk_overlap_total_engagement_per_user || {{}}).top10, 'fraction of top-10 highest-engagement users shared with GT'),
    card('Likes Top-Share RMSE', (efp.likes || {{}}).top_share_curve_rmse, 'deviation from GT likes concentration curve (lower = better)'),
    card('Replies Top-Share RMSE', (efp.replies || {{}}).top_share_curve_rmse, 'deviation from GT replies concentration curve (lower = better)'),
    card('Reposts Top-Share RMSE', (efp.retweets_reposts || {{}}).top_share_curve_rmse, 'deviation from GT retweet concentration curve (lower = better)'),
  ]);

  const ch = METRICS.community_health || {{}};
  if (Object.keys(ch).length > 0) {{
    html += section('Community Health (Absolute)', [
      card('Total Community Posts', ch.total_community_posts, 'posts authored by persona agents (excl. synthetic)'),
      card('Total Community Engagement', ch.total_community_engagement, 'likes + comments + reposts by persona agents'),
      card('Avg Engagement / Post', ch.avg_engagement_per_community_post, 'mean engagement actions received per community post'),
      card('Reply Rate', ch.reply_rate, 'fraction of posts that received at least one comment'),
      card('Unanswered Post Rate', ch.unanswered_post_rate, 'fraction of posts with zero engagement (lower = better)'),
      card('User Participation Rate', ch.user_participation_rate, 'fraction of persona users who posted at least once'),
      card('Follow Growth / Timestep', ch.follow_growth_per_timestep, 'avg new follow relationships created per timestep'),
    ]);
  }}

  const oe = METRICS.official_post_engagement || {{}};
  if (!oe.error && Object.keys(oe).length > 0) {{
    html += section('Official Post Engagement (@vechainofficial)', [
      card('GT Official Posts', oe.gt_official_post_count, 'real @vechainofficial root posts in test period'),
      card('Sim Posts Injected', oe.sim_official_post_count, 'official posts replayed into simulation'),
      card('GT Total Community Engagement', oe.gt_total_engagement, 'likes + retweets + replies + quotes in ground truth'),
      card('Sim Total Community Engagement', oe.sim_total_engagement, 'likes + comments + reposts by agents in sim'),
      card('GT Mean Engagement / Post', oe.gt_mean_engagement_per_post),
      card('Sim Mean Engagement / Post', oe.sim_mean_engagement_per_post),
      card('Spearman (per-post)', oe.spearman_engagement_per_post, 'rank correlation — higher means sim correctly identifies which VeChain posts get most community response'),
    ]);
  }}

  mc.innerHTML = html;
}})();

// ── Network graph (built lazily on first Timeline tab visit) ─────────────────
window._networkBuilt = false;
function buildNetwork() {{
  window._networkBuilt = true;
  const svg = d3.select('#network-svg');
  const W = 360, H = 320;
  const nodes = DATA.users.map(u => ({{
    id: u.user_id, name: u.user_name, bio: u.bio,
    r: 3 + Math.min(u.num_followers / 4, 7),
    active: activeUserIds.has(u.user_id),
    followers: u.num_followers, followings: u.num_followings,
  }}));
  const edgeSample = DATA.follows.length > 400
    ? DATA.follows.filter((_,i) => i % Math.ceil(DATA.follows.length/400) === 0)
    : DATA.follows;
  const links = edgeSample.map(f => ({{source: f.follower_id, target: f.followee_id}}));
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(22).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-35))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide(d => d.r + 2));
  const g = svg.append('g');
  const link = g.append('g').attr('stroke','#21262d').attr('stroke-width',0.8)
    .selectAll('line').data(links).join('line').attr('stroke-opacity',0.5);
  const node = g.append('g').selectAll('circle').data(nodes).join('circle')
    .attr('r', d => d.r)
    .attr('fill', d => d.active ? avatarColor(d.id) : '#30363d')
    .attr('stroke', d => d.active ? '#fff' : 'none')
    .attr('stroke-width', d => d.active ? 1.5 : 0)
    .style('cursor','pointer');
  const tt = document.getElementById('tooltip');
  node.on('mousemove', (evt,d) => {{
    document.getElementById('tt-name').textContent = '@' + d.name;
    document.getElementById('tt-bio').textContent = (userById[d.id]?.bio||'').slice(0,100);
    document.getElementById('tt-stats').textContent = `${{d.followers}} followers · ${{d.followings}} following${{d.active ? ' · ✓ Active' : ''}}`;
    tt.style.display='block'; tt.style.left=(evt.clientX+12)+'px'; tt.style.top=(evt.clientY+12)+'px';
  }}).on('mouseleave',()=>{{tt.style.display='none';}});
  sim.on('tick',()=>{{
    link.attr('x1',d=>Math.max(5,Math.min(W-5,d.source.x))).attr('y1',d=>Math.max(5,Math.min(H-5,d.source.y)))
        .attr('x2',d=>Math.max(5,Math.min(W-5,d.target.x))).attr('y2',d=>Math.max(5,Math.min(H-5,d.target.y)));
    node.attr('cx',d=>Math.max(d.r+2,Math.min(W-d.r-2,d.x))).attr('cy',d=>Math.max(d.r+2,Math.min(H-d.r-2,d.y)));
  }});
  svg.call(d3.zoom().scaleExtent([0.5,5]).on('zoom',e=>g.attr('transform',e.transform)));
  document.getElementById('legend').innerHTML = `
    <div class="leg-item"><div class="leg-dot" style="background:#58a6ff"></div> Active</div>
    <div class="leg-item"><div class="leg-dot" style="background:#30363d"></div> Inactive</div>
    <div class="leg-item" style="font-size:10px;color:var(--muted)">Node size ∝ followers · Scroll to zoom</div>`;
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate simulation timeline HTML.")
    parser.add_argument("--db",  default=_DEFAULT_DB,  help="Path to simulation SQLite DB")
    parser.add_argument("--csv", default=_DEFAULT_CSV, help="Path to persona CSV")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="Output HTML file path")
    args = parser.parse_args()

    db_path  = args.db  if os.path.isabs(args.db)  else os.path.join(_REPO_ROOT, args.db)
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO_ROOT, args.csv)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(_REPO_ROOT, args.out)

    if not os.path.exists(db_path):
        print(f"Error: DB not found at {db_path}")
        raise SystemExit(1)

    print(f"Loading data from {db_path}...")
    data = load_data(db_path, csv_path)
    print(f"  {len(data['users'])} agents, {len(data['sim_posts'])} sim posts, "
          f"{len(data['timesteps'])} timesteps, {len(data['personas'])} personas")

    html = generate_html(data, db_path, metrics=None)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
