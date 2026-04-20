#!/usr/bin/env python3
"""
Generate an HTML timeline dashboard from a simulation SQLite database.

Usage (from repo root):
    python simulation/generate_timeline.py
    python simulation/generate_timeline.py --db data/simu_db/yue_run.db
    python simulation/generate_timeline.py --db data/simu_db/yue_run.db --out simulation/timeline_yue.html
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_DB  = os.path.join(_REPO_ROOT, "data", "simu_db", "yue_run.db")
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "simulation_timeline.html")


def load_data(db_path: str) -> dict:
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

    # Observation log (may not exist in older DBs)
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

    timesteps = sorted(set(p["created_at"] for p in sim_posts))
    # Also include timesteps that only have observations (agents active but no posts)
    obs_timesteps = sorted(set(o["timestep"] for o in observations))
    all_timesteps = sorted(set(timesteps) | set(obs_timesteps))

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
    }


def generate_html(data: dict, db_path: str) -> str:
    data_json = json.dumps(data)
    db_label = Path(db_path).name

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Simulation — Agent Timeline ({db_label})</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {{
    --bg:#0d1117;--surface:#161b22;--surface2:#21262d;--border:#30363d;
    --text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;
    --yellow:#d29922;--purple:#bc8cff;--orange:#f78166;--pink:#f778ba;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}}
  .header{{background:var(--surface);border-bottom:1px solid var(--border);padding:24px 40px;display:flex;align-items:center;gap:20px;}}
  .header h1{{font-size:22px;font-weight:600;}}
  .header .subtitle{{color:var(--muted);font-size:13px;margin-top:2px;}}
  .badge{{background:var(--accent);color:#0d1117;font-size:11px;font-weight:700;border-radius:12px;padding:2px 10px;}}
  .stats{{display:flex;gap:0;border-bottom:1px solid var(--border);background:var(--surface);}}
  .stat{{flex:1;padding:16px 24px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:4px;}}
  .stat:last-child{{border-right:none;}}
  .stat .val{{font-size:28px;font-weight:700;color:var(--accent);}}
  .stat .lbl{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;}}
  main{{display:grid;grid-template-columns:1fr 380px;gap:0;min-height:calc(100vh - 140px);}}
  .left{{padding:32px 40px;border-right:1px solid var(--border);}}
  .right{{padding:24px;}}
  .section-title{{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:20px;display:flex;align-items:center;gap:8px;}}
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
  .seed-summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:0;}}
  .seed-card{{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:12px;}}
  .seed-card .sname{{font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .seed-card .sbio{{color:var(--muted);font-size:11px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .seed-card .scnt{{color:var(--green);font-size:11px;margin-top:4px;font-weight:600;}}
  .see-more{{font-size:12px;color:var(--muted);text-align:center;padding:8px;cursor:pointer;border:1px dashed var(--border);border-radius:6px;margin-top:8px;}}
  .see-more:hover{{color:var(--accent);border-color:var(--accent);}}
  .right h2{{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:16px;}}
  #network-svg{{width:100%;border-radius:8px;background:var(--surface);border:1px solid var(--border);}}
  .legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}}
  .leg-item{{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted);}}
  .leg-dot{{width:8px;height:8px;border-radius:50%;}}
  #tooltip{{position:fixed;background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-size:12px;pointer-events:none;z-index:100;max-width:240px;display:none;line-height:1.5;}}
  #tooltip .tt-name{{font-weight:600;color:var(--text);margin-bottom:2px;}}
  #tooltip .tt-bio{{color:var(--muted);}}
  #tooltip .tt-stats{{color:var(--accent);margin-top:4px;}}
  .agent-list{{margin-top:24px;}}
  .agent-scroll{{max-height:280px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;scrollbar-width:thin;scrollbar-color:var(--border) transparent;}}
  .agent-row{{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:6px;background:var(--surface);border:1px solid var(--border);font-size:12px;transition:border-color .1s;}}
  .agent-row:hover{{border-color:var(--accent);}}
  .agent-row .aname{{font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .agent-row .afol{{color:var(--muted);white-space:nowrap;}}
  .aactive{{width:6px;height:6px;border-radius:50%;flex-shrink:0;}}
  .aactive.yes{{background:var(--green);}}
  .aactive.no{{background:var(--border);}}
  .empty-ts{{color:var(--muted);font-size:13px;padding:12px;background:var(--surface2);border-radius:6px;border:1px solid var(--border);}}
  .obs-panel{{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin-bottom:14px;}}
  .obs-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:8px;}}
  .obs-meta{{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:4px;}}
  .obs-meta strong{{color:var(--text);}}
  .obs-legend{{display:flex;flex-wrap:wrap;gap:8px;font-size:11px;margin-top:4px;}}
  .obs-table{{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px;}}
  .obs-table th{{text-align:left;color:var(--muted);font-weight:600;padding:4px 6px;border-bottom:1px solid var(--border);}}
  .obs-table td{{padding:4px 6px;border-bottom:1px solid rgba(48,54,61,.5);vertical-align:middle;}}
  .obs-table tr:last-child td{{border-bottom:none;}}
</style>
</head>
<body>
<div id="tooltip"><div class="tt-name" id="tt-name"></div><div class="tt-bio" id="tt-bio"></div><div class="tt-stats" id="tt-stats"></div></div>
<header class="header">
  <div>
    <h1>LLM Simulation — Agent Timeline</h1>
    <div class="subtitle">qwen3.5-plus · DashScope · DB: {db_label}</div>
  </div>
  <span class="badge">RUN OUTPUT</span>
</header>
<div class="stats" id="stat-bar"></div>
<main>
  <div class="left">
    <div class="section-title">Agent Activity Timeline</div>
    <div class="timeline" id="timeline"></div>
  </div>
  <div class="right">
    <h2>Follow Network</h2>
    <svg id="network-svg" viewBox="0 0 360 340"></svg>
    <div class="legend" id="legend"></div>
    <div class="agent-list">
      <h2 style="margin-bottom:12px">All Agents</h2>
      <div class="agent-scroll" id="agent-list"></div>
    </div>
  </div>
</main>
<script>
const DATA = {data_json};
const PALETTE = ['#58a6ff','#3fb950','#bc8cff','#f78166','#f778ba','#39d353','#d29922','#79c0ff','#56d364','#e3b341'];
function avatarColor(uid) {{ return PALETTE[uid % PALETTE.length]; }}
function initials(name) {{ return (name||'?').slice(0,2).toUpperCase(); }}

const activeUserIds = new Set(DATA.sim_posts.map(p => p.user_id));
const userById = {{}};
DATA.users.forEach(u => {{ userById[u.user_id] = u; }});

// Stat bar
const statDefs = [
  {{val: DATA.users.length,           lbl: 'Total Agents'}},
  {{val: DATA.timesteps.length,        lbl: 'Timesteps Run'}},
  {{val: DATA.sim_posts.length,        lbl: 'New Posts'}},
  {{val: DATA.total_seed_posts,        lbl: 'Seed Posts'}},
  {{val: DATA.follows.length,          lbl: 'Follow Edges'}},
  {{val: activeUserIds.size,           lbl: 'Active Agents'}},
  {{val: DATA.comments.length,         lbl: 'Comments'}},
  {{val: DATA.likes.length,            lbl: 'Likes'}},
];
const statBar = document.getElementById('stat-bar');
statDefs.forEach(s => {{
  statBar.innerHTML += `<div class="stat"><div class="val">${{s.val}}</div><div class="lbl">${{s.lbl}}</div></div>`;
}});

function postType(p) {{
  if (p.original_post_id != null && p.quote_content) return 'quote';
  if (p.original_post_id != null) return 'repost';
  return 'post';
}}
const typeLabel = {{post:'Original Post',repost:'Repost',quote:'Quote',comment:'Comment'}};

function renderPostCard(p) {{
  const u = userById[p.user_id] || {{}};
  const t = postType(p);
  const raw = (p.content || '').replace(/</g,'&lt;');
  const content = t === 'quote'
    ? `<div style="border-left:2px solid var(--border);padding-left:10px;color:var(--muted);font-size:12px;margin-bottom:6px">${{raw.slice(0,120)}}…</div>${{(p.quote_content||'').replace(/</g,'&lt;')}}`
    : raw;
  return `<div class="post-card">
    <div class="post-header">
      <div class="avatar" style="background:${{avatarColor(p.user_id)}};color:#0d1117">${{initials(p.user_name)}}</div>
      <div style="flex:1;min-width:0">
        <div class="post-user">@${{p.user_name}}</div>
        <div class="post-bio">${{(u.bio||'').replace(/</g,'&lt;')}}</div>
      </div>
      <span class="post-type-badge type-${{t}}">${{typeLabel[t]}}</span>
    </div>
    <div class="post-content">${{content}}</div>
    <div class="post-footer">
      <span>♥ ${{p.num_likes}}</span><span>🔁 ${{p.num_shares}}</span>
      <span style="margin-left:auto">post #${{p.post_id}}</span>
    </div>
  </div>`;
}}

// Timeline
const tl = document.getElementById('timeline');
const seedUsers = DATA.users.filter(u => DATA.seed_counts[String(u.user_id)] > 0);
const seedHtml = seedUsers.slice(0,9).map(u =>
  `<div class="seed-card">
    <div class="sname">@${{u.user_name}}</div>
    <div class="sbio">${{(u.bio||'').slice(0,50)}}</div>
    <div class="scnt">📝 ${{DATA.seed_counts[String(u.user_id)]}} seed posts · ${{u.num_followers}} followers</div>
  </div>`).join('');
const extraHtml = seedUsers.slice(9).map(u =>
  `<div class="seed-card">
    <div class="sname">@${{u.user_name}}</div>
    <div class="sbio">${{(u.bio||'').slice(0,50)}}</div>
    <div class="scnt">📝 ${{DATA.seed_counts[String(u.user_id)]}} posts · ${{u.num_followers}} followers</div>
  </div>`).join('');

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

// ── Observation helpers ──────────────────────────────────────────────────────
const ACTION_COLORS = {{
  create_post:    '#58a6ff',
  create_comment: '#f78166',
  repost:         '#3fb950',
  like_post:      '#f778ba',
  quote_post:     '#bc8cff',
  follow:         '#39d353',
  do_nothing:     '#30363d',
}};

function renderObsPanel(obs) {{
  if (!obs || obs.length === 0) return '';

  // Action distribution
  const actionCounts = {{}};
  let totalFeedSeen = 0, totalFollowing = 0, totalRec = 0;
  obs.forEach(o => {{
    actionCounts[o.action_taken] = (actionCounts[o.action_taken] || 0) + 1;
    totalFeedSeen += o.feed_total;
    totalFollowing += o.feed_following;
    totalRec += o.feed_rec;
  }});
  const totalActions = obs.length;
  const avgFeed = totalActions ? (totalFeedSeen / totalActions).toFixed(1) : 0;
  const avgFollowing = totalActions ? (totalFollowing / totalActions).toFixed(1) : 0;
  const avgRec = totalActions ? (totalRec / totalActions).toFixed(1) : 0;

  // Action bar
  const actionOrder = ['create_post','create_comment','repost','like_post','quote_post','follow','do_nothing'];
  const barSegments = actionOrder
    .filter(a => actionCounts[a])
    .map(a => {{
      const pct = ((actionCounts[a] / totalActions) * 100).toFixed(1);
      return `<div title="${{a}}: ${{actionCounts[a]}} (${{pct}}%)" style="flex:${{actionCounts[a]}};background:${{ACTION_COLORS[a] || '#888'}};height:100%;border-radius:2px;"></div>`;
    }}).join('');

  // Agent observation rows
  const agentRows = obs.map(o => {{
    const srcBar = o.feed_total > 0
      ? `<div style="display:flex;gap:2px;height:6px;width:60px;border-radius:2px;overflow:hidden;background:var(--border)">
           <div style="flex:${{o.feed_following}};background:var(--accent);"></div>
           <div style="flex:${{o.feed_rec}};background:var(--purple);"></div>
         </div>`
      : `<span style="color:var(--muted);font-size:10px">—</span>`;
    const successDot = o.action_success
      ? `<span style="color:var(--green)">✓</span>`
      : `<span style="color:var(--muted)">·</span>`;
    return `<tr>
      <td>@${{o.username}}</td>
      <td><span style="color:${{ACTION_COLORS[o.action_taken]||'#888'}};font-size:11px">${{o.action_taken}}</span> ${{successDot}}</td>
      <td style="text-align:center">${{o.feed_total}}</td>
      <td>${{srcBar}}</td>
    </tr>`;
  }}).join('');

  return `<div class="obs-panel">
    <div class="obs-title">Observation Space</div>
    <div class="obs-meta">
      <span>${{totalActions}} agent actions</span>
      <span>avg feed <strong>${{avgFeed}}</strong> posts</span>
      <span style="color:var(--accent)">■ following ${{avgFollowing}}</span>
      <span style="color:var(--purple)">■ rec ${{avgRec}}</span>
    </div>
    <div style="display:flex;height:14px;gap:2px;border-radius:4px;overflow:hidden;margin:8px 0">${{barSegments}}</div>
    <div class="obs-legend">
      ${{Object.entries(actionCounts).map(([a,c])=>`<span style="color:${{ACTION_COLORS[a]||'#888'}}">${{a}} ${{c}}</span>`).join('')}}
    </div>
    <details style="margin-top:10px">
      <summary style="font-size:11px;color:var(--muted);cursor:pointer;user-select:none">Per-agent breakdown</summary>
      <table class="obs-table">
        <thead><tr><th>Agent</th><th>Action</th><th>Feed</th><th>Following/Rec</th></tr></thead>
        <tbody>${{agentRows}}</tbody>
      </table>
    </details>
  </div>`;
}}

DATA.timesteps.forEach(ts => {{
  const posts    = DATA.sim_posts.filter(p => p.created_at === ts);
  const comments = DATA.comments.filter(c => c.created_at === ts);
  const likes    = DATA.likes.filter(l => l.created_at === ts);
  const obs      = (DATA.observations || []).filter(o => o.timestep === ts);
  const unique   = new Set(obs.map(o => o.agent_id));
  const stepHours = ts * 12;  // half-day steps
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
  tl.innerHTML += `<div class="empty-ts" style="margin-top:8px">No simulation timesteps recorded yet. Run the simulation first.</div>`;
}}

// Network graph
const svg = d3.select('#network-svg');
const W = 360, H = 340;
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
  document.getElementById('tt-stats').textContent =
    `${{d.followers}} followers · ${{d.followings}} following${{d.active ? ' · ✓ Active' : ''}}`;
  tt.style.display='block'; tt.style.left=(evt.clientX+12)+'px'; tt.style.top=(evt.clientY+12)+'px';
}}).on('mouseleave',()=>{{tt.style.display='none';}});
sim.on('tick',()=>{{
  link.attr('x1',d=>Math.max(5,Math.min(W-5,d.source.x))).attr('y1',d=>Math.max(5,Math.min(H-5,d.source.y)))
      .attr('x2',d=>Math.max(5,Math.min(W-5,d.target.x))).attr('y2',d=>Math.max(5,Math.min(H-5,d.target.y)));
  node.attr('cx',d=>Math.max(d.r+2,Math.min(W-d.r-2,d.x))).attr('cy',d=>Math.max(d.r+2,Math.min(H-d.r-2,d.y)));
}});
svg.call(d3.zoom().scaleExtent([0.5,5]).on('zoom',e=>g.attr('transform',e.transform)));
document.getElementById('legend').innerHTML = `
  <div class="leg-item"><div class="leg-dot" style="background:#58a6ff"></div> Active agent</div>
  <div class="leg-item"><div class="leg-dot" style="background:#30363d"></div> Inactive agent</div>
  <div class="leg-item" style="font-size:10px;color:var(--muted)">Node size ∝ followers · Scroll to zoom</div>`;

// Agent list
const agentList = document.getElementById('agent-list');
const sorted = [...DATA.users].sort((a,b) => {{
  const d = (activeUserIds.has(b.user_id)?1:0) - (activeUserIds.has(a.user_id)?1:0);
  return d || b.num_followers - a.num_followers;
}});
agentList.innerHTML = sorted.map(u => `
  <div class="agent-row">
    <div class="aactive ${{activeUserIds.has(u.user_id)?'yes':'no'}}"></div>
    <div class="avatar" style="width:24px;height:24px;border-radius:50%;background:${{avatarColor(u.user_id)}};color:#0d1117;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0">${{initials(u.user_name)}}</div>
    <div class="aname">@${{u.user_name}}</div>
    <div class="afol">${{u.num_followers}} · ${{u.num_followings}}</div>
  </div>`).join('');
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate simulation timeline HTML.")
    parser.add_argument("--db",  default=_DEFAULT_DB,  help="Path to simulation SQLite DB")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="Output HTML file path")
    args = parser.parse_args()

    db_path = args.db if os.path.isabs(args.db) else os.path.join(_REPO_ROOT, args.db)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(_REPO_ROOT, args.out)

    if not os.path.exists(db_path):
        print(f"Error: DB not found at {db_path}")
        raise SystemExit(1)

    print(f"Loading data from {db_path}...")
    data = load_data(db_path)
    print(f"  {len(data['users'])} agents, {len(data['sim_posts'])} sim posts, {len(data['timesteps'])} timesteps")

    html = generate_html(data, db_path)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
