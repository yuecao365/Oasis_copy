#!/usr/bin/env python3
"""
Generate an HTML dashboard for persona outputs.

Supports:
- persona_generation/outputs/user_personas.json
- persona_generation/outputs/user_personas.csv

Output is written to the existing visualizations directory.
"""

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import OUTPUT_DIR, VISUALIZATIONS_DIR

DEFAULT_JSON = OUTPUT_DIR / "user_personas.json"
DEFAULT_CSV = OUTPUT_DIR / "user_personas.csv"
DEFAULT_HTML = VISUALIZATIONS_DIR / "personas_dashboard.html"

ACTION_PROB_COLUMNS = [
    ("create_post", "p_create_post"),
    ("create_comment", "p_create_comment"),
    ("do_nothing", "p_do_nothing"),
    ("repost", "p_repost"),
    ("quote_post", "p_quote_post"),
    ("like_post", "p_like_post"),
    ("follow", "p_follow"),
]

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def parse_json_list(value: str) -> List:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def normalize_json_personas(data: List[Dict]) -> List[Dict]:
    personas: List[Dict] = []
    for idx, p in enumerate(data):
        categories = p.get("all_categories")
        if not isinstance(categories, list):
            categories = [str(p.get("category", "Uncategorized"))]

        sample_tweets = p.get("sample_tweets")
        if not isinstance(sample_tweets, list):
            sample_tweets = []

        ap = p.get("action_probs") or p.get("action_probabilities") or {}
        if not isinstance(ap, dict):
            ap = {}
        
        personas.append(
            {
                "id": str(p.get("original_user_id") or p.get("user_id") or idx),
                "name": str(p.get("realname", p.get("username", "Unknown"))),
                "username": str(p.get("username", "unknown")),
                "bio": str(p.get("bio", "")),
                "persona": str(p.get("persona", "")),
                "category": str(p.get("category", "Uncategorized")),
                "all_categories": [str(c) for c in categories],
                "age": p.get("age", ""),
                "gender": str(p.get("gender", "")),
                "mbti": str(p.get("mbti", "")),
                "country": str(p.get("country", "")),
                "profession": str(p.get("profession", "")),
                "tweet_count": p.get("tweet_count", 0),
                "sample_tweets": [str(t) for t in sample_tweets],
                "following_count": 0,
                "source_format": "json",
                "action_probs": {k: safe_float(v, 0.0) for k, v in ap.items()},
            }
        )


    return personas

def normalize_csv_personas(rows: List[Dict]) -> List[Dict]:
    personas: List[Dict] = []
    for idx, r in enumerate(rows):
        following = parse_json_list(r.get("following_agentid_list", "[]"))
        tweets = parse_json_list(r.get("previous_tweets", "[]"))

        # PRIORITY: 1) category_label (new CSV) 2) category_description 3) category 4) fallback
        category = (
            r.get("category_label")
            or r.get("category_description")
            or r.get("category")
            or "Unspecified"
        )
        category = str(category).strip() or "Unspecified"

        all_categories = r.get("all_categories")
        if isinstance(all_categories, str):
            all_categories = parse_json_list(all_categories)
        
        # Ensure we have at least the primary category in the list
        if not isinstance(all_categories, list) or not all_categories:
            all_categories = [category]

        action_probs = {}
        for action, col in ACTION_PROB_COLUMNS:
            action_probs[action] = safe_float(r.get(col, 0.0), 0.0)

        personas.append(
            {
                "id": str(r.get("user_id", idx)),
                "name": str(r.get("name") or r.get("username") or "Unknown"),
                "username": str(r.get("username", "unknown")),
                "bio": str(r.get("description", "")),
                "persona": str(r.get("user_char", "")),
                "category": category,
                "all_categories": [str(c) for c in all_categories],
                "age": "",
                "gender": "",
                "mbti": "",
                "country": "",
                "profession": "",
                "tweet_count": len(tweets),
                "sample_tweets": [str(t) for t in tweets],
                "following_count": len(following),
                "source_format": "csv",
                "action_probs": action_probs,
            }
        )
    return personas

def load_personas(input_path: Path | None) -> List[Dict]:
    if input_path:
        source = input_path
    else:
        candidates = [p for p in (DEFAULT_JSON, DEFAULT_CSV) if p.exists()]
        if not candidates:
            raise FileNotFoundError(
                f"No persona files found. Expected one of: {DEFAULT_JSON} or {DEFAULT_CSV}"
            )
        source = max(candidates, key=lambda p: p.stat().st_mtime)

    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {source}")
        personas = normalize_json_personas(data)
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            personas = normalize_csv_personas(list(reader))
    else:
        raise ValueError(f"Unsupported input format: {source.suffix}")

    print(f"Loaded {len(personas)} personas from {source}")
    return personas


def build_html(personas: List[Dict]) -> str:
    total = len(personas)
    categories = sorted({p["category"] for p in personas})
    avg_tweets = round(sum(int(p.get("tweet_count", 0) or 0) for p in personas) / total, 2) if total else 0

    def render_action_probs(ap: Dict[str, float]) -> str:
        if not isinstance(ap, dict) or not ap:
            return '<div class="action-probs muted">No action probabilities</div>'

        # Ensure stable order + clamp values to [0,1]
        items = []
        for action, col in ACTION_PROB_COLUMNS:
            v = ap.get(action, 0.0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            v = max(0.0, min(1.0, v))
            pct = int(round(v * 100))
            items.append((action, v, pct))

        rows = []
        for action, v, pct in items:
            rows.append(f"""
              <div class="ap-row">
                <div class="ap-label">{html.escape(action)}</div>
                <div class="ap-bar"><div class="ap-fill" style="width:{pct}%"></div></div>
                <div class="ap-val">{pct}%</div>
              </div>
            """)
        return f"""
          <div class="action-probs">
            <strong>Action probabilities</strong>
            <div class="ap-grid">
              {''.join(rows)}
            </div>
          </div>
        """

    category_options = ['<option value="all">All categories</option>']
    for c in categories:
        category_options.append(f'<option value="{html.escape(c)}">{html.escape(c)}</option>')
    category_options_html = "\n".join(category_options)

    cards = []
    for p in personas:
        tags = [p.get("age"), p.get("gender"), p.get("mbti"), p.get("country"), p.get("profession")]
        tags = [html.escape(str(t)) for t in tags if str(t).strip()]
        if p.get("source_format") == "csv":
            tags.append(f"following:{int(p.get('following_count', 0) or 0)}")

        tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags) if tags else '<span class="tag muted">No demographics</span>'
        
        action_probs_html = render_action_probs(p.get("action_probs", {}))

        tweets = p.get("sample_tweets", [])
        tweets_html = "".join(f"<li>{html.escape(str(t))}</li>" for t in tweets[:5]) if tweets else "<li>No sample tweets</li>"
        categories_html = ", ".join(html.escape(c) for c in p.get("all_categories", []))



        card = f"""
        <article class="card" data-category="{html.escape(p['category'])}" data-search="{html.escape((p['name'] + ' ' + p['username'] + ' ' + p['bio'] + ' ' + p['persona']).lower())}">
          <div class="header-row">
            <div>
              <h2>{html.escape(p['name'])}</h2>
              <div class="handle">@{html.escape(p['username'])}</div>
            </div>
            <div class="pill">{html.escape(p['category'])}</div>
          </div>
          <p class="bio">{html.escape(p['bio'])}</p>
          <p class="persona">{html.escape(p['persona'])}</p>
          <div class="meta">{tags_html}</div>
          {action_probs_html}
          <div class="categories"><strong>All categories:</strong> {categories_html or "N/A"}</div>
          <div class="tweets">
            <strong>Sample tweets</strong>
            <ul>{tweets_html}</ul>
          </div>
        </article>
        """
        cards.append(card)

    cards_html = "\n".join(cards)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Persona Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f2;
      --panel: #fffdf7;
      --ink: #11212d;
      --muted: #5d6b75;
      --accent: #0d9488;
      --accent-2: #f59e0b;
      --line: #d8ddd2;
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
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 2rem; }}
    .sub {{ color: var(--muted); margin-bottom: 14px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }}
    .stat {{ background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 12px; }}
    .stat .n {{ font-size: 1.3rem; font-weight: 700; color: var(--accent); }}
    .controls {{
      margin: 18px 0;
      display: grid;
      grid-template-columns: 1fr 220px;
      gap: 10px;
    }}
    input, select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 0.95rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 6px 18px rgba(17, 33, 45, 0.06);
    }}
    .header-row {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: start;
    }}
    h2 {{ margin: 0; font-size: 1.1rem; }}
    .handle {{ color: var(--muted); font-size: 0.9rem; }}
    .pill {{
      background: linear-gradient(120deg, var(--accent), #14b8a6);
      color: #fff;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.76rem;
      white-space: nowrap;
    }}
    .bio {{ margin: 12px 0 8px; color: #22313d; }}
    .persona {{ margin: 0 0 10px; color: #22313d; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .tag {{
      font-size: 0.75rem;
      background: #ecfeff;
      border: 1px solid #99f6e4;
      color: #134e4a;
      border-radius: 999px;
      padding: 3px 8px;
    }}
    .tag.muted {{ background: #f6f7f2; border-color: var(--line); color: var(--muted); }}
    .categories {{ font-size: 0.86rem; color: var(--muted); margin-bottom: 8px; }}
    .tweets strong {{ display: block; margin-bottom: 5px; }}
    ul {{ margin: 0; padding-left: 18px; color: #2f3e48; }}
    li {{ margin: 4px 0; }}
    .hidden {{ display: none; }}
    @media (max-width: 760px) {{
      .controls {{ grid-template-columns: 1fr; }}
    }}
    
    .action-probs {{ margin: 10px 0 8px; border-top: 1px dashed var(--line); padding-top: 10px; }}
    .action-probs.muted {{ color: var(--muted); }}
    .action-probs strong {{ display: block; margin-bottom: 8px; color: var(--ink); }}

    .ap-grid {{ display: grid; gap: 6px; }}
    .ap-row {{
      display: grid;
      grid-template-columns: 120px 1fr 44px;
      gap: 8px;
      align-items: center;
      font-size: 0.84rem;
      color: #2f3e48;
    }}
    .ap-label {{
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.78rem;
    }}
    .ap-bar {{
      height: 10px;
      border-radius: 999px;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      overflow: hidden;
    }}
    .ap-fill {{
      height: 100%;
      background: linear-gradient(120deg, var(--accent), #14b8a6);
      border-radius: 999px;
    }}
    .ap-val {{
      text-align: right;
      color: #0f766e;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Persona Dashboard</h1>
      <p class="sub">Interactive view of generated personas.</p>
      <div class="stats">
        <div class="stat"><div class="n">{total}</div><div>Personas</div></div>
        <div class="stat"><div class="n">{len(categories)}</div><div>Primary Categories</div></div>
        <div class="stat"><div class="n">{avg_tweets}</div><div>Avg Sample Tweets</div></div>
      </div>
    </section>

    <section class="controls">
      <input id="searchBox" placeholder="Search name, username, bio, persona..." />
      <select id="categoryFilter">
        {category_options_html}
      </select>
    </section>

    <section class="grid" id="cards">
      {cards_html}
    </section>
  </div>

  <script>
    const searchBox = document.getElementById("searchBox");
    const categoryFilter = document.getElementById("categoryFilter");
    const cards = Array.from(document.querySelectorAll(".card"));

    function applyFilters() {{
      const q = (searchBox.value || "").toLowerCase().trim();
      const cat = categoryFilter.value;
      for (const card of cards) {{
        const text = card.dataset.search || "";
        const c = card.dataset.category || "";
        const matchesText = !q || text.includes(q);
        const matchesCat = cat === "all" || c === cat;
        card.classList.toggle("hidden", !(matchesText && matchesCat));
      }}
    }}

    searchBox.addEventListener("input", applyFilters);
    categoryFilter.addEventListener("change", applyFilters);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an HTML dashboard for persona outputs.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Optional input persona file (.json or .csv). Defaults to latest of outputs/user_personas.(json|csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_HTML,
        help=f"Output HTML path (default: {DEFAULT_HTML}).",
    )
    args = parser.parse_args()

    personas = load_personas(args.input)

    output_path = args.output
    if output_path.resolve().parent != VISUALIZATIONS_DIR.resolve():
        output_path = VISUALIZATIONS_DIR / output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_doc = build_html(personas)
    output_path.write_text(html_doc, encoding="utf-8")

    print(f"Wrote dashboard to {output_path}")


if __name__ == "__main__":
    main()
