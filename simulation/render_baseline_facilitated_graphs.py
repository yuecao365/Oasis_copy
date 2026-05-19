#!/usr/bin/env python3
"""Render compact baseline-vs-facilitated graph assets from metrics JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


BLUE = "#4f8edb"
GREEN = "#37a66b"
INK = "#243044"
MUTED = "#6b7280"
GRID = "#d8dee9"
BG = "#ffffff"


def _fmt(v: float, pct: bool = False) -> str:
    if pct:
        return f"{v * 100:.0f}%"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bar(
    x: float,
    y: float,
    w: float,
    label: str,
    baseline: float,
    facilitated: float,
    maxv: float,
    lower_is_better: bool = False,
    pct: bool = False,
) -> str:
    maxv = max(maxv, baseline, facilitated, 1e-9)
    b_w = w * baseline / maxv
    f_w = w * facilitated / maxv
    label = html.escape(label)
    return f"""
  <text x="{x:.1f}" y="{y:.1f}" class="label">{label}</text>
  <rect x="{x + 168:.1f}" y="{y - 12:.1f}" width="{b_w:.1f}" height="12" rx="2" fill="{BLUE}"/>
  <rect x="{x + 168:.1f}" y="{y + 8:.1f}" width="{f_w:.1f}" height="12" rx="2" fill="{GREEN}"/>
  <text x="{x + 168 + w + 8:.1f}" y="{y - 2:.1f}" class="value">{_fmt(baseline, pct)}</text>
  <text x="{x + 168 + w + 8:.1f}" y="{y + 18:.1f}" class="value">{_fmt(facilitated, pct)}</text>
"""


def _panel(title: str, x: int, y: int, w: int, h: int, body: str) -> str:
    return f"""
<g transform="translate({x},{y})">
  <rect width="{w}" height="{h}" rx="8" fill="{BG}" stroke="{GRID}" stroke-width="1.5"/>
  <text x="18" y="28" class="title">{html.escape(title)}</text>
  {body}
</g>
"""


def _legend(x: int, y: int) -> str:
    return f"""
<g transform="translate({x},{y})">
  <rect x="0" y="-10" width="13" height="13" rx="2" fill="{BLUE}"/>
  <text x="19" y="1" class="legend">Baseline</text>
  <rect x="88" y="-10" width="13" height="13" rx="2" fill="{GREEN}"/>
  <text x="107" y="1" class="legend">Facilitated</text>
</g>
"""


def render(baseline: dict, facilitated: dict, output: Path) -> None:
    b_ul = baseline["engagement_fit"]["user_level"]
    f_ul = facilitated["engagement_fit"]["user_level"]
    b_ch = baseline["community_health"]
    f_ch = facilitated["community_health"]
    volume_rows = [
        ("Community engagement", b_ch["total_community_engagement"], f_ch["total_community_engagement"], 150, False, False),
        ("Engagement per post", b_ch["avg_engagement_per_community_post"], f_ch["avg_engagement_per_community_post"], 0.60, False, False),
        ("User total engagement", b_ul["sim_total_engagement"], f_ul["sim_total_engagement"], 150, False, False),
        ("Official post engagement", baseline["official_post_engagement"]["sim_total_engagement"], facilitated["official_post_engagement"]["sim_total_engagement"], 20, False, False),
    ]
    volume_body = _legend(305, 29)
    for i, row in enumerate(volume_rows):
        volume_body += _bar(18, 70 + i * 42, 150, *row)

    response_rows = [
        ("Reply rate", b_ch["reply_rate"], f_ch["reply_rate"], 0.25, False, True),
        ("Unanswered posts", b_ch["unanswered_post_rate"], f_ch["unanswered_post_rate"], 1.00, True, True),
    ]
    response_body = _legend(305, 29)
    for i, row in enumerate(response_rows):
        response_body += _bar(18, 82 + i * 56, 150, *row)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="640" viewBox="0 0 980 640">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; fill: {INK}; }}
  .suptitle {{ font-size: 27px; font-weight: 700; }}
  .subtitle {{ font-size: 15px; fill: {MUTED}; }}
  .title {{ font-size: 18px; font-weight: 700; }}
  .label {{ font-size: 14px; font-weight: 600; }}
  .legend {{ font-size: 13px; fill: {MUTED}; }}
  .value {{ font-size: 13px; font-family: Consolas, Menlo, monospace; fill: {INK}; }}
</style>
<rect width="980" height="640" fill="#f8fafc"/>
<text x="28" y="40" class="suptitle">Engagement Performance: Baseline vs Facilitated</text>
{_panel("Engagement Volume", 28, 90, 925, 220, volume_body)}
{_panel("Response Health", 28, 330, 925, 170, response_body)}
</svg>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("simulation/metrics_history/baseline_metrics.json"))
    parser.add_argument("--facilitated", type=Path, default=Path("simulation/metrics_history/facilitated_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("simulation/graphs/baseline_vs_facilitated_results.svg"))
    args = parser.parse_args()

    render(_load(args.baseline), _load(args.facilitated), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
