"""
Generate all simulation result figures for the slide deck.

Outputs (in simulation/figures/):
  fig1_engagement_improvement.png  — % improvement from baseline, both persona sets
  fig2_community_health.png        — absolute metric bars side-by-side, both persona sets
  fig3_agent_roles.png             — three synthetic agent role cards
  fig4_pipeline.png                — simulation loop diagram (baseline vs facilitated)
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__))
METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "metrics_history")

# ── palette ──────────────────────────────────────────────────────────────────
C_BASE   = "#94A3B8"   # slate gray
C_FACIL  = "#F97316"   # orange
C_TEXT   = "#1E293B"
C_MUTED  = "#64748B"
C_GREEN  = "#16A34A"
C_RED    = "#DC2626"
C_BG     = "white"
C_GRID   = "#F1F5F9"
C_BORDER = "#E2E8F0"

# Use Calibri if available, fall back to DejaVu Sans
_calibri = next(
    (f.name for f in font_manager.fontManager.ttflist if "Calibri" in f.name),
    "DejaVu Sans",
)
plt.rcParams["font.family"] = _calibri


def load(name):
    path = os.path.join(METRICS_DIR, name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    if isinstance(d, float):
        return round(d, 4)
    return d


def pct_change(new, old, invert=False):
    """Return % improvement. invert=True for lower-is-better metrics."""
    if old is None or new is None or old == 0:
        return None
    if invert:
        return (old - new) / abs(old) * 100
    return (new - old) / abs(old) * 100


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Improvement bars
# ─────────────────────────────────────────────────────────────────────────────
def fig1_improvement():
    yb = load("baseline_metrics.json")
    yf = load("facilitated_metrics.json")
    eb = load("evelyn_baseline_run_metrics.json")
    ef = load("evelyn_facilitated_metrics.json")

    # (display label, calculation note, bl_val, fa_val, invert, fmt)
    def all_metrics(bl, fa):
        return [
            ("Reply Rate",
             "posts with ≥1 comment ÷ total posts",
             get(bl, "community_health", "reply_rate"),
             get(fa, "community_health", "reply_rate"),
             False, "{:.1%}"),
            ("Posts with Replies",
             "1 − (posts with 0 replies ÷ total posts)",
             1 - (get(bl, "community_health", "unanswered_post_rate") or 0),
             1 - (get(fa, "community_health", "unanswered_post_rate") or 0),
             False, "{:.1%}"),
            ("Avg Engagement / Post",
             "(likes + comments + reposts) ÷ total posts",
             get(bl, "community_health", "avg_engagement_per_community_post"),
             get(fa, "community_health", "avg_engagement_per_community_post"),
             False, "{:.2f}"),
            ("Total Community Engagement",
             "sum of all likes + comments + reposts",
             get(bl, "community_health", "total_community_engagement"),
             get(fa, "community_health", "total_community_engagement"),
             False, "{:.0f}"),
        ]

    def positive_only(bl, fa):
        return [
            m for m in all_metrics(bl, fa)
            if (pct_change(m[3], m[2], invert=m[4]) or 0) > 0
        ]

    datasets = [
        ("Coarse-Grained Personas\n(k=4, 40 personas)", positive_only(yb, yf)),
        ("Fine-Grained Personas\n(k=19, 143 personas)", positive_only(eb, ef)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=C_BG)
    fig.suptitle(
        "Community Engagement: Original vs. Facilitated (3 Synthetic Agents)",
        fontsize=13, fontweight="bold", color=C_TEXT, y=1.03,
    )

    for ax, (title, mlist) in zip(axes, datasets):
        ax.set_facecolor(C_BG)
        labels   = [m[0] for m in mlist]
        calcs    = [m[1] for m in mlist]
        changes  = [pct_change(m[3], m[2], invert=m[4]) for m in mlist]
        abs_base = [m[2] for m in mlist]
        abs_facil= [m[3] for m in mlist]
        fmts     = [m[5] for m in mlist]

        n = len(labels)
        # Spread bars out more so calc notes have room
        y = np.arange(n) * 1.6
        vals = [c if c is not None else 0 for c in changes]

        bars = ax.barh(y, vals, height=0.6, color=C_GREEN, zorder=3, alpha=0.88)
        ax.axvline(0, color=C_TEXT, linewidth=1.0, zorder=4)

        # Annotations to the right of each bar
        for bar, bv, fv, fmt, chg in zip(bars, abs_base, abs_facil, fmts, changes):
            if bv is not None and fv is not None:
                bstr = fmt.format(bv)
                fstr = fmt.format(fv)
                chg_str = f"+{chg:.0f}%" if chg is not None else ""
                x_pos = bar.get_width()
                ax.text(
                    x_pos + max(vals) * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bstr} → {fstr}  ({chg_str})",
                    va="center", ha="left", fontsize=10, color=C_TEXT,
                    fontweight="bold" if chg is not None and chg >= 15 else "normal",
                )

        # Custom y-axis labels: metric name + calc note below
        ax.set_yticks([])
        for i, (lbl, calc, yi) in enumerate(zip(labels, calcs, y)):
            ax.text(-max(vals) * 0.04, yi + 0.15, lbl,
                    ha="right", va="center", fontsize=11, fontweight="bold",
                    color=C_TEXT)
            ax.text(-max(vals) * 0.04, yi - 0.22, calc,
                    ha="right", va="center", fontsize=8, color=C_MUTED,
                    style="italic")

        # X-axis
        max_val = max(vals) if vals else 10
        ax.set_xlim(-max_val * 0.6, max_val * 2.2)
        ax.set_ylim(-0.8, max(y) + 1.0)
        ax.set_xlabel("% improvement from original", fontsize=9, color=C_MUTED)
        ax.tick_params(axis="x", colors=C_MUTED, labelsize=8.5)

        # Spines
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(C_BORDER)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)

        ax.set_title(title, fontsize=11.5, fontweight="bold", color=C_TEXT, pad=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_engagement_improvement.svg")
    plt.savefig(out, format="svg", bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Absolute metric comparison (grouped bars)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_absolute():
    yb = load("baseline_metrics.json")
    yf = load("facilitated_metrics.json")
    eb = load("evelyn_baseline_run_metrics.json")
    ef = load("evelyn_facilitated_metrics.json")

    # Two rows of charts: top = yue, bottom = evelyn
    # Each row: 4 metric mini-charts side by side
    metric_defs = [
        ("Reply Rate",                   "community_health.reply_rate",             "{:.1%}"),
        ("Posts with\nReplies",          "community_health.unanswered_post_rate",   "{:.1%}", True),  # invert
        ("Avg Engagement\n/ Post",       "community_health.avg_engagement_per_community_post", "{:.2f}"),
        ("Total Community\nEngagement",  "community_health.total_community_engagement", "{:.0f}"),
    ]

    def _get(d, dotpath, invert=False):
        keys = dotpath.split(".")
        v = get(d, *keys)
        if invert and v is not None:
            v = round(1 - v, 4)
        return v

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor=C_BG)
    fig.suptitle(
        "Community Health Metrics — Baseline vs. Facilitated",
        fontsize=14, fontweight="bold", color=C_TEXT, y=1.01,
    )

    row_data = [
        ("Yue (40 personas)", yb, yf),
        ("Evelyn (143 personas)", eb, ef),
    ]

    for row_i, (row_label, bl, fa) in enumerate(row_data):
        # Row label on leftmost axis
        axes[row_i][0].set_ylabel(row_label, fontsize=11, fontweight="bold",
                                   color=C_TEXT, labelpad=12)
        for col_i, mdef in enumerate(metric_defs):
            ax = axes[row_i][col_i]
            ax.set_facecolor(C_BG)

            label = mdef[0]
            dotpath = mdef[1]
            fmt = mdef[2] if len(mdef) > 2 else None
            invert = mdef[3] if len(mdef) > 3 else False

            bv = _get(bl, dotpath, invert)
            fv = _get(fa, dotpath, invert)

            if bv is None or fv is None:
                ax.set_visible(False)
                continue

            bars = ax.bar(
                ["Baseline", "Facilitated"], [bv, fv],
                color=[C_BASE, C_FACIL], width=0.5, zorder=3,
            )

            # Value labels on bars
            for bar, val in zip(bars, [bv, fv]):
                label_str = fmt.format(val) if fmt else f"{val:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(bv, fv) * 0.02,
                    label_str, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C_TEXT,
                )

            # Delta arrow annotation
            delta = fv - bv
            sign = "+" if delta >= 0 else ""
            color = C_GREEN if delta >= 0 else C_RED
            delta_str = fmt.format(abs(delta)) if fmt else f"{abs(delta):.1f}"
            ax.text(
                0.5, 0.97,
                f"{sign}{fmt.format(delta) if fmt else f'{delta:+.1f}'}",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=color,
            )

            # Style
            ax.set_title(mdef[0].replace("\n", " "), fontsize=9.5, color=C_TEXT, pad=6)
            ax.set_ylim(0, max(bv, fv) * 1.25)
            ax.tick_params(axis="x", labelsize=8.5, colors=C_MUTED)
            ax.tick_params(axis="y", labelsize=7.5, colors=C_MUTED)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color(C_BORDER)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_BASE, label="Baseline (no agents)"),
        mpatches.Patch(color=C_FACIL, label="Facilitated (3 synthetic agents)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_community_health.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: Agent role cards
# ─────────────────────────────────────────────────────────────────────────────
def fig3_agent_roles():
    agents = [
        {
            "name": "Community Manager",
            "handle": "@community_manager",
            "color": "#3B82F6",
            "light": "#EFF6FF",
            "actions": ["Likes unanswered posts", "Comments to acknowledge authors", "Reposts quality content"],
            "feed": "Prioritizes posts with 0 replies",
            "metric": "Unanswered post rate",
            "effect": "Ensures no voice goes unheard;\nreduces posts with zero engagement",
        },
        {
            "name": "Domain Expert",
            "handle": "@vechain_analyst",
            "color": "#8B5CF6",
            "light": "#F5F3FF",
            "actions": ["Likes high-quality posts", "Reposts insightful content", "Adds expert commentary"],
            "feed": "Sees full default feed ranked by likes",
            "metric": "Engagement per post",
            "effect": "Amplifies the best community content;\ndrives engagement toward substantive posts",
        },
        {
            "name": "Discussion Facilitator",
            "handle": "@vechain_connector",
            "color": "#F97316",
            "light": "#FFF7ED",
            "actions": ["Comments with open-ended questions", "Quote-tweets to spark discussion", "Follows active community members"],
            "feed": "Sees full default feed",
            "metric": "Reply rate",
            "effect": "Turns monologues into dialogues;\npulls community members into conversation",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor=C_BG)
    fig.suptitle(
        "Synthetic Engagement Agents",
        fontsize=20, fontweight="bold", color=C_TEXT, y=1.02,
    )

    for ax, agent in zip(axes, agents):
        ax.set_facecolor(agent["light"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Border
        ax.add_patch(FancyBboxPatch(
            (0.08, 0.08), 9.84, 9.84,
            boxstyle="round,pad=0.2",
            facecolor=agent["light"], edgecolor=agent["color"],
            linewidth=3.0, transform=ax.transData,
        ))

        # Header bar
        ax.add_patch(FancyBboxPatch(
            (0.08, 7.9), 9.84, 2.02,
            boxstyle="round,pad=0.1",
            facecolor=agent["color"], edgecolor="none",
            transform=ax.transData,
        ))

        ax.text(5, 9.18, agent["name"], ha="center", va="center",
                fontsize=18, fontweight="bold", color="white")
        ax.text(5, 8.28, agent["handle"], ha="center", va="center",
                fontsize=12, color="white", alpha=0.88)

        # Actions section
        ax.text(0.55, 7.55, "ACTIONS", fontsize=10, fontweight="bold",
                color=agent["color"], va="top")
        for i, action in enumerate(agent["actions"]):
            ax.text(0.65, 7.0 - i * 1.0, f"• {action}",
                    fontsize=12.5, color=C_TEXT, va="top")

        # Divider
        ax.plot([0.4, 9.6], [3.9, 3.9], color=agent["color"], alpha=0.3, linewidth=1.2)

        # Feed mode
        ax.text(0.55, 3.65, "FEED MODE", fontsize=10, fontweight="bold",
                color=agent["color"], va="top")
        ax.text(0.65, 3.1, agent["feed"], fontsize=12, color=C_TEXT, va="top",
                style="italic")

        # Divider
        ax.plot([0.4, 9.6], [2.35, 2.35], color=agent["color"], alpha=0.3, linewidth=1.2)

        # Key metric
        ax.text(0.55, 2.12, "KEY METRIC IMPROVED", fontsize=10, fontweight="bold",
                color=agent["color"], va="top")

        ax.add_patch(FancyBboxPatch(
            (0.4, 0.85), 9.2, 0.95,
            boxstyle="round,pad=0.1",
            facecolor=agent["color"], edgecolor="none", alpha=0.15,
            transform=ax.transData,
        ))
        ax.text(5, 1.32, agent["metric"], ha="center", va="center",
                fontsize=14, fontweight="bold", color=agent["color"])

    plt.tight_layout(pad=1.5)
    out = os.path.join(OUT_DIR, "fig3_agent_roles.svg")
    plt.savefig(out, format="svg", bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Pipeline diagram (baseline vs facilitated)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_pipeline():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=C_BG)
    fig.suptitle(
        "Simulation Pipeline",
        fontsize=16, fontweight="bold", color=C_TEXT, y=1.02,
    )

    # Shared geometry
    BW = 5.2   # box width
    BH = 1.1   # box height
    # Y positions (plenty of breathing room)
    Y_OFF  = 11.5
    Y_FEED = 9.0
    Y_AGT  = 6.0
    Y_DB   = 3.0
    CX     = 5.0   # center x for single-column layouts
    LOOP_X = 0.8   # x for the feedback loop line

    def mkbox(ax, cx, cy, w, h, title, subtitle=None,
              bg="#F8FAFC", border=C_BORDER, title_color=C_TEXT,
              title_size=11, sub_size=9.5):
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.2",
            facecolor=bg, edgecolor=border, linewidth=2.0,
            transform=ax.transData,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(cx, cy + 0.18, title, ha="center", va="center",
                    fontsize=title_size, fontweight="bold", color=title_color)
            ax.text(cx, cy - 0.28, subtitle, ha="center", va="center",
                    fontsize=sub_size, color=C_MUTED)
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=title_size, fontweight="bold", color=title_color)

    def arr(ax, x0, y0, x1, y1, color=C_MUTED, lw=1.8, rad=0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14,
                        connectionstyle=f"arc3,rad={rad}",
                    ))

    def feedback_loop(ax, y_top, y_bot):
        # L-shaped feedback: DB left edge → left margin → Feed left edge
        db_left  = CX - BW / 2
        feed_left = CX - BW / 2
        ax.plot([db_left, LOOP_X], [y_bot, y_bot],
                color=C_MUTED, lw=1.6, solid_capstyle="round")
        ax.plot([LOOP_X, LOOP_X], [y_bot, y_top],
                color=C_MUTED, lw=1.6, solid_capstyle="round")
        ax.annotate("", xy=(feed_left, y_top), xytext=(LOOP_X, y_top),
                    arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.6,
                                    mutation_scale=14))
        ax.text(LOOP_X - 0.45, (y_top + y_bot) / 2, "updates\nfeed",
                ha="center", va="center", fontsize=9, color=C_MUTED,
                style="italic")

    def draw(ax, title, synthetic=False):
        ax.set_xlim(0, 10)
        ax.set_ylim(1.5, 13)
        ax.axis("off")
        ax.set_facecolor(C_BG)
        ax.set_title(title, fontsize=13, fontweight="bold", color=C_TEXT, pad=14)

        # ── Official Posts ──
        mkbox(ax, CX, Y_OFF, BW, BH,
              "Official Posts",
              "VeChain account — injected each timestep",
              bg="#F0FDF4", border="#4ADE80")

        # ── Feed Builder ──
        mkbox(ax, CX, Y_FEED, BW, BH,
              "Recommendation Feed",
              "following + random",
              bg="#F8FAFC", border=C_BORDER)

        arr(ax, CX, Y_OFF - BH/2, CX, Y_FEED + BH/2)

        if synthetic:
            # ── Persona Agents (left) ──
            px = 3.0
            mkbox(ax, px, Y_AGT, 4.6, BH + 0.3,
                  "Persona Agents",
                  "LLM-driven action each timestep",
                  bg="#F8FAFC", border=C_BORDER)

            # Feed → Persona
            arr(ax, px, Y_FEED - BH/2, px, Y_AGT + (BH+0.3)/2,
                rad=0.15)

            # ── Synthetic Agents (right) ──
            sx = 7.6
            synth_h = BH + 1.4
            synth_box = FancyBboxPatch(
                (sx - 2.5, Y_AGT - synth_h/2), 5.0, synth_h,
                boxstyle="round,pad=0.2",
                facecolor="#FFF7ED", edgecolor=C_FACIL, linewidth=2.5,
                transform=ax.transData,
            )
            ax.add_patch(synth_box)
            ax.text(sx, Y_AGT + synth_h/2 - 0.38, "Synthetic Agents",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=C_FACIL)
            ax.text(sx, Y_AGT + synth_h/2 - 0.78, "always active · LLM-driven",
                    ha="center", va="center", fontsize=9,
                    color=C_FACIL, style="italic")
            for i, name in enumerate(["Community Manager",
                                      "Domain Expert",
                                      "Discussion Facilitator"]):
                ax.text(sx, Y_AGT + 0.18 - i * 0.52, f"• {name}",
                        ha="center", va="center", fontsize=9.5, color=C_TEXT)

            # Feed → Synthetic
            arr(ax, sx, Y_FEED - BH/2, sx, Y_AGT + synth_h/2,
                color=C_FACIL, lw=2.2, rad=-0.15)

            # ── Database ──
            mkbox(ax, CX, Y_DB, BW + 1.0, BH,
                  "Simulation Database",
                  "posts · likes · comments · reposts · follows",
                  bg="#F8FAFC", border=C_BORDER)

            # Persona → DB
            arr(ax, px, Y_AGT - (BH+0.3)/2, CX - 0.8, Y_DB + BH/2)
            # Synthetic → DB
            arr(ax, sx, Y_AGT - synth_h/2, CX + 0.8, Y_DB + BH/2,
                color=C_FACIL, lw=2.0)

        else:
            # ── Persona Agents (centered) ──
            mkbox(ax, CX, Y_AGT, BW, BH + 0.3,
                  "Persona Agents",
                  "LLM-driven action each timestep",
                  bg="#F8FAFC", border=C_BORDER)

            arr(ax, CX, Y_FEED - BH/2, CX, Y_AGT + (BH+0.3)/2)

            # ── Database ──
            mkbox(ax, CX, Y_DB, BW + 1.0, BH,
                  "Simulation Database",
                  "posts · likes · comments · reposts · follows",
                  bg="#F8FAFC", border=C_BORDER)

            arr(ax, CX, Y_AGT - (BH+0.3)/2, CX, Y_DB + BH/2)

        # Feedback loop
        feedback_loop(ax, Y_FEED, Y_DB)

        # Timestep note
        ax.text(CX, 2.1, "48 timesteps  x  12 hours each",
                ha="center", va="center", fontsize=9.5,
                color=C_MUTED, style="italic")

    draw(axes[0], "Original", synthetic=False)
    draw(axes[1], "Facilitated  (+3 Synthetic Agents)", synthetic=True)

    plt.tight_layout(w_pad=3)
    out = os.path.join(OUT_DIR, "fig4_pipeline.svg")
    plt.savefig(out, format="svg", bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: Combined — agent cards (top) + engagement improvement bars (bottom)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_combined():
    yb = load("baseline_metrics.json")
    yf = load("facilitated_metrics.json")
    eb = load("evelyn_baseline_run_metrics.json")
    ef = load("evelyn_facilitated_metrics.json")

    fig = plt.figure(figsize=(24, 12), facecolor=C_BG)  # 18:9 (2:1)

    # Cards span full figure width; bars have a wide left margin for label text.
    # Use add_axes([left, bottom, width, height]) for independent control.
    cw, cgap = 0.313, 0.020   # card width, gap between cards
    card_axes = [fig.add_axes([0.010 + i * (cw + cgap), 0.46, cw, 0.47])
                 for i in range(3)]

    bw, bgap = 0.355, 0.045   # bar panel width, gap between panels
    bar_axes = [
        fig.add_axes([0.215,             0.06, bw, 0.36]),
        fig.add_axes([0.215 + bw + bgap, 0.06, bw, 0.36]),
    ]

    # ── Section labels ────────────────────────────────────────────────────────
    fig.text(0.5, 0.975, "Synthetic Engagement Agents",
             ha="center", va="top", fontsize=30, fontweight="bold", color=C_TEXT)

    # ── Agent cards ───────────────────────────────────────────────────────────
    agents = [
        {
            "name": "Community Manager",
            "handle": "@community_manager",
            "color": "#3B82F6",
            "light": "#EFF6FF",
            "actions": ["Likes unanswered posts",
                        "Comments to acknowledge authors",
                        "Reposts quality content"],
            "feed": "Prioritizes posts with 0 replies",
            "metric": "Unanswered post rate",
        },
        {
            "name": "Domain Expert",
            "handle": "@vechain_analyst",
            "color": "#8B5CF6",
            "light": "#F5F3FF",
            "actions": ["Likes high-quality posts",
                        "Reposts insightful content",
                        "Adds expert commentary"],
            "feed": "Sees full feed ranked by likes",
            "metric": "Engagement per post",
        },
        {
            "name": "Discussion Facilitator",
            "handle": "@vechain_connector",
            "color": "#F97316",
            "light": "#FFF7ED",
            "actions": ["Comments with open-ended questions",
                        "Quote-tweets to spark discussion",
                        "Follows active community members"],
            "feed": "Sees full default feed",
            "metric": "Reply rate",
        },
    ]

    for ax, agent in zip(card_axes, agents):
        ax.set_facecolor(agent["light"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Card border
        ax.add_patch(FancyBboxPatch(
            (0.08, 0.08), 9.84, 9.84,
            boxstyle="round,pad=0.2",
            facecolor=agent["light"], edgecolor=agent["color"],
            linewidth=3.0,
        ))

        # Header
        ax.add_patch(FancyBboxPatch(
            (0.08, 7.85), 9.84, 2.07,
            boxstyle="round,pad=0.1",
            facecolor=agent["color"], edgecolor="none",
        ))
        ax.text(5, 9.15, agent["name"], ha="center", va="center",
                fontsize=30, fontweight="bold", color="white")
        ax.text(5, 8.22, agent["handle"], ha="center", va="center",
                fontsize=19, color="white", alpha=0.88)

        # Actions
        ax.text(0.55, 7.52, "ACTIONS", fontsize=16, fontweight="bold",
                color=agent["color"], va="top")
        for i, action in enumerate(agent["actions"]):
            ax.text(0.65, 6.98 - i * 0.98, f"• {action}",
                    fontsize=19, color=C_TEXT, va="top")

        # Divider
        ax.plot([0.4, 9.6], [3.92, 3.92], color=agent["color"], alpha=0.25, lw=1.2)

        # Feed mode
        ax.text(0.55, 3.68, "FEED MODE", fontsize=16, fontweight="bold",
                color=agent["color"], va="top")
        ax.text(0.65, 3.14, agent["feed"], fontsize=19, color=C_TEXT,
                va="top", style="italic")

        # Divider
        ax.plot([0.4, 9.6], [2.38, 2.38], color=agent["color"], alpha=0.25, lw=1.2)

        # Key metric
        ax.text(0.55, 2.14, "KEY METRIC IMPROVED", fontsize=16, fontweight="bold",
                color=agent["color"], va="top")
        ax.add_patch(FancyBboxPatch(
            (0.4, 0.82), 9.2, 0.98,
            boxstyle="round,pad=0.1",
            facecolor=agent["color"], edgecolor="none", alpha=0.15,
        ))
        ax.text(5, 1.31, agent["metric"], ha="center", va="center",
                fontsize=24, fontweight="bold", color=agent["color"])

    # ── Engagement bars ───────────────────────────────────────────────────────
    # 3 agent-specific metrics (always shown, even if negative) + total engagement.
    # Tuple: (label, calc_note, bl_val, fa_val, invert, fmt, color)
    def agent_metrics(bl, fa):
        return [
            ("Total Community Engagement",
             "sum of all likes + comments + reposts",
             get(bl, "community_health", "total_community_engagement"),
             get(fa, "community_health", "total_community_engagement"),
             False, "{:.0f}", C_GREEN),
            ("Reply Rate",
             "posts with ≥1 comment ÷ total posts",
             get(bl, "community_health", "reply_rate"),
             get(fa, "community_health", "reply_rate"),
             False, "{:.1%}", "#F97316"),   # Discussion Facilitator
            ("Avg Engagement / Post",
             "(likes + comments + reposts) ÷ total posts",
             get(bl, "community_health", "avg_engagement_per_community_post"),
             get(fa, "community_health", "avg_engagement_per_community_post"),
             False, "{:.2f}", "#8B5CF6"),   # Domain Expert
            ("Posts with Replies",
             "1 − (posts with 0 replies ÷ total posts)",
             1 - (get(bl, "community_health", "unanswered_post_rate") or 0),
             1 - (get(fa, "community_health", "unanswered_post_rate") or 0),
             False, "{:.1%}", "#3B82F6"),   # Community Manager
        ]

    datasets = [
        ("Coarse-Grained  (k=4, 40 personas)", agent_metrics(yb, yf)),
        ("Fine-Grained  (k=19, 143 personas)",  agent_metrics(eb, ef)),
    ]

    for ax, (title, mlist) in zip(bar_axes, datasets):
        ax.set_facecolor(C_BG)
        changes    = [pct_change(m[3], m[2], invert=m[4]) for m in mlist]
        abs_base   = [m[2] for m in mlist]
        abs_facil  = [m[3] for m in mlist]
        fmts       = [m[5] for m in mlist]
        labels     = [m[0] for m in mlist]
        calcs      = [m[1] for m in mlist]
        clrs       = [m[6] for m in mlist]

        y       = np.arange(len(mlist)) * 1.7
        vals    = [c if c is not None else 0 for c in changes]
        max_abs = max(abs(v) for v in vals) if vals else 10

        bar_colors = [c if v >= 0 else C_RED for c, v in zip(clrs, vals)]
        bars = ax.barh(y, vals, height=0.62, color=bar_colors, zorder=3, alpha=0.88)
        ax.axvline(0, color=C_TEXT, linewidth=1.0, zorder=4)

        for bar, bv, fv, fmt, chg in zip(bars, abs_base, abs_facil, fmts, changes):
            if bv is not None and fv is not None:
                sign = "+" if chg >= 0 else ""
                ann_x = max(chg, 0) + max_abs * 0.03
                ax.text(
                    ann_x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{fmt.format(bv)} → {fmt.format(fv)}  ({sign}{chg:.0f}%)",
                    va="center", ha="left", fontsize=19,
                    color=C_TEXT,
                    fontweight="bold" if abs(chg) >= 10 else "normal",
                )

        # Custom y labels colored by agent, clipped off so they render in margin
        ax.set_yticks([])
        for lbl, calc, yi, col in zip(labels, calcs, y, clrs):
            ax.text(-max_abs * 0.04, yi + 0.17, lbl,
                    ha="right", va="center", fontsize=21, fontweight="bold",
                    color=col, clip_on=False)
            ax.text(-max_abs * 0.04, yi - 0.24, calc,
                    ha="right", va="center", fontsize=15,
                    color=C_MUTED, style="italic", clip_on=False)

        ax.set_xlim(-max_abs * 0.9, max_abs * 2.3)
        ax.set_ylim(-0.9, max(y) + 1.1)
        ax.set_xlabel("% improvement from original", fontsize=17, color=C_MUTED)
        ax.tick_params(axis="x", colors=C_MUTED, labelsize=16)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(C_BORDER)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=22, fontweight="bold", color=C_TEXT, pad=12)

    out = os.path.join(OUT_DIR, "fig5_combined.svg")
    plt.savefig(out, format="svg", facecolor=C_BG)
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    fig1_improvement()
    fig2_absolute()
    fig3_agent_roles()
    fig4_pipeline()
    fig5_combined()
    print("\nAll figures saved to simulation/figures/")
