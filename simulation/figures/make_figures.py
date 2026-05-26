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
    # ── load all metrics ──────────────────────────────────────────────────────
    yb  = load("baseline_metrics.json")
    yf  = load("facilitated_metrics.json")
    eb  = load("evelyn_baseline_run_metrics.json")
    ef  = load("evelyn_facilitated_metrics.json")
    ybs = load("yue_baseline_semantic.json")
    yfs = load("yue_facilitated_semantic.json")
    ebs = load("evelyn_baseline_semantic.json")
    efs = load("evelyn_facilitated_semantic.json")

    # ── metric definitions ────────────────────────────────────────────────────
    # (short_label, full_name, description, color, bl_yue, fa_yue, bl_ev, fa_ev)
    METRICS = [
        ("Total\nEngagement",
         "Total Community Engagement",
         "Sum of all likes, comments,\nand reposts by community members",
         C_GREEN,
         get(yb, "community_health", "total_community_engagement"),
         get(yf, "community_health", "total_community_engagement"),
         get(eb, "community_health", "total_community_engagement"),
         get(ef, "community_health", "total_community_engagement")),
        ("Posts with\nReplies",
         "Posts with Replies",
         "Fraction of posts that received\nat least one reply",
         "#3B82F6",
         1 - (get(yb, "community_health", "unanswered_post_rate") or 0),
         1 - (get(yf, "community_health", "unanswered_post_rate") or 0),
         1 - (get(eb, "community_health", "unanswered_post_rate") or 0),
         1 - (get(ef, "community_health", "unanswered_post_rate") or 0)),
        ("Reply\nRelevance",
         "Reply Relevance",
         "Avg cosine similarity between\na post and its direct replies",
         "#F97316",
         ybs["reply_relevance"], yfs["reply_relevance"],
         ebs["reply_relevance"], efs["reply_relevance"]),
        ("Intra-thread\nSimilarity",
         "Intra-thread Similarity",
         "Avg pairwise similarity of\nmessages within a thread",
         "#8B5CF6",
         ybs["intra_thread_sim"], yfs["intra_thread_sim"],
         ebs["intra_thread_sim"], efs["intra_thread_sim"]),
        ("Topic\nCoherence",
         "Topic Coherence",
         "Avg similarity of each post\nto its 5 nearest neighbor posts",
         "#0EA5E9",
         ybs["topic_coherence"], yfs["topic_coherence"],
         ebs["topic_coherence"], efs["topic_coherence"]),
    ]

    # ── figure & axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 12), facecolor=C_BG)  # 18:9 (2:1)

    fig.text(0.5, 0.975, "Synthetic Engagement Agents",
             ha="center", va="top", fontsize=28, fontweight="bold", color=C_TEXT)

    # Combined agent card (left column)
    ax_card = fig.add_axes([0.01, 0.24, 0.295, 0.70])
    # Radar charts — square in physical space: width*24 == height*12 → height=2*width
    rw, rh = 0.215, 0.43
    ax_r1 = fig.add_axes([0.355,        0.30, rw, rh], projection="polar")
    ax_r2 = fig.add_axes([0.355 + rw + 0.04, 0.30, rw, rh], projection="polar")
    # Metric description cards (bottom strip) — taller for bigger text
    mw = 0.182
    metric_axes = [fig.add_axes([0.01 + i * (mw + 0.008), 0.02, mw, 0.25])
                   for i in range(5)]

    # ── combined agent card ───────────────────────────────────────────────────
    agents = [
        {"name": "Community Manager",   "color": "#3B82F6", "light": "#EFF6FF",
         "handle": "@community_manager",
         "actions": ["Likes unanswered posts",
                     "Comments to acknowledge authors",
                     "Reposts quality content"],
         "feed": "Prioritizes posts with 0 replies"},
        {"name": "Domain Expert",        "color": "#8B5CF6", "light": "#F5F3FF",
         "handle": "@vechain_analyst",
         "actions": ["Likes high-quality posts",
                     "Reposts insightful content",
                     "Adds expert commentary"],
         "feed": "Sees full feed ranked by likes"},
        {"name": "Discussion Facilitator", "color": "#F97316", "light": "#FFF7ED",
         "handle": "@vechain_connector",
         "actions": ["Comments with open-ended questions",
                     "Quote-tweets to spark discussion",
                     "Follows active community members"],
         "feed": "Sees full default feed"},
    ]

    ax_card.set_facecolor("#F8FAFC")
    ax_card.set_xlim(0, 10)
    ax_card.set_ylim(0, 10)
    ax_card.axis("off")

    # Card border
    ax_card.add_patch(FancyBboxPatch(
        (0.05, 0.05), 9.9, 9.9, boxstyle="round,pad=0.2",
        facecolor="#F8FAFC", edgecolor=C_BORDER, linewidth=2.5,
    ))
    # Header bar
    ax_card.add_patch(FancyBboxPatch(
        (0.05, 8.8), 9.9, 1.15, boxstyle="round,pad=0.1",
        facecolor=C_TEXT, edgecolor="none",
    ))
    ax_card.text(5, 9.37, "Synthetic Engagement Agents",
                 ha="center", va="center", fontsize=22, fontweight="bold",
                 color="white")
    ax_card.text(5, 8.95, "3 always-active LLM-driven bots",
                 ha="center", va="center", fontsize=15, color="white", alpha=0.75)

    # Agent rows (3 equal sections)
    section_h = 2.55
    for i, ag in enumerate(agents):
        y0 = 8.6 - (i + 1) * section_h
        # Colored left accent bar
        ax_card.add_patch(FancyBboxPatch(
            (0.15, y0 + 0.12), 0.45, section_h - 0.25,
            boxstyle="square,pad=0",
            facecolor=ag["color"], edgecolor="none",
        ))
        # Agent name
        ax_card.text(0.9, y0 + section_h - 0.45, ag["name"],
                     ha="left", va="center", fontsize=20, fontweight="bold",
                     color=ag["color"])
        ax_card.text(0.9, y0 + section_h - 0.82, ag["handle"],
                     ha="left", va="center", fontsize=14, color=C_MUTED,
                     style="italic")
        # Actions
        for j, action in enumerate(ag["actions"]):
            ax_card.text(0.9, y0 + section_h - 1.28 - j * 0.50,
                         f"• {action}", ha="left", va="center",
                         fontsize=15, color=C_TEXT)
        # Feed mode
        ax_card.text(0.9, y0 + 0.32, ag["feed"],
                     ha="left", va="center", fontsize=14,
                     color=ag["color"], style="italic")
        # Divider (skip after last)
        if i < 2:
            ax_card.plot([0.15, 9.85], [y0 + 0.07, y0 + 0.07],
                         color=C_BORDER, lw=1.2)

    # ── radar charts ──────────────────────────────────────────────────────────
    N      = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang_c  = np.concatenate([angles, [angles[0]]])   # closed loop

    # Pin baseline at radius 0.60; facilitated scales proportionally from there.
    # Rings at 0.3 / 0.6 / 0.9 → labeled "–50%" / "baseline" / "+50%".
    BL_R = 0.60

    def norm_radar(bl_vals, fa_vals):
        nb, nf = [], []
        for b, f in zip(bl_vals, fa_vals):
            if b == 0:
                nb.append(BL_R); nf.append(BL_R)
            else:
                nb.append(BL_R)
                nf.append(min(BL_R * (f / b), 0.97))
        return nb, nf

    radar_datasets = [
        ("Coarse-Grained  (k=4, 40 personas)",
         [m[4] for m in METRICS], [m[5] for m in METRICS]),
        ("Fine-Grained  (k=19, 143 personas)",
         [m[6] for m in METRICS], [m[7] for m in METRICS]),
    ]

    spoke_labels = [m[0] for m in METRICS]
    spoke_colors = [m[3] for m in METRICS]

    for ax_r, (title, bl_vals, fa_vals) in zip([ax_r1, ax_r2], radar_datasets):
        nb, nf = norm_radar(bl_vals, fa_vals)
        nb_c = nb + [nb[0]]
        nf_c = nf + [nf[0]]

        ax_r.set_theta_offset(np.pi / 2)
        ax_r.set_theta_direction(-1)
        ax_r.set_ylim(0, 1)

        # Ring labels: 0.3 = −50% of baseline, 0.6 = baseline, 0.9 = +50%
        ax_r.set_yticks([0.3, BL_R, 0.9])
        ax_r.set_yticklabels(["−50%", "baseline", "+50%"],
                             size=11, color=C_MUTED)
        ax_r.set_rlabel_position(12)          # offset labels off the first spoke
        ax_r.grid(color="#CBD5E1", linewidth=0.8, alpha=0.6)
        ax_r.set_facecolor("#F8FAFC")

        # Fill + lines
        ax_r.fill(ang_c, nb_c, alpha=0.22, color=C_BASE)
        ax_r.fill(ang_c, nf_c, alpha=0.28, color=C_FACIL)
        ax_r.plot(ang_c, nb_c, "o-", lw=2.2, color=C_BASE,
                  markersize=7, label="Original")
        ax_r.plot(ang_c, nf_c, "o-", lw=2.2, color=C_FACIL,
                  markersize=7, label="Facilitated")

        # % change annotation at each facilitated vertex
        for angle, nf_val, b, f, col in zip(angles, nf, bl_vals, fa_vals,
                                             spoke_colors):
            if b and b != 0:
                pct  = (f - b) / abs(b) * 100
                sign = "+" if pct >= 0 else ""
                ann_col = C_GREEN if pct >= 0 else C_RED
                r_ann = nf_val + 0.09
                ax_r.annotate(
                    f"{sign}{pct:.0f}%",
                    xy=(angle, r_ann),
                    fontsize=10.5, ha="center", va="center",
                    fontweight="bold", color=ann_col,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor=ann_col, alpha=0.88, linewidth=1.2),
                )

        # Spoke labels colored by metric
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(spoke_labels, size=13, fontweight="bold")
        for lbl, col in zip(ax_r.get_xticklabels(), spoke_colors):
            lbl.set_color(col)

        ax_r.set_title(title, pad=22, fontsize=16, fontweight="bold",
                       color=C_TEXT, loc="center")
        ax_r.legend(loc="upper right", fontsize=12,
                    framealpha=0.90, edgecolor=C_BORDER,
                    handlelength=1.6, handleheight=0.9)

    # ── metric description cards (bottom strip) ───────────────────────────────
    for ax_m, m in zip(metric_axes, METRICS):
        _, full_name, desc, color, *_ = m
        ax_m.set_facecolor("#F8FAFC")
        ax_m.set_xlim(0, 10)
        ax_m.set_ylim(0, 10)
        ax_m.axis("off")

        # Colored header band
        ax_m.add_patch(FancyBboxPatch(
            (0.1, 7.2), 9.8, 2.7, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="none", alpha=0.15,
        ))
        ax_m.add_patch(FancyBboxPatch(
            (0.1, 7.2), 9.8, 2.7, boxstyle="round,pad=0.1",
            facecolor="none", edgecolor=color, linewidth=2.0,
        ))
        ax_m.text(5, 8.65, full_name, ha="center", va="center",
                  fontsize=16, fontweight="bold", color=color)

        # Description
        ax_m.text(5, 5.5, desc, ha="center", va="center",
                  fontsize=14, color=C_TEXT, linespacing=1.5)

        # Computation type tag
        tag = "semantic" if _ and full_name not in ("Total Community Engagement",
                                                      "Posts with Replies") else "community health"
        ax_m.add_patch(FancyBboxPatch(
            (2.5, 0.5), 5.0, 1.4, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="none", alpha=0.2,
        ))
        ax_m.text(5, 1.2, tag, ha="center", va="center",
                  fontsize=11, color=color, fontweight="bold")

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
