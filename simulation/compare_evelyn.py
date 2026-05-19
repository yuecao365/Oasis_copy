"""Four-way community engagement comparison (all runs: 48ts, 12h, apples-to-apples)."""
import json

def load(path):
    with open(path) as f:
        return json.load(f)

yb = load('simulation/metrics_history/baseline_metrics.json')
yf = load('simulation/metrics_history/facilitated_metrics.json')
eb = load('simulation/metrics_history/evelyn_baseline_run_metrics.json')
ef = load('simulation/metrics_history/evelyn_facilitated_metrics.json')

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

def pct(new, old):
    if old is not None and old != 0 and new is not None:
        delta = (new - old) / abs(old) * 100
        sign = '+' if delta >= 0 else ''
        return f"{sign}{round(delta):.0f}%"
    if old == 0 and new and new > 0:
        return "new"
    return "N/A"

def fmt(v):
    return str(v) if v is not None else "N/A"

print("COMMUNITY ENGAGEMENT: BASELINE vs FACILITATED (48 timesteps, 12h each)")
print("All runs use identical simulation settings; only synthetic agents differ.")
print()

for label, personas, bl, fa in [
    ("YUE", "40 personas", yb, yf),
    ("EVELYN", "143 personas", eb, ef),
]:
    print(f"== {label} ({personas}) ==")
    print(f"  {'Metric':<42} {'Baseline':>10} {'Facilitated':>12} {'Change':>8}")
    print(f"  {'-'*42} {'-'*10} {'-'*12} {'-'*8}")

    def row(label, *keys):
        bv = get(bl, *keys)
        fv = get(fa, *keys)
        print(f"  {label:<42} {fmt(bv):>10} {fmt(fv):>12} {pct(fv, bv):>8}")

    print(f"  -- Participation --")
    row("Active persona users",            "coverage", "sim_active_users")
    row("Total community posts",           "community_health", "total_community_posts")
    print(f"  -- Engagement --")
    row("Total community engagement",      "community_health", "total_community_engagement")
    row("Avg engagement per post",         "community_health", "avg_engagement_per_community_post")
    row("Reply rate",                      "community_health", "reply_rate")
    row("Unanswered post rate",            "community_health", "unanswered_post_rate")
    print(f"  -- Activity Distribution --")
    row("JSD posts/user  (lower=better)",  "user_activity_fit", "jsd_posts_per_user")
    row("Spearman posts/user",             "user_activity_fit", "spearman_posts_per_user")
    row("Top-10 active user overlap",      "user_activity_fit", "top10_overlap")
    row("Top-20 active user overlap",      "user_activity_fit", "top20_overlap")
    print()
