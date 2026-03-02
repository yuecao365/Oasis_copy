import json
import os
import math
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# Config
# =========================

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

INPUT_FILE = "filtered_tweets_train_2025-04_to_07.jsonl"

TOP_K = 10
BOTTOM_K = 5
MIN_CONTENT_WORDS = 2   # used only by dedupe_and_filter_tweets if re-enabled

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Helpers
# =========================

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def dedupe_and_filter_tweets(tweets):
    """
    Deduplicate by tweet_id (upstream may have same tweet repeated).
    Filter out tweets with no real content: empty, "URL", or very few words.
    Returns list of tweets (one per tweet_id, first occurrence kept).
    """
    seen_ids = set()
    out = []
    dropped_dupe = 0
    dropped_content = 0
    for tw in tweets:
        tid = tw.get("id")
        if tid is None:
            continue
        tid_str = str(tid)
        if tid_str in seen_ids:
            dropped_dupe += 1
            continue
        text = (tw.get("clean_text") or tw.get("text") or "").strip()
        if not text or text == "URL":
            dropped_content += 1
            continue
        if len(text.split()) < MIN_CONTENT_WORDS:
            dropped_content += 1
            continue
        seen_ids.add(tid_str)
        out.append(tw)
    if dropped_dupe or dropped_content:
        print(f"  Deduped {dropped_dupe} duplicate tweet_ids, filtered {dropped_content} low-content tweets.")
    return out


def safe_log(x):
    """log(1 + x), guard against None"""
    if x is None:
        return 0.0
    return math.log1p(max(x, 0))


# =========================
# Step 1: Extract features
# =========================

def build_feature_matrix(tweets):
    """
    Build feature matrix X and metadata list.

    Features used:
    - likes
    - retweets
    - replies
    - quotes
    - views

    All features use log(1 + x).
    """

    X = []
    meta = []

    for tw in tweets:
        # Support both publicMetrics and top-level engagement fields
        metrics = tw.get("publicMetrics") or tw

        likes = safe_log(metrics.get("likeCount", 0))
        retweets = safe_log(metrics.get("retweetCount", 0))
        replies = safe_log(metrics.get("replyCount", 0))
        quotes = safe_log(metrics.get("quoteCount", 0))
        views = safe_log(metrics.get("viewCount", 0))

        X.append([likes, retweets, replies, quotes, views])

        meta.append({
            "tweet_id": tw.get("id"),
            "author_id": (tw.get("author") or {}).get("id"),
            "clean_text": tw.get("clean_text", ""),
            "likes": metrics.get("likeCount", 0),
            "retweets": metrics.get("retweetCount", 0),
            "replies": metrics.get("replyCount", 0),
            "quotes": metrics.get("quoteCount", 0),
            "views": metrics.get("viewCount", 0),
        })

    return np.array(X), meta


# =========================
# Step 2: PCA ranking
# =========================

def run_pca_ranking(X):
    """
    Standardize features, run PCA (1 component),
    return scores + PCA metadata.
    """

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    scores = pca.fit_transform(X_norm).flatten()

    pca_info = {
        "feature_names": ["likes", "retweets", "replies", "quotes", "views"],
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "pca_weights": pca.components_[0].tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_[0],
    }

    return scores, pca_info


# =========================
# Step 3: Save outputs
# =========================

def save_ranked_tweets(meta, scores):
    rows = []
    for m, s in zip(meta, scores):
        row = dict(m)
        row["popularity_score"] = float(s)
        rows.append(row)

    rows.sort(key=lambda r: r["popularity_score"], reverse=True)

    # Full ranking
    with open(os.path.join(OUTPUT_DIR, "tweet_popularity_scores.csv"), "w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    # Top-K
    with open(os.path.join(OUTPUT_DIR, "top_ranked_tweets.csv"), "w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows[:TOP_K])

    # Bottom-K (for contrast; teacher asked for bottom 5)
    with open(os.path.join(OUTPUT_DIR, "bottom_ranked_tweets.csv"), "w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows[-BOTTOM_K:])

    return rows


def save_pca_metadata(pca_info):
    with open(os.path.join(OUTPUT_DIR, "pca_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(pca_info, f, indent=2)


def write_pca_ranking_report(rows, pca_info):
    """
    Write PCA ranking report: ranking formula (with weights), Top 10, Bottom 5.
    """
    path = os.path.join(OUTPUT_DIR, "pca_ranking_report.md")
    names = pca_info.get("feature_names", ["likes", "retweets", "replies", "quotes", "views"])
    weights = pca_info.get("pca_weights", [])
    mean = pca_info.get("mean", [0] * 5)
    std = pca_info.get("std", [1] * 5)
    ev = pca_info.get("explained_variance_ratio", 0)

    lines = []
    lines.append("# PCA ranking report")
    lines.append("")
    lines.append("## Ranking formula")
    lines.append("")
    lines.append("Features use **log(1 + raw_count)**; then standardized; then one PCA component:")
    lines.append("")
    lines.append("```")
    lines.append("z_i = (log(1 + x_i) - mean_i) / std_i   for each feature i")
    lines.append("popularity_score = w1*z1 + w2*z2 + w3*z3 + w4*z4 + w5*z5")
    lines.append("```")
    lines.append("")
    lines.append("**Weights (PCA first component):**")
    for i, (n, w) in enumerate(zip(names, weights)):
        lines.append(f"- {n}: **{w:.6f}**")
    lines.append("")
    lines.append("**Standardization:** mean_i and std_i are from StandardScaler (see pca_metadata.json).")
    lines.append("")
    if isinstance(ev, (int, float)) and math.isfinite(ev):
        lines.append(f"**Explained variance ratio (1 component):** {ev:.4f}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top 10 tweet IDs + text + score")
    lines.append("")
    for i, r in enumerate(rows[:TOP_K], 1):
        tid = r.get("tweet_id", "")
        text = (r.get("clean_text", "") or "")[:120]
        if len((r.get("clean_text") or "")) > 120:
            text += "..."
        score = r.get("popularity_score", 0)
        lines.append(f"**{i}.** `{tid}`  score = {score:.4f}")
        lines.append(f"    {text}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Bottom 5 tweet IDs + text + score (for contrast)")
    lines.append("")
    for i, r in enumerate(rows[-BOTTOM_K:], 1):
        tid = r.get("tweet_id", "")
        text = (r.get("clean_text", "") or "")[:120]
        if len((r.get("clean_text") or "")) > 120:
            text += "..."
        score = r.get("popularity_score", 0)
        lines.append(f"**{i}.** `{tid}`  score = {score:.4f}")
        lines.append(f"    {text}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# Main
# =========================

def main():
    print("Loading filtered tweets...")
    tweets = load_jsonl(os.path.join(INPUT_DIR, INPUT_FILE))
    print(f"Loaded {len(tweets)} tweets")

    print("Building feature matrix (content filter done in user_filtering)...")
    print(f"Using {len(tweets)} tweets for PCA")
    X, meta = build_feature_matrix(tweets)

    print("Running PCA for popularity ranking...")
    scores, pca_info = run_pca_ranking(X)

    print("Saving ranked tweets...")
    rows = save_ranked_tweets(meta, scores)

    print("Saving PCA metadata...")
    save_pca_metadata(pca_info)

    print("Writing PCA ranking report (formula + Top 10 + Bottom 5)...")
    write_pca_ranking_report(rows, pca_info)

    print("DONE.")
    print("Outputs:")
    print(" - tweet_popularity_scores.csv")
    print(" - top_ranked_tweets.csv")
    print(" - bottom_ranked_tweets.csv")
    print(" - pca_metadata.json")
    print(" - pca_ranking_report.md")


if __name__ == "__main__":
    main()
