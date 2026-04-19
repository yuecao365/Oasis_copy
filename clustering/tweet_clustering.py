"""
Step 5: Clustering (root tweets only) + representative posts per cluster.

- Uses precomputed root embeddings from root_tweet_embeddings.csv (no model call here).
- KMeans(n_clusters=20) on L2-normalized root embeddings.
- Representative posts: from ALL tweets in each cluster (roots + replies), top % by popularity.
- Optional: exclude low-content roots (MIN_CONTENT_TOKENS).
- Outputs: cluster_assignments_roots.csv, cluster_assignments_kmeans.json,
  cluster_assignments.jsonl (same rows as kmeans JSON; JSON Lines for persona pipeline),
  representative_posts_kmeans.json, excluded roots (optional). (Noise outputs skipped for KMeans.)

Requires: sentence-transformers, sklearn
  pip install sentence-transformers scikit-learn
"""

import json
import os
import csv
import random
import numpy as np
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# =========================
# Config
# =========================

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

# Inputs (from previous pipeline steps)
# Use train-only subset (2025-04 to 2025-07) for clustering roots.
FILTERED_TWEETS_FILE = "filtered_tweets_train_2025-04_to_07.jsonl"
REPLY_CHAIN_FILE = "reply_chain.jsonl"
POPULARITY_SCORES_FILE = "tweet_popularity_scores.csv"

# Embedding: precomputed root embeddings CSV (one row per root tweet)
EMBEDDING_CSV = "root_tweet_embeddings.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Clustering: KMeans on L2-normalized embeddings (no UMAP)
KMEANS_N_CLUSTERS = 20
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

# Representative posts: top fraction by popularity per cluster (from ALL tweets in cluster, incl. replies)
REP_TOP_PCT = 0.10  # top 10%

# Exclude low-information roots from clustering (e.g. mostly @mentions, "URL", very short).
MIN_CONTENT_TOKENS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Helpers: load data
# =========================

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_popularity_scores(path):
    """Return dict tweet_id_str -> popularity_score (float)."""
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("tweet_id", "").strip()
            try:
                scores[tid] = float(row.get("popularity_score", 0))
            except (ValueError, TypeError):
                scores[tid] = 0.0
    return scores


def get_root_tweets(filtered_tweets, reply_chain):
    """
    From reply_chain, select rows with is_root == True; return full tweet objects
    for those tweet_ids from filtered_tweets.
    """
    id_to_tweet = {str(t.get("id")): t for t in filtered_tweets if t.get("id") is not None}
    root_ids = set()
    for r in reply_chain:
        if r.get("is_root") is True:
            root_ids.add(str(r["tweet_id"]))
    root_tweets = []
    for rid in root_ids:
        if rid in id_to_tweet:
            root_tweets.append(id_to_tweet[rid])
    return root_tweets


def _content_token_count(text):
    """
    Count tokens that are "content": strip @mentions, "URL", and http(s) tokens.
    Used to exclude low-information tweets (e.g. only @user URL) from clustering.
    """
    if not text or not isinstance(text, str):
        return 0
    tokens = text.strip().split()
    n = 0
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t.startswith("@"):
            continue
        if t.upper() == "URL":
            continue
        if t.startswith("http://") or t.startswith("https://"):
            continue
        n += 1
    return n


def is_substantive_tweet(tweet, min_tokens=None):
    """True if tweet has at least min_tokens content tokens (default MIN_CONTENT_TOKENS)."""
    min_tokens = min_tokens if min_tokens is not None else MIN_CONTENT_TOKENS
    if min_tokens <= 0:
        return True
    text = (tweet.get("clean_text") or tweet.get("text") or "").strip()
    return _content_token_count(text) >= min_tokens


def filter_substantive_roots(root_tweets, min_tokens=None):
    """
    Split root tweets into those with enough content for clustering vs excluded.
    Returns (substantive_root_tweets, excluded_root_ids).
    """
    min_tokens = min_tokens if min_tokens is not None else MIN_CONTENT_TOKENS
    if min_tokens <= 0:
        return root_tweets, []
    substantive = []
    excluded_ids = []
    for t in root_tweets:
        if is_substantive_tweet(t, min_tokens=min_tokens):
            substantive.append(t)
        else:
            excluded_ids.append(str(t.get("id")))
    return substantive, excluded_ids


# =========================
# Step 1: Embeddings (root tweets only)
# =========================

def _load_precomputed_embeddings(csv_path):
    """
    Load precomputed root embeddings from CSV.
    Expects columns:
      - root_tweet_id
      - embedding  (JSON-like list string)
    Returns dict[str, np.ndarray].
    """
    import ast

    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("root_tweet_id") or row.get("tweet_id") or row.get("id")
            if not tid:
                continue
            emb_str = row.get("embedding")
            if not emb_str:
                continue
            try:
                # embedding is stored as a stringified Python/JSON list
                vec = ast.literal_eval(emb_str)
            except Exception:
                try:
                    vec = json.loads(emb_str)
                except Exception:
                    continue
            try:
                arr = np.asarray(vec, dtype=np.float32)
            except Exception:
                continue
            mapping[str(tid)] = arr
    return mapping


def get_embeddings(root_tweets, cache_dir):
    """
    Get embeddings for root tweets using precomputed CSV.
    Only roots that have an embedding row are kept; others are skipped.
    """
    csv_path = os.path.join(cache_dir, EMBEDDING_CSV)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Precomputed embeddings CSV not found: {csv_path}")

    print(f"  Loading precomputed embeddings from {csv_path} ...")
    emb_map = _load_precomputed_embeddings(csv_path)
    print(f"  Loaded {len(emb_map)} embedding vectors from CSV.")

    X_list = []
    tweet_ids = []
    missing = 0

    for t in root_tweets:
        tid = str(t.get("id"))
        if not tid:
            continue
        emb = emb_map.get(tid)
        if emb is None:
            missing += 1
            continue
        tweet_ids.append(tid)
        X_list.append(emb)

    if not X_list:
        raise RuntimeError("No embeddings found for any root tweets; check IDs and CSV.")

    if missing:
        print(f"  Warning: {missing} root tweets had no embedding in {EMBEDDING_CSV} and were skipped.")

    X = np.vstack(X_list)
    print(f"  Using embeddings for {len(tweet_ids)} root tweets (matrix shape {X.shape[0]} x {X.shape[1]}).")
    return X, tweet_ids


# =========================
# Step 2: Clustering (KMeans on L2-normalized embeddings)
# =========================

def run_kmeans(X):
    """Run KMeans. Returns (labels,) with labels in 0..n_clusters-1."""
    kmeans = KMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    labels = np.array(kmeans.fit_predict(X), dtype=np.intp)
    return labels


# =========================
# Cluster size distribution
# =========================

def cluster_size_stats(labels, method_name, quiet=False):
    """Return (and optionally print) cluster size distribution."""
    from collections import Counter
    c = Counter(labels)
    noise = c.get(-1, 0)
    clusters = {k: v for k, v in c.items() if k >= 0}
    sizes = sorted(clusters.values(), reverse=True)

    if not quiet:
        lines = [
            f"[{method_name}]",
            f"  n_clusters: {len(clusters)}",
            f"  noise (label=-1): {noise}",
            f"  cluster sizes (sorted desc): {sizes[:20]}{'...' if len(sizes) > 20 else ''}",
            f"  min/median/max size: {min(sizes)} / {np.median(sizes):.0f} / {max(sizes)}",
        ]
        for line in lines:
            print(line)
    return {
        "n_clusters": len(clusters),
        "n_noise": int(noise),
        "sizes": [int(s) for s in sizes],
        "min_size": int(min(sizes)) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        "median_size": int(np.median(sizes)) if sizes else 0,
    }


# =========================
# Representative posts per cluster
# =========================

def build_tweet_to_root(reply_chain):
    """tweet_id -> root_tweet_id for every tweet in reply_chain."""
    return {str(r["tweet_id"]): str(r["root_tweet_id"]) for r in reply_chain}


def assign_all_tweets_to_cluster(reply_chain, root_tweet_ids, root_labels, excluded_root_ids=None):
    """
    root_tweet_ids: list of root tweet ids in same order as root_labels.
    Returns dict tweet_id_str -> cluster_id (int). -2 = root excluded from clustering.
    All tweets (roots + replies) get a cluster via their root_tweet_id.
    """
    root_to_cluster = {}
    for tid, lab in zip(root_tweet_ids, root_labels):
        root_to_cluster[str(tid)] = int(lab)
    default_for_missing = -2 if (excluded_root_ids and len(excluded_root_ids) > 0) else -1
    tweet_to_root = build_tweet_to_root(reply_chain)
    tweet_to_cluster = {}
    for tweet_id, root_id in tweet_to_root.items():
        tweet_to_cluster[tweet_id] = root_to_cluster.get(root_id, default_for_missing)
    return tweet_to_cluster


def select_representative_posts(tweet_to_cluster, popularity_scores, top_pct=0.10):
    """
    For each cluster, take all tweets in that cluster and keep top `top_pct` by
    popularity_score. Returns dict cluster_id -> list of (tweet_id, popularity_score).
    """
    cluster_tweets = defaultdict(list)  # cluster_id -> [(tweet_id, score), ...]
    for tweet_id, cluster_id in tweet_to_cluster.items():
        if cluster_id < 0:
            continue
        score = popularity_scores.get(tweet_id, 0.0)
        cluster_tweets[cluster_id].append((tweet_id, score))

    rep = {}
    for cid, pairs in cluster_tweets.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        n = max(1, int(len(pairs) * top_pct))
        rep[cid] = pairs[:n]
    return rep


# =========================
# Save outputs
# =========================

def save_root_cluster_assignments(root_tweet_ids, labels_cluster, out_dir):
    """Save CSV: tweet_id, cluster (roots only)."""
    path = os.path.join(out_dir, "cluster_assignments_roots.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "cluster"])
        for tid, lab in zip(root_tweet_ids, labels_cluster):
            w.writerow([tid, lab])
    print(f"  Saved {path}")


def save_all_tweet_cluster_assignments_json(tweet_to_cluster, filtered_tweets, out_dir):
    """
    Save JSON with cluster assignments for all tweets (roots + replies) that:
    - appear in the filtered_tweets train subset, and
    - have a non-negative cluster_id (0..K-1).

    Format is a list of objects like:
      { "tweet_id": "...", "cluster_id": 3 }

    This mirrors the semantics of cluster_assignments_k4.csv, but in JSON.
    """
    # Restrict to tweets in the current (train) filtered_tweets file
    valid_ids = {
        str(t.get("id"))
        for t in filtered_tweets
        if t.get("id") is not None
    }

    rows = []
    for tid, cid in tweet_to_cluster.items():
        if cid < 0:
            continue
        if tid not in valid_ids:
            continue
        rows.append({"tweet_id": str(tid), "cluster_id": int(cid)})

    path = os.path.join(out_dir, "cluster_assignments_kmeans.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"  Saved {path} ({len(rows)} tweet-level assignments)")

    # JSON Lines: same records as kmeans JSON; matches persona_generation load_cluster_assignments
    jsonl_path = os.path.join(out_dir, "cluster_assignments.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {jsonl_path} ({len(rows)} lines; use as CLUSTER_ASSIGNMENTS_PATH for hard KMeans)")


def save_cluster_size_report(stats, out_dir):
    path = os.path.join(out_dir, "cluster_size_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"kmeans": stats}, f, indent=2)
    print(f"  Saved {path}")


def save_noise_to_cluster_reassignments(reassignments, root_tweets, out_dir):
    """
    Save CSV: tweet_id, cluster_id, membership_prob, clean_text.
    Lists tweets that were moved from noise to a cluster by soft clustering.
    """
    if not reassignments:
        print("  No noise->cluster reassignments; skipping noise_to_cluster_reassignments.csv")
        return
    id_to_tweet = {str(t.get("id")): t for t in root_tweets if t.get("id") is not None}
    path = os.path.join(out_dir, "noise_to_cluster_reassignments.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "cluster_id", "membership_prob", "clean_text"])
        for tid, cluster_id, prob in reassignments:
            t = id_to_tweet.get(tid)
            text = (t.get("clean_text") or t.get("text") or "").strip() if t else ""
            w.writerow([tid, cluster_id, f"{prob:.4f}", text])
    print(f"  Saved {path} ({len(reassignments)} tweets moved from noise to cluster)")


def save_noise_tweets_sample(root_tweets, root_tweet_ids, labels_hdbscan, out_dir, sample_size=100):
    """
    Save which root tweets are labeled as noise (label=-1) so you can inspect them.
    - noise_tweet_ids_primary.txt: all noise tweet IDs (one per line).
    - noise_tweets_sample_100.jsonl: sample_size random noise tweets with tweet_id and clean_text.
    - noise_tweets_sample_100.csv: same 100 rows, columns id, clean_text.
    """
    id_to_tweet = {str(t.get("id")): t for t in root_tweets if t.get("id") is not None}
    noise_indices = [i for i in range(len(labels_hdbscan)) if labels_hdbscan[i] == -1]
    noise_ids = [root_tweet_ids[i] for i in noise_indices]
    if not noise_ids:
        print("  No noise points (label=-1); skipping noise export.")
        return
    # All noise IDs
    ids_path = os.path.join(out_dir, "noise_tweet_ids_primary.txt")
    with open(ids_path, "w", encoding="utf-8") as f:
        f.write("\n".join(noise_ids))
    print(f"  Saved {ids_path} ({len(noise_ids)} noise tweet IDs)")
    # Sample with text (same 100 for both jsonl and csv)
    sample_size = min(sample_size, len(noise_ids))
    chosen_ids = random.Random(42).sample(noise_ids, sample_size)
    rows = []
    for tid in chosen_ids:
        t = id_to_tweet.get(tid)
        text = (t.get("clean_text") or t.get("text") or "").strip() if t else ""
        rows.append({"tweet_id": tid, "clean_text": text})
    sample_path = os.path.join(out_dir, "noise_tweets_sample_{}.jsonl".format(sample_size))
    with open(sample_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {sample_path} (sample of {sample_size} noise tweets for manual inspection)")
    # CSV: id, clean_text (noise sample rows)
    csv_path = os.path.join(out_dir, "noise_tweets_sample_{}.csv".format(sample_size))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "clean_text"])
        for r in rows:
            w.writerow([r["tweet_id"], r["clean_text"]])
    print(f"  Saved {csv_path} (same {sample_size} rows: id, clean_text)")


def save_representative_posts(rep_by_method, out_dir):
    """rep_by_method: {'hdbscan': {cluster_id: [(tweet_id, score), ...]}, 'kmeans': ...}"""
    for method, rep in rep_by_method.items():
        path = os.path.join(out_dir, f"representative_posts_{method}.json")
        # JSON-serializable: cluster_id -> list of {tweet_id, popularity_score}
        out = {
            str(cid): [{"tweet_id": tid, "popularity_score": float(score)} for tid, score in posts]
            for cid, posts in rep.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {path}")


def save_parameter_rationale(out_dir):
    path = os.path.join(out_dir, "parameter_rationale.md")
    lines = [
        "# Clustering parameter rationale",
        "",
        f"- **Embedding**: {EMBEDDING_MODEL} (sentence-transformers). L2-normalized; no UMAP.",
        "",
        "## Clustering",
        f"- **KMeans** n_clusters={KMEANS_N_CLUSTERS}, on root-tweet embeddings only.",
        "",
        "## Excluding low-information roots",
        f"- **MIN_CONTENT_TOKENS** = {MIN_CONTENT_TOKENS}: only root tweets with at least this many \"content\" tokens are clustered. Excluded root IDs in `excluded_root_ids_from_clustering.txt` when non-empty.",
        "",
        "## Representative posts",
        f"- Top {REP_TOP_PCT*100:.0f}% by popularity within each cluster. Selected from **all tweets** in the cluster (roots + replies).",
        "",
        "See **clustering_hyperparameters.json** for full config.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {path}")


def save_hyperparameters_json(out_dir):
    """Save clustering hyperparameters for reference."""
    data = {
        "embedding": {"model": EMBEDDING_MODEL},
        "kmeans": {
            "n_clusters": KMEANS_N_CLUSTERS,
            "random_state": KMEANS_RANDOM_STATE,
            "n_init": KMEANS_N_INIT,
        },
        "min_content_tokens": MIN_CONTENT_TOKENS,
        "rep_top_pct": REP_TOP_PCT,
    }
    path = os.path.join(out_dir, "clustering_hyperparameters.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")


# =========================
# Main
# =========================

def main():
    print("Loading data...")
    filtered_tweets = load_jsonl(os.path.join(INPUT_DIR, FILTERED_TWEETS_FILE))
    reply_chain = load_jsonl(os.path.join(INPUT_DIR, REPLY_CHAIN_FILE))
    popularity_scores = load_popularity_scores(os.path.join(INPUT_DIR, POPULARITY_SCORES_FILE))
    print(f"  filtered_tweets: {len(filtered_tweets)}, reply_chain: {len(reply_chain)}, popularity entries: {len(popularity_scores)}")

    print("Getting root tweets...")
    root_tweets_all = get_root_tweets(filtered_tweets, reply_chain)
    print(f"  Root tweets (total): {len(root_tweets_all)}")

    if MIN_CONTENT_TOKENS > 0:
        root_tweets, excluded_root_ids = filter_substantive_roots(root_tweets_all)
        print(f"  Excluded {len(excluded_root_ids)} roots (content tokens < {MIN_CONTENT_TOKENS}); clustering {len(root_tweets)} substantive roots.")
        if excluded_root_ids:
            excl_path = os.path.join(OUTPUT_DIR, "excluded_root_ids_from_clustering.txt")
            with open(excl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(excluded_root_ids))
            print(f"  Saved {excl_path}")
    else:
        root_tweets = root_tweets_all
        excluded_root_ids = []

    print("Embedding root tweets (substantive only)...")
    X, root_tweet_ids = get_embeddings(root_tweets, OUTPUT_DIR)
    print(f"  Embedding shape: {X.shape}")

    # L2-normalize for cosine-like distance in KMeans
    X_cluster = normalize(X, norm="l2", axis=1)

    # ---------- KMeans ----------
    print("\n--- KMeans (n_clusters={}) ---".format(KMEANS_N_CLUSTERS))
    labels = run_kmeans(X_cluster)
    stats = cluster_size_stats(labels, "KMeans")

    # ---------- Save cluster assignments ----------
    save_root_cluster_assignments(root_tweet_ids, labels, OUTPUT_DIR)

    # ---------- Representative posts (from ALL tweets in each cluster: roots + replies) ----------
    print("\nRepresentative posts per cluster (top {}% by popularity, from all tweets in cluster)...".format(int(REP_TOP_PCT * 100)))
    tweet_to_cluster = assign_all_tweets_to_cluster(reply_chain, root_tweet_ids, labels, excluded_root_ids=excluded_root_ids)
    rep = select_representative_posts(tweet_to_cluster, popularity_scores, REP_TOP_PCT)
    save_representative_posts({"kmeans": rep}, OUTPUT_DIR)

    # ---------- Save full tweet-level assignments (train subset only) ----------
    save_all_tweet_cluster_assignments_json(tweet_to_cluster, filtered_tweets, OUTPUT_DIR)

    print("\nDONE.")
    print(
        "Outputs: cluster_assignments_roots.csv, cluster_assignments_kmeans.json, "
        "cluster_assignments.jsonl, representative_posts_kmeans.json"
    )


if __name__ == "__main__":
    main()
