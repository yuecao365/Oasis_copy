"""
Step 5: Clustering (root tweets only) + representative posts per cluster.

- Embeds root tweets (sentence-transformers). UMAP (n_components=10, n_neighbors=15, cosine);
  HDBSCAN (euclidean on UMAP output).
- Optional: exclude low-content roots (MIN_CONTENT_TOKENS).
- Outputs: cluster_assignments_roots.csv, cluster_size_report.json, representative_posts_hdbscan.json,
  noise sample, excluded roots, parameter rationale, hyperparameters.

Requires: sentence-transformers, hdbscan, umap-learn
  pip install sentence-transformers hdbscan umap-learn
"""

import json
import os
import csv
import random
import numpy as np
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# =========================
# Config
# =========================

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

# Inputs (from previous pipeline steps)
FILTERED_TWEETS_FILE = "filtered_tweets.jsonl"
REPLY_CHAIN_FILE = "reply_chain.jsonl"
POPULARITY_SCORES_FILE = "tweet_popularity_scores.csv"

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, ~80MB
EMBEDDING_CACHE = "root_embeddings.npy"   # under OUTPUT_DIR
EMBEDDING_META_CACHE = "root_embedding_meta.json"  # tweet_id order for embeddings

# Dimensionality reduction before clustering
# REDUCE_METHOD: "umap" (cosine, good for text) or "pca". None = no reduction (cluster on raw 384-d).
REDUCE_METHOD = "umap"
REDUCE_DIM = 50  # used when REDUCE_METHOD == "pca" (number of components)
# UMAP (fixed): n_components=10, n_neighbors=15; cosine for text; HDBSCAN uses euclidean on UMAP output
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# Clustering (used for cluster_assignments + representative_posts)
# Target ~10–30 clusters for interpretable topic generation.
# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 100    # larger → fewer clusters, used as primary
HDBSCAN_MIN_SAMPLES = 3
# Use "euclidean" (BallTree supports it). We L2-normalize X_cluster so euclidean = cosine semantics.
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0

# Representative posts: top fraction by popularity per cluster
REP_TOP_PCT = 0.10  # top 10%

# Exclude low-information roots from clustering (e.g. mostly @mentions, "URL", very short).
# Only roots with at least this many "content" tokens (after dropping @mentions and "URL") are clustered.
# Set to 0 to disable and cluster all roots.
MIN_CONTENT_TOKENS = 3

# Soft clustering: reassign noise points to clusters when membership prob >= threshold.
# 0 = disabled (hard clustering only).
SOFT_CLUSTERING_NOISE_THRESHOLD = 0.15

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

def get_embeddings(root_tweets, cache_dir):
    """
    Get sentence embeddings for root tweet texts. Cache to disk so we can skip
    re-encoding when tuning clustering only.
    """
    cache_path = os.path.join(cache_dir, EMBEDDING_CACHE)
    meta_path = os.path.join(cache_dir, EMBEDDING_META_CACHE)

    texts = []
    tweet_ids = []
    for t in root_tweets:
        text = (t.get("clean_text") or t.get("text") or "").strip()
        texts.append(text or " ")
        tweet_ids.append(str(t.get("id")))

    if os.path.isfile(cache_path) and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("tweet_ids") == tweet_ids and meta.get("model") == EMBEDDING_MODEL:
            X = np.load(cache_path)
            print(f"  Loaded cached embeddings from {cache_path} ({X.shape[0]} x {X.shape[1]})")
            return X, tweet_ids

    from sentence_transformers import SentenceTransformer
    print(f"  Encoding with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    X = model.encode(texts, show_progress_bar=True)
    np.save(cache_path, X)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"tweet_ids": tweet_ids, "model": EMBEDDING_MODEL}, f, indent=2)
    print(f"  Saved embeddings to {cache_path}")
    return X, tweet_ids


# =========================
# Step 2: Dimensionality reduction (optional)
# =========================

def reduce_dim_pca(X):
    """PCA to REDUCE_DIM components. Returns (X_reduced, X_2d)."""
    n_components = min(REDUCE_DIM, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {X.shape[1]} -> {n_components} components, explained variance: {explained:.2%}")
    X_2d = X_reduced[:, :2] if X_reduced.shape[1] >= 2 else X_reduced
    return X_reduced, X_2d


def reduce_dim_umap(X, n_components=None, n_neighbors=None, metric=None):
    """
    UMAP with cosine (good for text). Returns (X_reduced, X_2d).
    HDBSCAN on output typically uses euclidean (common practice).
    """
    import umap
    n_components = n_components or UMAP_N_COMPONENTS
    n_neighbors = n_neighbors or UMAP_N_NEIGHBORS
    metric = metric or UMAP_METRIC
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=UMAP_RANDOM_STATE,
    )
    X_reduced = reducer.fit_transform(X)
    print(f"  UMAP: {X.shape[1]} -> {n_components} components, n_neighbors={n_neighbors}, metric={metric}")
    X_2d = X_reduced[:, :2] if X_reduced.shape[1] >= 2 else X_reduced
    return X_reduced, X_2d


def reduce_dim_for_clustering(X):
    """Dispatch to PCA or UMAP based on REDUCE_METHOD. Returns X_reduced (or None)."""
    if REDUCE_METHOD == "umap":
        X_red, _ = reduce_dim_umap(X)
        return X_red
    if REDUCE_METHOD == "pca" and REDUCE_DIM is not None:
        X_red, _ = reduce_dim_pca(X)
        return X_red
    return None


# =========================
# Step 3: Clustering
# =========================

def run_hdbscan(X, min_cluster_size=None, min_samples=None, metric=None,
                cluster_selection_method=None, cluster_selection_epsilon=None,
                prediction_data=True):
    """Run HDBSCAN. Returns (labels, clusterer). labels: noise=-1.
    prediction_data=True enables soft clustering via all_points_membership_vectors."""
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size or HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=min_samples or HDBSCAN_MIN_SAMPLES,
        metric=metric or HDBSCAN_METRIC,
        cluster_selection_method=cluster_selection_method or HDBSCAN_CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=cluster_selection_epsilon if cluster_selection_epsilon is not None else HDBSCAN_CLUSTER_SELECTION_EPSILON,
        prediction_data=prediction_data,
    )
    labels = np.array(clusterer.fit_predict(X), dtype=np.intp)
    return labels, clusterer


def apply_soft_clustering_to_noise(labels, clusterer, root_tweet_ids, threshold):
    """
    Reassign noise points (label=-1) to clusters when soft membership prob >= threshold.
    Uses hdbscan.prediction.all_points_membership_vectors.
    Returns (updated_labels, reassignments) where reassignments is list of
    (tweet_id, new_cluster_id, membership_prob).
    """
    if threshold <= 0:
        return labels, []

    try:
        from hdbscan import prediction as hdbscan_prediction
        soft = hdbscan_prediction.all_points_membership_vectors(clusterer)
    except (ImportError, AttributeError):
        return labels, []

    clusters = np.array(sorted([c for c in np.unique(clusterer.labels_) if c >= 0])).astype(np.intp)
    if clusters.size == 0:
        return labels, []

    # Column j in soft corresponds to clusters[j]
    updated = labels.copy()
    reassignments = []
    for i in range(len(labels)):
        if labels[i] != -1:
            continue
        probs = soft[i]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        if best_prob >= threshold:
            new_label = int(clusters[best_idx])
            updated[i] = new_label
            reassignments.append((root_tweet_ids[i], new_label, best_prob))
    return updated, reassignments


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
    Returns dict tweet_id_str -> cluster_id (int). -1 = HDBSCAN noise, -2 = root excluded from clustering.
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

def save_root_cluster_assignments(root_tweet_ids, labels_hdbscan, out_dir):
    """Save CSV: tweet_id, cluster_hdbscan (roots only)."""
    path = os.path.join(out_dir, "cluster_assignments_roots.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "cluster_hdbscan"])
        for tid, lh in zip(root_tweet_ids, labels_hdbscan):
            w.writerow([tid, lh])
    print(f"  Saved {path}")


def save_cluster_size_report(stats_hdbscan, out_dir):
    path = os.path.join(out_dir, "cluster_size_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"hdbscan": stats_hdbscan}, f, indent=2)
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
    reduce_desc = (
        f"- **REDUCE_METHOD**: {REDUCE_METHOD}. "
        + (f"UMAP: n_components={UMAP_N_COMPONENTS}, n_neighbors={UMAP_N_NEIGHBORS}, metric={UMAP_METRIC} (good for text). "
           f"HDBSCAN then uses euclidean on UMAP output (common practice)." if REDUCE_METHOD == "umap" else
           f"PCA: REDUCE_DIM={REDUCE_DIM} components, then L2-normalize so euclidean = cosine-like.")
    )
    lines = [
        "# Clustering parameter rationale",
        "",
        f"- **Embedding**: {EMBEDDING_MODEL} (sentence-transformers).",
        "",
        "## Dimensionality reduction (before clustering)",
        reduce_desc,
        "",
        "## Excluding low-information roots from clustering",
        f"- **MIN_CONTENT_TOKENS** = {MIN_CONTENT_TOKENS}: only root tweets with at least this many \"content\" tokens are clustered (tokens after dropping @mentions, \"URL\", and http(s) links). 0 = cluster all roots.",
        "- Reduces \"garbage\" clusters of short / mention-only tweets. Excluded root IDs are written to `excluded_root_ids_from_clustering.txt` when non-empty.",
        "",
        "## Soft clustering (noise reassignment)",
        f"- **SOFT_CLUSTERING_NOISE_THRESHOLD** = {SOFT_CLUSTERING_NOISE_THRESHOLD}: after HDBSCAN, noise points (label=-1) are reassigned when their soft membership prob (from `all_points_membership_vectors`) >= this threshold. 0 = disabled.",
        "",
        "## HDBSCAN (primary)",
        f"- `min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}`, `min_samples={HDBSCAN_MIN_SAMPLES}`.",
        f"- `metric={HDBSCAN_METRIC}` (euclidean on UMAP output, or on L2-normalized PCA output).",
        f"- `cluster_selection_method={HDBSCAN_CLUSTER_SELECTION_METHOD}`, `cluster_selection_epsilon={HDBSCAN_CLUSTER_SELECTION_EPSILON}`.",
        "",
        "## Representative posts",
        f"- Top {REP_TOP_PCT*100:.0f}% by popularity within each cluster.",
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
        "reduce_method": REDUCE_METHOD,
        "reduce_dim": REDUCE_DIM,
        "umap": {"n_components": UMAP_N_COMPONENTS, "n_neighbors": UMAP_N_NEIGHBORS, "metric": UMAP_METRIC} if REDUCE_METHOD == "umap" else None,
        "hdbscan": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": HDBSCAN_METRIC,
            "cluster_selection_method": HDBSCAN_CLUSTER_SELECTION_METHOD,
            "cluster_selection_epsilon": HDBSCAN_CLUSTER_SELECTION_EPSILON,
        },
        "min_content_tokens": MIN_CONTENT_TOKENS,
        "soft_clustering_noise_threshold": SOFT_CLUSTERING_NOISE_THRESHOLD,
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

    # ---------- Dimensionality reduction ----------
    if REDUCE_METHOD and (REDUCE_METHOD == "umap" or (REDUCE_METHOD == "pca" and REDUCE_DIM is not None)):
        print("\nReducing dimension ({}: n_components={}, n_neighbors={})...".format(
            REDUCE_METHOD.upper(),
            UMAP_N_COMPONENTS if REDUCE_METHOD == "umap" else REDUCE_DIM,
            UMAP_N_NEIGHBORS if REDUCE_METHOD == "umap" else "N/A"))
        X_cluster = reduce_dim_for_clustering(X)
        if X_cluster is None:
            X_cluster = X
    else:
        X_cluster = X

    # L2-normalize only for PCA so euclidean = cosine-like; UMAP output is used as-is with euclidean HDBSCAN
    if REDUCE_METHOD == "pca" and X_cluster is not None:
        X_cluster = normalize(X_cluster, norm="l2", axis=1)

    # ---------- HDBSCAN ----------
    print("\n--- HDBSCAN ---")
    labels_hdbscan, clusterer = run_hdbscan(X_cluster)
    n_noise_original = int(np.sum(labels_hdbscan == -1))
    stats_hdbscan = cluster_size_stats(labels_hdbscan, "HDBSCAN")

    # ---------- Soft clustering: reassign noise to clusters when prob >= threshold ----------
    if SOFT_CLUSTERING_NOISE_THRESHOLD > 0:
        print("\n--- Soft clustering (noise -> cluster when membership >= {}) ---".format(SOFT_CLUSTERING_NOISE_THRESHOLD))
        labels_hdbscan, reassignments = apply_soft_clustering_to_noise(
            labels_hdbscan, clusterer, root_tweet_ids, SOFT_CLUSTERING_NOISE_THRESHOLD
        )
        n_noise_after = int(np.sum(labels_hdbscan == -1))
        save_noise_to_cluster_reassignments(reassignments, root_tweets, OUTPUT_DIR)
        print("  Noise: {} -> {} (reassigned {} tweets to clusters)".format(
            n_noise_original, n_noise_after, n_noise_original - n_noise_after))
    else:
        n_noise_after = n_noise_original
        print("  Soft clustering disabled (SOFT_CLUSTERING_NOISE_THRESHOLD=0)")
        print("  Noise: {}".format(n_noise_original))

    stats_hdbscan = cluster_size_stats(labels_hdbscan, "HDBSCAN", quiet=True)
    save_noise_tweets_sample(root_tweets, root_tweet_ids, labels_hdbscan, OUTPUT_DIR, sample_size=100)

    # ---------- Save cluster assignments and reports ----------
    save_root_cluster_assignments(root_tweet_ids, labels_hdbscan, OUTPUT_DIR)
    save_cluster_size_report(stats_hdbscan, OUTPUT_DIR)
    save_parameter_rationale(OUTPUT_DIR)
    save_hyperparameters_json(OUTPUT_DIR)

    # ---------- Representative posts (HDBSCAN only) ----------
    print("\nRepresentative posts per cluster (top {}% by popularity)...".format(int(REP_TOP_PCT * 100)))
    tweet_to_cluster_h = assign_all_tweets_to_cluster(reply_chain, root_tweet_ids, labels_hdbscan, excluded_root_ids=excluded_root_ids)
    rep_hdbscan = select_representative_posts(tweet_to_cluster_h, popularity_scores, REP_TOP_PCT)
    save_representative_posts({"hdbscan": rep_hdbscan}, OUTPUT_DIR)

    print("\nDONE.")
    print("  noise (original): {}, noise (after soft clustering): {}".format(n_noise_original, n_noise_after))
    print("Outputs: cluster_assignments_roots.csv, cluster_size_report.json, clustering_hyperparameters.json,")
    print("  parameter_rationale.md, noise_to_cluster_reassignments.csv, noise_tweet_ids_primary.txt,")
    print("  noise_tweets_sample_100.jsonl/csv, representative_posts_hdbscan.json")


if __name__ == "__main__":
    main()
