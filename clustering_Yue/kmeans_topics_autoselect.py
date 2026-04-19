"""
Auto-select KMeans topic pipeline (Python script version).

Derived from: clustering/kmeans_topics_autoselect.ipynb
Run: python clustering/kmeans_topics_autoselect.py

Notes:
- Starts at k=2 and keeps increasing k until stop condition is met.
- Embedding fallback: cached embeddings -> local model -> remote model download (if enabled).
"""

import csv
import json
import os
import argparse
from collections import defaultdict
from datetime import datetime, time, timezone, timedelta

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Config
# =========================
INPUT_DIR = "clustering/data"
OUTPUT_DIR = "clustering/data/processed/kmeans_topic_search"

FILTERED_TWEETS_FILE = "filtered_tweets.jsonl"
REPLY_CHAIN_FILE = "reply_chain.jsonl"
POPULARITY_SCORES_FILE = "tweet_popularity_scores.csv"

# Pipeline switches (run in two stages if needed)
RUN_KMEANS_SWEEP = True
RUN_TOPIC_GENERATION = True

# Optional time cutoff for data inclusion.
# Set directly here if you prefer in-file config.
# Examples: "Jul 2025", "Jul 22 2025", "2025-07", "2025-07-22", "2025-07-22T07:30:00Z"
CUTOFF_DATE = "2025-08"
# Environment variable has higher priority than this in-file value.
# Examples:
#   --cutoff-date "Jul 2025"   -> keeps data before July 2025
#   --cutoff-date "Jul 22 2025"
#   --cutoff-date "2025-07"    -> keeps data before July 2025
#   --cutoff-date "2025-07-22"
#   --cutoff-date "2025-07-22T07:30:00Z"
CUTOFF_DATE_RAW = os.environ.get("CUTOFF_DATE", CUTOFF_DATE).strip()

# Root embedding model for clustering.
# Use a fully-qualified HF repo id by default to avoid SentenceTransformer
# rewriting bare names into "sentence-transformers/<name>".
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_TRUST_REMOTE_CODE = True
EMBEDDING_BATCH_SIZE = 8
ALLOW_REMOTE_EMBEDDING_MODEL = (os.environ.get("ALLOW_REMOTE_EMBEDDING_MODEL", "1") == "1")

# Local embedding artifacts (priority order)
EMBEDDING_CACHE_NPY = os.path.join(OUTPUT_DIR, "root_embeddings.npy")
EMBEDDING_CACHE_META = os.path.join(OUTPUT_DIR, "root_embedding_meta.json")
ROOT_EMBEDDING_CSV_CANDIDATES = [
    os.path.join(INPUT_DIR, "processed", "root_tweet_embeddings.csv"),
    "clustering/root_tweet_embeddings.csv",
]

# KMeans sweep
K_START = 3
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

# Auto-stop rules
TOPIC_SIM_THRESHOLD = float(os.environ.get("TOPIC_SIM_THRESHOLD", "0.75"))
STOP_CONSECUTIVE = int(os.environ.get("STOP_CONSECUTIVE", "3"))
SIMILAR_CLUSTER_RATIO_LIMIT = float(os.environ.get("SIMILAR_CLUSTER_RATIO_LIMIT", "0.34"))

# Filtering
MIN_CONTENT_TOKENS = 3

# Representative selection
REP_TOP_PCT = 0.10

# Topic generation
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "qwen3.5-plus")
MAX_POSTS_PER_CLUSTER = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Helpers: data loading
# =========================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_popularity_scores(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get("tweet_id", "")).strip()
            try:
                scores[tid] = float(row.get("popularity_score", 0))
            except (TypeError, ValueError):
                scores[tid] = 0.0
    return scores


def parse_cutoff_datetime(raw_value):
    if not raw_value:
        return None

    raw = raw_value.strip()
    if not raw:
        return None

    datetime_formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%b %d %Y %H:%M:%S",
        "%b %d %Y %H:%M",
    ]
    date_only_formats = [
        "%b %d %Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    month_only_formats = [
        "%b %Y",
        "%B %Y",
        "%Y-%m",
    ]

    iso_candidate = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in datetime_formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    # Date-only values are interpreted as end-of-day UTC to include that date.
    for fmt in date_only_formats:
        try:
            d = datetime.strptime(raw, fmt).date()
            dt = datetime.combine(d, time.max, tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    # Month-only values are interpreted as "before this month".
    # Example: "Jul 2025" -> keep data up to 2025-06-30 23:59:59.999999 UTC.
    for fmt in month_only_formats:
        try:
            month_start = datetime.strptime(raw, fmt).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
            return (month_start - timedelta(microseconds=1)).astimezone(timezone.utc)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid cutoff date: {raw_value!r}. "
        "Use formats like 'Jul 2025', 'Jul 22 2025', '2025-07', '2025-07-22', or '2025-07-22T07:30:00Z'."
    )


def parse_tweet_created_at(created_at):
    if not created_at:
        return None
    raw = str(created_at).strip()
    if not raw:
        return None

    try:
        dt = datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def filter_data_by_cutoff(filtered_tweets, reply_chain, cutoff_dt):
    if cutoff_dt is None:
        return filtered_tweets, reply_chain, None, 0

    kept_tweets = []
    allowed_ids = set()
    parse_failed = 0
    for t in filtered_tweets:
        tid = str(t.get("id", "")).strip()
        if not tid:
            continue
        created_at = t.get("createdAt")
        dt = parse_tweet_created_at(created_at)
        if dt is None:
            parse_failed += 1
            continue
        if dt <= cutoff_dt:
            kept_tweets.append(t)
            allowed_ids.add(tid)

    kept_reply_chain = []
    for r in reply_chain:
        tid = str(r.get("tweet_id", "")).strip()
        rid = str(r.get("root_tweet_id", "")).strip()
        if tid in allowed_ids and rid in allowed_ids:
            kept_reply_chain.append(r)

    return kept_tweets, kept_reply_chain, allowed_ids, parse_failed


def get_root_tweets(filtered_tweets, reply_chain):
    id_to_tweet = {str(t.get("id")): t for t in filtered_tweets if t.get("id") is not None}
    root_ids = {str(r["tweet_id"]) for r in reply_chain if r.get("is_root") is True}
    return [id_to_tweet[rid] for rid in root_ids if rid in id_to_tweet]


def _content_token_count(text):
    if not text or not isinstance(text, str):
        return 0
    n = 0
    for tok in text.strip().split():
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("@"):
            continue
        if tok.upper() == "URL":
            continue
        if tok.startswith("http://") or tok.startswith("https://"):
            continue
        n += 1
    return n


def filter_substantive_roots(root_tweets, min_tokens=MIN_CONTENT_TOKENS):
    if min_tokens <= 0:
        return root_tweets, []
    keep, excluded = [], []
    for t in root_tweets:
        text = (t.get("clean_text") or t.get("text") or "").strip()
        if _content_token_count(text) >= min_tokens:
            keep.append(t)
        else:
            excluded.append(str(t.get("id")))
    return keep, excluded


def build_tweet_to_root(reply_chain):
    return {str(r["tweet_id"]): str(r["root_tweet_id"]) for r in reply_chain}


def assign_all_tweets_to_cluster(reply_chain, root_tweet_ids, root_labels, excluded_root_ids=None):
    root_to_cluster = {str(tid): int(lab) for tid, lab in zip(root_tweet_ids, root_labels)}
    default_for_missing = -2 if (excluded_root_ids and len(excluded_root_ids) > 0) else -1
    tweet_to_root = build_tweet_to_root(reply_chain)
    return {tid: root_to_cluster.get(root_id, default_for_missing) for tid, root_id in tweet_to_root.items()}


def assign_filtered_tweets_to_cluster(filtered_tweets, reply_chain, root_tweet_ids, root_labels, excluded_root_ids=None):
    """Propagate root cluster labels to all filtered tweets via reply_chain/root mapping."""
    root_to_cluster = {str(tid): int(lab) for tid, lab in zip(root_tweet_ids, root_labels)}
    default_for_missing = -2 if (excluded_root_ids and len(excluded_root_ids) > 0) else -1

    tweet_to_cluster = assign_all_tweets_to_cluster(
        reply_chain,
        root_tweet_ids,
        root_labels,
        excluded_root_ids=excluded_root_ids,
    )

    out = {}
    for t in filtered_tweets:
        tid = str(t.get("id", "")).strip()
        if not tid:
            continue
        if tid in tweet_to_cluster:
            out[tid] = int(tweet_to_cluster[tid])
        elif tid in root_to_cluster:
            out[tid] = int(root_to_cluster[tid])
        else:
            out[tid] = int(default_for_missing)
    return out


def select_representative_posts(tweet_to_cluster, popularity_scores, top_pct=REP_TOP_PCT):
    by_cluster = defaultdict(list)
    for tid, cid in tweet_to_cluster.items():
        if cid < 0:
            continue
        by_cluster[cid].append((tid, popularity_scores.get(tid, 0.0)))

    out = {}
    for cid, pairs in by_cluster.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        n = max(1, int(len(pairs) * top_pct))
        out[cid] = pairs[:n]
    return out


def build_id_to_text(filtered_tweets):
    m = {}
    for row in filtered_tweets:
        tid = row.get("id")
        if tid is None:
            continue
        txt = (row.get("clean_text") or row.get("text") or "").strip()
        if txt:
            m[str(tid)] = txt
    return m


# =========================
# Embedding helpers
# =========================
def _load_local_model(local_files_only):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
    )
    # Older sentence-transformers versions do not accept `model_kwargs`.
    # Move the underlying transformer to fp16 after load when CUDA is available.
    if device == "cuda":
        try:
            model.half()
        except Exception:
            pass
    return model, device


def load_embedder_with_fallback():
    # 1) Try local downloaded model first.
    try:
        model, device = _load_local_model(local_files_only=True)
        print("  embedding model loaded from local files")
        return model, device, "local"
    except Exception as e:
        print(f"  local model unavailable: {e}")

    # 2) Allow remote model download if configured.
    if not ALLOW_REMOTE_EMBEDDING_MODEL:
        raise RuntimeError(
            "Local embedding model not found and remote download is disabled. "
            "Set EMBEDDING_MODEL_PATH to a local model directory (e.g. /root/oasis/models/Qwen3-Embedding-0.6B), "
            "or set ALLOW_REMOTE_EMBEDDING_MODEL=1 and use a valid repo id like Qwen/Qwen3-Embedding-0.6B."
        )

    model, device = _load_local_model(local_files_only=False)
    print("  embedding model loaded via remote download")
    return model, device, "remote"


def _load_cached_embeddings_from_npy(root_tweet_ids):
    if not (os.path.isfile(EMBEDDING_CACHE_NPY) and os.path.isfile(EMBEDDING_CACHE_META)):
        return None

    with open(EMBEDDING_CACHE_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cached_ids = [str(x) for x in meta.get("tweet_ids", [])]
    if not cached_ids:
        return None

    id_to_idx = {tid: i for i, tid in enumerate(cached_ids)}
    if any(tid not in id_to_idx for tid in root_tweet_ids):
        return None

    X_all = np.load(EMBEDDING_CACHE_NPY)
    if X_all.ndim != 2 or X_all.shape[0] != len(cached_ids):
        return None

    reorder = [id_to_idx[tid] for tid in root_tweet_ids]
    X = X_all[reorder]
    print(f"  loaded cached embeddings: {EMBEDDING_CACHE_NPY} ({X.shape[0]} x {X.shape[1]})")
    return X


def _load_cached_embeddings_from_csv(root_tweet_ids):
    for path in ROOT_EMBEDDING_CSV_CANDIDATES:
        if not os.path.isfile(path):
            continue

        id_to_emb = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = str(row.get("root_tweet_id", "")).strip()
                emb_raw = row.get("embedding", "")
                if not rid or not emb_raw:
                    continue
                try:
                    vec = np.asarray(json.loads(emb_raw), dtype=np.float32)
                except Exception:
                    continue
                if vec.ndim != 1:
                    continue
                id_to_emb[rid] = vec

        if any(tid not in id_to_emb for tid in root_tweet_ids):
            continue

        dim = len(id_to_emb[root_tweet_ids[0]])
        if any(len(id_to_emb[tid]) != dim for tid in root_tweet_ids):
            continue

        X = np.vstack([id_to_emb[tid] for tid in root_tweet_ids]).astype(np.float32)
        print(f"  loaded cached embeddings: {path} ({X.shape[0]} x {X.shape[1]})")
        return X

    return None


def save_embedding_cache(X, root_tweet_ids):
    np.save(EMBEDDING_CACHE_NPY, X)
    with open(EMBEDDING_CACHE_META, "w", encoding="utf-8") as f:
        json.dump({"tweet_ids": root_tweet_ids}, f, ensure_ascii=False, indent=2)
    print(f"  saved embedding cache: {EMBEDDING_CACHE_NPY}")


def get_root_embeddings_with_fallback(root_tweets):
    texts = [(t.get("clean_text") or t.get("text") or "").strip() or " " for t in root_tweets]
    ids = [str(t.get("id")) for t in root_tweets]

    # Priority 1: local precomputed embeddings.
    X = _load_cached_embeddings_from_npy(ids)
    if X is None:
        X = _load_cached_embeddings_from_csv(ids)
    if X is not None:
        return X, ids, "cache"

    # Priority 2/3: local model -> remote model.
    model, device, source = load_embedder_with_fallback()
    print(f"  root embedder device: {device}")
    X = model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    X = np.asarray(X)
    save_embedding_cache(X, ids)
    return X, ids, source


def load_openai_client():
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY or DASHSCOPE_API_KEY before running topic generation."
        )
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


# =========================
# Stage 1 helpers: single-k clustering
# =========================
def run_kmeans_single_k(k, X, root_tweet_ids, filtered_tweets, reply_chain, excluded_root_ids, popularity_scores):
    labels = KMeans(
        n_clusters=k,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    ).fit_predict(X)
    labels = np.array(labels, dtype=np.intp)

    tweet_to_cluster = assign_all_tweets_to_cluster(
        reply_chain,
        root_tweet_ids,
        labels,
        excluded_root_ids=excluded_root_ids,
    )
    rep = select_representative_posts(tweet_to_cluster, popularity_scores, REP_TOP_PCT)

    all_tweet_to_cluster = assign_filtered_tweets_to_cluster(
        filtered_tweets,
        reply_chain,
        root_tweet_ids,
        labels,
        excluded_root_ids=excluded_root_ids,
    )
    all_assign_rows = [
        {"tweet_id": str(tid), "cluster_kmeans": int(cid)}
        for tid, cid in all_tweet_to_cluster.items()
    ]
    write_jsonl(all_assign_jsonl, all_assign_rows)

    return {
        "k": k,
        "n_clusters": int(len(set(labels.tolist()))),
        "root_assignments": [{"tweet_id": str(tid), "cluster_kmeans": int(lab)} for tid, lab in zip(root_tweet_ids, labels)],
        "all_assignments": [
            {"tweet_id": str(tid), "cluster_kmeans": int(cid)}
            for tid, cid in all_tweet_to_cluster.items()
        ],
        "rep_posts": rep,
    }


# =========================
# Stage 2: Topic generation
# =========================
def generate_topics_for_k(rep_posts, id_to_text, client):
    topics = {}
    for cid in sorted(rep_posts.keys()):
        tweet_ids = [tid for tid, _ in rep_posts[cid]][:MAX_POSTS_PER_CLUSTER]
        texts = [id_to_text.get(str(tid), "").strip() for tid in tweet_ids]
        texts = [t for t in texts if t]
        if not texts:
            topics[str(cid)] = "No posts available."
            continue

        posts_block = "\n".join(f"- {t}" for t in texts)
        prompt = f"""Given the following posts from one Twitter cluster, generate a concise topic description.

Requirements:
- Description should be 1-2 sentences, under 100 words, neutral, no opinions
- Be specific and concise
- Capture the shared core topic
- Output only the description text, no label, no preamble

Cluster ID: {cid}

Sample posts:
{posts_block}
"""
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a topic summarization assistant. Return only a brief neutral topic description "
                            "(1-2 sentences, under 100 words). No label. No extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            topics[str(cid)] = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            topics[str(cid)] = f"[Error: {e}]"
    return topics


# =========================
# Auto-select via topic similarity
# =========================
def overlap_stats(topics_dict, embedder, threshold):
    cids = sorted(topics_dict.keys(), key=lambda x: int(x))
    texts = [topics_dict[c] for c in cids]
    if len(texts) < 2:
        return {
            "similar_pairs": [],
            "similar_pairs_count": 0,
            "involved_clusters": [],
            "involved_clusters_count": 0,
            "involved_ratio": 0.0,
        }

    emb = embedder.encode(texts, convert_to_numpy=True, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)
    sim = cosine_similarity(emb)
    pairs = []
    involved = set()
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            s = float(sim[i, j])
            if s >= threshold:
                pairs.append({"c1": int(cids[i]), "c2": int(cids[j]), "cosine": s})
                involved.add(int(cids[i]))
                involved.add(int(cids[j]))
    return {
        "similar_pairs": pairs,
        "similar_pairs_count": len(pairs),
        "involved_clusters": sorted(list(involved)),
        "involved_clusters_count": len(involved),
        "involved_ratio": (len(involved) / len(cids)) if cids else 0.0,
    }


def run_autoselect_pipeline(filtered_tweets, reply_chain, popularity_scores, X, root_tweet_ids, excluded_root_ids):
    id_to_text = build_id_to_text(filtered_tweets)
    client = load_openai_client()

    topic_embedder, _, _ = load_embedder_with_fallback()

    k_reports = []
    best_run_info = None
    best_topics = None

    consecutive_trigger = 0
    best_k = None

    max_k = len(root_tweet_ids)
    if max_k < K_START:
        raise RuntimeError(f"Not enough root tweets for KMeans: {max_k}")

    for k in range(K_START, max_k + 1):
        print(f"\n=== KMeans k={k} ===")
        run_info = run_kmeans_single_k(
            k,
            X,
            root_tweet_ids,
            filtered_tweets,
            reply_chain,
            excluded_root_ids,
            popularity_scores,
        )

        print(f"  generating topics for k={k}...")
        topics = generate_topics_for_k(run_info["rep_posts"], id_to_text, client)
        print(f"  generated topics for {len(topics)} clusters")

        stats = overlap_stats(topics, topic_embedder, TOPIC_SIM_THRESHOLD)
        n_clusters = len(topics)
        involved = stats["involved_clusters_count"]
        trigger = involved > (n_clusters * SIMILAR_CLUSTER_RATIO_LIMIT) if n_clusters else False

        if trigger:
            consecutive_trigger += 1
        else:
            consecutive_trigger = 0
            best_k = k
            best_run_info = run_info
            best_topics = topics

        row = {
            "k": k,
            "n_clusters": n_clusters,
            "threshold": TOPIC_SIM_THRESHOLD,
            "similar_pairs_count": stats["similar_pairs_count"],
            "involved_clusters_count": involved,
            "involved_ratio": stats["involved_ratio"],
            "trigger": trigger,
            "consecutive_trigger": consecutive_trigger,
            "similar_pairs": stats["similar_pairs"],
        }
        k_reports.append(row)

        print(
            f"  overlap: pairs={row['similar_pairs_count']}, involved={involved}/{n_clusters}, "
            f"limit={SIMILAR_CLUSTER_RATIO_LIMIT:.2f}, trigger={trigger}, consecutive={consecutive_trigger}"
        )

        if consecutive_trigger >= STOP_CONSECUTIVE:
            print(f"  stop condition reached at k={k}")
            break

    if best_k is None and k_reports:
        best_k = k_reports[0]["k"]
        fallback_run_info = run_kmeans_single_k(
            best_k,
            X,
            root_tweet_ids,
            filtered_tweets,
            reply_chain,
            excluded_root_ids,
            popularity_scores,
        )
        best_run_info = fallback_run_info
        best_topics = generate_topics_for_k(fallback_run_info["rep_posts"], id_to_text, client)

    if best_k is None or best_run_info is None or best_topics is None:
        raise RuntimeError("Failed to select a valid k and build final outputs.")

    assignments_path = os.path.join(OUTPUT_DIR, "cluster_assignments.jsonl")
    topics_path = os.path.join(OUTPUT_DIR, "cluster_topics.json")

    write_jsonl(assignments_path, best_run_info["all_assignments"])
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(best_topics, f, ensure_ascii=False, indent=2)

    print(f"\nBest k: {best_k}")
    print(f"Saved selected assignments: {assignments_path}")
    print(f"Saved selected topics: {topics_path}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Auto-select KMeans topic pipeline")
    parser.add_argument(
        "--cutoff-date",
        default=CUTOFF_DATE_RAW,
        help=(
            "Only include tweets on/before this UTC time. "
            "Examples: 'Jul 2025' (before July), '2025-07' (before July), "
            "'Jul 22 2025', '2025-07-22', '2025-07-22T07:30:00Z'."
        ),
    )
    args = parser.parse_args()

    cutoff_dt = parse_cutoff_datetime(args.cutoff_date)

    filtered_path = os.path.join(INPUT_DIR, FILTERED_TWEETS_FILE)
    reply_path = os.path.join(INPUT_DIR, REPLY_CHAIN_FILE)
    pop_path = os.path.join(INPUT_DIR, POPULARITY_SCORES_FILE)

    print("Loading data...")
    filtered_tweets = load_jsonl(filtered_path)
    reply_chain = load_jsonl(reply_path)
    popularity_scores = load_popularity_scores(pop_path)
    print(f"  filtered_tweets: {len(filtered_tweets)}, reply_chain: {len(reply_chain)}, popularity entries: {len(popularity_scores)}")

    filtered_tweets, reply_chain, allowed_ids, parse_failed = filter_data_by_cutoff(
        filtered_tweets,
        reply_chain,
        cutoff_dt,
    )
    if cutoff_dt is not None:
        print(f"  cutoff_date_utc: {cutoff_dt.isoformat()}")
        print(
            f"  after cutoff filter -> filtered_tweets: {len(filtered_tweets)}, "
            f"reply_chain: {len(reply_chain)}, createdAt parse_failed: {parse_failed}"
        )
        popularity_scores = {tid: score for tid, score in popularity_scores.items() if tid in allowed_ids}
        print(f"  popularity entries after cutoff: {len(popularity_scores)}")

    root_tweets_all = get_root_tweets(filtered_tweets, reply_chain)
    root_tweets, excluded_root_ids = filter_substantive_roots(root_tweets_all, MIN_CONTENT_TOKENS)
    print(f"  root_tweets total: {len(root_tweets_all)}, substantive: {len(root_tweets)}, excluded: {len(excluded_root_ids)}")

    if not (RUN_KMEANS_SWEEP and RUN_TOPIC_GENERATION):
        raise RuntimeError("Auto-select requires both RUN_KMEANS_SWEEP=True and RUN_TOPIC_GENERATION=True")

    print("\n[AutoSelect] KMeans from k=2 upward until stop condition...")
    X, root_tweet_ids, emb_source = get_root_embeddings_with_fallback(root_tweets)
    print(f"  root embeddings source: {emb_source}")
    print(f"  root embedding shape: {X.shape}")

    run_autoselect_pipeline(
        filtered_tweets,
        reply_chain,
        popularity_scores,
        X,
        root_tweet_ids,
        excluded_root_ids,
    )


if __name__ == "__main__":
    main()
