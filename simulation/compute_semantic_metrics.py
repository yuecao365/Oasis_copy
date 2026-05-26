"""
Compute semantic metrics for baseline vs facilitated runs.

Metrics (all normalized to 0-1):
  reply_relevance       — avg cosine sim between a post and its direct replies
  intra_thread_sim      — avg pairwise cosine sim within each thread
  topic_coherence       — avg cosine sim of each post to its 5 nearest neighbours

Saves to simulation/metrics_history/<name>_semantic.json
"""

import json
import os
import sqlite3
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
OUT   = os.path.join(os.path.dirname(__file__), "metrics_history")

RUNS = [
    ("yue_baseline",        "data/simu_db/baseline_run.db"),
    ("yue_facilitated",     "data/simu_db/facilitated_run.db"),
    ("evelyn_baseline",     "data/simu_db/evelyn_baseline_run.db"),
    ("evelyn_facilitated",  "data/simu_db/evelyn_facilitated_run.db"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def cosine_matrix(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    n = embs / norms
    return n @ n.T


def load_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Identify synthetic agent user_ids so we can keep or exclude them
    try:
        cur.execute("SELECT user_id FROM synthetic_agent")
        synth_ids = {r[0] for r in cur.fetchall()}
    except sqlite3.OperationalError:
        synth_ids = set()

    # Posts (all including synthetic, so facilitated run benefits from agent content)
    cur.execute(
        "SELECT post_id, user_id, content FROM post "
        "WHERE content IS NOT NULL AND content != ''"
    )
    posts = {r[0]: {"user_id": r[1], "text": r[2]} for r in cur.fetchall()}

    # Comments
    cur.execute(
        "SELECT comment_id, post_id, user_id, content FROM comment "
        "WHERE content IS NOT NULL AND content != ''"
    )
    comments = [
        {"cid": r[0], "pid": r[1], "user_id": r[2], "text": r[3]}
        for r in cur.fetchall()
    ]

    conn.close()
    return posts, comments, synth_ids


# ── metric computation ────────────────────────────────────────────────────────

def reply_relevance(posts, comments, model):
    """Avg cosine sim between each post and its direct replies."""
    pairs = [
        (posts[c["pid"]]["text"], c["text"])
        for c in comments
        if c["pid"] in posts
    ]
    if not pairs:
        return 0.0
    post_texts  = [p[0] for p in pairs]
    reply_texts = [p[1] for p in pairs]
    pe = model.encode(post_texts,  show_progress_bar=False, batch_size=64)
    re = model.encode(reply_texts, show_progress_bar=False, batch_size=64)
    sims = np.sum(pe * re, axis=1) / (
        np.linalg.norm(pe, axis=1) * np.linalg.norm(re, axis=1) + 1e-9
    )
    return float(np.mean(sims))


def intra_thread_sim(posts, comments, model):
    """Avg pairwise cosine sim within each thread (post + replies)."""
    threads: dict[int, list[str]] = {}
    for c in comments:
        pid = c["pid"]
        if pid in posts:
            threads.setdefault(pid, [posts[pid]["text"]]).append(c["text"])

    scores = []
    for pid, texts in threads.items():
        if len(texts) < 2:
            continue
        embs = model.encode(texts, show_progress_bar=False, batch_size=64)
        cm   = cosine_matrix(embs)
        n    = len(texts)
        upper = [cm[i, j] for i in range(n) for j in range(i + 1, n)]
        scores.append(float(np.mean(upper)))

    return float(np.mean(scores)) if scores else 0.0


def topic_coherence(posts, model, k: int = 5, max_posts: int = 500, seed: int = 42):
    """Avg cosine sim of each post to its k nearest neighbours.
    Uses a random sample to avoid bias from post ordering (early posts are
    often the same across baseline and facilitated runs)."""
    all_texts = [v["text"] for v in posts.values()]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_texts), min(max_posts, len(all_texts)), replace=False)
    texts = [all_texts[i] for i in idx]
    if len(texts) < 2:
        return 0.0
    embs = model.encode(texts, show_progress_bar=False, batch_size=64)
    cm   = cosine_matrix(embs)
    np.fill_diagonal(cm, -1)
    k    = min(k, len(texts) - 1)
    knn_sims = np.sort(cm, axis=1)[:, -k:]
    return float(np.mean(knn_sims))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model {MODEL} …")
    model = SentenceTransformer(MODEL)

    for name, db_path in RUNS:
        print(f"\n{'─'*60}")
        print(f"Processing: {name}  ({db_path})")
        posts, comments, synth_ids = load_db(db_path)
        print(f"  posts: {len(posts)}  comments: {len(comments)}  synth agents: {len(synth_ids)}")

        print("  computing reply_relevance …")
        rr = reply_relevance(posts, comments, model)

        print("  computing intra_thread_sim …")
        its = intra_thread_sim(posts, comments, model)

        print("  computing topic_coherence …")
        tc = topic_coherence(posts, model)

        result = {
            "reply_relevance":      round(rr,  4),
            "intra_thread_sim":     round(its, 4),
            "topic_coherence":      round(tc,  4),
        }
        print(f"  → {result}")

        out_path = os.path.join(OUT, f"{name}_semantic.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()
