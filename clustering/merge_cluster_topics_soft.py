"""
Soft-merge cluster topics at the topic level (no change to original K=20 clustering).

Workflow:
1. Run tweet_clustering.py (K=20) and topic_generation.py to produce:
   - data/processed/cluster_topics.json              (cluster_id -> topic sentence)
   - data/processed/cluster_assignments_roots.csv    (optional, for size stats)
   - data/processed/cluster_assignments_kmeans.json  (tweet_id -> hard cluster_id, JSON)
2. Run this script:
   python merge_cluster_topics_soft.py

It will:
- Embed the 20 topic sentences using sentence-transformers/all-mpnet-base-v2.
- Compute cosine similarity between topics.
- Greedily group topics whose pairwise similarity >= SIM_THRESHOLD (default 0.88).
- Write three JSON files:
   - data/processed/cluster_topic_groups_soft.json       (soft-merge groups, with original cluster_ids)
   - data/processed/cluster_topics_soft.json             (soft groups: list[{cluster_id, topic}])
   - data/processed/cluster_assignments_soft_groups.json (tweet_id -> soft group_id mapping)
"""

import json
import os
import csv
from typing import Dict, List, Any

import numpy as np


INPUT_DIR = "data/processed"
TOPIC_FILE = "cluster_topics.json"
ASSIGNMENTS_FILE = "cluster_assignments_roots.csv"  # optional, for size stats
KMEANS_ASSIGNMENTS_FILE = "cluster_assignments_kmeans.json"  # tweet_id -> hard cluster_id (JSON)
OUTPUT_GROUPS_FILE = "cluster_topic_groups_soft.json"
OUTPUT_SOFT_TOPICS_FILE = "cluster_topics_soft.json"
OUTPUT_SOFT_ASSIGN_FILE = "cluster_assignments_soft_groups.json"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SIM_THRESHOLD = 0.88  # soft-merge threshold (cosine similarity)


def load_topics(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    topics: Dict[int, str] = {}
    for k, v in raw.items():
        topics[int(k)] = str(v)
    return topics


def load_cluster_sizes(path: str) -> Dict[int, int]:
    """
    Optional: count how many root tweets in each cluster from cluster_assignments_roots.csv.
    Returns cluster_id -> count. If file missing, returns {}.
    """
    if not os.path.isfile(path):
        return {}

    sizes: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["cluster"])
            sizes[cid] = sizes.get(cid, 0) + 1
    return sizes


def compute_embeddings(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(texts, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    # L2-normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb


def build_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    # Cosine similarity for L2-normalized embeddings is just dot product
    return emb @ emb.T


def soft_merge_topics(
    topics: Dict[int, str],
    sims: np.ndarray,
    cluster_ids: List[int],
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Greedy grouping:
    - iterate cluster_ids in order
    - for each not-yet-assigned id, start a new group
    - add any other ids whose similarity >= threshold to that group
    """
    n = len(cluster_ids)
    assigned = [False] * n
    groups: List[Dict[str, Any]] = []

    for i in range(n):
        if assigned[i]:
            continue
        base_cid = cluster_ids[i]
        group_ids = [base_cid]
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if sims[i, j] >= threshold:
                group_ids.append(cluster_ids[j])
                assigned[j] = True

        # Representative topic: just use the first cluster's topic
        rep_topic = topics[base_cid]
        all_topics = [topics[cid] for cid in group_ids]

        groups.append(
            {
                "cluster_ids": group_ids,
                "representative_topic": rep_topic,
                "all_topics": all_topics,
            }
        )

    return groups


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.normpath(os.path.join(base_dir, "..", INPUT_DIR))

    topic_path = os.path.join(input_dir, TOPIC_FILE)
    assign_path = os.path.join(input_dir, ASSIGNMENTS_FILE)
    kmeans_assign_path = os.path.join(input_dir, KMEANS_ASSIGNMENTS_FILE)

    if not os.path.isfile(topic_path):
        raise FileNotFoundError(f"{topic_path} not found; run topic_generation.py first.")

    print(f"Loading topics from {topic_path} ...")
    topics = load_topics(topic_path)
    cluster_ids = sorted(topics.keys())
    topic_texts = [topics[cid] for cid in cluster_ids]
    print(f"  Loaded {len(cluster_ids)} topics.")

    print("Embedding topics with sentence-transformers...")
    emb = compute_embeddings(topic_texts)
    sims = build_similarity_matrix(emb)

    print(f"Soft merging topics with similarity >= {SIM_THRESHOLD} ...")
    groups = soft_merge_topics(topics, sims, cluster_ids, SIM_THRESHOLD)

    # Optional: add size stats if assignments file is available
    sizes = load_cluster_sizes(assign_path)
    for g in groups:
        g["num_clusters"] = len(g["cluster_ids"])
        if sizes:
            root_counts = [sizes.get(cid, 0) for cid in g["cluster_ids"]]
            g["root_tweet_count"] = int(sum(root_counts))

    # 1) Full soft-merge payload
    groups_payload = {
        "embedding_model": EMBEDDING_MODEL,
        "similarity_threshold": SIM_THRESHOLD,
        "num_original_clusters": len(cluster_ids),
        "num_groups": len(groups),
        "groups": groups,
    }
    os.makedirs(input_dir, exist_ok=True)
    groups_out_path = os.path.join(input_dir, OUTPUT_GROUPS_FILE)
    with open(groups_out_path, "w", encoding="utf-8") as f:
        json.dump(groups_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved soft-merged topic groups to {groups_out_path}")

    # 2) FINAL topics file: list[{cluster_id (soft group id), topic}]
    soft_topics = []
    for gid, g in enumerate(groups):
        soft_topics.append(
            {
                "cluster_id": gid,
                "topic": g.get("representative_topic", ""),
            }
        )
    soft_topics_path = os.path.join(input_dir, OUTPUT_SOFT_TOPICS_FILE)
    with open(soft_topics_path, "w", encoding="utf-8") as f:
        json.dump(soft_topics, f, ensure_ascii=False, indent=2)
    print(f"Saved soft topics list to {soft_topics_path}")

    # 3) FINAL tweet-level mapping: tweet_id -> soft group_id
    if not os.path.isfile(kmeans_assign_path):
        print(
            f"WARNING: {kmeans_assign_path} not found; skipping {OUTPUT_SOFT_ASSIGN_FILE}. "
            "Run tweet_clustering.py to generate cluster_assignments_kmeans.json."
        )
    else:
        # Build hard cluster_id -> soft group_id map
        cluster_to_group = {}
        for gid, g in enumerate(groups):
            for cid in g.get("cluster_ids", []):
                cluster_to_group[int(cid)] = int(gid)

        with open(kmeans_assign_path, "r", encoding="utf-8") as f:
            kmeans_rows = json.load(f)

        soft_rows = []
        skipped_missing = 0
        for r in kmeans_rows:
            tid = str(r.get("tweet_id", "")).strip()
            if not tid:
                continue
            cid = r.get("cluster_id")
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_int not in cluster_to_group:
                skipped_missing += 1
                continue
            gid = cluster_to_group[cid_int]
            soft_rows.append({"tweet_id": tid, "cluster_id": gid})

        soft_assign_path = os.path.join(input_dir, OUTPUT_SOFT_ASSIGN_FILE)
        with open(soft_assign_path, "w", encoding="utf-8") as f:
            json.dump(soft_rows, f, ensure_ascii=False, indent=2)
        print(
            f"Saved {len(soft_rows)} soft-group assignments to {soft_assign_path} "
            f"(skipped {skipped_missing} tweets without a soft group, if any)."
        )


if __name__ == "__main__":
    main()

