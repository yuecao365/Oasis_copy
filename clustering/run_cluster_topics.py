#!/usr/bin/env python3
"""
Run topic generation with multiple clustering settings and write summaries.

Scenarios:
- HDBSCAN with min_cluster_size: 200, 400, 600
- KMeans with n_clusters: 3, 4, 5, 6

Output: tweet_processing/cluster_topics.txt

Notes:
- Uses embed_texts / load_tweets / build_prompt / maybe_call_openai from topic_generation.py.
- Requires SBERT model downloadable from Hugging Face (or cached locally per SentenceTransformer).
- If OPENAI_API_KEY is unset, topics may be empty strings.
"""

from pathlib import Path
from typing import Dict, List

import hdbscan
import numpy as np
from sklearn.cluster import KMeans

from topic_generation import (
    MAX_TWEETS,
    build_prompt,
    embed_texts,
    load_tweets,
    maybe_call_openai,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TXT = REPO_ROOT / "tweet_processing" / "cluster_topics.txt"


def topics_for_labels(labels: np.ndarray, tweets) -> List[str]:
    clusters: Dict[int, List] = {}
    for label, tw in zip(labels, tweets):
        if label == -1:
            continue  # ignore noise for HDBSCAN
        clusters.setdefault(int(label), []).append(tw)

    topics: List[str] = []
    for cid in sorted(clusters.keys()):
        tws = clusters[cid]
        sorted_tws = sorted(tws, key=lambda t: t.popularity, reverse=True)[:MAX_TWEETS]
        record = build_prompt(cid, sorted_tws)
        topic = maybe_call_openai(record["prompt"]) or ""
        topics.append(topic)
    return topics


def main():
    tweets = load_tweets()
    if not tweets:
        raise SystemExit("No tweets loaded.")
    embeddings = embed_texts([t.text for t in tweets])

    lines: List[str] = []

    # HDBSCAN runs
    for min_size in (200, 400, 600):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=None,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)
        topics = topics_for_labels(labels, tweets)
        lines.append(f"HDBSCAN {min_size}:")
        lines.extend([f"topic {i+1}: {t}" for i, t in enumerate(topics)])
        lines.append("")  # blank line

    # KMeans runs
    for k in (3, 4, 5, 6):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(embeddings)
        topics = topics_for_labels(labels, tweets)
        lines.append(f"KMEANS {k}:")
        lines.extend([f"topic {i+1}: {t}" for i, t in enumerate(topics)])
        lines.append("")

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote topics to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
