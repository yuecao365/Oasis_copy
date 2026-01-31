#!/usr/bin/env python3
"""
Pipeline (no external API for embeddings):
- Load non-duplicate, root tweets from tweet_December.txt (Nov+Dec combined).
- Compute log(1+c) popularity score using PCA-derived weights.
- Embed texts locally with Sentence-BERT (no API).
- Cluster embeddings with HDBSCAN.
- For each cluster, take top-20 tweets by popularity, build a prompt, and optionally call OpenAI
  to generate a topic label.

Notes:
- Requires hdbscan + scikit-learn (`pip install hdbscan scikit-learn` if missing).
- If OPENAI_API_KEY is set, will call OpenAI chat for topic labels; otherwise only prompts are written.
- This script writes JSONL with one record per cluster: {cluster_id, prompt, top_tweets, topic?}.
"""

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import hdbscan
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "tweet_processing" / "tweet_Nov+Dec.txt"
OUTPUT_JSONL = REPO_ROOT / "tweet_processing" / "cluster_topics.jsonl"
PCA_STATS_FILE = REPO_ROOT / "tweet_processing" / "pca_stats.json"

# PCA-derived positive weights (abs-normalized first component).
PCA_WEIGHTS = {
    "likes": 0.220255,
    "retweets": 0.214415,
    "replies": 0.201749,
    "quotes": 0.180303,
    "impressions": 0.183278,
}
PCA_MEANS = None
PCA_STDS = None

MAX_TWEETS = 20
OPENAI_MODEL = "gpt-4o-mini"
SBERT_MODEL = "all-MiniLM-L6-v2"
TFIDF_MAX_FEATURES = 4096
SVD_COMPONENTS = 128


@dataclass
class TweetRow:
    text: str
    popularity: float
    id: str
    topic_id: str
    user_name: str


def load_pca_stats():
    global PCA_WEIGHTS, PCA_MEANS, PCA_STDS
    if not PCA_STATS_FILE.exists():
        return
    with PCA_STATS_FILE.open() as f:
        stats = json.load(f)
    PCA_WEIGHTS = stats.get("weights", PCA_WEIGHTS)
    PCA_MEANS = stats.get("means")
    PCA_STDS = stats.get("stds")


def load_tweets() -> List[TweetRow]:
    load_pca_stats()
    rows: List[TweetRow] = []
    with DATA_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dup") != "0":
                continue
            if row.get("is_root") != "1":
                continue
            metrics = {
                k: float(row.get(k, 0) or 0)
                for k in ["likes", "retweets", "replies", "quotes", "impressions"]
            }
            logs = {k: math.log1p(v) for k, v in metrics.items()}
            if PCA_MEANS and PCA_STDS:
                popularity = sum(
                    PCA_WEIGHTS[k]
                    * (
                        (logs[k] - PCA_MEANS.get(k, 0))
                        / (PCA_STDS.get(k, 1) or 1)
                    )
                    for k in PCA_WEIGHTS.keys()
                )
            else:
                popularity = sum(
                    PCA_WEIGHTS[k] * logs[k] for k in PCA_WEIGHTS.keys()
                )
            rows.append(
                TweetRow(
                    text=row["text"],
                    popularity=popularity,
                    id=row.get("id", ""),
                    topic_id=row.get("topic_id", ""),
                    user_name=row.get("user_name", ""),
                )
            )
    return rows


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64
    )
    return embeddings.astype(float)


def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=500,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def build_prompt(cluster_id: int, tweets: List[TweetRow]) -> Dict[str, object]:
    header = (
        "Given the following tweets from the same discussion, generate a concise topic label.\n"
        "Generate a concise topic label that summarizes their shared core idea.\n\n"
        "Requirements:\n"
        "- Use a noun or noun phrase only\n"
        "- 3–6 words maximum\n"
        "- No verbs, no opinions, no sentiment\n"
        "- Capture the shared core topic\n\n"
        "Tweets:"
    )
    body_lines = []
    for idx, tw in enumerate(tweets, 1):
        body_lines.append(f"{idx}. {tw.text}")
    prompt = header + "\n" + "\n".join(body_lines)
    return {
        "cluster_id": int(cluster_id),
        "prompt": prompt,
        "top_tweets": [
            {
                "id": tw.id,
                "topic_id": tw.topic_id,
                "user_name": tw.user_name,
                "popularity": tw.popularity,
                "text": tw.text,
            }
            for tw in tweets
        ],
    }


def maybe_call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a topic labeling assistant. Output a short noun phrase only.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip()


def main():
    tweets = load_tweets()
    if not tweets:
        raise SystemExit("No tweets loaded.")
    embeddings = embed_texts([t.text for t in tweets])
    labels = cluster_embeddings(embeddings)

    clusters: Dict[int, List[TweetRow]] = {}
    for label, tw in zip(labels, tweets):
        if label == -1:
            continue  # noise
        clusters.setdefault(label, []).append(tw)

    outputs: List[Dict[str, object]] = []
    for cid, tws in clusters.items():
        sorted_tws = sorted(tws, key=lambda t: t.popularity, reverse=True)[:MAX_TWEETS]
        record = build_prompt(cid, sorted_tws)
        topic = maybe_call_openai(record["prompt"])
        outputs.append({"cluster_id": record["cluster_id"], "topic": topic or ""})

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(outputs)} cluster prompts to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
