import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path so llm_client is importable.
_REPO_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_DIR))

from llm_client import chat_completion, LLM_MODEL

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLUSTER_ROOT = (
    REPO_ROOT / "Clustering_and_topic_generation" / "cluster_outputs"
)
MAX_TWEETS = 5


@dataclass
class TweetRow:
    text: str
    popularity: float


CLUSTER_FILE_RE = re.compile(r"cluster_(\d+)_top(\d+)\.txt$")


def load_cluster_tweets(cluster_dir: Path) -> Dict[int, List[TweetRow]]:
    clusters: Dict[int, List[TweetRow]] = {}
    files = sorted(cluster_dir.glob("cluster_*_top*.txt"))
    for path in files:
        match = CLUSTER_FILE_RE.search(path.name)
        if not match:
            continue
        cluster_id = int(match.group(1))
        rows: List[TweetRow] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            score, _, text = line.partition("\t")
            if not text:
                # Fallback: first whitespace split
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    score, text = parts
                else:
                    continue
            try:
                popularity = float(score)
            except Exception:
                continue
            rows.append(TweetRow(text=text, popularity=popularity))
        if rows:
            clusters[cluster_id] = rows
    return clusters


def build_prompt(cluster_id: int, tweets: List[TweetRow]) -> Dict[str, object]:
    header = (
        "Given the following tweets from the same discussion, generate a concise topic label and a short description.\n"
        "Generate a topic label (3–6 words) that summarizes their shared core idea, then add a brief description.\n\n"
        "Requirements:\n"
        "- Label must be a noun or noun phrase\n"
        "- Label must be 3–6 words\n"
        "- Description should be 1–2 sentences, neutral, no opinions\n"
        "- Capture the shared core topic\n"
        "- Output format: \"Label — Description\"\n\n"
        "Tweets:"
    )
    body_lines = [f"{idx}. {tw.text}" for idx, tw in enumerate(tweets, 1)]
    prompt = header + "\n" + "\n".join(body_lines)
    return {
        "cluster_id": int(cluster_id),
        "prompt": prompt,
        "top_tweets": [
            {"popularity": tw.popularity, "text": tw.text} for tw in tweets
        ],
    }


def maybe_call_openai(prompt: str, model: str = LLM_MODEL) -> str:
    try:
        return chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a topic labeling assistant. Create a 3–6 word noun-phrase label "
                        "and a brief neutral description (1–2 sentences). "
                        "Output exactly as: Label — Description. No extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception:
        return ""


def _write_topics(topics: List[str], out_path: Path) -> None:
    out_path.write_text("\n".join(topics), encoding="utf-8")


def _process_cluster_dir(
    cluster_dir: Path,
    output_path: Path,
    model: str,
    max_tweets: int,
    use_openai: bool,
) -> int:
    clusters = load_cluster_tweets(cluster_dir)
    if not clusters:
        return 0

    topics: List[str] = []
    for cid, tws in clusters.items():
        sorted_tws = sorted(tws, key=lambda t: t.popularity, reverse=True)[:max_tweets]
        record = build_prompt(cid, sorted_tws)
        topic = (
            maybe_call_openai(record["prompt"], model) if use_openai else ""
        )
        if topic:
            topics.append(topic)
    _write_topics(topics, output_path)
    return len(topics)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate topic labels from cluster representative tweets."
    )
    parser.add_argument(
        "--cluster-root",
        type=Path,
        default=DEFAULT_CLUSTER_ROOT,
        help="Root directory containing cluster output subdirectories.",
    )
    parser.add_argument("--openai-model", type=str, default=OPENAI_MODEL)
    parser.add_argument("--max-tweets", type=int, default=MAX_TWEETS)
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Call OpenAI to generate topic labels.",
    )
    args = parser.parse_args()

    root = args.cluster_root
    if not root.exists():
        raise SystemExit(f"Cluster root not found: {root}")

    # Process only KMeans runs (k*).
    target_dirs = sorted(root.glob("kmeans_k*"))

    if not target_dirs:
        raise SystemExit("No matching cluster directories found.")

    total = 0
    for d in target_dirs:
        out_path = d / "cluster_topics.txt"
        count = _process_cluster_dir(
            d, out_path, args.openai_model, args.max_tweets, args.use_openai
        )
        total += count
        print(f"Wrote {count} topics to {out_path}")
    print(f"Total topics written: {total}")


if __name__ == "__main__":
    main()
