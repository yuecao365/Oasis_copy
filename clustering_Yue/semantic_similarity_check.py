from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def load_topics(topics_path: Path) -> List[str]:
    lines = [
        line.strip()
        for line in topics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines


def compute_similar_pairs(
    lines: List[str], threshold: float
) -> List[Tuple[float, int, int]]:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeds = model.encode(lines, normalize_embeddings=True)
    sims = np.matmul(embeds, embeds.T)

    pairs: List[Tuple[float, int, int]] = []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sims[i, j])
            if sim >= threshold:
                pairs.append((sim, i, j))
    pairs.sort(reverse=True)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check semantic similarity among topic lines using all-MiniLM-L6-v2."
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Path to cluster_topics.txt (single file mode).",
    )
    parser.add_argument(
        "--cluster-root",
        type=str,
        default="Clustering_and_topic_generation/cluster_outputs",
        help="Root directory containing kmeans_k* folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Clustering_and_topic_generation/semantic_similarity_report.txt",
        help="Output report path for all kmeans runs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold.",
    )
    args = parser.parse_args()

    # Single file mode if --topics is provided.
    if args.topics:
        topics_path = Path(args.topics)
        if not topics_path.exists():
            raise SystemExit(f"Topics file not found: {topics_path}")
        lines = load_topics(topics_path)
        if not lines:
            raise SystemExit("No topic lines found.")
        pairs = compute_similar_pairs(lines, args.threshold)
        print(f"Total topics: {len(lines)}")
        if not pairs:
            print(f"No pairs >= {args.threshold:.2f}")
            return
        for sim, i, j in pairs:
            print(f"sim={sim:.3f} | {i+1}: {lines[i]} || {j+1}: {lines[j]}")
        return

    # Batch mode: scan all kmeans_k* folders and write report.
    root = Path(args.cluster_root)
    run_dirs = sorted(root.glob("kmeans_k*"))
    if not run_dirs:
        raise SystemExit(f"No kmeans_k* folders found under {root}")

    report_lines: List[str] = []
    for d in run_dirs:
        topics_path = d / "cluster_topics.txt"
        if not topics_path.exists():
            continue
        lines = load_topics(topics_path)
        if not lines:
            continue
        pairs = compute_similar_pairs(lines, args.threshold)
        report_lines.append(f"{d.name}: {len(lines)} topics")
        if not pairs:
            report_lines.append(f"  No pairs >= {args.threshold:.2f}")
            continue
        for sim, i, j in pairs:
            report_lines.append(f"  sim={sim:.3f}")
            report_lines.append(f"    {i+1}: {lines[i]}")
            report_lines.append(f"    {j+1}: {lines[j]}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
