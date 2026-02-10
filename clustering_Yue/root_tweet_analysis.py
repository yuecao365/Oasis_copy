from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

DATA_SOURCE = "Clustering_and_topic_generation/data/filtered_tweets.jsonl"
DEFAULT_SCORE_CSV = "Clustering_and_topic_generation/data/tweet_popularity_scores.csv"
DEFAULT_PLOT_DIR = "Clustering_and_topic_generation/popularity_score_analysis"


def load_root_tweet_ids(source: str) -> Set[str]:
    root_ids: Set[str] = set()
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tweet = json.loads(line)
            except Exception:
                continue
            tweet_id = str(tweet.get("id", "")).strip()
            conversation_id = str(tweet.get("conversationId", "")).strip()
            if tweet_id and conversation_id and tweet_id == conversation_id:
                root_ids.add(tweet_id)
    return root_ids


def add_is_root_column(
    csv_path: Path, root_ids: Set[str], id_col: str = "tweet_id"
) -> None:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {csv_path}")
        if id_col not in reader.fieldnames:
            raise ValueError(f"Column '{id_col}' not found in {csv_path}")
        fieldnames = list(reader.fieldnames)
        if "is_root" not in fieldnames:
            fieldnames.append("is_root")

        tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                tweet_id = (row.get(id_col) or "").strip()
                row["is_root"] = "1" if tweet_id in root_ids else "0"
                writer.writerow(row)
    tmp_path.replace(csv_path)


def load_popularity_scores(
    csv_path: Path, root_ids: Set[str], id_col: str = "tweet_id"
) -> tuple[List[float], List[float]]:
    all_scores: List[float] = []
    root_scores: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or id_col not in reader.fieldnames:
            raise ValueError(f"Column '{id_col}' not found in {csv_path}")
        if "popularity_score" not in reader.fieldnames:
            raise ValueError(f"Column 'popularity_score' not found in {csv_path}")
        has_is_root = "is_root" in reader.fieldnames
        for row in reader:
            score_raw = row.get("popularity_score", "")
            try:
                score = float(score_raw)
            except Exception:
                continue
            all_scores.append(score)
            if has_is_root:
                is_root = str(row.get("is_root", "0")).strip()
                if is_root == "1":
                    root_scores.append(score)
            else:
                tweet_id = (row.get(id_col) or "").strip()
                if tweet_id in root_ids:
                    root_scores.append(score)
    return all_scores, root_scores


def plot_popularity_density_overlay(
    all_scores: List[float],
    root_scores: List[float],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency not guaranteed
        raise ImportError("matplotlib is not installed.") from exc

    if not all_scores:
        raise ValueError("No popularity scores found for all tweets.")
    if not root_scores:
        raise ValueError("No popularity scores found for root tweets.")

    plt.figure(figsize=(9, 5))
    bins = 40
    plt.hist(
        all_scores,
        bins=bins,
        alpha=0.6,
        color="#4C78A8",
        label="All tweets",
        density=True,
    )
    plt.hist(
        root_scores,
        bins=bins,
        alpha=0.6,
        color="#F58518",
        label="Root tweets",
        density=True,
    )
    plt.title("Popularity Score Distribution: Root vs All Tweets")
    plt.xlabel("Popularity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_popularity_count_hist(
    scores: List[float],
    out_path: Path,
    title: str,
    color: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency not guaranteed
        raise ImportError("matplotlib is not installed.") from exc

    if not scores:
        raise ValueError("No popularity scores found for plotting.")

    total = len(scores)
    plt.figure(figsize=(9, 5))
    bins = 40
    plt.hist(scores, bins=bins, alpha=0.85, color=color, density=False)
    plt.title(f"{title} (Total: {total})")
    plt.xlabel("Popularity Score")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count root tweets and tag tweet popularity CSV with is_root."
    )
    parser.add_argument(
        "--tweets",
        type=str,
        default=DATA_SOURCE,
        help="Path to filtered_tweets.jsonl",
    )
    parser.add_argument(
        "--scores-csv",
        type=str,
        default=DEFAULT_SCORE_CSV,
        help="CSV to add is_root column.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=DEFAULT_PLOT_DIR,
        help="Output directory for popularity distribution plots.",
    )
    args = parser.parse_args()

    root_ids = load_root_tweet_ids(args.tweets)
    print(f"root_tweet count: {len(root_ids)}")

    csv_path = Path(args.scores_csv)
    if csv_path.exists():
        add_is_root_column(csv_path, root_ids)
        print(f"Updated is_root column in {csv_path}")

        all_scores, root_scores = load_popularity_scores(csv_path, root_ids)
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        density_path = plot_dir / "root_vs_all_density.png"
        plot_popularity_density_overlay(all_scores, root_scores, density_path)

        all_count_path = plot_dir / "all_tweets_count.png"
        plot_popularity_count_hist(
            all_scores, all_count_path, "Popularity Score Count: All Tweets", "#4C78A8"
        )

        root_count_path = plot_dir / "root_tweets_count.png"
        plot_popularity_count_hist(
            root_scores,
            root_count_path,
            "Popularity Score Count: Root Tweets",
            "#F58518",
        )

        print(f"Wrote density plot to {density_path}")
        print(f"Wrote count plot to {all_count_path}")
        print(f"Wrote count plot to {root_count_path}")
    else:
        print(f"CSV not found, skipped: {csv_path}")


if __name__ == "__main__":
    main()
