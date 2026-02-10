from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

INPUT_SOURCE = "Clustering_and_topic_generation/tweet_popularity_scores.csv"


def extract_text(source: str, text_col: str = "clean_text") -> List[str]:
    texts: List[str] = []
    with open(source, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found in {source}.")
        for row in reader:
            raw = row.get(text_col, "")
            if not raw:
                continue
            # Remove surrounding quotes if present: "..." or “...”
            cleaned = raw.strip().strip('"\u201c\u201d').strip()
            if cleaned:
                texts.append(cleaned)
    return texts


def extract_text_and_popularity(
    source: str,
    text_col: str = "clean_text",
    score_col: str = "popularity_score",
    root_only: bool = True,
    root_col: str = "is_root",
) -> Tuple[List[str], List[float]]:
    texts: List[str] = []
    scores: List[float] = []
    with open(source, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_cols = [c for c in (text_col, score_col) if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} in {source}.")
        if root_only and root_col not in reader.fieldnames:
            raise ValueError(f"Missing column '{root_col}' in {source}.")
        for row in reader:
            if root_only:
                is_root = str(row.get(root_col, "0")).strip()
                if is_root != "1":
                    continue
            raw = row.get(text_col, "")
            if not raw:
                continue
            cleaned = raw.strip().strip('"\u201c\u201d').strip()
            if not cleaned:
                continue
            score_raw = row.get(score_col, "")
            try:
                score = float(score_raw)
            except Exception:
                continue
            texts.append(cleaned)
            scores.append(score)
    return texts, scores


def embed_texts(
    texts: Iterable[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), normalize_embeddings=True)


def clustering(
    texts: Iterable[str],
    method: str = "kmeans",
    num_clusters: int = 10,
    min_cluster_size: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    texts_list = list(texts)
    if not texts_list:
        return np.array([]), np.array([])

    embeddings = embed_texts(texts_list, model_name=model_name)
    method = method.lower()

    if method in {"kmeans", "k-means"}:
        n_clusters = min(num_clusters, len(texts_list))
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels = kmeans.fit_predict(embeddings)
        return labels, embeddings

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency not guaranteed
            raise ImportError("hdbscan is not installed.") from exc
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)
        return labels, embeddings

    raise ValueError("method must be 'kmeans' or 'hdbscan'.")


def project_embeddings(
    embeddings: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
) -> np.ndarray:
    if embeddings.size == 0:
        return np.empty((0, 2))
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(embeddings)
    raise ValueError("method must be 'pca'.")


def visualize_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    method: str = "pca",
    random_state: int = 42,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency not guaranteed
        raise ImportError("matplotlib is not installed.") from exc

    coords = project_embeddings(embeddings, method=method, random_state=random_state)
    if coords.size == 0:
        raise ValueError("No embeddings to visualize.")

    clusters = sorted(set(int(x) for x in labels))
    counts = Counter(int(x) for x in labels)
    cmap = plt.get_cmap("tab20", max(len(clusters), 1))
    palette = [cmap(i) for i in range(len(clusters))]
    color_map = {cluster_id: palette[i] for i, cluster_id in enumerate(clusters)}
    point_colors = [color_map[int(x)] for x in labels]

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=10, alpha=0.8)
    plt.title(f"Clustering Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_map[cluster_id],
            markersize=6,
            label=f"Cluster {cluster_id} (n={counts[cluster_id]})",
        )
        for cluster_id in clusters
    ]
    plt.legend(
        handles=handles,
        title="Clusters",
        loc="best",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def visualize_cluster_popularity(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency not guaranteed
        raise ImportError("matplotlib is not installed.") from exc

    if labels.size == 0 or scores.size == 0:
        raise ValueError("labels and scores must be non-empty.")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length.")

    clusters = sorted(set(int(x) for x in labels))
    data = [scores[labels == c] for c in clusters]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=[str(c) for c in clusters], showfliers=False)
    plt.title("Popularity Score by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Popularity Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_projection_all(
    coords: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency not guaranteed
        raise ImportError("matplotlib is not installed.") from exc

    clusters = sorted(set(int(x) for x in labels))
    counts = Counter(int(x) for x in labels)
    cmap = plt.get_cmap("tab20", max(len(clusters), 1))
    palette = [cmap(i) for i in range(len(clusters))]
    color_map = {cluster_id: palette[i] for i, cluster_id in enumerate(clusters)}
    point_colors = [color_map[int(x)] for x in labels]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=10,
        alpha=0.8,
    )
    plt.title("Cluster Projection (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_map[cluster_id],
            markersize=6,
            label=f"Cluster {cluster_id} (n={counts[cluster_id]})",
        )
        for i, cluster_id in enumerate(clusters)
    ]
    plt.legend(
        handles=handles,
        title="Clusters",
        loc="best",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_cluster_counts(labels: np.ndarray, out_path: Path) -> None:
    counts = Counter(int(x) for x in labels)
    rows = [("cluster_id", "tweet_count")]
    for cluster_id in sorted(counts.keys()):
        rows.append((str(cluster_id), str(counts[cluster_id])))
    out_path.write_text("\n".join([",".join(r) for r in rows]), encoding="utf-8")


def _save_cluster_popularity_hist(
    scores: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency not guaranteed
        raise ImportError("matplotlib is not installed.") from exc

    data = scores[labels == cluster_id]
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=20, color="#1f77b4", alpha=0.85)
    plt.title(f"Cluster {cluster_id} Popularity Distribution")
    plt.xlabel("Popularity Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_top_tweets(
    texts: List[str],
    scores: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    out_path: Path,
    max_representative_tweet: int,
    min_popularity_score: float,
) -> None:
    idxs = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
    ranked = sorted(
        (i for i in idxs if scores[i] >= min_popularity_score),
        key=lambda i: scores[i],
        reverse=True,
    )
    top = ranked[:max_representative_tweet]
    lines = [f"{scores[i]:.6f}\t{texts[i]}" for i in top]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_clustering_experiments(
    source: str = INPUT_SOURCE,
    text_col: str = "clean_text",
    score_col: str = "popularity_score",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    kmeans_clusters: Iterable[int] = (3, 4, 5),
    hdbscan_min_cluster_sizes: Iterable[int] = (300, 600, 900),
    max_representative_tweet: int = 5,
    min_popularity_score: float = 0.0,
    out_dir: str = "Clustering_and_topic_generation/cluster_outputs",
    random_state: int = 42,
    root_only: bool = True,
    skip_existing: bool = True,
) -> None:
    texts, scores_list = extract_text_and_popularity(
        source, text_col=text_col, score_col=score_col, root_only=root_only
    )
    scores = np.array(scores_list)
    embeddings = embed_texts(texts, model_name=model_name)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for k in kmeans_clusters:
        runs.append(("kmeans", {"num_clusters": int(k)}))
    for m in hdbscan_min_cluster_sizes:
        runs.append(("hdbscan", {"min_cluster_size": int(m)}))

    for method, params in runs:
        labels, _ = clustering(
            texts,
            method=method,
            num_clusters=params.get("num_clusters", 10),
            min_cluster_size=params.get("min_cluster_size", 5),
            model_name=model_name,
            random_state=random_state,
        )
        if labels.size == 0:
            continue

        run_name = (
            f"kmeans_k{params['num_clusters']}"
            if method == "kmeans"
            else f"hdbscan_min{params['min_cluster_size']}"
        )
        run_dir = out_root / run_name
        if skip_existing and run_dir.exists():
            if (run_dir / "clusters_projection.png").exists() or (
                run_dir / "cluster_counts.csv"
            ).exists():
                continue
        run_dir.mkdir(parents=True, exist_ok=True)

        coords = project_embeddings(
            embeddings, method="pca", random_state=random_state
        )

        clusters = sorted(set(int(x) for x in labels))
        _save_projection_all(coords, labels, run_dir / "clusters_projection.png")
        _save_cluster_counts(labels, run_dir / "cluster_counts.csv")
        for cluster_id in clusters:
            if cluster_id == -1:
                continue
            _save_cluster_popularity_hist(
                scores, labels, cluster_id, run_dir / f"cluster_{cluster_id}_pop.png"
            )
            _save_top_tweets(
                texts,
                scores,
                labels,
                cluster_id,
                run_dir / f"cluster_{cluster_id}_top{max_representative_tweet}.txt",
                max_representative_tweet,
                min_popularity_score,
            )


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run clustering experiments and save projections/popularity/top tweets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (overrides defaults, CLI overrides config).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=INPUT_SOURCE,
        help="Input CSV with clean_text and popularity_score.",
    )
    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--score-col", type=str, default="popularity_score")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--kmeans-clusters", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument(
        "--hdbscan-min-cluster-sizes", type=int, nargs="+", default=[300, 600, 900]
    )
    parser.add_argument("--max-representative-tweet", type=int, default=5)
    parser.add_argument(
        "--min-popularity-score",
        type=float,
        default=0.0,
        help="Minimum popularity_score for a tweet to be selected as representative.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Clustering_and_topic_generation/cluster_outputs",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Do not skip runs that already have outputs.",
    )
    parser.add_argument(
        "--all-tweets",
        action="store_true",
        help="Use all tweets instead of only root tweets.",
    )

    # Load YAML config first (if provided) and set defaults.
    config_path = None
    if "--config" in sys.argv:
        try:
            config_path = sys.argv[sys.argv.index("--config") + 1]
        except Exception:
            config_path = None
    root_only_cfg = True
    if config_path:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency not guaranteed
            raise ImportError("pyyaml is not installed.") from exc
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "root_only" in cfg:
            root_only_cfg = bool(cfg.pop("root_only"))
        # Map config keys to argparse names
        if "input_source" in cfg and "source" not in cfg:
            cfg["source"] = cfg["input_source"]
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    root_only = root_only_cfg and not args.all_tweets
    run_clustering_experiments(
        source=args.source,
        text_col=args.text_col,
        score_col=args.score_col,
        model_name=args.model_name,
        kmeans_clusters=args.kmeans_clusters,
        hdbscan_min_cluster_sizes=args.hdbscan_min_cluster_sizes,
        max_representative_tweet=args.max_representative_tweet,
        min_popularity_score=args.min_popularity_score,
        out_dir=args.out_dir,
        random_state=args.random_state,
        root_only=root_only,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
