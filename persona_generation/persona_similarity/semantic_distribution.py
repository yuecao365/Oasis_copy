"""
Semantic overlap between a real-world text corpus (e.g. train tweets) and another
corpus (e.g. simulated posts or persona narrative+tweet snippets).

Uses the same embedding family as simulation/compare_sim_to_test.py when available.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

DEFAULT_MODEL = "paraphrase-MiniLM-L6-v2"


def _encode_normalized(texts: List[str], model_name: str):
    if np is None:
        return None
    if not _HAS_ST or not texts:
        return None
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def _mean_pairwise_cosine_sample(emb, max_pairs: int = 5000, seed: int = 42) -> Optional[float]:
    """Unbiased estimate of mean cosine similarity over random pairs (upper triangle), excluding diagonal."""
    n = emb.shape[0]
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    pairs = min(max_pairs, n * (n - 1) // 2)
    sims: List[float] = []
    for _ in range(pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sims.append(float(np.dot(emb[i], emb[j])))
    return float(np.mean(sims)) if sims else None


def semantic_distribution_report(
    real_texts: List[str],
    sim_texts: List[str],
    *,
    model_name: str = DEFAULT_MODEL,
    max_intra_pairs: int = 8000,
) -> Dict:
    """
    Compare semantic *distribution* of two text corpora.

    Returns centroid / max-match overlap (sim vs real) plus within-corpus cohesion
    (mean random-pair cosine): tighter real cohesion with divergent sim cohesion
    often indicates different generative behavior even when centroids align.
    """
    if np is None:
        return {"error": "numpy not installed"}
    if not _HAS_ST:
        return {"error": "sentence-transformers not installed"}
    if not real_texts or not sim_texts:
        return {"error": "empty text corpus"}

    gt_emb = _encode_normalized(real_texts, model_name)
    sim_emb = _encode_normalized(sim_texts, model_name)
    if gt_emb is None or sim_emb is None:
        return {"error": "encoding failed"}

    gt_centroid = gt_emb.mean(axis=0)
    sim_centroid = sim_emb.mean(axis=0)
    gc = np.linalg.norm(gt_centroid)
    sc = np.linalg.norm(sim_centroid)
    centroid_sim = float(np.dot(gt_centroid, sim_centroid) / (gc * sc + 1e-9))

    chunk = 512
    max_sims: List[float] = []
    for i in range(0, len(sim_emb), chunk):
        block = sim_emb[i : i + chunk]
        max_sims.extend((block @ gt_emb.T).max(axis=1).tolist())
    mean_max_match = float(np.mean(max_sims))

    real_cohesion = _mean_pairwise_cosine_sample(gt_emb, max_pairs=max_intra_pairs)
    sim_cohesion = _mean_pairwise_cosine_sample(sim_emb, max_pairs=max_intra_pairs)

    gt_norms = np.linalg.norm(gt_emb, axis=1)
    sm_norms = np.linalg.norm(sim_emb, axis=1)

    return {
        "model": model_name,
        "real_post_count": len(real_texts),
        "sim_post_count": len(sim_texts),
        "sim_to_real_centroid_cosine": round(centroid_sim, 4),
        "sim_to_real_mean_max_match": round(mean_max_match, 4),
        "real_mean_random_pair_cosine": round(real_cohesion, 4) if real_cohesion is not None else None,
        "sim_mean_random_pair_cosine": round(sim_cohesion, 4) if sim_cohesion is not None else None,
        "real_embedding_l2_norm_mean": round(float(np.mean(gt_norms)), 4),
        "sim_embedding_l2_norm_mean": round(float(np.mean(sm_norms)), 4),
    }
