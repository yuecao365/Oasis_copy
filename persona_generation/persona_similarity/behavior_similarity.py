"""
Pairwise similarity from persona action distributions (p_create_post, p_repost, …).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .io import action_probs_from_row, CSV_ACTION_KEYS


def _ordered_action_vector(probs: Mapping[str, float]) -> Tuple[List[str], List[float]]:
    keys = [a for _, a in CSV_ACTION_KEYS]
    vec = [float(probs.get(k, 0.0)) for k in keys]
    return keys, vec


def cosine_similarity_actions(
    row_a: Mapping[str, Any],
    row_b: Mapping[str, Any],
) -> float:
    """Cosine similarity between two personas' action probability vectors (same order as OASIS CSV)."""
    _, va = _ordered_action_vector(action_probs_from_row(row_a))
    _, vb = _ordered_action_vector(action_probs_from_row(row_b))
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def pairwise_action_cosine_matrix(rows: Sequence[Mapping[str, Any]]) -> List[List[float]]:
    """Full n×n matrix of action-vector cosines (diagonal 1.0 when vectors non-zero)."""
    n = len(rows)
    mat: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = cosine_similarity_actions(rows[i], rows[j])
            mat[i][j] = s
            mat[j][i] = s
    return mat
