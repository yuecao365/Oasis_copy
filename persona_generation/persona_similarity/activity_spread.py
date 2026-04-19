"""
How concentrated activity is among top-ranked users — "spread depth" of influence/volume.

Useful when comparing real train activity vs. a simulated population that may be
more or less top-heavy.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _normalized_shares(counts: Sequence[int]) -> List[float]:
    total = float(sum(counts))
    if total <= 0:
        return []
    return [c / total for c in counts]


def herfindahl_hirschman(shares: Sequence[float]) -> float:
    return float(sum(s * s for s in shares))


def effective_users_from_counts(counts: Sequence[int]) -> Optional[float]:
    """Inverse HHI in user space (1 / sum p_i^2); None if empty."""
    shares = _normalized_shares(counts)
    if not shares:
        return None
    hhi = herfindahl_hirschman(shares)
    if hhi <= 0:
        return None
    return 1.0 / hhi


def gini_coefficient(values: Sequence[float]) -> Optional[float]:
    """Gini in [0,1]; 0 = perfectly equal, 1 = maximally unequal."""
    x = sorted(float(v) for v in values if v >= 0)
    n = len(x)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    mean = sum(x) / n
    if mean <= 0:
        return None
    cum = 0.0
    for i, v in enumerate(x, start=1):
        cum += (2 * i - n - 1) * v
    return cum / (n * n * mean)


def top_k_volume_share(counts: Mapping[str, int], top_k: int) -> Dict[str, Optional[float]]:
    """
    Among all users, what fraction of total tweets sits in the top-k by count?
    """
    if top_k <= 0:
        return {"top_k": top_k, "volume_share_top_k": None, "user_count": 0}
    items = sorted(((uid, c) for uid, c in counts.items() if c > 0), key=lambda t: -t[1])
    total = sum(c for _, c in items)
    if total <= 0:
        return {"top_k": top_k, "volume_share_top_k": None, "user_count": len(items)}
    head = items[: min(top_k, len(items))]
    head_sum = sum(c for _, c in head)
    return {
        "top_k": top_k,
        "volume_share_top_k": round(head_sum / total, 4),
        "user_count": len(items),
        "top_k_actual": len(head),
    }


def activity_spread_depth(
    user_tweet_counts: Mapping[str, int],
    *,
    top_k: Optional[int] = 10,
) -> Dict:
    """
    Summarize how "spread out" activity is across users.

    - effective_users: diversity of mass (higher = deeper spread)
    - normalized_entropy: Shannon entropy of the share vector / log(n)
    - gini_tweet_counts: inequality of per-user tweet totals
    - optional top_k_volume_share: mass captured by busiest k users
    """
    counts = [c for c in user_tweet_counts.values() if c > 0]
    n = len(counts)
    if n == 0:
        return {"error": "no positive counts"}

    shares = _normalized_shares(counts)
    hhi = herfindahl_hirschman(shares)
    eff = 1.0 / hhi if hhi > 0 else None

    ent = 0.0
    for s in shares:
        if s > 0:
            ent -= s * math.log(s)
    max_ent = math.log(n) if n > 1 else 1.0
    norm_ent = ent / max_ent if max_ent > 0 else 0.0

    g = gini_coefficient([float(c) for c in counts])

    out: Dict = {
        "user_count": n,
        "total_tweets": int(sum(counts)),
        "herfindahl_users": round(hhi, 6),
        "effective_number_of_users": round(eff, 4) if eff is not None else None,
        "normalized_entropy_shares": round(norm_ent, 4),
        "gini_tweet_counts": round(g, 4) if g is not None else None,
    }
    if top_k is not None:
        out["top_k_concentration"] = top_k_volume_share(user_tweet_counts, top_k)
    return out


def compare_activity_spread(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
    *,
    top_k: int = 10,
) -> Dict:
    """Side-by-side spread metrics for two populations (e.g. real vs simulated)."""
    ra = activity_spread_depth(counts_a, top_k=top_k)
    rb = activity_spread_depth(counts_b, top_k=top_k)
    delta: Dict[str, Optional[float]] = {}
    if "error" not in ra and "error" not in rb:
        for key in ("effective_number_of_users", "normalized_entropy_shares", "gini_tweet_counts"):
            va, vb = ra.get(key), rb.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta[f"delta_{key}"] = round(float(vb) - float(va), 4)
            else:
                delta[f"delta_{key}"] = None
    return {"real": ra, "sim": rb, "delta_sim_minus_real": delta}
