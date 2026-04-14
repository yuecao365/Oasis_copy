"""
Compare repost-related behavior: persona-level action probabilities and optional
real retweet rates from train CSV vs. another corpus or simulation export.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .io import action_probs_from_row, iter_train_rows


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _histogram_01(values: Sequence[float], bins: int = 20) -> Tuple[List[float], List[int]]:
    """Fixed [0,1] bins for probability masses."""
    edges = [i / bins for i in range(bins + 1)]
    hist = [0] * bins
    for v in values:
        if v is None or math.isnan(v):
            continue
        v = max(0.0, min(1.0, float(v)))
        idx = min(int(v * bins), bins - 1)
        hist[idx] += 1
    total = sum(hist)
    dens = [c / total for c in hist] if total > 0 else [0.0] * bins
    return dens, hist


def _kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    s = 0.0
    for a, b in zip(p, q):
        aa = max(a, eps)
        bb = max(b, eps)
        s += aa * math.log(aa / bb)
    return s


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    m = [(a + b) * 0.5 for a, b in zip(p, q)]
    return 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))


def summarize_repost_propensity(persona_rows: Sequence[Mapping[str, Any]]) -> Dict:
    """Stats on CSV p_repost across personas."""
    vals: List[float] = []
    for row in persona_rows:
        probs = action_probs_from_row(row)
        r = probs.get("repost")
        if r is not None:
            vals.append(r)
    if not vals:
        return {"count": 0, "error": "no p_repost values"}
    dens, hist = _histogram_01(vals)
    return {
        "count": len(vals),
        "mean_p_repost": round(sum(vals) / len(vals), 6),
        "std_p_repost": round(
            (sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / len(vals)) ** 0.5,
            6,
        ),
        "hist_bins_20": hist,
        "hist_density_20": [round(x, 6) for x in dens],
    }


def compare_persona_repost_distributions(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    bins: int = 20,
) -> Dict:
    """
    Jensen–Shannon divergence between binned p_repost distributions of two persona tables.
    """
    def collect(rows: Sequence[Mapping[str, Any]]) -> List[float]:
        out: List[float] = []
        for row in rows:
            p = action_probs_from_row(row).get("repost")
            if p is not None:
                out.append(p)
        return out

    a = collect(rows_a)
    b = collect(rows_b)
    if not a or not b:
        return {"error": "need non-empty p_repost in both sets"}

    da, _ = _histogram_01(a, bins=bins)
    db, _ = _histogram_01(b, bins=bins)
    js = jensen_shannon_divergence(da, db)
    return {
        "bins": bins,
        "jensen_shannon_p_repost": round(js, 6),
        "summary_a": summarize_repost_propensity(rows_a),
        "summary_b": summarize_repost_propensity(rows_b),
    }


def _parse_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def weekly_retweet_dynamics_from_train_csv(
    path: Any,
    *,
    timestamp_key: str = "timestamp",
    retweets_key: str = "retweets",
) -> Dict:
    """
    Aggregate mean retweets-per-tweet by ISO calendar week (UTC).

    Compare two runs (e.g. real train vs. simulated export with the same columns)
    using L2 distance on aligned week keys in your own code, or extend this module.
    """
    from pathlib import Path

    weekly_sum: Dict[Tuple[int, int], float] = defaultdict(float)
    weekly_cnt: Dict[Tuple[int, int], int] = defaultdict(int)

    for row in iter_train_rows(Path(path)):
        ts = _parse_ts(row.get(timestamp_key) or "")
        if ts is None:
            continue
        iso = ts.isocalendar()
        key = (int(iso.year), int(iso.week))
        r = _safe_float(row.get(retweets_key)) or 0.0
        weekly_sum[key] += r
        weekly_cnt[key] += 1

    weeks = sorted(weekly_cnt.keys())
    series = []
    for w in weeks:
        c = weekly_cnt[w]
        series.append(
            {
                "year": w[0],
                "iso_week": w[1],
                "tweet_count": c,
                "mean_retweets_per_tweet": round(weekly_sum[w] / c, 6) if c else 0.0,
            }
        )
    return {"weeks": len(series), "series": series}


def compare_weekly_retweet_series(series_a: Dict, series_b: Dict) -> Dict:
    """
    L2 distance between mean-retweet curves on the intersection of (year, week) keys.
    """
    def to_map(obj: Dict) -> Dict[Tuple[int, int], float]:
        m: Dict[Tuple[int, int], float] = {}
        for item in obj.get("series") or []:
            y = item.get("year")
            w = item.get("iso_week")
            v = item.get("mean_retweets_per_tweet")
            if y is not None and w is not None and v is not None:
                m[(int(y), int(w))] = float(v)
        return m

    ma, mb = to_map(series_a), to_map(series_b)
    keys = sorted(set(ma) & set(mb))
    if len(keys) < 2:
        return {"error": "insufficient overlapping weeks", "overlap": len(keys)}

    va = [ma[k] for k in keys]
    vb = [mb[k] for k in keys]
    diff = sum((a - b) ** 2 for a, b in zip(va, vb)) ** 0.5
    mean_a = sum(va) / len(va)
    mean_b = sum(vb) / len(vb)
    return {
        "overlapping_weeks": len(keys),
        "l2_mean_retweets_curve": round(diff, 6),
        "mean_level_a": round(mean_a, 6),
        "mean_level_b": round(mean_b, 6),
        "delta_mean_level_b_minus_a": round(mean_b - mean_a, 6),
    }
