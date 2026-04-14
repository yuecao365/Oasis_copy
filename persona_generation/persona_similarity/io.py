"""
Load personas and tweet corpora for similarity metrics.

CSV columns follow OASIS export from generate_personas (p_* action probabilities).
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

# CSV header names used by save_personas → Twitter/OASIS format
CSV_ACTION_KEYS = [
    ("p_create_post", "create_post"),
    ("p_create_comment", "create_comment"),
    ("p_do_nothing", "do_nothing"),
    ("p_repost", "repost"),
    ("p_quote_post", "quote_post"),
    ("p_like_post", "like_post"),
    ("p_follow", "follow"),
]


def _parse_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_personas_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load persona rows from CSV as dicts (utf-8)."""
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_personas_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSON list of persona dicts (e.g. user_personas.json)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def action_probs_from_row(row: Dict[str, Any]) -> Dict[str, float]:
    """Map CSV row to canonical action name → probability."""
    out: Dict[str, float] = {}
    for csv_key, action in CSV_ACTION_KEYS:
        v = _parse_float(row.get(csv_key))
        if v is not None:
            out[action] = v
    return out


def persona_text_bundle(row: Dict[str, Any]) -> List[str]:
    """Collect narrative fields used for lightweight / embedding comparisons."""
    texts: List[str] = []
    for key in ("user_char", "description", "category_description"):
        t = (row.get(key) or "").strip()
        if t:
            texts.append(t)
    raw_prev = row.get("previous_tweets")
    if raw_prev:
        try:
            prev = ast.literal_eval(raw_prev) if isinstance(raw_prev, str) else raw_prev
            if isinstance(prev, list):
                for t in prev:
                    s = str(t).strip()
                    if s:
                        texts.append(s)
        except (ValueError, SyntaxError):
            pass
    return texts


def corpus_from_persona_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Flatten all persona-derived texts into one list (one string per line of evidence)."""
    out: List[str] = []
    for row in rows:
        out.extend(persona_text_bundle(row))
    return [t for t in out if t]


def load_train_tweet_texts(
    path: Union[str, Path],
    *,
    max_rows: Optional[int] = None,
    text_key: str = "text",
) -> List[str]:
    """Load tweet bodies from train_tweets.csv (or similar)."""
    path = Path(path)
    texts: List[str] = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            t = (row.get(text_key) or "").strip()
            if t:
                texts.append(t)
    return texts


def per_user_tweet_counts_from_train_csv(
    path: Union[str, Path],
    user_key: str = "user_id",
) -> Dict[str, int]:
    """Count tweets per user_id in a train export CSV."""
    path = Path(path)
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get(user_key) or "").strip()
            if not uid:
                continue
            counts[uid] = counts.get(uid, 0) + 1
    return counts


def iter_train_rows(path: Union[str, Path]) -> Iterable[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)
