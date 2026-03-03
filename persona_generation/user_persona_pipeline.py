#!/usr/bin/env python3
"""
Main pipeline to:
1. Load filtered_tweets.jsonl and split into train/test sets (time-based)
2. Load clustering output (cluster_assignments, cluster_topics)
3. Rank top users per category
4. Generate personas for top users

Output: user_personas.json compatible with OASIS simulation format
"""

import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta  # Standard for handling month increments

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    CLUSTER_ASSIGNMENTS_PATH,
    CLUSTER_TOPICS_PATH,
    FILTERED_TWEETS_PATH,
    MIN_TEST_TWEETS,
    MIN_TRAIN_TWEETS,
    MAX_FOLLOWEES,
    OUTPUT_DIR,
    OUTPUT_FORMAT,
    TEST_MONTH,
    TEST_END_DATE,
    TEST_START_DATE,
    TEST_YEAR,
    TOP_N,
    TRAIN_MONTH,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
    TRAIN_YEAR,
    TWEET_POPULARITY_PATH,
    USE_TWITTER_API,
    # ranking strategy controls
    RANK_MODE,
    BLEND_ALPHA,
    MIN_MIX_WEIGHT,
)
from data_split import (
    filter_active_users_in_both_periods,
    load_tweets_with_timestamps,
    save_split_info,
    save_tweets_to_csv,
    split_tweets_by_time,
)
from generate_personas import generate_personas_for_users, save_personas
from rank_users_by_category import (
    rank_users_by_category,
    save_user_rankings,
    compute_user_cluster_mixture,
    save_user_cluster_mix,
)

REPO_ROOT = _REPO_ROOT


def _parse_iso_utc(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def load_cluster_assignments(path: Path) -> Dict[str, int]:
    """Load cluster_assignments.jsonl. Returns tweet_id -> cluster_id."""
    result: Dict[str, int] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tweet_id = obj.get("tweet_id")
                cluster_id = obj.get("cluster_id")
                if tweet_id is not None and cluster_id is not None:
                    result[str(tweet_id)] = int(cluster_id)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return result


def load_cluster_topics(path: Path) -> Dict[int, str]:
    """Load cluster_topics.json. Returns cluster_id -> topic string."""
    result: Dict[int, str] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in (data or {}).items():
        try:
            result[int(k)] = str(v)
        except (ValueError, TypeError):
            continue
    return result


def load_train_tweets_with_popularity_and_clusters(
    train_csv_path: Path,
    popularity_csv_path: Path,
    tweet_to_cluster: Dict[str, int],
) -> Dict[int, List[Dict]]:
    """
    Load train tweets from CSV, join with popularity scores and cluster assignments.
    Returns clusters: Dict[cluster_id, List[tweet_dict]] where tweet_dict has
    id, text, user_id, user_name, popularity.
    """
    # Load popularity: tweet_id -> score
    popularity_map: Dict[str, float] = {}
    if popularity_csv_path.exists():
        with popularity_csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("tweet_id")
                score = row.get("popularity_score")
                if tid:
                    try:
                        popularity_map[str(tid)] = float(score) if score else 0.0
                    except (ValueError, TypeError):
                        popularity_map[str(tid)] = 0.0

    # Build clusters from train CSV
    clusters: Dict[int, List[Dict]] = {}

    with train_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dup") != "0":
                continue
            if row.get("is_root") != "1":
                continue

            tweet_id = row.get("id", "")
            if not tweet_id:
                continue

            cluster_id = tweet_to_cluster.get(str(tweet_id))
            if cluster_id is None or cluster_id < 0:
                continue  # Skip noise or unassigned

            popularity = popularity_map.get(str(tweet_id), 0.0)

            tweet_dict = {
                "id": tweet_id,
                "text": row.get("text", ""),
                "user_id": row.get("user_id", ""),
                "user_name": row.get("user_name", ""),
                "popularity": popularity,
            }

            clusters.setdefault(cluster_id, []).append(tweet_dict)

    return clusters


def load_all_train_tweets_by_user(train_csv_path: Path) -> Dict[str, List[str]]:
    """
    Load all non-duplicate tweets from the train split and group by user_id.

    Returns:
        Dict[user_id, List[tweet_text]]
    """
    user_tweets: Dict[str, List[str]] = {}
    with train_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dup") != "0":
                continue
            user_id = str(row.get("user_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if not user_id or not text:
                continue
            user_tweets.setdefault(user_id, []).append(text)
    return user_tweets


def build_reply_follow_map_for_train(
    filtered_tweets_path: Path,
    train_tweet_ids: Set[str],
) -> Dict[str, Dict[str, int]]:
    """
    Build reply-derived follow edges for train split only.

    Rule:
      if user A replied to user B in train tweets, A should follow B.

    Returns:
      Dict[replier_user_id, Dict[followed_user_id, reply_count]]
    """
    follow_map: Dict[str, Dict[str, int]] = {}
    if not filtered_tweets_path.exists():
        return follow_map

    with filtered_tweets_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            tweet_id = str(obj.get("id", ""))
            if tweet_id not in train_tweet_ids:
                continue

            if not bool(obj.get("isReply", False)):
                continue

            author = obj.get("author") or {}
            replier_user_id = str(author.get("id", "")).strip()
            replied_to_user_id = str(obj.get("inReplyToUserId", "")).strip()

            if not replier_user_id or not replied_to_user_id:
                continue
            if replier_user_id == replied_to_user_id:
                continue

            if replier_user_id not in follow_map:
                follow_map[replier_user_id] = {}
            follow_map[replier_user_id][replied_to_user_id] = (
                follow_map[replier_user_id].get(replied_to_user_id, 0) + 1
            )

    return follow_map


def _cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, av in a.items():
        dot += av * b.get(k, 0.0)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def build_follow_map_for_personas(
    personas: List,
    direct_reply_map: Dict[str, Dict[str, int]],
    user_all_train_tweets: Dict[str, List[str]],
    user_cluster_mix: Dict[str, Dict[int, float]],
    max_followees: int = 15,
) -> Dict[str, Dict[str, float]]:
    """
    Deterministic follow graph generation (non-random):
    1) direct replies
    2) shared reply-target similarity
    3) @mention interactions in train tweets
    4) topic-mixture cosine similarity
    5) deterministic fallback for empty users
    """
    selected_users = [p.original_user_id for p in personas]
    selected_set = set(selected_users)
    if len(selected_users) <= 1:
        return {u: {} for u in selected_users}

    username_to_user_ids: Dict[str, Set[str]] = {}
    for p in personas:
        handle = str(p.original_user_name or "").strip().lower()
        if not handle:
            continue
        username_to_user_ids.setdefault(handle, set()).add(p.original_user_id)

    # Precompute reply target sets (can include users outside selected set).
    reply_target_sets: Dict[str, Set[str]] = {}
    for uid in selected_users:
        reply_target_sets[uid] = set((direct_reply_map.get(uid) or {}).keys())

    scores: Dict[str, Dict[str, float]] = {uid: {} for uid in selected_users}

    for uid in selected_users:
        user_scores = scores[uid]

        # (1) Direct replies inside selected users.
        for target_uid, cnt in (direct_reply_map.get(uid) or {}).items():
            if target_uid in selected_set and target_uid != uid:
                user_scores[target_uid] = user_scores.get(target_uid, 0.0) + 1000.0 * float(cnt)

        # (2) Shared reply-target similarity across selected users.
        u_targets = reply_target_sets.get(uid, set())
        if u_targets:
            for other_uid in selected_users:
                if other_uid == uid:
                    continue
                v_targets = reply_target_sets.get(other_uid, set())
                if not v_targets:
                    continue
                inter = len(u_targets & v_targets)
                if inter == 0:
                    continue
                union = len(u_targets | v_targets)
                jacc = (inter / union) if union > 0 else 0.0
                if jacc > 0.0:
                    user_scores[other_uid] = user_scores.get(other_uid, 0.0) + 100.0 * jacc

        # (3) Mentions from all train tweets for this user.
        mention_counts: Dict[str, int] = {}
        for text in user_all_train_tweets.get(uid, []):
            for handle in re.findall(r"@([A-Za-z0-9_]{1,15})", text or ""):
                h = handle.lower()
                for target_uid in username_to_user_ids.get(h, set()):
                    if target_uid == uid:
                        continue
                    mention_counts[target_uid] = mention_counts.get(target_uid, 0) + 1
        for target_uid, cnt in mention_counts.items():
            user_scores[target_uid] = user_scores.get(target_uid, 0.0) + 300.0 * float(cnt)

        # (4) Topic-mixture cosine similarity.
        u_mix = user_cluster_mix.get(uid, {})
        for other_uid in selected_users:
            if other_uid == uid:
                continue
            sim = _cosine_sparse(u_mix, user_cluster_mix.get(other_uid, {}))
            if sim > 0.0:
                user_scores[other_uid] = user_scores.get(other_uid, 0.0) + 10.0 * sim

    # Keep only top-K and guarantee non-empty with deterministic fallback.
    persona_by_uid = {p.original_user_id: p for p in personas}
    final_map: Dict[str, Dict[str, float]] = {}
    for uid in selected_users:
        ranked = sorted(
            ((tid, sc) for tid, sc in scores.get(uid, {}).items() if tid != uid and sc > 0),
            key=lambda x: (-x[1], x[0]),
        )

        if not ranked:
            # (5) Fallback: similar category first, then higher tweet_count.
            u_cat = str(getattr(persona_by_uid[uid], "category", ""))
            candidates = [
                other_uid
                for other_uid in selected_users
                if other_uid != uid
            ]
            candidates.sort(
                key=lambda other_uid: (
                    0 if str(getattr(persona_by_uid[other_uid], "category", "")) == u_cat else 1,
                    -int(getattr(persona_by_uid[other_uid], "tweet_count", 0) or 0),
                    other_uid,
                )
            )
            ranked = [(other_uid, 1.0) for other_uid in candidates[:max_followees]]

        top_ranked = ranked[:max_followees]
        final_map[uid] = {tid: sc for tid, sc in top_ranked}

    return final_map


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("User Persona Generation Pipeline")
    print("=" * 60)

    # Step 1: Load and split tweets
    print("\n[Step 1] Loading tweets and splitting by time...")
    all_tweets = load_tweets_with_timestamps(FILTERED_TWEETS_PATH)
    if not all_tweets:
        print("No tweets found. Exiting.")
        return
    print(f"Loaded {len(all_tweets)} tweets")

    # --- NEW RELATIVE SPLIT LOGIC ---
    # 1. Find the earliest tweet timestamp to anchor the split
    global_start_dt = min(t.timestamp for t in all_tweets)
    
    # 2. Define training as first 2 months, test as the 1 month following that
    # Note: Using relativedelta is safer for calendar months than timedelta(days=60)
    train_end_dt = global_start_dt + relativedelta(months=2)
    test_end_dt = train_end_dt + relativedelta(months=1)

    train_tweets = [
        t for t in all_tweets if global_start_dt <= t.timestamp < train_end_dt
    ]
    test_tweets = [
        t for t in all_tweets if train_end_dt <= t.timestamp < test_end_dt
    ]

    print(f"Relative split based on data start: {global_start_dt.strftime('%Y-%m-%d')}")
    print(f"Train window: {global_start_dt.isoformat()} to {train_end_dt.isoformat()}")
    print(f"Test window:  {train_end_dt.isoformat()} to {test_end_dt.isoformat()}")
    # --------------------------------

    print(f"Initial split - Train: {len(train_tweets)}, Test: {len(test_tweets)}")

    train_filtered, test_filtered = filter_active_users_in_both_periods(
        train_tweets,
        test_tweets,
        min_train_tweets=MIN_TRAIN_TWEETS,
        min_test_tweets=MIN_TEST_TWEETS,
    )
    
    print(f"After filtering - Train: {len(train_filtered)}, Test: {len(test_filtered)}")

    train_file = OUTPUT_DIR / "train_tweets.csv"
    test_file = OUTPUT_DIR / "test_tweets.csv"
    save_tweets_to_csv(train_filtered, train_file)
    save_tweets_to_csv(test_filtered, test_file)
    save_split_info(
        train_filtered,
        test_filtered,
        OUTPUT_DIR / "split_info.json",
        train_months=TRAIN_MONTH,
        test_months=TEST_MONTH,
    )
    print(f"Saved train/test splits to {train_file} and {test_file}")

    # Step 2: Load clustering output
    print("\n[Step 2] Loading clustering output...")
    cluster_assignments_path = CLUSTER_ASSIGNMENTS_PATH
    cluster_topics_path = CLUSTER_TOPICS_PATH

    if not cluster_assignments_path.exists():
        fallback = REPO_ROOT / "clustering" / "cluster_assignments.jsonl"
        if fallback.exists():
            print(f"Cluster assignments not found at {cluster_assignments_path}, using {fallback}")
            cluster_assignments_path = fallback

    if not cluster_topics_path.exists():
        fallback = REPO_ROOT / "clustering" / "cluster_topics.json"
        if fallback.exists():
            print(f"Cluster topics not found at {cluster_topics_path}, using {fallback}")
            cluster_topics_path = fallback

    tweet_to_cluster = load_cluster_assignments(cluster_assignments_path)
    topics_dict = load_cluster_topics(cluster_topics_path)
    print(f"Loaded {len(tweet_to_cluster)} cluster assignments, {len(topics_dict)} topics")

    clusters = load_train_tweets_with_popularity_and_clusters(
        train_file, TWEET_POPULARITY_PATH, tweet_to_cluster
    )
    print(f"Built {len(clusters)} clusters from train tweets")

    if not clusters:
        print("Warning: No clusters with assigned tweets. Ensure clustering has been run.")
        print("See persona_generation/CLUSTERING_INPUT_SPEC.md for required output format.")

    # Step 3: Rank users by category
    print(f"\n[Step 3] Ranking top {TOP_N} users per category...")

    # (Optional) Persist mixture for analysis/QA (independent of rank mode)
    user_cluster_mix = compute_user_cluster_mixture(clusters)
    mix_file = OUTPUT_DIR / "user_cluster_mix.json"
    save_user_cluster_mix(user_cluster_mix, mix_file)
    print(f"Saved user->cluster mixture to {mix_file}")

    # Rank using strategy from config.py
    rankings = rank_users_by_category(
        clusters,
        topics_dict,
        top_n=TOP_N,
        rank_mode=RANK_MODE,
        alpha=BLEND_ALPHA,
        min_weight=MIN_MIX_WEIGHT,
    )

    rankings_file = OUTPUT_DIR / "user_rankings.json"
    save_user_rankings(rankings, rankings_file)
    print(f"Saved rankings to {rankings_file}")

    # Step 4: Generate personas
    print("\n[Step 4] Generating personas...")
    with rankings_file.open("r", encoding="utf-8") as f:
        rankings_data = json.load(f)

    user_all_train_tweets = load_all_train_tweets_by_user(train_file)
    print(f"Loaded full train-tweet context for {len(user_all_train_tweets)} users")

    personas = generate_personas_for_users(
        rankings_data,
        use_twitter_api=USE_TWITTER_API,
        user_all_train_tweets=user_all_train_tweets,
    )

    train_tweet_ids = {t.id for t in train_filtered}
    reply_follow_map = build_reply_follow_map_for_train(FILTERED_TWEETS_PATH, train_tweet_ids)
    follow_map = build_follow_map_for_personas(
        personas,
        direct_reply_map=reply_follow_map,
        user_all_train_tweets=user_all_train_tweets,
        user_cluster_mix=user_cluster_mix,
        max_followees=MAX_FOLLOWEES,
    )
    nonempty = sum(1 for uid in follow_map if follow_map[uid])
    print(
        f"Built deterministic follow graph for {len(follow_map)} users "
        f"({nonempty} with non-empty follow lists)"
    )

    if OUTPUT_FORMAT == "json":
        personas_file = OUTPUT_DIR / "user_personas.json"
    else:
        personas_file = OUTPUT_DIR / "user_personas.csv"

    save_personas(
        personas,
        personas_file,
        format=OUTPUT_FORMAT,
        following_by_user=follow_map,
    )
    print(f"Generated {len(personas)} personas and saved to {personas_file}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Train tweets: {train_file}")
    print(f"  - Test tweets: {test_file}")
    print(f"  - Split info: {OUTPUT_DIR / 'split_info.json'}")
    print(f"  - User rankings: {rankings_file}")
    print(f"  - User personas: {personas_file}")


if __name__ == "__main__":
    main()