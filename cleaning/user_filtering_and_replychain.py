import json
import os
import re
from collections import defaultdict

# =========================
# Config
# =========================

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

MIN_TWEETS_PER_USER = 5   # minimum tweets to support persona generation
JACCARD_THRESHOLD = 0.85  # within-user near-duplicate threshold (same as tweet_cleaning)
MIN_CONTENT_WORDS = 2     # drop tweets with empty text, "URL", or very few words

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load helpers
# =========================

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def filter_tweets_by_content(tweets):
    """
    Drop tweets with no real content: empty, "URL", or very few words.
    So filtered_tweets (and downstream) only get tweets with basic content.
    """
    out = []
    dropped = 0
    for tw in tweets:
        if tw.get("id") is None:
            continue
        text = (tw.get("clean_text") or tw.get("text") or "").strip()
        if not text or text == "URL":
            dropped += 1
            continue
        if len(text.split()) < MIN_CONTENT_WORDS:
            dropped += 1
            continue
        out.append(tw)
    if dropped:
        print(f"  Filtered {dropped} low-content tweets (empty/URL/<{MIN_CONTENT_WORDS} words).")
    return out


# =========================
# Within-user deduplication (repeat-rate check)
# =========================

def tokenize(text):
    """Same tokenization as tweet_cleaning."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate_tweets_within_user(user_tweets):
    """
    Within one user's tweets: drop near-duplicates (Jaccard >= threshold).
    No bins — compare each tweet only to already-kept tweets (all-pairs within user
    is fine for per-user scale). Keeps exactly one per duplicate group.
    Returns (kept_tweets, kept_tweet_ids).
    """
    if not user_tweets:
        return [], set()

    kept = []  # list of (tweet_dict, token_set) for already-kept tweets

    for tw in user_tweets:
        tw = dict(tw)
        text = tw.get("clean_text") or tw.get("text", "")
        token_set = set(tokenize(text))
        keep = True
        for _kept_tw, _kept_set in kept:
            if jaccard(token_set, _kept_set) >= JACCARD_THRESHOLD:
                keep = False
                break
        if keep:
            kept.append((tw, token_set))

    kept_tweets = [t for t, _ in kept]
    kept_ids = {t["id"] for t in kept_tweets}
    return kept_tweets, kept_ids


# =========================
# Step 1: User-level stats (after within-user dedup)
# =========================

def compute_user_tweet_counts_after_dedup(tweets):
    """
    Group tweets by user, run within-user dedup, then count effective tweets per user.
    Returns (user_counts, kept_tweet_ids).
    """
    by_user = defaultdict(list)
    for tw in tweets:
        user_id = (tw.get("author") or {}).get("id")
        if not user_id:
            continue
        by_user[user_id].append(tw)

    user_counts = defaultdict(int)
    kept_tweet_ids = set()

    for user_id, user_tweets in by_user.items():
        kept_tweets, kept_ids = deduplicate_tweets_within_user(user_tweets)
        user_counts[user_id] = len(kept_tweets)
        kept_tweet_ids.update(kept_ids)

    return user_counts, kept_tweet_ids


def select_users(user_counts):
    """
    Select users with enough tweets to support persona generation.
    """
    results = []

    for user_id, count in user_counts.items():
        is_selected = count >= MIN_TWEETS_PER_USER
        results.append({
            "user_id": user_id,
            "num_tweets": count,
            "is_selected": int(is_selected),
        })

    return results


# =========================
# Step 2: Filter tweets by selected users
# =========================

def filter_tweets_by_users(tweets, selected_user_ids, kept_tweet_ids=None):
    out = [
        tw for tw in tweets
        if (tw.get("author") or {}).get("id") in selected_user_ids
    ]
    if kept_tweet_ids is not None:
        out = [tw for tw in out if tw["id"] in kept_tweet_ids]
    # One row per tweet_id: within-user dedup already chose which ids to keep;
    # collapse duplicate rows (same id many times) to one.
    seen_id = set()
    result = []
    for tw in out:
        if tw["id"] in seen_id:
            continue
        seen_id.add(tw["id"])
        result.append(tw)
    return result


# =========================
# Step 3: Build reply chain
# =========================

def _in_reply_to(tw):
    """Reply target: data may use inReplyToId or inReplyToTweetId."""
    return tw.get("inReplyToId") or tw.get("inReplyToTweetId")


def build_reply_chain(tweets):
    """
    Build reply chain only among retained tweets.
    Follows inReplyToId (or inReplyToTweetId) only when the parent is in id_map.
    """
    id_map = {tw["id"]: tw for tw in tweets}

    def find_root(tw):
        depth = 0
        cur = tw
        parent_id = _in_reply_to(cur)
        while parent_id in id_map:
            cur = id_map[parent_id]
            depth += 1
            parent_id = _in_reply_to(cur)
        return cur["id"], depth

    chains = []

    for tw in tweets:
        root_id, depth = find_root(tw)
        chains.append({
            "tweet_id": tw["id"],
            "root_tweet_id": root_id,
            "reply_depth": depth,
            "is_root": depth == 0,
            "author_id": (tw.get("author") or {}).get("id"),
            "clean_text": tw.get("clean_text", ""),
        })

    return chains


# =========================
# Main
# =========================

def main():
    print("Loading cleaned tweets...")
    tweets = load_jsonl(os.path.join(INPUT_DIR, "clean_tweets.jsonl"))
    print(f"Loaded {len(tweets)} cleaned tweets")

    print("Filtering low-content tweets (empty/URL/too short)...")
    tweets = filter_tweets_by_content(tweets)
    print(f"Kept {len(tweets)} tweets with content")

    print("Within-user dedup (repeat-rate check, Jaccard >= {})...".format(JACCARD_THRESHOLD))
    user_counts, kept_tweet_ids = compute_user_tweet_counts_after_dedup(tweets)
    total_after_dedup = sum(user_counts.values())
    print(f"After within-user dedup: {total_after_dedup} tweets (from {len(tweets)} raw)")

    print("Selecting users (min {} tweets per user)...".format(MIN_TWEETS_PER_USER))
    user_candidates = select_users(user_counts)

    # Save user candidates (for inspection / reporting)
    candidates_path = os.path.join(OUTPUT_DIR, "user_candidates.csv")
    with open(candidates_path, "w", encoding="utf-8") as f:
        f.write("user_id,num_tweets,is_selected\n")
        for row in user_candidates:
            f.write(
                f"{row['user_id']},"
                f"{row['num_tweets']},"
                f"{row['is_selected']}\n"
            )

    selected_user_ids = {
        r["user_id"] for r in user_candidates if r["is_selected"] == 1
    }

    print(f"Selected {len(selected_user_ids)} users")

    print("Filtering tweets by selected users (and keeping only post-dedup tweets)...")
    filtered_tweets = filter_tweets_by_users(tweets, selected_user_ids, kept_tweet_ids)
    print(f"Kept {len(filtered_tweets)} tweets after user filtering")

    save_jsonl(
        os.path.join(OUTPUT_DIR, "filtered_tweets.jsonl"),
        filtered_tweets
    )

    print("Building reply chains...")
    reply_chain = build_reply_chain(filtered_tweets)

    save_jsonl(
        os.path.join(OUTPUT_DIR, "reply_chain.jsonl"),
        reply_chain
    )

    print("DONE.")
    print("Outputs:")
    print(" - user_candidates.csv")
    print(" - filtered_tweets.jsonl")
    print(" - reply_chain.jsonl")


if __name__ == "__main__":
    main()
