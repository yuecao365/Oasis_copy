import os
import re
import json
import csv
from collections import defaultdict
from typing import Dict, List, Set

# =========================
# Config
# =========================

INPUT_DIR = "/Users/sichengxu/Downloads/twitter"
OUTPUT_DIR = "data/processed"

MIN_TOKEN_LEN = 20          # skip very short tweets
BIN_SIZE = 10               # 0–10, 10–20, ...
JACCARD_THRESHOLD = 0.85    # high threshold for auto-gen
MIN_LIKES_THRESHOLD = 10    # duplicate cluster: if any tweet has likes >= this, keep one (the one with most likes); else drop all
LOG_EVERY = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Basic text cleaning
# =========================

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F700-\U0001F77F"  
    "\U0001F780-\U0001F7FF"  
    "\U0001F800-\U0001F8FF"  
    "\U0001F900-\U0001F9FF"  
    "\U0001FA00-\U0001FA6F"  
    "\U0001FA70-\U0001FAFF"  
    "\U00002702-\U000027B0"  
    "\U000024C2-\U0001F251" 
    "]+",
    flags=re.UNICODE,
)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = URL_PATTERN.sub("URL", text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def get_likes(tw: Dict) -> int:
    """Likes count from raw tweet (likeCount / likes / favorite_count)."""
    v = tw.get("likeCount") or tw.get("likes") or tw.get("favorite_count") or 0
    return int(v) if v is not None else 0

# =========================
# Load tweets
# =========================

def load_all_tweets(input_dir: str) -> List[Dict]:
    tweets = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.endswith(".json"):
                fp = os.path.join(root, fn)
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
                    tweets.extend(obj.get("tweets", []))
    return tweets

# =========================
# Deduplication
# =========================

def deduplicate_tweets(tweets: List[Dict]):
    """
    Tweet-level cleaning & near-duplicate removal.
    """

    bins = defaultdict(list)
    dropped = []
    processed = []

    for idx, tw in enumerate(tweets):
        raw = tw.get("text", "")
        clean = clean_text(raw)
        tokens = tokenize(clean)

        tw["clean_text"] = clean
        tw["tokens"] = tokens

        # Short tweets: keep but skip similarity check
        if len(tokens) < MIN_TOKEN_LEN:
            processed.append(tw)
            continue

        bin_id = (len(tokens) // BIN_SIZE) * BIN_SIZE
        bins[bin_id].append(tw)

        if idx % LOG_EVERY == 0 and idx > 0:
            print(f"Processed {idx} tweets...")

    
    for bin_id, group in bins.items():
        kept: List[tuple] = []  # (tw, token_set)
        dropped_reps: List[set] = []  

        for tw in group:
            tid = tw["id"]
            tokens = set(tw["tokens"])

            
            similar_kept_idx = None
            for idx, (kept_tw, kept_tokens) in enumerate(kept):
                if jaccard(tokens, kept_tokens) >= JACCARD_THRESHOLD:
                    similar_kept_idx = idx
                    break

            if similar_kept_idx is not None:
                
                kept_tw, kept_tokens = kept[similar_kept_idx]
                score = jaccard(tokens, kept_tokens)
                likes_tw = get_likes(tw)
                likes_kept = get_likes(kept_tw)

                if max(likes_tw, likes_kept) >= MIN_LIKES_THRESHOLD:
                    
                    if likes_tw >= likes_kept:
                        kept.pop(similar_kept_idx)
                        dropped.append({
                            "tweet_id": kept_tw["id"],
                            "reason": "near_duplicate",
                            "similar_to": tid,
                            "jaccard": round(score, 4),
                            "clean_text": kept_tw["clean_text"],
                        })
                        kept.append((tw, tokens))
                    else:
                        dropped.append({
                            "tweet_id": tid,
                            "reason": "near_duplicate",
                            "similar_to": kept_tw["id"],
                            "jaccard": round(score, 4),
                            "clean_text": tw["clean_text"],
                        })
                else:
                    # 簇里人气都低 → 两条都删，记代表进 dropped_reps
                    dropped.append({
                        "tweet_id": tid,
                        "reason": "near_duplicate",
                        "similar_to": kept_tw["id"],
                        "jaccard": round(score, 4),
                        "clean_text": tw["clean_text"],
                    })
                    kept.pop(similar_kept_idx)
                    dropped.append({
                        "tweet_id": kept_tw["id"],
                        "reason": "near_duplicate",
                        "similar_to": tid,
                        "jaccard": round(score, 4),
                        "clean_text": kept_tw["clean_text"],
                    })
                    dropped_reps.append(kept_tokens)
                continue

            
            similar_to_rep = False
            for rep_tokens in dropped_reps:
                if jaccard(tokens, rep_tokens) >= JACCARD_THRESHOLD:
                    similar_to_rep = True
                    if get_likes(tw) >= MIN_LIKES_THRESHOLD:
                        
                        kept.append((tw, tokens))
                    else:
                        dropped.append({
                            "tweet_id": tid,
                            "reason": "near_duplicate",
                            "similar_to": "(dropped_rep)",
                            "jaccard": round(jaccard(tokens, rep_tokens), 4),
                            "clean_text": tw["clean_text"],
                        })
                    break
            if similar_to_rep:
                continue

            kept.append((tw, tokens))

        for tw, _ in kept:
            processed.append(tw)

    return processed, dropped

# =========================
# Save helpers
# =========================

def save_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# =========================
# Main
# =========================

def main():
    print("Loading tweets...")
    tweets = load_all_tweets(INPUT_DIR)
    print(f"Loaded {len(tweets)} tweets")

    print("Cleaning & deduplicating tweets...")
    clean_tweets, dropped = deduplicate_tweets(tweets)

    print(f"Kept {len(clean_tweets)} tweets")
    print(f"Dropped {len(dropped)} near-duplicates")

    # Save outputs
    save_jsonl(os.path.join(OUTPUT_DIR, "clean_tweets.jsonl"), clean_tweets)
    save_jsonl(os.path.join(OUTPUT_DIR, "dropped_duplicates.jsonl"), dropped)

    # Optional CSVs (for inspection)
    save_csv(os.path.join(OUTPUT_DIR, "dropped_duplicates.csv"), dropped)

    print("DONE.")
    print("Outputs:")
    print(" - clean_tweets.jsonl")
    print(" - dropped_duplicates.jsonl")
    print(" - dropped_duplicates.csv")

if __name__ == "__main__":
    main()
