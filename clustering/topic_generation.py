"""
Step 6: Topic generation per cluster via LLM.

Reads representative posts (from tweet_clustering) and generates a brief topic
description for each cluster using an LLM (OpenAI API).

Run: python topic_generation.py
Requires: pip install openai, set OPENAI_API_KEY env
Output: data/processed/cluster_topics.json
"""

import json
import os

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"
REP_POSTS_FILE = "representative_posts_kmeans.json"
# Use train-only filtered tweets (2025-04 to 2025-07) for topic generation.
FILTERED_TWEETS_FILE = "filtered_tweets_train_2025-04_to_07.jsonl"
OUTPUT_FILE = "cluster_topics.json"

# LLM config
OPENAI_MODEL = "gpt-4o-mini"  # cost-efficient
MAX_POSTS_PER_CLUSTER = 50    # cap to avoid token limit
MAX_CHARS_PER_POST = 280      # truncate long tweets


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_id_to_text(filtered_tweets_path):
    """Build tweet_id -> text from filtered_tweets (clean_text or text)."""
    out = {}
    with open(filtered_tweets_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row.get("id")
            if tid is None:
                continue
            tid = str(tid).strip()
            text = (row.get("clean_text") or row.get("text") or "").strip()
            if text:
                if len(text) > MAX_CHARS_PER_POST:
                    text = text[:MAX_CHARS_PER_POST] + "..."
                out[tid] = text
    return out


def load_representative_posts(path):
    """cluster_id -> list of (tweet_id,) - order is already top-by-popularity from clustering."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for cid_str, posts in raw.items():
        out[int(cid_str)] = [p["tweet_id"] for p in posts]
    return out


def get_cluster_texts(rep_posts, id_to_text, max_posts=MAX_POSTS_PER_CLUSTER):
    """For each cluster, return list of post texts (first max_posts, already ranked by popularity)."""
    cluster_texts = {}
    for cid, tweet_ids in rep_posts.items():
        texts = []
        for tid in tweet_ids[:max_posts]:
            t = id_to_text.get(str(tid))
            if t and t.strip():
                texts.append(t)
        cluster_texts[cid] = texts
    return cluster_texts


def generate_topic_for_cluster(cluster_id, texts, model=OPENAI_MODEL):
    """Call LLM to generate a brief, direct topic sentence for the cluster."""
    if not texts:
        return "No posts available."

    posts_block = "\n".join(f"- {t}" for t in texts[:MAX_POSTS_PER_CLUSTER])
    prompt = f"""Below are sample posts from a Twitter cluster (conversation threads about similar topics).
Write a SHORT description in 1–2 sentences (max 40 words total).
- First sentence: the high-level topic (e.g. "VeChain ecosystem upgrades and community sentiment").
- Second sentence (optional): what is distinctive about this cluster (specific apps, campaigns, mechanisms, events, or behaviors).
- Start directly with the key subject (do NOT start with phrases like "This cluster focuses on" or "This cluster is about").
- Be specific and concise. Output only these 1–2 sentences, no preamble or extra commentary.

Cluster ID: {cluster_id}

Sample posts:
{posts_block}
"""

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[Error: {e}]"


def main():
    rep_path = os.path.join(INPUT_DIR, REP_POSTS_FILE)
    tweets_path = os.path.join(INPUT_DIR, FILTERED_TWEETS_FILE)
    if not os.path.isfile(rep_path):
        print("ERROR: {} not found. Run tweet_clustering.py first.".format(rep_path))
        return
    if not os.path.isfile(tweets_path):
        print("ERROR: {} not found.".format(tweets_path))
        return

    print("Loading representative posts and filtered_tweets...")
    rep_posts = load_representative_posts(rep_path)
    id_to_text = load_id_to_text(tweets_path)
    cluster_texts = get_cluster_texts(rep_posts, id_to_text)
    cluster_ids = sorted(rep_posts.keys())
    print("  Clusters: {}".format(len(cluster_ids)))

    print("Generating topics via {}...".format(OPENAI_MODEL))
    topics = {}
    for i, cid in enumerate(cluster_ids):
        texts = cluster_texts.get(cid, [])
        topic = generate_topic_for_cluster(cid, texts)
        topics[str(cid)] = topic
        print("  cluster {}: {}".format(cid, topic[:80] + "..." if len(topic) > 80 else topic))

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    print("\nSaved {}".format(out_path))


if __name__ == "__main__":
    main()
