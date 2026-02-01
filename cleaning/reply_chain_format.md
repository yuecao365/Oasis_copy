# reply_chain.jsonl — Data format & how to use it

**File:** `reply_chain.jsonl`  
**Format:** One JSON object per line. Each line corresponds to **one tweet** and gives its position in the reply chain (root vs reply, depth, root id).


---

## Example: root tweet (start of a thread)

```json
{
  "tweet_id": "1947552617514561676",
  "root_tweet_id": "1947552617514561676",
  "reply_depth": 0,
  "is_root": true,
  "author_id": "1450911771166969864",
  "clean_text": "Do it as it says. Bring friends and family to Greencart #VeFam. VebetterDAO $VET $B3TR #vechain."
}
```

- **`tweet_id`** = **`root_tweet_id`** → this tweet is the **root** of its thread (no parent in our data).
- **`reply_depth`** = 0 → top of the chain.
- **`is_root`** = true.

---

## Example: reply tweet (part of a thread)

```json
{
  "tweet_id": "1947492725957529743",
  "root_tweet_id": "1947431331706900702",
  "reply_depth": 1,
  "is_root": false,
  "author_id": "1055346238600568835",
  "clean_text": "@TheVirtualFlame @vechainofficial #Vechain #VeFam $VET $VTHO"
}
```

- **`tweet_id`** ≠ **`root_tweet_id`** → this tweet is a **reply**; its thread root is `1947431331706900702`.
- **`reply_depth`** = 1 → one step below the root (direct reply to root).
- **`is_root`** = false.
- **`root_tweet_id`** → use this to group all tweets that belong to the same thread.

---

## Field reference

| Field | Type | Meaning |
|-------|------|--------|
| **`tweet_id`** | string | Tweet ID. Use to join with `filtered_tweets.jsonl` (field `id`). |
| **`root_tweet_id`** | string | Tweet ID of the root of this tweet’s reply chain. Same as `tweet_id` when this row is a root. |
| **`reply_depth`** | int | 0 = root; 1 = direct reply to root; 2 = reply to a reply; etc. Max depth in this dataset is 19. |
| **`is_root`** | bool | true if this tweet is the root of its chain (`reply_depth` = 0). |
| **`author_id`** | string | Author’s user ID. |
| **`clean_text`** | string | Cleaned tweet text (same cleaning as in the rest of the pipeline). |

---

## How to use it for clustering

1. **Join with `filtered_tweets.jsonl`**  
   - Key: **`reply_chain.tweet_id`** = **`filtered_tweets.id`**  
   - You get: full tweet (text, engagement, author, etc.) **plus** `reply_depth`, `root_tweet_id`, `is_root`.

2. **Thread-level features**  
   - Group by **`root_tweet_id`** to get all tweets in a thread.  
   - Use **`reply_depth`** for depth-based features (e.g. root vs reply, or depth bins).  
   - Use **`is_root`** to separate roots from replies (e.g. cluster only roots, or only replies).

3. **Filtering**  
   - Keep only roots: `is_root == true` or `reply_depth == 0`.  
   - Keep only replies: `is_root == false` or `reply_depth >= 1`.  
   - Keep only deep replies: e.g. `reply_depth >= 2`.

4. **Consistency with `filtered_tweets`**  
   - Every row in `reply_chain.jsonl` corresponds to a tweet that is in `filtered_tweets.jsonl` (same set of tweets; reply_chain adds the chain structure).  
   - One row per tweet: each `tweet_id` appears at most once.

---

## Upstream 

- Built from **`filtered_tweets.jsonl`** only (no extra data).  
- Reply links: **`inReplyToId`** or **`inReplyToTweetId`** in the tweet object.  
- A tweet’s **root** is found by following reply-to parent until there is no parent in the set; **reply_depth** is the number of steps from that root.  
- So chains are **within** the filtered tweet set: if the parent of a reply is not in `filtered_tweets`, that reply is still present but its depth/root are computed from the nearest ancestor that is in the set.
