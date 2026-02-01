# filtered_tweets.jsonl — Data format

**File:** `filtered_tweets.jsonl`  
**Format:** One JSON object per line (one tweet per line). Each tweet `id` appears at most once.

---

## Example: one tweet (one line in the file)

```json
{
  "type": "tweet",
  "id": "1947552617514561676",
  "url": "https://x.com/GreeceVechain/status/1947552617514561676",
  "twitterUrl": "https://twitter.com/GreeceVechain/status/1947552617514561676",
  "text": "Do it as it says. \n\nBring friends and family to Greencart \n#VeFam. \n\nVebetterDAO $VET",
  "clean_text": "Do it as it says. Bring friends and family to Greencart #VeFam. VebetterDAO $VET",
  "tokens": ["do", "it", "as", "it", "says", "bring", "friends", "and", "family", "to", "greencart", "vefam", "vebetterdao", "vet"],
  "source": "Twitter for iPhone",
  "retweetCount": 0,
  "replyCount": 0,
  "likeCount": 3,
  "quoteCount": 0,
  "viewCount": 88,
  "bookmarkCount": 0,
  "createdAt": "Tue Jul 22 07:01:42 +0000 2025",
  "lang": "en",
  "isReply": false,
  "inReplyToId": null,
  "conversationId": "1947552617514561676",
  "inReplyToUserId": null,
  "inReplyToUsername": null,
  "author": {
    "type": "user",
    "id": "1450911771166969864",
    "userName": "GreeceVechain",
    "name": "vechain Greece 🇬🇷 Community",
    "url": "https://x.com/GreeceVechain",
    "description": "",
    "location": "Greece",
    "followers": 1670,
    "following": 498,
    "statusesCount": 3726,
    "createdAt": "Wed Oct 20 19:48:39 +0000 2021",
    "isVerified": false,
    "profilePicture": "https://pbs.twimg.com/..."
  },
  "entities": {
    "hashtags": [{"indices": [59, 65], "text": "VeFam"}, {"indices": [92, 100], "text": "vechain"}],
    "symbols": [{"indices": [81, 85], "text": "VET"}]
  },
  "quoted_tweet": { "id": "1945843484348273110", "text": "🚀 Big news: exclusive features...", "author": {...} },
  "retweeted_tweet": null,
  "extendedEntities": {},
  "card": null,
  "place": {}
}
```

---

## What each part is for

- **`id`** — Unique tweet ID. Use this to join with `reply_chain.jsonl` (field `tweet_id`).
- **`text`** — Raw tweet body (newlines, @mentions, URLs as-is).
- **`clean_text`** — Cleaned body (URLs replaced with "URL", emoji removed, whitespace normalized). Best for text clustering.
- **`tokens`** — List of lowercased words from `clean_text`. Also useful for clustering.
- **`author`** — Nested object: `author.id` is user ID, `author.userName` is handle, `author.name` is display name, `author.followers` / `author.following` / `author.statusesCount` for user-level features.
- **Engagement** — `likeCount`, `retweetCount`, `replyCount`, `quoteCount`, `viewCount`, `bookmarkCount`. In the file they may be numbers or strings; cast to int if needed.
- **`createdAt`** — Tweet timestamp string (e.g. `"Tue Jul 22 07:01:42 +0000 2025"`). Parse for time-based features.
- **Reply info** — `isReply`, `inReplyToId` (parent tweet ID or null), `conversationId`, `inReplyToUserId`, `inReplyToUsername`.
- **`entities`** — `hashtags`, `symbols`, and possibly `user_mentions`, `urls` (lists of objects with `text`, `indices`, etc.).
- **`quoted_tweet`** / **`retweeted_tweet`** — Nested tweet object if this tweet quotes or retweets another; otherwise `null`.

---

## Notes for clustering

- Use **`clean_text`** or **`tokens`** for text-based clustering.
- **`id`** is the unique key per tweet; **`author.id`** is the unique key per user.
- Engagement fields may be strings in some rows; use `int(x)` or `int(x) if x is not None else 0` when building features.
- **`reply_chain.jsonl`** has one row per tweet with `tweet_id`, `root_tweet_id`, `reply_depth`; join on `tweet_id` = `id` if you need reply structure.

---


