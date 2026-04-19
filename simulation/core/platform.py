"""
platform.py — synchronous SQLite-backed social media platform.

Replaces OASIS's async Channel + Platform stack with a plain synchronous
class.  The DB schema is identical to OASIS so the comparison scripts and
dashboard continue to work without changes.

Actions return a result dict: {"success": True, ...} or
{"success": False, "error": "..."}.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

platform_log = logging.getLogger("social.platform")

# ── Schema SQL (inlined to avoid depending on the oasis package path) ──────
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=OFF;

CREATE TABLE IF NOT EXISTS user (
    user_id         INTEGER PRIMARY KEY,
    agent_id        INTEGER,
    user_name       TEXT,
    name            TEXT,
    bio             TEXT,
    created_at      REAL,
    num_followings  INTEGER DEFAULT 0,
    num_followers   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS post (
    post_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER,
    original_post_id INTEGER,
    content         TEXT DEFAULT '',
    quote_content   TEXT,
    created_at      REAL,
    num_likes       INTEGER DEFAULT 0,
    num_dislikes    INTEGER DEFAULT 0,
    num_shares      INTEGER DEFAULT 0,
    num_reports     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS follow (
    follow_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    follower_id INTEGER,
    followee_id INTEGER,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS 'like' (
    like_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id     INTEGER,
    user_id     INTEGER,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS dislike (
    dislike_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id     INTEGER,
    user_id     INTEGER,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS comment (
    comment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id     INTEGER,
    user_id     INTEGER,
    content     TEXT,
    created_at  REAL,
    num_likes   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS comment_like (
    comment_like_id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id      INTEGER,
    user_id         INTEGER,
    created_at      REAL
);

CREATE TABLE IF NOT EXISTS mute (
    mute_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    muter_id    INTEGER,
    mutee_id    INTEGER,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS report (
    report_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id     INTEGER,
    user_id     INTEGER,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS rec (
    user_id INTEGER,
    post_id INTEGER,
    PRIMARY KEY (user_id, post_id)
);

CREATE TABLE IF NOT EXISTS trace (
    user_id     INTEGER,
    created_at  REAL,
    action      TEXT,
    info        TEXT,
    PRIMARY KEY (user_id, created_at, action, info)
);
"""


class Platform:
    """Synchronous social media platform backed by SQLite.

    Args:
        db_path:              File path for the SQLite DB (":memory:" also works).
        recsys_type:          "twhin-bert", "twitter", "random", or "reddit".
        max_rec_post_len:     Max posts per user in the rec table.
        refresh_rec_post_count: Number of recsys posts surfaced per feed refresh.
        following_post_count: Number of posts fetched from followed users per refresh.
        use_openai_embedding: Use OpenAI embeddings instead of TWHIN-BERT.
    """

    def __init__(
        self,
        db_path: str,
        recsys_type: str = "twhin-bert",
        max_rec_post_len: int = 10,
        refresh_rec_post_count: int = 5,
        following_post_count: int = 10,
        use_openai_embedding: bool = False,
    ) -> None:
        self.db_path = db_path
        self.recsys_type = recsys_type.lower().strip()
        self.max_rec_post_len = max_rec_post_len
        self.refresh_rec_post_count = refresh_rec_post_count
        self.following_post_count = following_post_count
        self.use_openai_embedding = use_openai_embedding
        self.time_step: float = 0.0

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.executescript(_SCHEMA_SQL)
        self.db.commit()
        self._cur = self.db.cursor()

    # ── Internal helpers ────────────────────────────────────────────────

    def _exec(self, sql: str, params: tuple = (), commit: bool = False) -> None:
        self._cur.execute(sql, params)
        if commit:
            self.db.commit()

    def _record_trace(self, user_id: int, action: str, info: dict) -> None:
        info_str = json.dumps(info, ensure_ascii=False)
        try:
            self._exec(
                "INSERT OR IGNORE INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)",
                (user_id, self.time_step, action, info_str),
                commit=True,
            )
        except Exception:
            pass

    def _get_root_post_id(self, post_id: int) -> int:
        """Walk the repost chain and return the root (original) post_id."""
        self._exec(
            "SELECT original_post_id, quote_content FROM post WHERE post_id = ?",
            (post_id,),
        )
        row = self._cur.fetchone()
        if not row or row[0] is None:
            return post_id
        # It's a repost or quote — recurse to find root.
        return self._get_root_post_id(row[0])

    def _post_type(self, post_id: int) -> dict | None:
        self._exec(
            "SELECT original_post_id, quote_content FROM post WHERE post_id = ?",
            (post_id,),
        )
        row = self._cur.fetchone()
        if row is None:
            return None
        orig_id, quote_content = row
        if orig_id is None:
            return {"type": "common", "root_post_id": post_id}
        if quote_content and quote_content.strip():
            root = self._get_root_post_id(orig_id)
            return {"type": "quote", "root_post_id": root}
        root = self._get_root_post_id(orig_id)
        return {"type": "repost", "root_post_id": root}

    def _posts_with_comments(self, rows: list[tuple]) -> list[dict]:
        """Attach comment lists to a set of post rows."""
        results = []
        for row in rows:
            post_id, user_id, orig_id, content, quote_content, created_at, \
                num_likes, num_dislikes, num_shares = row
            self._exec(
                "SELECT comment_id, user_id, content, created_at, num_likes "
                "FROM comment WHERE post_id = ?",
                (post_id,),
            )
            comments = [
                {"comment_id": c[0], "user_id": c[1], "content": c[2],
                 "created_at": c[3], "num_likes": c[4]}
                for c in self._cur.fetchall()
            ]
            results.append({
                "post_id": post_id,
                "user_id": user_id,
                "original_post_id": orig_id,
                "content": content,
                "quote_content": quote_content,
                "created_at": created_at,
                "num_likes": num_likes,
                "num_dislikes": num_dislikes,
                "num_shares": num_shares,
                "comments": comments,
            })
        return results

    # ── Agent / user management ─────────────────────────────────────────

    def sign_up(self, agent_id: int, username: str, name: str, bio: str) -> dict:
        """Register an agent as a user in the DB (idempotent — safe to call on resume)."""
        try:
            self._exec(
                "INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, "
                "created_at, num_followings, num_followers) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
                (agent_id, agent_id, username, name, bio, self.time_step),
                commit=True,
            )
            self._record_trace(agent_id, "sign_up",
                               {"user_name": username, "name": name, "bio": bio})
            return {"success": True, "user_id": agent_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_follow(self, follower_id: int, followee_id: int) -> dict:
        """Create an initial follow relationship (used during agent setup)."""
        return self.follow(follower_id, followee_id)

    # ── Feed ────────────────────────────────────────────────────────────

    def get_feed(self, user_id: int) -> dict:
        """Return a mixed feed of recsys + following posts for a user.

        Combines up to ``refresh_rec_post_count`` posts from the rec table
        with up to ``following_post_count`` posts from followed users,
        deduplicates, and returns full post dicts with comments attached.
        """
        # Posts from rec table.
        self._exec("SELECT post_id FROM rec WHERE user_id = ?", (user_id,))
        rec_ids = [r[0] for r in self._cur.fetchall()]
        if len(rec_ids) >= self.refresh_rec_post_count:
            rec_ids = random.sample(rec_ids, self.refresh_rec_post_count)

        # Posts from followed users (sorted by likes — most-liked first).
        self._exec(
            "SELECT post.post_id FROM post "
            "JOIN follow ON post.user_id = follow.followee_id "
            "WHERE follow.follower_id = ? "
            "ORDER BY post.num_likes DESC LIMIT ?",
            (user_id, self.following_post_count),
        )
        following_ids = [r[0] for r in self._cur.fetchall()]

        # Merge, deduplicate, preserve following-first order.
        seen: set[int] = set()
        post_ids: list[int] = []
        for pid in following_ids + rec_ids:
            if pid not in seen:
                seen.add(pid)
                post_ids.append(pid)

        # Fallback: if rec table is empty or unfollowed, serve recent original
        # posts from anyone so agents always have something to act on.
        if not post_ids:
            self._exec(
                "SELECT post_id FROM post "
                "WHERE original_post_id IS NULL AND user_id != ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, self.following_post_count + self.refresh_rec_post_count),
            )
            post_ids = [r[0] for r in self._cur.fetchall()]

        if not post_ids:
            return {"success": True, "posts": []}

        placeholders = ",".join("?" * len(post_ids))
        self._exec(
            f"SELECT post_id, user_id, original_post_id, content, quote_content,"
            f" created_at, num_likes, num_dislikes, num_shares "
            f"FROM post WHERE post_id IN ({placeholders})",
            tuple(post_ids),
        )
        rows = self._cur.fetchall()
        posts = self._posts_with_comments(rows)
        self._record_trace(user_id, "refresh", {"post_count": len(posts)})
        return {"success": True, "posts": posts}

    # ── Actions ─────────────────────────────────────────────────────────

    def create_post(self, user_id: int, content: str) -> dict:
        try:
            self._exec(
                "INSERT INTO post (user_id, content, created_at, "
                "num_likes, num_dislikes, num_shares) VALUES (?, ?, ?, 0, 0, 0)",
                (user_id, content, self.time_step),
                commit=True,
            )
            post_id = self._cur.lastrowid
            self._record_trace(user_id, "create_post",
                               {"post_id": post_id, "content": content})
            return {"success": True, "post_id": post_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def like_post(self, user_id: int, post_id: int) -> dict:
        try:
            pt = self._post_type(post_id)
            if not pt:
                return {"success": False, "error": "Post not found."}
            # Likes always target the root post.
            target_id = pt["root_post_id"]

            self._exec(
                "SELECT like_id FROM 'like' WHERE post_id = ? AND user_id = ?",
                (target_id, user_id),
            )
            if self._cur.fetchone():
                return {"success": False, "error": "Like record already exists."}

            self._exec(
                "UPDATE post SET num_likes = num_likes + 1 WHERE post_id = ?",
                (target_id,), commit=True,
            )
            self._exec(
                "INSERT INTO 'like' (post_id, user_id, created_at) VALUES (?,?,?)",
                (target_id, user_id, self.time_step), commit=True,
            )
            like_id = self._cur.lastrowid
            self._record_trace(user_id, "like_post",
                               {"post_id": target_id, "like_id": like_id})
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def follow(self, follower_id: int, followee_id: int) -> dict:
        if follower_id == followee_id:
            return {"success": False, "error": "Cannot follow yourself."}
        try:
            self._exec(
                "SELECT follow_id FROM follow WHERE follower_id = ? AND followee_id = ?",
                (follower_id, followee_id),
            )
            if self._cur.fetchone():
                return {"success": False, "error": "Follow record already exists."}

            self._exec(
                "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?,?,?)",
                (follower_id, followee_id, self.time_step), commit=True,
            )
            follow_id = self._cur.lastrowid
            self._exec(
                "UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?",
                (follower_id,), commit=True,
            )
            self._exec(
                "UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?",
                (followee_id,), commit=True,
            )
            self._record_trace(follower_id, "follow",
                               {"followee_id": followee_id, "follow_id": follow_id})
            return {"success": True, "follow_id": follow_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_comment(self, user_id: int, post_id: int, content: str) -> dict:
        try:
            pt = self._post_type(post_id)
            if not pt:
                return {"success": False, "error": "Post not found."}
            target_id = pt["root_post_id"]

            self._exec(
                "INSERT INTO comment (post_id, user_id, content, created_at, num_likes) "
                "VALUES (?, ?, ?, ?, 0)",
                (target_id, user_id, content, self.time_step), commit=True,
            )
            comment_id = self._cur.lastrowid
            self._record_trace(user_id, "create_comment",
                               {"post_id": target_id, "comment_id": comment_id,
                                "content": content})
            return {"success": True, "comment_id": comment_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def repost(self, user_id: int, post_id: int) -> dict:
        try:
            pt = self._post_type(post_id)
            if not pt:
                return {"success": False, "error": "Post not found."}
            root_id = pt["root_post_id"]

            # Check for existing repost of the root.
            self._exec(
                "SELECT post_id FROM post WHERE original_post_id = ? AND user_id = ? "
                "AND (quote_content IS NULL OR TRIM(quote_content) = '')",
                (root_id, user_id),
            )
            if self._cur.fetchone():
                return {"success": False, "error": "Repost record already exists."}

            self._exec(
                "INSERT INTO post (user_id, original_post_id, created_at) VALUES (?,?,?)",
                (user_id, root_id, self.time_step), commit=True,
            )
            new_post_id = self._cur.lastrowid
            self._exec(
                "UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?",
                (root_id,), commit=True,
            )
            self._record_trace(user_id, "repost",
                               {"post_id": root_id, "new_post_id": new_post_id})
            return {"success": True, "post_id": new_post_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def quote_post(self, user_id: int, post_id: int, quote_content: str) -> dict:
        try:
            pt = self._post_type(post_id)
            if not pt:
                return {"success": False, "error": "Post not found."}
            root_id = pt["root_post_id"]

            self._exec("SELECT content FROM post WHERE post_id = ?", (root_id,))
            row = self._cur.fetchone()
            original_content = row[0] if row else ""

            self._exec(
                "INSERT INTO post (user_id, original_post_id, content, quote_content,"
                " created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, root_id, original_content, quote_content, self.time_step),
                commit=True,
            )
            new_post_id = self._cur.lastrowid
            self._exec(
                "UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?",
                (root_id,), commit=True,
            )
            self._record_trace(user_id, "quote_post",
                               {"post_id": root_id, "new_post_id": new_post_id})
            return {"success": True, "post_id": new_post_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def do_nothing(self, user_id: int) -> dict:
        self._record_trace(user_id, "do_nothing", {})
        return {"success": True}

    # ── Recommendation system ───────────────────────────────────────────

    def update_rec_table(self) -> None:
        """Re-run the recommendation algorithm and refresh the rec table."""
        from .db_utils import fetch_table_from_db, fetch_rec_table_as_matrix
        from .recsys import run_recsys

        platform_log.info("Refreshing recommendation table...")
        user_table = fetch_table_from_db(self._cur, "user")
        post_table = fetch_table_from_db(self._cur, "post")
        trace_table = fetch_table_from_db(self._cur, "trace")
        rec_matrix = fetch_rec_table_as_matrix(self._cur)

        try:
            new_rec = run_recsys(
                recsys_type=self.recsys_type,
                user_table=user_table,
                post_table=post_table,
                trace_table=trace_table,
                rec_matrix=rec_matrix,
                max_rec_post_len=self.max_rec_post_len,
                time_step=int(self.time_step),
                use_openai_embedding=self.use_openai_embedding,
            )
        except Exception as e:
            platform_log.error("update_rec_table failed: %s", e, exc_info=True)
            return

        self._exec("DELETE FROM rec", commit=True)
        insert_vals = [
            (user_id, post_id)
            for user_id, post_list in enumerate(new_rec)
            for post_id in post_list
        ]
        self._cur.executemany(
            "INSERT OR IGNORE INTO rec (user_id, post_id) VALUES (?, ?)",
            insert_vals,
        )
        self.db.commit()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Lifecycle ───────────────────────────────────────────────────────

    def close(self) -> None:
        self.db.commit()
        self._cur.close()
        self.db.close()

    def __enter__(self) -> "Platform":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
