"""
db_utils.py — lightweight SQLite helpers used by the recsys and platform.

These are lifted from oasis/oasis/social_platform/database.py so that the
simulation core no longer depends on the oasis package for these utilities.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List


def fetch_table_from_db(cursor: sqlite3.Cursor,
                        table_name: str) -> List[Dict[str, Any]]:
    """Return every row in *table_name* as a list of column-keyed dicts."""
    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def fetch_rec_table_as_matrix(cursor: sqlite3.Cursor) -> List[List[int]]:
    """Return the rec table as a list-of-lists indexed by user_id order."""
    cursor.execute("SELECT user_id FROM user ORDER BY user_id")
    user_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT user_id, post_id FROM rec ORDER BY user_id, post_id")
    user_posts: dict[int, list[int]] = {uid: [] for uid in user_ids}
    for uid, pid in cursor.fetchall():
        if uid in user_posts:
            user_posts[uid].append(pid)

    return [user_posts[uid] for uid in user_ids]
