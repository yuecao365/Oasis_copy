"""
LLM client for the OASIS-free Twitter simulation.

Uses the OpenAI-compatible DashScope API to decide what action a simulated
Twitter user takes at each simulation timestep.

Usage:
    client = get_client(base_url=..., api_key=..., timeout=60.0, max_retries=2)
    decision = await decide_action(
        username="...", name="...", bio="...", user_char="...",
        category_description="...",
        previous_tweets=[...], feed_posts=[...],
        available_actions=["create_post", "like_post", ...],
        action_probs={"create_post": 0.25, ...},
        model="qwen3.5-plus",
        client=client,
    )
    # decision: {"action": "create_post", "content": "...", "target_post_id": None, "target_user_id": None}
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI

_log = logging.getLogger("social")

# Module-level client singleton (keyed by (base_url, api_key) to allow multiple configs).
_clients: dict[tuple[str, str], AsyncOpenAI] = {}

_FALLBACK_DECISION: dict[str, Any] = {
    "action": "do_nothing",
    "content": "",
    "target_post_id": None,
    "target_user_id": None,
}

_VALID_ACTIONS = frozenset([
    "create_post", "create_comment", "repost", "like_post",
    "quote_post", "follow", "do_nothing",
])

_SYSTEM_PROMPT = """\
You are simulating a realistic Twitter/X user. Given their persona and their current feed, decide ONE action to take right now.

Return ONLY a valid JSON object — no markdown, no explanation, no extra text.

JSON schema:
{
  "action": "<one of the valid actions>",
  "content": "<tweet/comment/quote text, or empty string>",
  "target_post_id": <integer post_id or null>,
  "target_user_id": <integer user_id or null>
}

Valid actions and requirements:
- "create_post"    → content required; target_post_id=null; target_user_id=null
- "create_comment" → content required; target_post_id=<id of post to comment on>; target_user_id=null
- "repost"         → content="";       target_post_id=<id of post to repost>;    target_user_id=null
- "like_post"      → content="";       target_post_id=<id of post to like>;      target_user_id=null
- "quote_post"     → content required; target_post_id=<id of post to quote>;     target_user_id=null
- "follow"         → content="";       target_post_id=null; target_user_id=<id of user to follow>
- "do_nothing"     → content="";       target_post_id=null; target_user_id=null

For create_post: keep content under 280 characters and write in the user's authentic voice.
For create_comment/quote_post: keep content under 200 characters.
Only choose feed-dependent actions (like_post, repost, create_comment, quote_post, follow) when the feed is non-empty.
"""


def get_client(
    base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
    api_key: str = "",
    timeout: float = 60.0,
    max_retries: int = 2,
) -> AsyncOpenAI:
    """Return (or create) the AsyncOpenAI singleton for this (base_url, api_key) pair."""
    key = (base_url, api_key)
    if key not in _clients:
        _clients[key] = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    return _clients[key]


def _build_user_message(
    username: str,
    user_char: str,
    bio: str,
    category_description: str,
    previous_tweets: list[str],
    feed_posts: list[dict[str, Any]],
    available_actions: list[str],
    action_probs: dict[str, float],
) -> str:
    parts: list[str] = []

    # --- Persona block ---
    parts.append("## Your Persona")
    parts.append(f"Username: @{username}")
    if bio:
        parts.append(f"Bio: {bio}")
    if user_char:
        parts.append(f"Character: {user_char[:600]}")
    if category_description:
        parts.append(f"Topics you care about: {category_description[:300]}")

    # --- Writing style ---
    sample_tweets = [t.strip() for t in previous_tweets if t.strip()][:3]
    if sample_tweets:
        parts.append("\n## Your Writing Style (sample past tweets)")
        for t in sample_tweets:
            parts.append(f'- "{t[:180]}"')

    # --- Feed ---
    if feed_posts:
        parts.append("\n## Your Current Feed")
        for p in feed_posts[:8]:
            pid = p.get("post_id", "?")
            uid = p.get("user_id", "?")
            content = str(p.get("content", "")).strip()[:200]
            likes = p.get("num_likes", 0)
            parts.append(f'[post_id={pid}, author_id={uid}] "{content}" (likes: {likes})')
    else:
        parts.append("\n## Your Current Feed\n(empty — no posts visible yet)")

    # --- Action hints ---
    parts.append(f"\n## Available Actions\n{', '.join(available_actions)}")
    if action_probs:
        sorted_probs = sorted(action_probs.items(), key=lambda kv: -kv[1])
        hint = ", ".join(f"{a} ({v:.0%})" for a, v in sorted_probs if v > 0)
        if hint:
            parts.append(f"Your typical tendencies: {hint}")

    parts.append("\nChoose ONE action that authentically fits this user's persona right now.")
    return "\n".join(parts)


def _extract_json(text: str) -> str:
    """Strip ```json ... ``` markdown fences if present."""
    text = text.strip()
    # Remove opening fence
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```\s*$", "", text)
    # If there's still JSON embedded in surrounding text, try to extract the first {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()


def _validate_decision(
    decision: dict[str, Any],
    available_actions: list[str],
    feed_post_ids: set[int],
    feed_user_ids: set[int],
) -> dict[str, Any]:
    """Validate and normalize an LLM-produced decision dict. Returns fallback on any issue."""
    result = {
        "action": "do_nothing",
        "content": "",
        "target_post_id": None,
        "target_user_id": None,
    }

    action = str(decision.get("action", "")).strip()
    if action not in _VALID_ACTIONS:
        return result
    if available_actions and action not in available_actions and action != "do_nothing":
        return result

    result["action"] = action
    result["content"] = str(decision.get("content") or "").strip()

    # Validate/normalize target_post_id
    raw_pid = decision.get("target_post_id")
    if raw_pid is not None:
        try:
            pid = int(raw_pid)
            # Only accept if it's actually in the feed (prevents hallucinations)
            if feed_post_ids and pid in feed_post_ids:
                result["target_post_id"] = pid
        except (TypeError, ValueError):
            pass

    # Validate/normalize target_user_id
    raw_uid = decision.get("target_user_id")
    if raw_uid is not None:
        try:
            uid = int(raw_uid)
            if feed_user_ids and uid in feed_user_ids:
                result["target_user_id"] = uid
        except (TypeError, ValueError):
            pass

    # Enforce required fields; fall back to do_nothing if missing
    needs_post_id = action in ("create_comment", "repost", "like_post", "quote_post")
    needs_user_id = action == "follow"
    needs_content = action in ("create_post", "create_comment", "quote_post")

    if needs_post_id and result["target_post_id"] is None:
        # Pick the first feed post as a fallback if available
        if feed_post_ids:
            result["target_post_id"] = next(iter(sorted(feed_post_ids)))
        else:
            return {**_FALLBACK_DECISION}

    if needs_user_id and result["target_user_id"] is None:
        if feed_user_ids:
            result["target_user_id"] = next(iter(sorted(feed_user_ids)))
        else:
            return {**_FALLBACK_DECISION}

    if needs_content and not result["content"]:
        return {**_FALLBACK_DECISION}

    return result


async def decide_action(
    username: str,
    name: str,
    bio: str,
    user_char: str,
    category_description: str,
    previous_tweets: list[str],
    feed_posts: list[dict[str, Any]],
    available_actions: list[str],
    action_probs: dict[str, float],
    model: str,
    client: AsyncOpenAI,
) -> dict[str, Any]:
    """
    Call the LLM to decide what action a simulated Twitter user takes.

    Returns a dict with keys: action, content, target_post_id, target_user_id.
    Never raises — returns {"action": "do_nothing", ...} on any failure.
    """
    feed_post_ids: set[int] = set()
    feed_user_ids: set[int] = set()
    for p in feed_posts:
        try:
            feed_post_ids.add(int(p["post_id"]))
        except (KeyError, TypeError, ValueError):
            pass
        try:
            feed_user_ids.add(int(p["user_id"]))
        except (KeyError, TypeError, ValueError):
            pass

    # Filter available_actions to only feed-dependent ones when feed is empty
    effective_actions = list(available_actions) if available_actions else list(_VALID_ACTIONS)
    if not feed_posts:
        feed_dependent = {"like_post", "create_comment", "repost", "quote_post", "follow"}
        effective_actions = [a for a in effective_actions if a not in feed_dependent]
    if not effective_actions:
        effective_actions = ["create_post", "do_nothing"]

    user_message = _build_user_message(
        username=username,
        user_char=user_char,
        bio=bio,
        category_description=category_description,
        previous_tweets=previous_tweets,
        feed_posts=feed_posts,
        available_actions=effective_actions,
        action_probs=action_probs,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=400,
            temperature=0.8,
            extra_body={"enable_thinking": False},
        )
        raw_text = response.choices[0].message.content or ""
        json_text = _extract_json(raw_text)
        decision = json.loads(json_text)
        return _validate_decision(
            decision=decision,
            available_actions=effective_actions,
            feed_post_ids=feed_post_ids,
            feed_user_ids=feed_user_ids,
        )
    except Exception as exc:
        _log.warning("LLM decide_action failed for @%s: %s", username, exc)
        return {**_FALLBACK_DECISION}
