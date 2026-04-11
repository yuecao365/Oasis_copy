"""
Agent — a single simulated user in the social media simulation.

Replaces OASIS's SocialAgent / CAMEL ChatAgent stack with a plain Python
class that has no external framework dependencies.  The key addition over
the OASIS version is *state tracking*: the agent remembers what it has
already liked, followed, commented on, and reposted so that the simulation
loop can avoid wasting action slots on operations the platform will reject
as duplicates.
"""

from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass, field
from typing import Any

# ── Tweet cleaning ───────────────────────────────────────────────────────────
# Patterns that add noise without contributing meaningful content.
_RE_URL = re.compile(r"\bURL\b|https?://\S+", re.IGNORECASE)
_RE_ISO_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
_RE_METADATA_SEP = re.compile(r"\s*-:\s*:\s*")   # " -: : " artifact
_RE_WHITESPACE = re.compile(r"\s{2,}")


def _clean_tweet(text: str) -> str:
    """Strip metadata artefacts from a raw previous_tweet string."""
    text = html.unescape(text)           # &amp; → & etc.
    text = _RE_ISO_TS.sub("", text)      # remove ISO timestamps
    text = _RE_URL.sub("", text)         # remove URL tokens
    text = _RE_METADATA_SEP.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip(" :-")


def _build_clean_pool(tweets: list[str], min_len: int = 15) -> list[str]:
    """Return the subset of tweets that still have meaningful content after cleaning."""
    cleaned = [_clean_tweet(t) for t in tweets]
    return [t for t in cleaned if len(t) >= min_len]


# Post-content variation templates (used when no clean tweets remain).
_POST_TEMPLATES = [
    "Thinking about: {bio}",
    "Hot take: {bio}",
    "Worth discussing — {bio}",
    "{bio}",
    "Something on my mind: {bio}",
]

_COMMENT_TEMPLATES = [
    "Interesting perspective.",
    "I see your point.",
    "Worth thinking about.",
    "Agreed, that's a good take.",
    "Hadn't considered that angle.",
    "Thanks for sharing this.",
]


@dataclass
class Agent:
    """A stateful simulated social-media user.

    Attributes
    ----------
    agent_id:
        Zero-based index used as the primary key in the simulation loop.
    user_id:
        The row ID in the platform's ``user`` table (set after sign-up,
        usually agent_id + 1 for 1-indexed SQLite auto-increment).
    username:
        Lowercase handle, matched against persona CSV and test data.
    name:
        Display name.
    bio:
        Free-text profile description drawn from the persona CSV.
    previous_tweets:
        Historical tweets from the real user's timeline, used as a pool
        for content generation so posts are varied and persona-faithful.
    action_probabilities:
        Mapping of action name → probability (must sum to 1.0).
        Per-agent overrides from the personas CSV take priority over the
        global config defaults.
    activity_profile:
        List of per-bin activation probabilities.
        Length 24 for hourly granularity, 7 for weekly.
    extra_info:
        Any additional persona fields (follower counts, location, etc.)
        preserved for logging and future use.

    State (mutated during simulation)
    ----------------------------------
    liked_post_ids:
        Post IDs this agent has successfully liked.  Checked before
        attempting a like so the action budget isn't wasted on duplicates.
    followed_user_ids:
        User IDs this agent follows.  Checked before attempting a follow.
    reposted_post_ids:
        Post IDs this agent has already reposted.
    commented_post_ids:
        Post IDs this agent has commented on at least once.
        (Commenting the same post twice is allowed by the platform, but
        tracking it enables future logic that limits spam.)
    posts_created:
        Number of original posts this agent has created this run.
        Used to enforce per-agent post budgets if needed.
    """

    # Identity
    agent_id: int
    user_id: int = -1          # filled in after platform sign-up
    username: str = ""
    name: str = ""
    bio: str = ""

    # Content pool & behavioural config
    previous_tweets: list[str] = field(default_factory=list)
    action_probabilities: dict[str, float] = field(default_factory=dict)
    activity_profile: list[float] = field(default_factory=list)
    extra_info: dict[str, Any] = field(default_factory=dict)

    # ── Runtime state (not part of persona definition) ─────────────────
    liked_post_ids: set[int] = field(default_factory=set)
    followed_user_ids: set[int] = field(default_factory=set)
    reposted_post_ids: set[int] = field(default_factory=set)
    commented_post_ids: set[int] = field(default_factory=set)
    posts_created: int = 0

    # Clean tweet pool and rotation cursor — populated on first use.
    _clean_pool: list[str] = field(default_factory=list, repr=False)
    _pool_index: int = field(default=0, repr=False)
    _pool_ready: bool = field(default=False, repr=False)

    # ── Activation ─────────────────────────────────────────────────────

    def activation_probability(self, bin_index: int) -> float:
        """Return the raw activation probability for a time bin.

        Args:
            bin_index: Hour-of-day (0-23) or day-of-week (0-6) depending
                       on the granularity configured for the simulation.

        Returns:
            Float in [0, 1].  Returns 0.0 if the profile is empty.
        """
        if not self.activity_profile:
            return 0.0
        idx = bin_index % len(self.activity_profile)
        return float(self.activity_profile[idx])

    def should_activate(self, bin_index: int, multiplier: float = 1.0,
                        min_threshold: float = 0.0) -> bool:
        """Sample whether this agent activates on a given time step.

        Args:
            bin_index:     Time bin (hour or weekday).
            multiplier:    Scales the base probability up (>1) or down.
            min_threshold: Floor applied after scaling, ensuring sparse
                           users still get a minimum activation chance.

        Returns:
            True if the agent should act this step.
        """
        base = self.activation_probability(bin_index)
        prob = max(min_threshold, min(1.0, base * multiplier))
        return random.random() < prob

    # ── Content generation ──────────────────────────────────────────────

    def _ensure_pool(self) -> None:
        """Build and shuffle the clean tweet pool on first call."""
        if not self._pool_ready:
            pool = _build_clean_pool(self.previous_tweets)
            if pool:
                random.shuffle(pool)
            self._clean_pool = pool
            self._pool_index = 0
            self._pool_ready = True

    def _next_tweet(self, max_len: int = 280) -> str | None:
        """Return the next tweet from the rotating pool, or None if empty.

        Advances the pool index so consecutive calls return different tweets,
        wrapping around and re-shuffling when the pool is exhausted to prevent
        long repetitive runs.
        """
        self._ensure_pool()
        if not self._clean_pool:
            return None
        tweet = self._clean_pool[self._pool_index % len(self._clean_pool)]
        self._pool_index += 1
        # Re-shuffle when we wrap around so the next cycle is in a new order.
        # Ensure the first tweet after shuffle differs from the last before it.
        if self._pool_index % len(self._clean_pool) == 0 and len(self._clean_pool) > 1:
            last = tweet
            random.shuffle(self._clean_pool)
            if self._clean_pool[0] == last:
                # Swap first with a random other position to avoid repeat.
                j = random.randint(1, len(self._clean_pool) - 1)
                self._clean_pool[0], self._clean_pool[j] = \
                    self._clean_pool[j], self._clean_pool[0]
        return tweet[:max_len].strip()

    def _bio_fallback(self) -> str:
        """Return a varied bio-based post when no clean tweets are available."""
        bio = (self.bio or "").strip()
        snippet = bio[:160] if bio else "life"
        template = random.choice(_POST_TEMPLATES)
        return template.format(bio=snippet)

    def generate_post_content(self) -> str:
        """Generate original post content.

        Uses the cleaned, rotating tweet pool so each post is different.
        Falls back to bio-based templates when the pool is empty.
        """
        tweet = self._next_tweet(max_len=280)
        return tweet if tweet else self._bio_fallback()

    def generate_comment_content(self, post_content: str = "") -> str:
        """Generate a reply.

        Samples a short excerpt from the tweet pool when available so
        comments feel persona-specific rather than generic.
        """
        tweet = self._next_tweet(max_len=140)
        if tweet:
            return tweet
        return random.choice(_COMMENT_TEMPLATES)

    def generate_quote_content(self, post_content: str = "") -> str:
        """Generate a quote-tweet comment."""
        tweet = self._next_tweet(max_len=120)
        if tweet:
            return f"My take: {tweet}"
        return "Worth thinking about."

    # ── State helpers ───────────────────────────────────────────────────

    def can_like(self, post_id: int) -> bool:
        return post_id not in self.liked_post_ids

    def can_follow(self, user_id: int) -> bool:
        return user_id != self.user_id and user_id not in self.followed_user_ids

    def can_repost(self, post_id: int) -> bool:
        return post_id not in self.reposted_post_ids

    def record_like(self, post_id: int) -> None:
        self.liked_post_ids.add(post_id)

    def record_follow(self, user_id: int) -> None:
        self.followed_user_ids.add(user_id)

    def record_repost(self, post_id: int) -> None:
        self.reposted_post_ids.add(post_id)

    def record_comment(self, post_id: int) -> None:
        self.commented_post_ids.add(post_id)

    def record_post(self) -> None:
        self.posts_created += 1

    # ── Action sampling ─────────────────────────────────────────────────

    def sample_action(self, blocked: set[str] | None = None) -> str:
        """Sample an action from this agent's probability distribution.

        Args:
            blocked: Set of action names to exclude (e.g. {"create_post"}
                     when the post budget for this step is exhausted).

        Returns:
            Action name string, e.g. "like_post", "create_post", etc.
            Returns "do_nothing" if all actions are blocked or the
            probability dict is empty.
        """
        blocked = blocked or set()
        probs = {k: v for k, v in self.action_probabilities.items()
                 if k not in blocked and v > 0}
        if not probs:
            return "do_nothing"
        total = sum(probs.values())
        r = random.random() * total
        cumulative = 0.0
        for action, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return action
        return "do_nothing"

    # ── Representation ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id}, user_id={self.user_id}, "
            f"username={self.username!r}, posts={self.posts_created})"
        )
