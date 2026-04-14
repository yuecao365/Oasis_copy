#!/usr/bin/env python3
"""
Generate personas for users using hybrid approach:
- Optional Twitter API for location/created_at hints
- LLM inference for demographics (age, gender, MBTI, country, profession)
- LLM generation for detailed persona
"""

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import DRY_RUN
import hashlib
import re

# Ensure repo root is on sys.path so llm_client is importable.
_REPO_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_DIR))

from llm_client import chat_completion, LLM_MODEL

# NOTE: This points to the outputs directory under the script's folder, not the repo root.
REPO_ROOT = Path(__file__).resolve().parent / "outputs"

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")  # Bearer token for Twitter API v2
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "20260215"))
NOISE_EPS = float(os.getenv("NOISE_EPS", "0.05"))  # for priors noise amplitude

# -----------------------------
# Diversity controls: add tweet sampling + prompt variants
# -----------------------------
MAX_TWEETS_DEMO = int(os.getenv("MAX_TWEETS_DEMO", "24"))      # tweets shown to demographics prompt
MAX_TWEETS_PERSONA = int(os.getenv("MAX_TWEETS_PERSONA", "28")) # tweets shown to persona prompt
MIN_TWEETS = int(os.getenv("MIN_TWEETS", "10"))               # if available
SHUFFLE_TWEETS = os.getenv("SHUFFLE_TWEETS", "1") == "1"

# Balancing sample controls
BALANCE_USERS_BY_ACTIVITY = os.getenv("BALANCE_USERS_BY_ACTIVITY", "1") == "1"
USERS_PER_CLUSTER = int(os.getenv("USERS_PER_CLUSTER", "40"))
LOW_ACTIVITY_FRACTION = float(os.getenv("LOW_ACTIVITY_FRACTION", "0.35"))
MID_ACTIVITY_FRACTION = float(os.getenv("MID_ACTIVITY_FRACTION", "0.35"))
HIGH_ACTIVITY_FRACTION = float(os.getenv("HIGH_ACTIVITY_FRACTION", "0.30"))


# Parallelism controls (major speedup with DashScope)
PERSONA_WORKERS = int(os.getenv("PERSONA_WORKERS", "6"))

# Reduce calls: by default, do a single LLM call per user (demographics + persona).
# Set to "0" to revert to legacy behavior (2 calls per user).
SINGLE_CALL_PER_USER = os.getenv("SINGLE_CALL_PER_USER", "1") == "1"

DEMO_PROMPT_VARIANTS = [
    # Variant A: emphasize linguistic/cultural cues
    "Prioritize linguistic/cultural cues (spelling, slang, holidays, sports, local institutions) and avoid defaulting to US unless evidence suggests it.",
    # Variant B: emphasize life stage & career signals
    "Prioritize life-stage and career signals (school, entry-level, parenting, management, retirement) and avoid clustering ages around late 20s/30s.",
    # Variant C: emphasize interests-to-profession mapping
    "Infer profession primarily from recurring topics/tools/industries mentioned and be willing to select less-common professions if plausible.",
    # Variant D: emphasize uncertainty and diversity
    "If evidence is weak for gender/country, choose a plausible non-default option; ensure outputs vary across users and avoid repeating the same MBTI frequently.",
    # Variant E: emphasize writing tone and online behavior
    "Use writing tone (formality, humor, emoji use, meme references) and online behavior to infer age/MBTI; ensure wide variance across users."
]

# -----------------------------
# Persona prompt: single polished instruction (no random variants)
# Combines all prior variants + adds explicit diversity constraints
# -----------------------------
PERSONA_PROMPT_MASTER = """
You must satisfy ALL of the following requirements in the bio + persona:

1) Grounding in routine & habits:
- Ground the persona in plausible daily routines and online habits.
- Include exactly one specific recurring habit (e.g., morning news scan, weekend hobby) that is consistent with the tweets.
- Do not invent implausible life facts.

2) Values & motivations:
- Center part of the persona around values/motivations and what they seek from social media.
- Include one explicit “why” behind their posting behavior.

3) Community participation:
- Describe how they engage with communities (replying, lurking, sharing resources, debating).
- Include a distinct engagement style (e.g., “helpful explainer,” “debate-driven,” “signal booster,” “quiet lurker who posts threads”).

4) Concise, punchy bio:
- Make the bio punchy and specific.
- Avoid generic phrases like “passionate about” unless strongly supported by the tweets.
- Prefer concrete topics, signals, and tone that match the user’s writing.

5) Light humor + realism:
- Use a light, witty tone where it fits the user’s tweets (don’t force humor if their tweets are formal).
- Include one quirky but believable detail consistent with the tweets.

6) Analytical/professional signal:
- Also incorporate an analytical/professional angle: highlight expertise areas and how they approach discussions (e.g., “evidence-driven,” “hands-on builder,” “big-picture strategist”), as supported by tweets.


7) Diffusion & engagement dynamics (critical):
- Describe not only what they post, but how their content tends to spread.
- Mention whether they amplify others (reposts/quote posts), spark replies, or mostly broadcast.
- Include one plausible “spread mechanism” (e.g., sharing links, signal-boosting certain accounts, asking questions that trigger replies).


DIVERSITY REQUIREMENT (critical):
- Do not use a fixed template. Vary sentence structure and voice across users.
- Let the user’s tweet tone drive the voice (more formal vs. meme-y vs. earnest).
- Prefer specific details anchored to the tweets over broad labels.
- Avoid repeating the same phrasing patterns across different personas (e.g., do not always start with “This user is…”).
"""

INVALID_CATEGORIES = {"", "unspecified", "unknown", "n/a", "none", "null"}

AVAILABLE_ACTIONS = [
    "create_post",
    "create_comment",
    "do_nothing",
    "repost",
    "quote_post",
    "like_post",
    "follow",
]

# Optional: baseline behavioral priors (tweak as you like)
DEFAULT_ACTION_PRIORS = {
    "create_post": 0.10,
    "create_comment": 0.14,
    "do_nothing": 0.42,
    "repost": 0.06,
    "quote_post": 0.04,
    "like_post": 0.20,
    "follow": 0.04,
}


ACTION_PRIOR_SMOOTHING = float(os.getenv("ACTION_PRIOR_SMOOTHING", "0.35"))
# 0.35 means: 65% user priors + 35% global priors

MIN_ACTION_FLOORS = {
    "repost": float(os.getenv("MIN_P_REPOST", "0.02")),
    "quote_post": float(os.getenv("MIN_P_QUOTE", "0.01")),
    "like_post": float(os.getenv("MIN_P_LIKE", "0.08")),
    "create_comment": float(os.getenv("MIN_P_REPLY", "0.04")),
}

ACTION_CONCENTRATION = float(os.getenv("ACTION_CONCENTRATION", "12.0"))
# (was 30.0) -> more diverse action profiles per persona

GLOBAL_ACTION_PRIORS = DEFAULT_ACTION_PRIORS.copy()

def infer_engagement_profile_from_tweets(tweets: List[str]) -> Dict[str, float]:
    if not tweets:
        return {"replyiness": 0.0, "amplifier": 0.0, "link_sharer": 0.0, "mention_rate": 0.0}

    n = max(1, len(tweets))
    replies = sum(1 for t in tweets if t.strip().startswith("@"))
    rts = sum(1 for t in tweets if t.strip().startswith("RT ") or " RT @" in t or "via @" in t)
    links = sum(1 for t in tweets if "http://" in t or "https://" in t)

    mentions = set()
    for t in tweets:
        mentions.update(re.findall(r"@\w+", t))
    

    return {
        "replyiness": replies / n if n > 0 else 0.0,
        "amplifier": rts / n if n > 0 else 0.0,
        "link_sharer": links / n if n > 0 else 0.0,
        "mention_rate": len(mentions) / n if n > 0 else 0.0
    }

def _normalize_dist(d: Dict[str, float]) -> Dict[str, float]:
    d = {k: max(1e-9, float(v)) for k, v in d.items()}
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}

def blend_action_priors(
    user_priors: Dict[str, float],
    global_priors: Dict[str, float],
    smoothing: float = 0.35
) -> Dict[str, float]:
    """
    Blends user priors with global priors to prevent collapse (e.g., repost->0).
    smoothing = weight on global priors.
    """
    user_priors = _normalize_dist(user_priors)
    global_priors = _normalize_dist(global_priors)
    out = {}
    for a in AVAILABLE_ACTIONS:
        up = user_priors.get(a, 0.0)
        gp = global_priors.get(a, 0.0)
        out[a] = (1.0 - smoothing) * up + smoothing * gp
    return _normalize_dist(out)

def apply_action_floors(priors: Dict[str, float], floors: Dict[str, float]) -> Dict[str, float]:
    priors = priors.copy()
    for k, floor in floors.items():
        if k in priors:
            priors[k] = max(priors[k], float(floor))
    return _normalize_dist(priors)


def extract_style_markers(tweets: List[str], k: int = 6) -> List[str]:
    """
    Pull lightweight, non-invasive style anchors: hashtags, emojis, common tokens, punctuation habits.
    """
    if not tweets:
        return []
    text = " ".join(tweets)
    hashtags = re.findall(r"#\w+", text)
    mentions = re.findall(r"@\w+", text)
    emojis = re.findall(r"[\U0001F300-\U0001FAFF]", text)

    # simple token freq (skip very short)
    toks = re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower())
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    common = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:k]]

    markers = []
    markers += hashtags[: min(2, len(hashtags))]
    markers += emojis[: min(2, len(emojis))]
    markers += common[: max(0, k - len(markers))]

    # de-dupe preserve
    seen = set()
    out = []
    for m in markers:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out[:k]


def sample_action_probabilities(
    rng: random.Random,
    priors: Optional[Dict[str, float]] = None,
    actions: Optional[List[str]] = None,
    concentration: float = 30.0,
) -> Dict[str, float]:
    """
    Sample a probability distribution over actions.
    Deterministic given rng.

    Uses a Dirichlet-like approach via Gamma sampling:
      alpha_i = concentration * prior_i (or uniform if no priors)
      p_i ~ Gamma(alpha_i, 1); normalize.

    Higher concentration => closer to priors (less variance).
    Lower concentration => more diverse behavior per persona.
    """
    actions = actions or AVAILABLE_ACTIONS
    if priors is None:
        priors = {a: 1.0 for a in actions}

    # Ensure all actions present and positive
    weights = []
    for a in actions:
        w = float(priors.get(a, 0.0))
        if w <= 0:
            w = 1e-6
        weights.append(w)

    s = sum(weights)
    base = [w / s for w in weights]

    # Dirichlet(alpha) via Gamma draws
    alphas = [max(1e-3, concentration * b) for b in base]
    draws = [rng.gammavariate(alpha, 1.0) for alpha in alphas]
    total = sum(draws)
    probs = [d / total for d in draws]

    return {a: float(p) for a, p in zip(actions, probs)}


def _balanced_pick_users(top_users: List[Dict], rng: random.Random, k: int) -> List[Dict]:
    if len(top_users) <= k:
        return top_users

    # sort by tweet_count
    sorted_users = sorted(top_users, key=lambda u: u.get("tweet_count", 0))
    n = len(sorted_users)
    lo = sorted_users[: max(1, n // 3)]
    mid = sorted_users[max(1, n // 3) : max(2, 2 * n // 3)]
    hi = sorted_users[max(2, 2 * n // 3) :]

    k_lo = int(k * LOW_ACTIVITY_FRACTION)
    k_mid = int(k * MID_ACTIVITY_FRACTION)
    k_hi = max(0, k - k_lo - k_mid)

    picked = []
    picked += rng.sample(lo, min(k_lo, len(lo)))
    picked += rng.sample(mid, min(k_mid, len(mid)))
    picked += rng.sample(hi, min(k_hi, len(hi)))

    # if still short, fill from remaining
    if len(picked) < k:
        remaining = [u for u in sorted_users if u not in picked]
        picked += rng.sample(remaining, min(k - len(picked), len(remaining)))

    return picked



def infer_action_priors_from_tweets(tweets: List[str]) -> Dict[str, float]:
    """
    Infer per-user action priors from tweet text heuristics.
    """
    if not tweets:
        return DEFAULT_ACTION_PRIORS.copy()

    n = len(tweets)
    replies = 0
    reposts = 0
    originals = 0

    for t in tweets:
        s = t.strip()
        if not s:
            continue
        if s.startswith("RT ") or " RT @" in s or "via @" in s:
            reposts += 1
        elif s.startswith("@"):
            replies += 1
        else:
            originals += 1

    reply_ratio = replies / n
    repost_ratio = reposts / n
    post_ratio = originals / n

    # Base priors from observed behavior
    priors = {
        "create_post": post_ratio,
        "create_comment": reply_ratio,
        "repost": repost_ratio,
        "quote_post": repost_ratio * 0.4,
        "like_post": 0.15 + 0.5 * (reply_ratio + repost_ratio),
        "follow": 0.05 + 0.2 * reply_ratio,
        "do_nothing": max(0.05, 1.0 - (reply_ratio + repost_ratio + post_ratio)),
    }

    # Clamp and normalize
    priors = {k: max(1e-6, float(v)) for k, v in priors.items()}
    s = sum(priors.values())
    return {k: v / s for k, v in priors.items()}


def normalize_category(cat: Optional[str]) -> Optional[str]:
    """Return a cleaned category string, or None if invalid/unspecified."""
    if cat is None:
        return None
    c = str(cat).strip()
    if not c:
        return None
    if c.lower() in INVALID_CATEGORIES:
        return None
    return c

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def category_label_from_topic(topic: str) -> str:
    """
    Extract a short category label from a long topic description.
    """
    t = normalize_category(topic) or ""
    if not t:
        return "general"

    tl = t.lower()

    # Keyword-based labeling for common clusters in this dataset.
    if "vebetterdao" in tl or "b3tr" in tl or "vot3" in tl:
        return "VeBetterDAO"
    if "stargate" in tl or "renaissance" in tl or "galactica" in tl:
        return "VeChain Renaissance/Stargate"
    if "vechain" in tl or "$vet" in tl or "$vtho" in tl:
        return "VeChain ecosystem"
    if "altcoin" in tl or "price" in tl or "trading" in tl or "chart" in tl:
        return "Crypto markets"
    if "sustain" in tl or "esg" in tl or "climate" in tl or "recycle" in tl:
        return "Sustainability"

    # Generic fallback: strip boilerplate lead-ins.
    cleaned = t
    cleaned = re.sub(r"^\s*the topic (centers around|revolves around)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*discussions?\s*(highlight|focus)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # Take text before em dash or colon
    for sep in ["—", ":", "-"]:
        if sep in cleaned:
            cleaned = cleaned.split(sep)[0].strip()
            break

    return cleaned[:50].strip() or "general"



def select_tweets_for_prompt(
    tweets: List[str],
    rng: Optional[random.Random],
    max_tweets: int,
    min_tweets: int = 10,
    shuffle: bool = True,
) -> List[str]:
    """
    Deterministically select a subset of tweets to increase diversity and reduce averaging.
    - If tweets are many, sample max_tweets.
    - If tweets are few, keep as many as available.
    - Shuffle (deterministic if rng provided).
    """
    if not tweets:
        return []

    tweets_clean = [t.strip() for t in tweets if t and t.strip()]
    if not tweets_clean:
        return []

    n = len(tweets_clean)

    # If plenty of tweets, sample a subset deterministically
    if rng and n > max_tweets:
        k = max(min_tweets, min(max_tweets, n))
        sampled = rng.sample(tweets_clean, k)
    else:
        sampled = tweets_clean[:max_tweets] if n > max_tweets else list(tweets_clean)

    # Shuffle order to avoid the model anchoring on early tweets
    if shuffle and rng:
        rng.shuffle(sampled)

    return sampled

@dataclass
class UserProfile:
    """User profile from Twitter API (optional)."""
    location: Optional[str] = None
    created_at: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Demographics:
    """Inferred demographics for a user."""
    age: int
    gender: str
    mbti: str
    country: str
    profession: str


@dataclass
class Persona:
    """Generated persona for a user."""
    username: str
    realname: str
    bio: str
    persona: str
    age: int
    gender: str
    mbti: str
    country: str
    profession: str
    category: str  # Primary category
    all_categories: List[str]  # All categories user appears in
    cluster_id: int
    original_user_name: str
    original_user_id: str
    tweet_count: int
    sample_tweets: List[str]  # Tweet texts for previous_tweets (OASIS simulation)
    action_probs: Dict[str, float]
    style_markers: List[str] = None
    engagement_profile: Dict[str, Any] = None


def pick_with_priors(options, priors, rng, noise_eps: float = 0.05):
    """
    Pick an option with small controlled noise around the given priors.
    - options: list[str]
    - priors: list[float] (same length as options), not necessarily normalized
    - rng: random.Random
    - noise_eps: +/- noise range added to each prior before normalization
    """
    if len(options) != len(priors):
        raise ValueError("options and priors must have the same length")

    # Add small noise but keep > 0
    noisy = [max(1e-9, p + rng.uniform(-noise_eps, noise_eps)) for p in priors]
    s = sum(noisy)
    probs = [x / s for x in noisy]
    return rng.choices(options, weights=probs, k=1)[0]


def persona_rng(user_id: str, global_seed: int = 20260215) -> random.Random:
    """
    Deterministic RNG for a user, stable across processes/machines.
    """
    msg = f"{global_seed}-{user_id}".encode("utf-8")
    h = hashlib.sha256(msg).digest()
    seed = int.from_bytes(h[:8], "big")  # use 64 bits of the hash
    return random.Random(seed)


def fetch_user_profile_from_api(user_id: str) -> Optional[UserProfile]:
    """
    Fetch user profile from Twitter API v2 (optional).

    Args:
        user_id: Twitter user ID

    Returns:
        UserProfile if API available, None otherwise
    """
    if not TWITTER_API_KEY:
        return None

    try:
        try:
            import requests
        except ImportError:
            print("Warning: requests library not installed. Install with: pip install requests")
            return None

        url = f"https://api.twitter.com/2/users/{user_id}"
        headers = {
            "Authorization": f"Bearer {TWITTER_API_KEY}",
        }
        params = {
            "user.fields": "location,created_at,description",
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            user_data = data.get("data", {})
            return UserProfile(
                location=user_data.get("location"),
                created_at=user_data.get("created_at"),
                description=user_data.get("description"),
            )
    except Exception as e:
        print(f"Warning: Failed to fetch user profile from API: {e}")

    return None


def generate_demographics_from_tweets(
    user_tweets: List[str],
    user_name: str,
    api_profile: Optional[UserProfile] = None,
    rng: Optional[random.Random] = None,
) -> Demographics:
    """
    Infer demographics from user tweets using LLM (or return synthetic values in DRY_RUN).

    Args:
        user_tweets: List of tweet texts from the user
        user_name: Username
        api_profile: Optional profile from Twitter API
        rng: Optional random.Random for reproducibility

    Returns:
        Demographics object
    """
    # -----------------------------
    # DRY RUN MODE: return fake demographics
    # -----------------------------
    if DRY_RUN:
        rng = rng or random.Random()  # if not provided, fall back to non-deterministic

        # Example priors (uniform by default). If you have real priors, plug them here.
        genders = ["male", "female", "other"]
        gender_priors = [1.0, 1.0, 0.2]  # slight tilt; tune as desired

        mbti_types = [
            "ISTJ","ISFJ","INFJ","INTJ","ISTP","ISFP","INFP","INTP",
            "ESTP","ESFP","ENFP","ENTP","ESTJ","ESFJ","ENFJ","ENTJ"
        ]
        mbti_priors = [1.0] * len(mbti_types)  # uniform unless you prefer different

        countries = ["US", "UK", "CA", "AU"]
        country_priors = [1.0, 0.6, 0.6, 0.4]  # example; edit for your dataset

        professions = [
            "Agriculture Food & Natural Resources",
            "Architecture & Construction",
            "Arts Audio/Video Technology & Communications",
            "Business Management & Administration",
            "Education & Training",
            "Finance",
            "Government & Public Administration",
            "Health Science",
            "Hospitality & Tourism",
            "Human Services",
            "Information Technology",
            "Law Public Safety Corrections & Security",
            "Manufacturing",
            "Marketing",
            "Science Technology Engineering & Mathematics",
            "Transportation Distribution & Logistics",
        ]
        profession_priors = [1.0] * len(professions)
        # Mild bias toward IT/STEAM (optional)
        profession_priors[professions.index("Information Technology")] = 1.4
        profession_priors[professions.index("Science Technology Engineering & Mathematics")] = 1.2

        fake = {
            "age": rng.randint(20, 60),
            "gender": pick_with_priors(genders, gender_priors, rng, noise_eps=NOISE_EPS),
            "mbti": pick_with_priors(mbti_types, mbti_priors, rng, noise_eps=NOISE_EPS),
            "country": pick_with_priors(countries, country_priors, rng, noise_eps=NOISE_EPS),
            "profession": pick_with_priors(professions, profession_priors, rng, noise_eps=NOISE_EPS),
        }
        return Demographics(**fake)

    # Choose a deterministic prompt variant (per user) to encourage diversity
    demo_variant = rng.choice(DEMO_PROMPT_VARIANTS) if rng else random.choice(DEMO_PROMPT_VARIANTS)

    # Sample a subset of tweets (reduces generic averaging & increases variance)
    tweets_for_prompt = select_tweets_for_prompt(
        user_tweets,
        rng=rng,
        max_tweets=MAX_TWEETS_DEMO,
        min_tweets=MIN_TWEETS,
        shuffle=SHUFFLE_TWEETS,
    )

    # Prepare context
    sample_tweets = "\n".join([f"- {tweet}" for tweet in tweets_for_prompt])

    
    style_markers = extract_style_markers(user_tweets)
    style_line = ""
    if style_markers:
        style_line = f"\nStyle anchors (must reflect in voice subtly): {', '.join(style_markers)}\n"


    api_context = ""
    if api_profile:
        api_context = f"""
Additional context from Twitter profile:
- Location: {api_profile.location or 'Not specified'}
- Account created: {api_profile.created_at or 'Not specified'}
- Bio: {api_profile.description or 'Not specified'}
"""

    prompt = f"""Analyze the following tweets from a Twitter user and infer their demographics.
    
{style_line}

User: @{user_name}
{api_context}

Diversity instruction (rotate styles across users):
{demo_variant}

Sample tweets:
{sample_tweets}

Based on the writing style, topics, language patterns, and any available profile information, infer:
1. Age (as an integer between 18-65, estimate based on writing style, topics, and language patterns. Ensure good variance across users - don't cluster around the same age)
2. Gender (one of: "male", "female", "other")
3. MBTI personality type (one of: ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ)
4. Country (ISO country code like "US", "UK", "CA", etc. - infer from location field if available, or from language/content)
5. Profession (choose from: Agriculture Food & Natural Resources, Architecture & Construction, Arts Audio/Video Technology & Communications, Business Management & Administration, Education & Training, Finance, Government & Public Administration, Health Science, Hospitality & Tourism, Human Services, Information Technology, Law Public Safety Corrections & Security, Manufacturing, Marketing, Science Technology Engineering & Mathematics, Transportation Distribution & Logistics)

IMPORTANT: For age, ensure diversity. Consider factors like:
- Writing formality and style
- References to life events (college, career milestones, family)
- Technology adoption patterns
- Cultural references
- Language complexity

Output ONLY valid JSON in this exact format:
{{
    "age": 28,
    "gender": "male",
    "mbti": "ENTP",
    "country": "US",
    "profession": "Information Technology"
}}
"""
    content = chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a demographics inference assistant. Output only valid JSON. Ensure age diversity across users.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        max_tokens=200,
    )
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    demographics_dict = json.loads(content)

    return Demographics(
        age=int(demographics_dict["age"]),
        gender=demographics_dict["gender"],
        mbti=demographics_dict["mbti"],
        country=demographics_dict["country"],
        profession=demographics_dict["profession"],
    )


def generate_demographics_and_persona_single_call(
    user_name: str,
    user_tweets: List[str],
    category_topic: str,
    all_categories: List[str],
    cluster_id: int,
    original_user_id: str,
    tweet_count: int,
    api_profile: Optional[UserProfile] = None,
    rng: Optional[random.Random] = None,
) -> Persona:
    """
    Faster path: demographics + persona in one LLM request (vs 2 sequential calls).
    """
    if DRY_RUN:
        demo = generate_demographics_from_tweets(
            user_tweets=user_tweets,
            user_name=user_name,
            api_profile=api_profile,
            rng=rng or random.Random(),
        )
        return generate_persona(
            user_name=user_name,
            demographics=demo,
            user_tweets=user_tweets,
            category_topic=category_topic,
            all_categories=all_categories,
            cluster_id=cluster_id,
            original_user_id=original_user_id,
            tweet_count=tweet_count,
            rng=rng or random.Random(),
        )

    category_topic = normalize_category(category_topic) or "general"
    all_categories = [normalize_category(c) for c in all_categories]
    all_categories = [c for c in all_categories if c]
    all_categories = dedupe_preserve_order(all_categories)
    if category_topic not in all_categories:
        all_categories.insert(0, category_topic)

    demo_variant = rng.choice(DEMO_PROMPT_VARIANTS) if rng else random.choice(DEMO_PROMPT_VARIANTS)

    tweets_for_prompt = select_tweets_for_prompt(
        user_tweets,
        rng=rng,
        max_tweets=max(MAX_TWEETS_DEMO, MAX_TWEETS_PERSONA),
        min_tweets=MIN_TWEETS,
        shuffle=SHUFFLE_TWEETS,
    )
    sample_tweets = "\n".join([f"- {tweet}" for tweet in tweets_for_prompt])

    api_context = ""
    if api_profile:
        api_context = f"""
Additional context from Twitter profile:
- Location: {api_profile.location or 'Not specified'}
- Account created: {api_profile.created_at or 'Not specified'}
- Bio: {api_profile.description or 'Not specified'}
"""

    if len(all_categories) > 1:
        categories_text = (
            f"Primary category: {category_topic}. Also active in: "
            f"{', '.join([c for c in all_categories if c != category_topic])}"
        )
        category_instruction = (
            f"the categories: {', '.join(all_categories)}. "
            "The persona should reflect their multi-faceted interests across these areas."
        )
    else:
        categories_text = category_topic
        category_instruction = f"the category: {category_topic}"

    prompt = f"""You will infer demographics AND write the persona for a Twitter user.

User: @{user_name}
{api_context}

Diversity instruction (rotate styles across users):
{demo_variant}

Interest Categories: {categories_text}

Persona writing requirements:
{PERSONA_PROMPT_MASTER}

Sample tweets:
{sample_tweets}

Infer demographics (grounded in tweets/profile, avoid US-by-default):
- age: integer 18-65
- gender: one of "male", "female", "other"
- mbti: one of ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ
- country: ISO code like "US", "UK", "CA", etc.
- profession: choose exactly one from:
  Agriculture Food & Natural Resources, Architecture & Construction, Arts Audio/Video Technology & Communications,
  Business Management & Administration, Education & Training, Finance, Government & Public Administration, Health Science,
  Hospitality & Tourism, Human Services, Information Technology, Law Public Safety Corrections & Security, Manufacturing,
  Marketing, Science Technology Engineering & Mathematics, Transportation Distribution & Logistics

Then generate:
1) bio (1-2 sentences, Twitter bio style) reflecting interests in {category_instruction}
2) persona (2-3 sentences) describing personality, habits, motivations, and how they engage with {category_instruction}

Output ONLY valid JSON in this exact format:
{{
  "age": 28,
  "gender": "male",
  "mbti": "ENTP",
  "country": "US",
  "profession": "Information Technology",
  "bio": "…",
  "persona": "…"
}}
"""

    content = chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a demographics + persona assistant. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=520,
    )

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    obj = json.loads(content)

    demo = Demographics(
        age=int(obj["age"]),
        gender=str(obj["gender"]),
        mbti=str(obj["mbti"]),
        country=str(obj["country"]),
        profession=str(obj["profession"]),
    )

    tweet_based_priors = infer_action_priors_from_tweets(user_tweets)

    # NEW: blend + floors
    priors = blend_action_priors(tweet_based_priors, GLOBAL_ACTION_PRIORS, smoothing=ACTION_PRIOR_SMOOTHING)
    priors = apply_action_floors(priors, MIN_ACTION_FLOORS)

    action_probs = sample_action_probabilities(
        rng=rng or random.Random(),
        priors=priors,
        actions=AVAILABLE_ACTIONS,
    )

    eng_profile = infer_engagement_profile_from_tweets(user_tweets)
    style_markers = extract_style_markers(user_tweets)

    return Persona(
        username=user_name,
        realname=user_name,
        bio=str(obj["bio"]),
        persona=str(obj["persona"]),
        age=demo.age,
        gender=demo.gender,
        mbti=demo.mbti,
        country=demo.country,
        profession=demo.profession,
        category=category_topic,
        all_categories=all_categories,
        cluster_id=cluster_id,
        original_user_name=user_name,
        original_user_id=original_user_id,
        tweet_count=tweet_count,
        sample_tweets=user_tweets,
        action_probs=action_probs,
        style_markers=style_markers,
        engagement_profile=eng_profile,
    )


def generate_persona(
    user_name: str,
    demographics: Demographics,
    user_tweets: List[str],
    category_topic: str,
    all_categories: List[str],
    cluster_id: int,
    original_user_id: str,
    tweet_count: int,
    rng: Optional[random.Random] = None,
) -> Persona:
    """
    Generate detailed persona for a user from demographics and sample tweets.

    Returns:
        Persona object (DRY_RUN returns a minimal persona with placeholders).
    """
    # -----------------------------
    # DRY RUN MODE: dummy persona
    # -----------------------------
    if DRY_RUN:
        category_topic = normalize_category(category_topic) or "general"
        all_categories = [normalize_category(c) for c in all_categories]
        all_categories = [c for c in all_categories if c]
        all_categories = dedupe_preserve_order(all_categories)
        if category_topic not in all_categories:
            all_categories.insert(0, category_topic)

        tweets_for_prompt = select_tweets_for_prompt(
            user_tweets,
            rng=rng,
            max_tweets=min(12, MAX_TWEETS_PERSONA),
            min_tweets=min(5, MIN_TWEETS),
            shuffle=SHUFFLE_TWEETS,
        )
        tweet_snippets = [t.strip() for t in tweets_for_prompt if t and t.strip()]
        tone_hint = "earnest" if any("#" in t or "$" in t for t in tweet_snippets) else "casual"
        topics_hint = ", ".join(all_categories[:3]) if all_categories else category_topic

        tweet_based_priors = infer_action_priors_from_tweets(user_tweets)

        # NEW: blend + floors
        priors = blend_action_priors(tweet_based_priors, GLOBAL_ACTION_PRIORS, smoothing=ACTION_PRIOR_SMOOTHING)
        priors = apply_action_floors(priors, MIN_ACTION_FLOORS)

        action_probs = sample_action_probabilities(
            rng=rng or random.Random(),
            priors=priors,
            actions=AVAILABLE_ACTIONS,
        )   

        return Persona(
            username=user_name,
            realname=user_name,
            bio=f"{user_name} | {demographics.profession} | {topics_hint} | {tone_hint} feed",
            persona=(
                f"{user_name} is a {demographics.age}-year-old {demographics.profession} based in {demographics.country} "
                f"who posts in a {tone_hint} voice about {topics_hint} and tends to engage through replies and shares."
            ),
            age=demographics.age,
            gender=demographics.gender,
            mbti=demographics.mbti,
            country=demographics.country,
            profession=demographics.profession,
            category=category_topic,
            all_categories=all_categories,
            cluster_id=cluster_id,
            original_user_name=user_name,
            original_user_id=original_user_id,
            tweet_count=tweet_count,
            sample_tweets=user_tweets,
            action_probs=action_probs,
        )

    category_topic = normalize_category(category_topic) or "general"
    all_categories = [normalize_category(c) for c in all_categories]
    all_categories = [c for c in all_categories if c]
    all_categories = dedupe_preserve_order(all_categories)
    if category_topic not in all_categories:
        all_categories.insert(0, category_topic)

    # Sample a subset of tweets for the persona prompt
    tweets_for_prompt = select_tweets_for_prompt(
        user_tweets,
        rng=rng,
        max_tweets=MAX_TWEETS_PERSONA,
        min_tweets=MIN_TWEETS,
        shuffle=SHUFFLE_TWEETS,
    )

    sample_tweets = "\n".join([f"- {tweet}" for tweet in tweets_for_prompt])

    # Build category description
    if len(all_categories) > 1:
        categories_text = f"Primary category: {category_topic}. Also active in: {', '.join([c for c in all_categories if c != category_topic])}"
        category_instruction = f"the categories: {', '.join(all_categories)}. The persona should reflect their multi-faceted interests across these areas."
    else:
        categories_text = category_topic
        category_instruction = f"the category: {category_topic}"

    prompt = f"""Generate a social media user profile based on the provided information.

User Information:
- Username: @{user_name}
- Age: {demographics.age}
- Gender: {demographics.gender}
- MBTI: {demographics.mbti}
- Country: {demographics.country}
- Profession: {demographics.profession}
- Interest Categories: {categories_text}

Persona writing requirements (single polished instruction)
{PERSONA_PROMPT_MASTER}

Sample tweets from this user:
{sample_tweets}

Generate:
1. A bio (1-2 sentences, similar to Twitter bio style) that reflects their interests in {category_instruction}
2. A detailed persona description (2-3 sentences describing personality, interests, background, and how they engage with {category_instruction})

IMPORTANT: Do NOT generate names or usernames. Use the provided username @{user_name} in your description if needed. Stay consistent with the tweets; do not invent major life events.

Output ONLY valid JSON in this exact format:
{{
    "bio": "Tech enthusiast passionate about blockchain and sustainability. Always learning.",
    "persona": "This user is a tech-savvy professional in their late 20s with an ENTP personality. They're passionate about blockchain technology and sustainable solutions, often engaging in discussions about VeChain and related ecosystems. Their analytical mind and outgoing nature make them an active participant in crypto communities."
}}
"""

    content = chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a persona generation assistant. Create realistic social media user profiles. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=400,
    )
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    persona_dict = json.loads(content)

    tweet_based_priors = infer_action_priors_from_tweets(user_tweets)
    action_probs = sample_action_probabilities(
        rng=rng or random.Random(),
        priors=tweet_based_priors,
        actions=AVAILABLE_ACTIONS,
        concentration=30.0,
    )

    # Use the REAL username from the data for both display name and handle.
    # We ignore any AI-generated names/usernames to keep personas grounded in the original dataset.
    return Persona(
        username=user_name,
        realname=user_name,
        bio=persona_dict["bio"],
        persona=persona_dict["persona"],
        age=demographics.age,
        gender=demographics.gender,
        mbti=demographics.mbti,
        country=demographics.country,
        profession=demographics.profession,
        category=category_topic,
        all_categories=all_categories,
        cluster_id=cluster_id,
        original_user_name=user_name,
        original_user_id=original_user_id,
        tweet_count=tweet_count,
        sample_tweets=user_tweets,
        action_probs=action_probs,
    )


def generate_personas_for_users(
    user_rankings: List[Dict],
    use_twitter_api: bool = False,
    user_all_train_tweets: Optional[Dict[str, List[str]]] = None,
) -> List[Persona]:
    """
    Generate personas for a list of ranked users.

    Each user gets only ONE persona, using their PRIMARY category (highest tweet count or best rank).
    If a user appears in multiple categories, include all categories in the persona.
    """
    # First pass: collect all users and their appearances across categories
    user_appearances = {}  # user_id -> list of (cluster_data, user_data, rank_in_category)

    for cluster_data in user_rankings:
        cluster_id = cluster_data["cluster_id"]

        # Try multiple possible keys (depending on how rankings are produced)
        raw_topic = (
            cluster_data.get("topic")
            or cluster_data.get("category")
            or cluster_data.get("label")
            or cluster_data.get("cluster_topic")
            or ""
        )
        topic = normalize_category(raw_topic) or "general"

        users = cluster_data["top_users"]

        if BALANCE_USERS_BY_ACTIVITY:
            cluster_rng = random.Random(GLOBAL_SEED + int(cluster_id))
            users = _balanced_pick_users(users, cluster_rng, USERS_PER_CLUSTER)


        for rank, user_data in enumerate(users, 1):
            user_id = user_data["user_id"]

            if user_id not in user_appearances:
                user_appearances[user_id] = []

            user_appearances[user_id].append({
                "cluster_data": cluster_data,
                "user_data": user_data,
                "rank": rank,
                "cluster_id": cluster_id,
                "topic": topic,
            })

    # Second pass: for each user, select their PRIMARY category (highest tweet count)
    # and generate one persona that reflects ALL their categories
    personas = []
    processed_users = set()

    def _build_work_item(user_id: str, appearances: List[Dict]) -> Dict:
        primary_appearance = max(
            appearances,
            key=lambda x: (x["user_data"]["tweet_count"], -x["rank"])
        )

        user_data = primary_appearance["user_data"]
        cluster_id = primary_appearance["cluster_id"]
        user_name = user_data["user_name"]
        tweet_count = user_data["tweet_count"]

        raw_all_categories = [a.get("topic") for a in appearances]
        clean_all_categories = [normalize_category(c) for c in raw_all_categories]
        clean_all_categories = [c for c in clean_all_categories if c]
        clean_all_categories = dedupe_preserve_order(clean_all_categories)

        primary_topic = normalize_category(primary_appearance.get("topic"))
        if not primary_topic:
            best_valid = None
            best_score = None
            for a in appearances:
                t = normalize_category(a.get("topic"))
                if not t:
                    continue
                score = (a["user_data"]["tweet_count"], -a["rank"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_valid = t
            primary_topic = best_valid
        if not primary_topic:
            primary_topic = "general"

        if primary_topic not in clean_all_categories:
            clean_all_categories.insert(0, primary_topic)

        if user_all_train_tweets and user_id in user_all_train_tweets:
            tweets = user_all_train_tweets[user_id]
        else:
            all_tweets = []
            for appearance in appearances:
                all_tweets.extend(appearance["user_data"]["sample_tweets"])
            seen = set()
            unique_tweets = []
            for tweet in all_tweets:
                if tweet not in seen:
                    seen.add(tweet)
                    unique_tweets.append(tweet)
            tweets = unique_tweets

        base_rng = persona_rng(user_id, global_seed=GLOBAL_SEED)
        user_rng = random.Random(base_rng.randrange(2**63))

        return {
            "user_id": user_id,
            "user_name": user_name,
            "cluster_id": cluster_id,
            "topic": primary_topic,
            "all_categories": clean_all_categories,
            "tweets": tweets,
            "tweet_count": tweet_count,
            "rng": user_rng,
        }

    work_items: List[Dict] = []
    for user_id, appearances in user_appearances.items():
        if user_id in processed_users:
            continue
        work_items.append(_build_work_item(user_id, appearances))

    def _run_one(w: Dict) -> Optional[Persona]:
        user_id = w["user_id"]
        user_name = w["user_name"]
        try:
            api_profile = fetch_user_profile_from_api(user_id) if use_twitter_api else None

            if SINGLE_CALL_PER_USER:
                return generate_demographics_and_persona_single_call(
                    user_name=user_name,
                    user_tweets=w["tweets"],
                    category_topic=w["topic"],
                    all_categories=w["all_categories"],
                    cluster_id=w["cluster_id"],
                    original_user_id=user_id,
                    tweet_count=w["tweet_count"],
                    api_profile=api_profile,
                    rng=w["rng"],
                )

            demo_rng = random.Random(w["rng"].randrange(2**63))
            persona_rng_stream = random.Random(w["rng"].randrange(2**63))
            demographics = generate_demographics_from_tweets(
                w["tweets"], user_name, api_profile, rng=demo_rng
            )
            return generate_persona(
                user_name=user_name,
                demographics=demographics,
                user_tweets=w["tweets"],
                category_topic=w["topic"],
                all_categories=w["all_categories"],
                cluster_id=w["cluster_id"],
                original_user_id=user_id,
                tweet_count=w["tweet_count"],
                rng=persona_rng_stream,
            )
        except Exception as e:
            print(f"Error generating persona for {user_name}: {e}")
            return None

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except Exception:
        ThreadPoolExecutor = None

    if ThreadPoolExecutor is None or PERSONA_WORKERS <= 1:
        for w in work_items:
            p = _run_one(w)
            if p is None:
                continue
            personas.append(p)
            processed_users.add(p.original_user_id)
            if len(p.all_categories) > 1:
                print(f"Generated multi-category persona for {p.original_user_name} (categories: {', '.join(p.all_categories)})")
            else:
                print(f"Generated persona for {p.original_user_name} (category: {p.category})")
    else:
        with ThreadPoolExecutor(max_workers=max(1, PERSONA_WORKERS)) as ex:
            futures = [ex.submit(_run_one, w) for w in work_items]
            for fut in as_completed(futures):
                p = fut.result()
                if p is None:
                    continue
                personas.append(p)
                processed_users.add(p.original_user_id)
                if len(p.all_categories) > 1:
                    print(f"Generated multi-category persona for {p.original_user_name} (categories: {', '.join(p.all_categories)})")
                else:
                    print(f"Generated persona for {p.original_user_name} (category: {p.category})")

    return personas


def save_personas(
    personas: List[Persona],
    output_path: Path,
    format: str = "json",
    following_by_user: Optional[Dict[str, Dict[str, int]]] = None,
):
    """
    Save personas to file in OASIS-compatible format.

    Args:
        personas: List of Persona objects
        output_path: Output file path
        format: "json" for Reddit format, "csv" for Twitter format
    """
    if format == "json":
        # Reddit format (JSON)
        output_data = []
        for persona in personas:
            output_data.append({
                "username": persona.username,
                "realname": persona.realname,
                "bio": persona.bio,
                "persona": persona.persona,
                "age": persona.age,
                "gender": persona.gender,
                "mbti": persona.mbti,
                "country": persona.country,
                "profession": persona.profession,
                "category": normalize_category(persona.category) or "general",
                "all_categories": dedupe_preserve_order(
                    [c for c in (normalize_category(x) for x in persona.all_categories) if c]
                ) or [normalize_category(persona.category) or "general"],
                "cluster_id": persona.cluster_id,
                "original_user_name": persona.original_user_name,
                "original_user_id": persona.original_user_id,
                "tweet_count": persona.tweet_count,
                "sample_tweets": persona.sample_tweets,
                "action_probs": persona.action_probs,
                "style_markers": persona.style_markers or [],
                "engagement_profile": persona.engagement_profile or {},
            })

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    elif format == "csv":
        # OASIS Twitter simulation format (CSV)
        # Required columns: username, name, description, user_char,
        # following_agentid_list, previous_tweets (see agents_generator.generate_agents)
        import csv

        n = len(personas)

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "user_id",
                "name",
                "username",
                "category_label",
                "category_description",
                "all_categories",
                "following_agentid_list",
                "previous_tweets",
                "user_char",
                "description",
                "p_create_post",
                "p_create_comment",
                "p_do_nothing",
                "p_repost",
                "p_quote_post",
                "p_like_post",
                "p_follow",
            ])
            user_id_to_agent_idx = {
                persona.original_user_id: idx for idx, persona in enumerate(personas)
            }

            for idx, persona in enumerate(personas):
                # If reply-derived edges are provided, use them:
                #   replied_user -> followed_user
                # and keep only users that exist in this persona file.
                following_list: List[int] = []
                if following_by_user:
                    target_counts = following_by_user.get(persona.original_user_id, {})
                    ranked_targets = sorted(
                        target_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                    for target_user_id, _count in ranked_targets:
                        target_idx = user_id_to_agent_idx.get(target_user_id)
                        if target_idx is None or target_idx == idx:
                            continue
                        following_list.append(target_idx)

                    # De-duplicate while preserving order.
                    seen = set()
                    following_list = [
                        x for x in following_list if not (x in seen or seen.add(x))
                    ]
                else:
                    # Backward-compatible fallback if no follow map is provided.
                    user_rng = persona_rng(persona.original_user_id, global_seed=GLOBAL_SEED)
                    other_ids = [i for i in range(n) if i != idx]
                    k = user_rng.randint(0, min(10, len(other_ids))) if other_ids else 0
                    following_list = user_rng.sample(other_ids, k) if k > 0 else []

                # previous_tweets: Python list literal for agents_generator.ast.literal_eval
                tweets_str = json.dumps(persona.sample_tweets) if persona.sample_tweets else "[]"

                ap = persona.action_probs or {}
                cat_desc = normalize_category(persona.category) or "general"
                cat_label = category_label_from_topic(cat_desc)
                writer.writerow([
                    idx,
                    persona.realname,
                    persona.username,
                    cat_label,
                    cat_desc,
                    json.dumps(persona.all_categories),
                    json.dumps(following_list),
                    tweets_str,
                    persona.persona,
                    persona.bio,

                    #NEW Values (default 0.0 if missing)
                    ap.get("create_post", 0.0),
                    ap.get("create_comment", 0.0),
                    ap.get("do_nothing", 0.0),
                    ap.get("repost", 0.0),
                    ap.get("quote_post", 0.0),
                    ap.get("like_post", 0.0),
                    ap.get("follow", 0.0),

                ])

    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    # Example usage
    rankings_file = REPO_ROOT / "user_rankings.json"
    output_file = REPO_ROOT / "user_personas.json"

    if not rankings_file.exists():
        print(f"Rankings file not found: {rankings_file}")
        print("Run rank_users_by_category.py first")
    else:
        REPO_ROOT.mkdir(parents=True, exist_ok=True)  # ensure outputs directory exists
        with rankings_file.open("r", encoding="utf-8") as f:
            rankings = json.load(f)

        use_api = bool(TWITTER_API_KEY)
        personas = generate_personas_for_users(rankings, use_twitter_api=use_api)
        save_personas(personas, output_file)
        print(f"Generated {len(personas)} personas and saved to {output_file}")
