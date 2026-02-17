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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from config import DRY_RUN
import hashlib

from openai import OpenAI

# NOTE: This points to the outputs directory under the script's folder, not the repo root.
REPO_ROOT = Path(__file__).resolve().parent / "outputs"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")  # Bearer token for Twitter API v2
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "20260215"))
NOISE_EPS = float(os.getenv("NOISE_EPS", "0.05"))  # for priors noise amplitude


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

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set - required for demographics inference")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare context
    sample_tweets = "\n".join([f"- {tweet}" for tweet in user_tweets])

    api_context = ""
    if api_profile:
        api_context = f"""
Additional context from Twitter profile:
- Location: {api_profile.location or 'Not specified'}
- Account created: {api_profile.created_at or 'Not specified'}
- Bio: {api_profile.description or 'Not specified'}
"""

    prompt = f"""Analyze the following tweets from a Twitter user and infer their demographics.

User: @{user_name}
{api_context}
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
    seed_int = rng.randrange(0, 2**31 - 1) if rng else None
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a demographics inference assistant. Output only valid JSON. Ensure age diversity across users.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,  # Increased for more variance
        max_tokens=200,
        **({"seed": seed_int} if seed_int is not None else {}),  # Optional seeding for reproducibility
    )

    content = response.choices[0].message.content.strip()
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
        return Persona(
            username=user_name,
            realname=user_name,
            bio=f"Bio placeholder for @{user_name} (DRY RUN)",
            persona=f"Persona description placeholder for @{user_name}. Categories: {all_categories}.",
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
        )

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set - required for persona generation")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare sample tweets (deterministic shuffle if rng is provided)
    tweets_for_prompt = list(user_tweets)
    if rng:
        rng.shuffle(tweets_for_prompt)
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

Sample tweets from this user:
{sample_tweets}

Generate:
1. A bio (1-2 sentences, similar to Twitter bio style) that reflects their interests in {category_instruction}
2. A detailed persona description (2-3 sentences describing personality, interests, background, and how they engage with {category_instruction})

IMPORTANT: Do NOT generate names or usernames. Use the provided username @{user_name} in your description if needed.

Output ONLY valid JSON in this exact format:
{{
    "bio": "Tech enthusiast passionate about blockchain and sustainability. Always learning.",
    "persona": "This user is a tech-savvy professional in their late 20s with an ENTP personality. They're passionate about blockchain technology and sustainable solutions, often engaging in discussions about VeChain and related ecosystems. Their analytical mind and outgoing nature make them an active participant in crypto communities."
}}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a persona generation assistant. Create realistic social media user profiles. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=400,
    )

    content = response.choices[0].message.content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    persona_dict = json.loads(content)

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
        topic = cluster_data.get("topic", "")

        for rank, user_data in enumerate(cluster_data["top_users"], 1):
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

    for user_id, appearances in user_appearances.items():
        if user_id in processed_users:
            continue

        # Select primary category: highest tweet count, or highest rank if tied
        primary_appearance = max(
            appearances,
            key=lambda x: (x["user_data"]["tweet_count"], -x["rank"])
        )

        cluster_data = primary_appearance["cluster_data"]
        user_data = primary_appearance["user_data"]
        cluster_id = primary_appearance["cluster_id"]
        topic = primary_appearance["topic"]
        user_name = user_data["user_name"]
        tweets = user_data["sample_tweets"]
        tweet_count = user_data["tweet_count"]

        # Collect ALL categories this user appears in
        all_categories = [a["topic"] for a in appearances]

        # Prefer full train-split tweet context for this user when available.
        # Fallback to merged ranking samples if full context is not provided.
        if user_all_train_tweets and user_id in user_all_train_tweets:
            tweets = user_all_train_tweets[user_id]
        else:
            all_tweets = []
            for appearance in appearances:
                all_tweets.extend(appearance["user_data"]["sample_tweets"])
            # Deduplicate while preserving order
            seen = set()
            unique_tweets = []
            for tweet in all_tweets:
                if tweet not in seen:
                    seen.add(tweet)
                    unique_tweets.append(tweet)
            tweets = unique_tweets

        # If user appears in multiple categories, note it
        if len(appearances) > 1:
            print(
                f"User {user_name} appears in {len(appearances)} categories: "
                f"{', '.join(all_categories)}. Primary: {topic} ({tweet_count} tweets). "
                "Generating multi-category persona."
            )

        # Optionally fetch from API
        api_profile = None
        if use_twitter_api:
            api_profile = fetch_user_profile_from_api(user_id)

        user_rng = persona_rng(user_id, global_seed=GLOBAL_SEED)

        # Infer demographics
        try:
            demographics = generate_demographics_from_tweets(
                tweets, user_name, api_profile, rng=user_rng
            )

            # Generate persona (with all categories)
            persona = generate_persona(
                user_name=user_name,
                demographics=demographics,
                user_tweets=tweets,
                category_topic=topic,
                all_categories=all_categories,
                cluster_id=cluster_id,
                original_user_id=user_id,
                tweet_count=tweet_count,
                rng=user_rng,
            )

            personas.append(persona)
            processed_users.add(user_id)
            if len(appearances) > 1:
                print(f"Generated multi-category persona for {user_name} (categories: {', '.join(all_categories)})")
            else:
                print(f"Generated persona for {user_name} (category: {topic})")

        except Exception as e:
            print(f"Error generating persona for {user_name}: {e}")
            continue

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
                "category": persona.category,
                "all_categories": persona.all_categories,
                "cluster_id": persona.cluster_id,
                "original_user_name": persona.original_user_name,
                "original_user_id": persona.original_user_id,
                "tweet_count": persona.tweet_count,
                "sample_tweets": persona.sample_tweets,
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
                "following_agentid_list",
                "previous_tweets",
                "user_char",
                "description",
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

                writer.writerow([
                    idx,
                    persona.realname,
                    persona.username,
                    json.dumps(following_list),
                    tweets_str,
                    persona.persona,
                    persona.bio,
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
