#!/usr/bin/env python3
"""
Generate personas for users using hybrid approach:
- Optional Twitter API for location/created_at hints
- LLM inference for demographics (age, gender, MBTI, country, profession)
- LLM generation for detailed persona
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")  # Bearer token for Twitter API v2


@dataclass
class UserProfile:
    """User profile from Twitter API (optional)."""
    location: Optional[str] = None
    created_at: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Demographics:
    """Inferred demographics for a user."""
    age: Optional[int]
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
    age: Optional[int]
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


ALLOWED_GENDERS = {"male", "female", "other", "unknown"}
ALLOWED_MBTI = {
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ",
}
ALLOWED_PROFESSIONS = {
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
}


def _normalize_age(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"unknown", "n/a", "na", ""}:
            return None
        try:
            value = int(re.sub(r"[^0-9]", "", cleaned))
        except ValueError:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = int(value)
        if 18 <= value <= 65:
            return value
    return None


def _normalize_gender(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    cleaned = str(value).strip().lower()
    if cleaned in {"nonbinary", "non-binary", "nb"}:
        cleaned = "other"
    if cleaned not in ALLOWED_GENDERS:
        return "unknown"
    return cleaned


def _normalize_mbti(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    cleaned = str(value).strip().upper()
    if cleaned.endswith("-A") or cleaned.endswith("-T"):
        cleaned = cleaned[:4]
    if cleaned not in ALLOWED_MBTI:
        return "unknown"
    return cleaned


def _normalize_country(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    cleaned = str(value).strip().upper()
    if cleaned in {"UNKNOWN", "N/A", "NA", ""}:
        return "unknown"
    cleaned = cleaned.replace(".", "")
    if cleaned in {"USA", "US"}:
        return "US"
    if cleaned in {"UK", "GB", "GBR"}:
        return "UK"
    if not re.fullmatch(r"[A-Z]{2,3}", cleaned):
        return "unknown"
    return cleaned


def _normalize_profession(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    cleaned = str(value).strip()
    if cleaned.lower() in {"unknown", "n/a", "na", ""}:
        return "unknown"
    if cleaned not in ALLOWED_PROFESSIONS:
        return "unknown"
    return cleaned


def _evidence_says_unknown(evidence: Dict, key: str) -> bool:
    if not evidence or key not in evidence:
        return False
    value = str(evidence.get(key, "")).strip().lower()
    return value in {"unknown", "n/a", "na", "", "none"}


def _normalize_demographics(demographics_dict: Dict) -> Demographics:
    evidence = demographics_dict.get("evidence", {}) if isinstance(demographics_dict, dict) else {}

    age = _normalize_age(demographics_dict.get("age") if isinstance(demographics_dict, dict) else None)
    gender = _normalize_gender(demographics_dict.get("gender") if isinstance(demographics_dict, dict) else None)
    mbti = _normalize_mbti(demographics_dict.get("mbti") if isinstance(demographics_dict, dict) else None)
    country = _normalize_country(demographics_dict.get("country") if isinstance(demographics_dict, dict) else None)
    profession = _normalize_profession(demographics_dict.get("profession") if isinstance(demographics_dict, dict) else None)

    if _evidence_says_unknown(evidence, "age"):
        age = None
    if _evidence_says_unknown(evidence, "gender"):
        gender = "unknown"
    if _evidence_says_unknown(evidence, "mbti"):
        mbti = "unknown"
    if _evidence_says_unknown(evidence, "country"):
        country = "unknown"
    if _evidence_says_unknown(evidence, "profession"):
        profession = "unknown"

    return Demographics(
        age=age,
        gender=gender,
        mbti=mbti,
        country=country,
        profession=profession,
    )


def _evidence_matches_text(evidence_value, text_lower: str) -> bool:
    if not evidence_value or not text_lower:
        return False
    if isinstance(evidence_value, list):
        candidates = [str(item) for item in evidence_value]
    else:
        raw = str(evidence_value)
        if "|" in raw:
            candidates = raw.split("|")
        elif ";" in raw:
            candidates = raw.split(";")
        else:
            candidates = [raw]
    for candidate in candidates:
        snippet = candidate.strip()
        if not snippet:
            continue
        if snippet.lower() in text_lower:
            return True
    return False


def _extract_anchor_terms(tweets: List[str]) -> List[str]:
    anchors: List[str] = []
    seen = set()

    for tweet in tweets:
        for token in re.findall(r"(#\w+|\$\w+)", tweet):
            if token.upper() == "URL":
                continue
            if token not in seen:
                seen.add(token)
                anchors.append(token)

    if len(anchors) >= 4:
        return anchors[:6]

    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "into", "your", "you", "are",
        "was", "were", "have", "has", "had", "will", "would", "could", "should", "about",
        "just", "like", "what", "when", "where", "their", "them", "they", "then", "than",
        "also", "here", "there", "over", "under", "into", "onto", "our", "out", "too",
        "very", "more", "most", "some", "much", "any", "can", "cant", "dont", "does",
        "did", "didnt", "im", "its", "its", "url", "http", "https",
    }

    word_counts: Dict[str, int] = {}
    for tweet in tweets:
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", tweet):
            lowered = word.lower()
            if lowered in stopwords:
                continue
            word_counts[lowered] = word_counts.get(lowered, 0) + 1

    for word, _ in sorted(word_counts.items(), key=lambda item: (-item[1], item[0])):
        if word not in seen:
            seen.add(word)
            anchors.append(word)
        if len(anchors) >= 6:
            break

    return anchors[:6]


def _extract_behavioral_signals(tweets: List[str]) -> List[str]:
    signals = []
    if any("?" in t for t in tweets):
        signals.append("asks questions to engage others")
    exclamation_count = sum(t.count("!") for t in tweets)
    if exclamation_count >= 3:
        signals.append("uses emphatic exclamations")
    hashtag_count = sum(len(re.findall(r"#\w+", t)) for t in tweets)
    if hashtag_count >= 5:
        signals.append("uses many hashtags")
    if any("http" in t or "URL" in t for t in tweets):
        signals.append("shares links to resources")
    if any(re.search(r"\b[A-Z]{3,}\b", t) for t in tweets):
        signals.append("uses all-caps emphasis")
    if any(re.search(r"\b(thank you|thanks)\b", t, re.IGNORECASE) for t in tweets):
        signals.append("often expresses gratitude")
    if any(re.search(r"\b(DYOR|NFA|not financial advice)\b", t, re.IGNORECASE) for t in tweets):
        signals.append("adds cautionary notes")

    return signals[:5]


def generate_demographics_from_tweets(
    user_tweets: List[str],
    user_name: str,
    api_profile: Optional[UserProfile] = None,
) -> Demographics:
    """
    Infer demographics from user tweets using LLM.
    
    Args:
        user_tweets: List of tweet texts from the user
        user_name: Username
        api_profile: Optional profile from Twitter API
        
    Returns:
        Demographics object
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set - required for demographics inference")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Prepare context
    sample_tweets = "\n".join([f"- {tweet}" for tweet in user_tweets[:10]])  # Use up to 10 tweets
    
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

Rules:
- Only set a value if it is explicitly stated or directly supported by a quoted phrase in the tweets or profile fields above.
- If there is no explicit evidence, output \"unknown\" for that field.
- Evidence MUST be a direct, exact substring from the tweets or profile fields (no paraphrase).
- Do NOT guess based on writing style alone.

Fields:
1. Age: integer 18-65 or \"unknown\" (only if explicitly stated, e.g., \"I'm 29\")
2. Gender: \"male\", \"female\", \"other\", or \"unknown\" (only if explicit, e.g., pronouns or self-ID)
3. MBTI: one of ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ, or \"unknown\"
4. Country: ISO code like \"US\", \"UK\", \"CA\" or \"unknown\" (only if location is explicit; cite the location string)
5. Profession: one of Agriculture Food & Natural Resources, Architecture & Construction, Arts Audio/Video Technology & Communications, Business Management & Administration, Education & Training, Finance, Government & Public Administration, Health Science, Hospitality & Tourism, Human Services, Information Technology, Law Public Safety Corrections & Security, Manufacturing, Marketing, Science Technology Engineering & Mathematics, Transportation Distribution & Logistics, or \"unknown\"

Output ONLY valid JSON in this exact format:
{{
    "age": "unknown",
    "gender": "unknown",
    "mbti": "unknown",
    "country": "unknown",
    "profession": "unknown",
    "evidence": {
        "age": "unknown",
        "gender": "unknown",
        "mbti": "unknown",
        "country": "unknown",
        "profession": "unknown"
    }
}}
"""
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a demographics inference assistant. Use only explicit evidence from the input. "
                    "If evidence is missing, return \"unknown\". Output only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=200,
    )
    
    content = response.choices[0].message.content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    
    demographics_dict = json.loads(content)
    normalized = _normalize_demographics(demographics_dict)

    evidence = demographics_dict.get("evidence", {}) if isinstance(demographics_dict, dict) else {}
    combined_text_parts = list(user_tweets)
    if api_profile:
        if api_profile.location:
            combined_text_parts.append(api_profile.location)
        if api_profile.description:
            combined_text_parts.append(api_profile.description)
        if api_profile.created_at:
            combined_text_parts.append(api_profile.created_at)
    combined_text_lower = "\n".join(combined_text_parts).lower()

    if not _evidence_matches_text(evidence.get("age"), combined_text_lower):
        normalized.age = None
    if not _evidence_matches_text(evidence.get("gender"), combined_text_lower):
        normalized.gender = "unknown"
    if not _evidence_matches_text(evidence.get("mbti"), combined_text_lower):
        normalized.mbti = "unknown"
    if not _evidence_matches_text(evidence.get("country"), combined_text_lower):
        normalized.country = "unknown"
    if not _evidence_matches_text(evidence.get("profession"), combined_text_lower):
        normalized.profession = "unknown"

    return normalized


def generate_persona(
    user_name: str,
    demographics: Demographics,
    user_tweets: List[str],
    category_topic: str,
    all_categories: List[str],
    cluster_id: int,
    original_user_id: str,
    tweet_count: int,
) -> Persona:
    """
    Generate detailed persona for a user.
    
    Args:
        user_name: Original username
        demographics: Inferred demographics
        user_tweets: List of tweet texts
        category_topic: Primary topic/category name
        all_categories: All categories the user appears in
        cluster_id: Cluster ID
        original_user_id: Original user ID
        tweet_count: Number of tweets
        
    Returns:
        Persona object
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set - required for persona generation")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Prepare sample tweets
    sample_tweets = "\n".join([f"- {tweet}" for tweet in user_tweets[:15]])  # Use up to 15 tweets
    
    # Build category description
    if len(all_categories) > 1:
        categories_text = f"Primary category: {category_topic}. Also active in: {', '.join([c for c in all_categories if c != category_topic])}"
        category_instruction = f"the categories: {', '.join(all_categories)}. The persona should reflect their multi-faceted interests across these areas."
    else:
        categories_text = category_topic
        category_instruction = f"the category: {category_topic}"
    
    age_value = demographics.age if demographics.age is not None else "unknown"
    gender_value = demographics.gender or "unknown"
    mbti_value = demographics.mbti or "unknown"
    country_value = demographics.country or "unknown"
    profession_value = demographics.profession or "unknown"

    anchors = _extract_anchor_terms(user_tweets)
    behavioral_signals = _extract_behavioral_signals(user_tweets)

    anchors_text = ", ".join(anchors) if anchors else "None"
    signals_text = ", ".join(behavioral_signals) if behavioral_signals else "None"

    prompt = f"""Generate a social media user profile grounded ONLY in the evidence below.

User Information:
- Username: @{user_name}
- Age: {age_value}
- Gender: {gender_value}
- MBTI: {mbti_value}
- Country: {country_value}
- Profession: {profession_value}
- Interest Categories: {categories_text}

Evidence anchors (exact strings from tweets; mention 1-2 if available):
{anchors_text}

Behavioral signals (from tweets; mention 1 if available):
{signals_text}

Sample tweets from this user:
{sample_tweets}

Generate:
1. A bio (1-2 sentences, similar to Twitter bio style) that reflects their interests in {category_instruction}
2. A detailed persona description (2-3 sentences describing personality, interests, and how they engage with {category_instruction})

Rules:
- Do NOT generate names or usernames. Use the provided username @{user_name} if needed.
- Do NOT invent demographics, education, family status, or life events.
- If a demographic field is \"unknown\", do NOT mention or imply it.
- Keep wording specific to tweet evidence; avoid generic stock phrases (e.g., \"tech-savvy professional\", \"passionate about\").
- Make this persona distinct by using anchors/behaviors unique to this user's tweets.

Output ONLY valid JSON in this exact format:
{{
    "bio": "Shares concise updates and calls to action around the topics they tweet about, often linking to resources.",
    "persona": "They engage the community with specific project mentions and a direct, update-driven tone. Their posts highlight concrete milestones and participation prompts drawn from their own tweets."
}}
"""
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a persona generation assistant. Use only provided evidence. "
                    "Do not invent demographics or life details. Output only valid JSON."
                ),
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
    # We ignore any AI-generated names/usernames to keep personas grounded
    # in the original dataset. The prompt no longer asks for names.
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
        sample_tweets=user_tweets[:15],
    )


def generate_personas_for_users(
    user_rankings: List[Dict],
    use_twitter_api: bool = False,
) -> List[Persona]:
    """
    Generate personas for a list of ranked users.
    
    Each user gets only ONE persona, using their PRIMARY category (where they rank highest
    or have the most tweets). If a user appears in multiple categories, we select the one
    where they have the highest tweet count.
    
    Args:
        user_rankings: List of cluster ranking dicts (from rank_users_by_category output)
        use_twitter_api: Whether to fetch profiles from Twitter API
        
    Returns:
        List of Persona objects (one per unique user)
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
        
        # Collect tweets from ALL categories for better persona generation
        # Combine sample tweets from all appearances (up to 15 total)
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
        tweets = unique_tweets[:15]  # Limit to 15 tweets
        
        # If user appears in multiple categories, note it
        if len(appearances) > 1:
            print(f"User {user_name} appears in {len(appearances)} categories: {', '.join(all_categories)}. "
                  f"Primary: {topic} ({tweet_count} tweets). Generating multi-category persona.")
        
        # Optionally fetch from API
        api_profile = None
        if use_twitter_api:
            api_profile = fetch_user_profile_from_api(user_id)
        
        # Infer demographics
        try:
            demographics = generate_demographics_from_tweets(
                tweets, user_name, api_profile
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


def save_personas(personas: List[Persona], output_path: Path, format: str = "json"):
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
                "age": persona.age if persona.age is not None else "unknown",
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
            for idx, persona in enumerate(personas):
                # previous_tweets: Python list literal for agents_generator.ast.literal_eval
                tweets_str = json.dumps(persona.sample_tweets) if persona.sample_tweets else "[]"
                writer.writerow([
                    idx,
                    persona.realname,
                    persona.username,
                    "[]",  # following_agentid_list (empty; no follow graph from pipeline)
                    tweets_str,
                    persona.persona,
                    persona.bio,
                ])
    
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    # Example usage
    rankings_file = REPO_ROOT / "tweet_processing" / "user_rankings.json"
    output_file = REPO_ROOT / "tweet_processing" / "user_personas.json"
    
    if not rankings_file.exists():
        print(f"Rankings file not found: {rankings_file}")
        print("Run rank_users_by_category.py first")
    else:
        with rankings_file.open() as f:
            rankings = json.load(f)
        
        use_api = bool(TWITTER_API_KEY)
        personas = generate_personas_for_users(rankings, use_twitter_api=use_api)
        save_personas(personas, output_file)
        print(f"Generated {len(personas)} personas and saved to {output_file}")
