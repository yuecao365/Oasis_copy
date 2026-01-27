#!/usr/bin/env python3
"""
Generate personas for users using hybrid approach:
- Optional Twitter API for location/created_at hints
- LLM inference for demographics (age, gender, MBTI, country, profession)
- LLM generation for detailed persona
"""

import json
import os
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
    category: str
    cluster_id: int
    original_user_name: str
    original_user_id: str
    tweet_count: int


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

Based on the writing style, topics, language patterns, and any available profile information, infer:
1. Age (as an integer, estimate based on writing style and topics)
2. Gender (one of: "male", "female", "other")
3. MBTI personality type (one of: ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ)
4. Country (ISO country code like "US", "UK", "CA", etc. - infer from location field if available, or from language/content)
5. Profession (choose from: Agriculture Food & Natural Resources, Architecture & Construction, Arts Audio/Video Technology & Communications, Business Management & Administration, Education & Training, Finance, Government & Public Administration, Health Science, Hospitality & Tourism, Human Services, Information Technology, Law Public Safety Corrections & Security, Manufacturing, Marketing, Science Technology Engineering & Mathematics, Transportation Distribution & Logistics)

Output ONLY valid JSON in this exact format:
{{
    "age": 28,
    "gender": "male",
    "mbti": "ENTP",
    "country": "US",
    "profession": "Information Technology"
}}
"""
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a demographics inference assistant. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
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
        category_topic: Topic/category name
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
    
    prompt = f"""Generate a social media user profile based on the provided information.

User Information:
- Age: {demographics.age}
- Gender: {demographics.gender}
- MBTI: {demographics.mbti}
- Country: {demographics.country}
- Profession: {demographics.profession}
- Primary Interest Category: {category_topic}

Sample tweets from this user:
{sample_tweets}

Generate:
1. A real name (first and last name)
2. A username (different from the original, suitable for social media)
3. A bio (1-2 sentences, similar to Twitter bio style)
4. A detailed persona description (2-3 sentences describing personality, interests, background, and how they relate to the category "{category_topic}")

Output ONLY valid JSON in this exact format:
{{
    "realname": "John Smith",
    "username": "john_crypto_enthusiast",
    "bio": "Tech enthusiast passionate about blockchain and sustainability. Always learning.",
    "persona": "John is a tech-savvy professional in his late 20s with an ENTP personality. He's passionate about blockchain technology and sustainable solutions, often engaging in discussions about VeChain and related ecosystems. His analytical mind and outgoing nature make him an active participant in crypto communities."
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
    
    return Persona(
        username=persona_dict["username"],
        realname=persona_dict["realname"],
        bio=persona_dict["bio"],
        persona=persona_dict["persona"],
        age=demographics.age,
        gender=demographics.gender,
        mbti=demographics.mbti,
        country=demographics.country,
        profession=demographics.profession,
        category=category_topic,
        cluster_id=cluster_id,
        original_user_name=user_name,
        original_user_id=original_user_id,
        tweet_count=tweet_count,
    )


def generate_personas_for_users(
    user_rankings: List[Dict],
    use_twitter_api: bool = False,
) -> List[Persona]:
    """
    Generate personas for a list of ranked users.
    
    Args:
        user_rankings: List of cluster ranking dicts (from rank_users_by_category output)
        use_twitter_api: Whether to fetch profiles from Twitter API
        
    Returns:
        List of Persona objects
    """
    personas = []
    
    for cluster_data in user_rankings:
        cluster_id = cluster_data["cluster_id"]
        topic = cluster_data.get("topic", "")
        
        for user_data in cluster_data["top_users"]:
            user_id = user_data["user_id"]
            user_name = user_data["user_name"]
            tweets = user_data["sample_tweets"]
            tweet_count = user_data["tweet_count"]
            
            # Optionally fetch from API
            api_profile = None
            if use_twitter_api:
                api_profile = fetch_user_profile_from_api(user_id)
            
            # Infer demographics
            try:
                demographics = generate_demographics_from_tweets(
                    tweets, user_name, api_profile
                )
                
                # Generate persona
                persona = generate_persona(
                    user_name=user_name,
                    demographics=demographics,
                    user_tweets=tweets,
                    category_topic=topic,
                    cluster_id=cluster_id,
                    original_user_id=user_id,
                    tweet_count=tweet_count,
                )
                
                personas.append(persona)
                print(f"Generated persona for {user_name} (cluster {cluster_id})")
                
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
                "age": persona.age,
                "gender": persona.gender,
                "mbti": persona.mbti,
                "country": persona.country,
                "profession": persona.profession,
                "category": persona.category,
                "cluster_id": persona.cluster_id,
                "original_user_name": persona.original_user_name,
                "original_user_id": persona.original_user_id,
                "tweet_count": persona.tweet_count,
            })
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    elif format == "csv":
        # Twitter format (CSV)
        import csv
        
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "user_id", "name", "username", "user_char", "description"
            ])
            for idx, persona in enumerate(personas):
                writer.writerow([
                    idx,  # user_id (agent_id)
                    persona.realname,  # name
                    persona.username,  # username
                    persona.persona,  # user_char
                    persona.bio,  # description
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
