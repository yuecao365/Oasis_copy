#!/usr/bin/env python3
"""
Time-based train/test split for tweets.

Extracts timestamps from Snowflake tweet IDs and splits data chronologically.
Only includes users who are active in BOTH train and test periods.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "tweet_processing" / "tweet_Nov+Dec.txt"

# Snowflake epoch: 1288834974657 milliseconds since Unix epoch
SNOWFLAKE_EPOCH = 1288834974657


@dataclass
class TweetWithTimestamp:
    """Tweet with extracted timestamp."""
    text: str
    id: str
    topic_id: str
    user_name: str
    user_id: str
    timestamp: datetime
    likes: int
    retweets: int
    replies: int
    quotes: int
    impressions: int
    is_root: bool
    dup: bool


def extract_timestamp_from_tweet_id(tweet_id: str) -> datetime:
    """
    Extract timestamp from Snowflake tweet ID.
    
    Snowflake IDs encode timestamp: (id >> 22) + SNOWFLAKE_EPOCH gives milliseconds since epoch.
    
    Args:
        tweet_id: Snowflake tweet ID as string
        
    Returns:
        datetime object representing tweet creation time
    """
    try:
        tweet_id_int = int(tweet_id)
        # Extract timestamp: shift right 22 bits, add Snowflake epoch
        timestamp_ms = (tweet_id_int >> 22) + SNOWFLAKE_EPOCH
        return datetime.fromtimestamp(timestamp_ms / 1000.0)
    except (ValueError, OverflowError):
        # If ID is invalid, return a default date (shouldn't happen with real data)
        return datetime.fromtimestamp(0)


def load_tweets_with_timestamps() -> List[TweetWithTimestamp]:
    """Load tweets and extract timestamps from IDs."""
    tweets = []
    with DATA_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = row.get("id", "")
            if not tweet_id:
                continue
                
            timestamp = extract_timestamp_from_tweet_id(tweet_id)
            
            tweets.append(
                TweetWithTimestamp(
                    text=row.get("text", ""),
                    id=tweet_id,
                    topic_id=row.get("topic_id", ""),
                    user_name=row.get("user_name", ""),
                    user_id=row.get("user_id", ""),
                    timestamp=timestamp,
                    likes=int(row.get("likes", 0) or 0),
                    retweets=int(row.get("retweets", 0) or 0),
                    replies=int(row.get("replies", 0) or 0),
                    quotes=int(row.get("quotes", 0) or 0),
                    impressions=int(row.get("impressions", 0) or 0),
                    is_root=row.get("is_root") == "1",
                    dup=row.get("dup") != "0",
                )
            )
    return tweets


def split_tweets_by_time(
    tweets: List[TweetWithTimestamp],
    train_months: int = 6,
    test_months: int = 1,
    train_year: int = None,
    train_month: int = None,
    test_year: int = None,
    test_month: int = None,
) -> Tuple[List[TweetWithTimestamp], List[TweetWithTimestamp]]:
    """
    Split tweets chronologically by time periods. Uses the MOST RECENT data for training (last N months) and the period before that for testing.
    
    If train_year/train_month/test_year/test_month are provided, splits by specific months instead.
    
    If the backward-from-end approach results in insufficient test data or no user overlap,
    falls back to a percentage-based split (80% train, 20% test) to ensure meaningful splits.
    
    Args:
        tweets: List of tweets with timestamps
        train_months: Number of months for training period (ignored if train_year/train_month provided)
        test_months: Number of months for test period (ignored if test_year/test_month provided)
        train_year: Year for training period (optional, for month-based splitting)
        train_month: Month for training period 1-12 (optional, for month-based splitting)
        test_year: Year for test period (optional, for month-based splitting)
        test_month: Month for test period 1-12 (optional, for month-based splitting)
        
    Returns:
        Tuple of (train_tweets, test_tweets)
    """
    if not tweets:
        return [], []
    
    # If specific months are provided, use month-based splitting
    if train_year is not None and train_month is not None and test_year is not None and test_month is not None:
        from datetime import datetime, timezone
        
        # Create date boundaries (inclusive start, exclusive end)
        # Use UTC timezone to match tweet timestamps
        train_start = datetime(train_year, train_month, 1, tzinfo=timezone.utc)
        if train_month == 12:
            train_end = datetime(train_year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            train_end = datetime(train_year, train_month + 1, 1, tzinfo=timezone.utc)
        
        test_start = datetime(test_year, test_month, 1, tzinfo=timezone.utc)
        if test_month == 12:
            test_end = datetime(test_year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            test_end = datetime(test_year, test_month + 1, 1, tzinfo=timezone.utc)
        
        train_tweets = []
        test_tweets = []
        
        for tweet in tweets:
            ts = tweet.timestamp
            # Ensure timestamp has timezone for comparison
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            
            if train_start <= ts < train_end:
                train_tweets.append(tweet)
            elif test_start <= ts < test_end:
                test_tweets.append(tweet)
        
        return train_tweets, test_tweets
    
    # Otherwise, use the original time-based splitting logic
    # Sort tweets chronologically
    sorted_tweets = sorted(tweets, key=lambda t: t.timestamp)

    # Use END date (most recent) as reference
    end_date = sorted_tweets[-1].timestamp
    start_date = sorted_tweets[0].timestamp
    total_days = (end_date - start_date).days
    
    # Calculate boundaries working backwards from the end
    # Train period: last train_months months
    train_start_date = end_date - timedelta(days=train_months * 30)
    
    # Test period: test_months months before train period
    test_end_date = train_start_date
    test_start_date = test_end_date - timedelta(days=test_months * 30)
    
    # Check if we have sufficient data in test period
    test_count = sum(1 for t in sorted_tweets if test_start_date <= t.timestamp < test_end_date)
    train_count = sum(1 for t in sorted_tweets if t.timestamp >= train_start_date)
    
    # If test period would go before data start or has very few tweets (< 100), 
    # use a different strategy: find a split point that ensures both periods have enough data
    if test_start_date < start_date or test_count < 100:
        # Strategy: Use last 80% for train, previous 20% for test
        # This ensures test is immediately before train chronologically
        train_start_date = start_date + timedelta(days=int(total_days * 0.8))
        test_end_date = train_start_date
        # Test period: 20% of total time, immediately before train
        test_start_date = train_start_date - timedelta(days=int(total_days * 0.2))
        
        # Ensure test_start_date doesn't go before start_date
        if test_start_date < start_date:
            test_start_date = start_date
    
    train_tweets = []
    test_tweets = []
    
    for tweet in sorted_tweets:
        if tweet.timestamp >= train_start_date:
            train_tweets.append(tweet)
        elif tweet.timestamp >= test_start_date and tweet.timestamp < train_start_date:
            test_tweets.append(tweet)
    
    return train_tweets, test_tweets


def filter_active_users_in_both_periods(
    train_tweets: List[TweetWithTimestamp],
    test_tweets: List[TweetWithTimestamp],
    min_train_tweets: int = 1,
    min_test_tweets: int = 1,
) -> Tuple[List[TweetWithTimestamp], List[TweetWithTimestamp]]:
    """
    Filter to only include users active in BOTH train and test periods.
    
    This ensures meaningful evaluation - users must be active in test window.
    
    Args:
        train_tweets: Training period tweets
        test_tweets: Test period tweets
        min_train_tweets: Minimum tweets in train period
        min_test_tweets: Minimum tweets in test period
        
    Returns:
        Tuple of filtered (train_tweets, test_tweets)
    """
    # Count tweets per user in each period
    train_user_counts: Dict[str, int] = {}
    test_user_counts: Dict[str, int] = {}
    
    for tweet in train_tweets:
        train_user_counts[tweet.user_id] = train_user_counts.get(tweet.user_id, 0) + 1
    
    for tweet in test_tweets:
        test_user_counts[tweet.user_id] = test_user_counts.get(tweet.user_id, 0) + 1
    
    # Find users active in both periods with minimum thresholds
    active_users = set()
    for user_id in train_user_counts:
        if (
            train_user_counts[user_id] >= min_train_tweets
            and test_user_counts.get(user_id, 0) >= min_test_tweets
        ):
            active_users.add(user_id)
    
    # Filter tweets to only active users
    filtered_train = [t for t in train_tweets if t.user_id in active_users]
    filtered_test = [t for t in test_tweets if t.user_id in active_users]
    
    return filtered_train, filtered_test


def save_tweets_to_csv(tweets: List[TweetWithTimestamp], output_path: Path):
    """Save tweets to CSV file."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dup", "id", "topic_id", "reply_to_id", "reply_to_user",
            "likes", "retweets", "replies", "quotes", "impressions",
            "user_name", "user_id", "is_root", "text", "timestamp"
        ])
        for tweet in tweets:
            writer.writerow([
                "1" if tweet.dup else "0",
                tweet.id,
                tweet.topic_id,
                "",  # reply_to_id not in our dataclass
                "",  # reply_to_user not in our dataclass
                tweet.likes,
                tweet.retweets,
                tweet.replies,
                tweet.quotes,
                tweet.impressions,
                tweet.user_name,
                tweet.user_id,
                "1" if tweet.is_root else "0",
                tweet.text,
                tweet.timestamp.isoformat(),
            ])


def save_split_info(
    train_tweets: List[TweetWithTimestamp],
    test_tweets: List[TweetWithTimestamp],
    output_path: Path,
    train_months: int,
    test_months: int,
):
    """Save split metadata to JSON."""
    import json
    
    train_users = set(t.user_id for t in train_tweets)
    test_users = set(t.user_id for t in test_tweets)
    active_users = train_users & test_users
    
    train_start = min(t.timestamp for t in train_tweets) if train_tweets else None
    train_end = max(t.timestamp for t in train_tweets) if train_tweets else None
    test_start = min(t.timestamp for t in test_tweets) if test_tweets else None
    test_end = max(t.timestamp for t in test_tweets) if test_tweets else None
    
    info = {
        "train_months": train_months,
        "test_months": test_months,
        "train_period": {
            "start": train_start.isoformat() if train_start else None,
            "end": train_end.isoformat() if train_end else None,
            "tweet_count": len(train_tweets),
            "user_count": len(train_users),
        },
        "test_period": {
            "start": test_start.isoformat() if test_start else None,
            "end": test_end.isoformat() if test_end else None,
            "tweet_count": len(test_tweets),
            "user_count": len(test_users),
        },
        "active_users_count": len(active_users),
        "active_user_ids": sorted(list(active_users)),
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    """Main function for testing."""
    print("Loading tweets...")
    tweets = load_tweets_with_timestamps()
    print(f"Loaded {len(tweets)} tweets")
    
    if tweets:
        sorted_tweets = sorted(tweets, key=lambda t: t.timestamp)
        print(f"Date range: {sorted_tweets[0].timestamp} to {sorted_tweets[-1].timestamp}")
    
    print("Splitting by time (6 months train, 1 month test)...")
    train_tweets, test_tweets = split_tweets_by_time(tweets, train_months=6, test_months=1)
    print(f"Train: {len(train_tweets)} tweets, Test: {len(test_tweets)} tweets")
    
    if train_tweets:
        sorted_train = sorted(train_tweets, key=lambda t: t.timestamp)
        print(f"Train date range: {sorted_train[0].timestamp} to {sorted_train[-1].timestamp}")
        print(f"Train unique users: {len(set(t.user_id for t in train_tweets))}")
    
    if test_tweets:
        sorted_test = sorted(test_tweets, key=lambda t: t.timestamp)
        print(f"Test date range: {sorted_test[0].timestamp} to {sorted_test[-1].timestamp}")
        print(f"Test unique users: {len(set(t.user_id for t in test_tweets))}")
    
    # Check user overlap before filtering
    train_users = set(t.user_id for t in train_tweets)
    test_users = set(t.user_id for t in test_tweets)
    overlap = train_users & test_users
    print(f"User overlap before filtering: {len(overlap)} users")
    
    print("Filtering to users active in both periods...")
    train_filtered, test_filtered = filter_active_users_in_both_periods(
        train_tweets, test_tweets, min_train_tweets=1, min_test_tweets=1
    )
    print(f"After filtering - Train: {len(train_filtered)} tweets, Test: {len(test_filtered)} tweets")
    
    # Save outputs
    output_dir = REPO_ROOT / "tweet_processing"
    save_tweets_to_csv(train_filtered, output_dir / "train_tweets.csv")
    save_tweets_to_csv(test_filtered, output_dir / "test_tweets.csv")
    save_split_info(
        train_filtered, test_filtered, output_dir / "split_info.json",
        train_months=6, test_months=1
    )
    print("Saved train_tweets.csv, test_tweets.csv, and split_info.json")


if __name__ == "__main__":
    main()
