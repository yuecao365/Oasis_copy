import json
import csv
import os
from datetime import datetime
from typing import Optional

"""
Split filtered_tweets.jsonl into a 2025-04..07 training subset,
and derive a train-scoped user_candidates CSV.

- Input tweets:  data/processed/filtered_tweets.jsonl
- Train tweets:  data/processed/filtered_tweets_train_2025-04_to_07.jsonl
- Input users:   data/processed/user_candidates.csv
- Train users:   data/processed/user_candidates_train_2025-04_to_07.csv
"""

INPUT_PATH = "data/processed/filtered_tweets.jsonl"
OUTPUT_PATH_TRAIN = "data/processed/filtered_tweets_train_2025-04_to_07.jsonl"
USER_CANDIDATES_INPUT = "data/processed/user_candidates.csv"
USER_CANDIDATES_TRAIN_OUTPUT = "data/processed/user_candidates_train_2025-04_to_07.csv"

# Inclusive date range for train data
TRAIN_START = datetime(2025, 4, 1)
TRAIN_END = datetime(2025, 7, 31, 23, 59, 59)


def parse_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()

    # ISO-like: 2025-12-30T23:05:08+00:00 or ...Z
    if "T" in raw and raw[0].isdigit():
        try:
            s = raw.replace("Z", "+00:00")[:19]
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

    # Twitter-style: Tue Jul 22 07:01:42 +0000 2025
    try:
        parts = raw.split()
        if len(parts) >= 6 and parts[-1].isdigit():
            year = int(parts[-1])
            s = " ".join(parts[:4])  # Tue Jul 22 07:01:42
            dt = datetime.strptime(s, "%a %b %d %H:%M:%S")
            return dt.replace(year=year)
    except Exception:
        return None

    return None


def main() -> None:
    total = 0
    kept_train = 0
    no_date = 0

    # Step 1: split tweets into train subset
    with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as fin, open(
        OUTPUT_PATH_TRAIN, "w", encoding="utf-8"
    ) as ftrain:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            try:
                tw = json.loads(line)
            except Exception:
                continue

            raw = tw.get("createdAt") or tw.get("creation_time")
            dt = parse_date(raw)
            if dt is None:
                no_date += 1
                continue

            if TRAIN_START <= dt <= TRAIN_END:
                ftrain.write(line + "\n")
                kept_train += 1

    print(f"Input file: {INPUT_PATH}")
    print(f"Total lines read: {total}")
    print(f"  - Without parsable date: {no_date}")
    print(
        f"Train range: [{TRAIN_START.date()} .. {TRAIN_END.date()}], "
        f"written to {OUTPUT_PATH_TRAIN}"
    )
    print(f"  - Train tweets: {kept_train}")

    # Step 2: filter user_candidates down to users present in the train tweets
    train_author_ids = set()
    try:
        with open(OUTPUT_PATH_TRAIN, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tw = json.loads(line)
                except Exception:
                    continue
                author = tw.get("author") or {}
                uid = author.get("id")
                if uid:
                    train_author_ids.add(str(uid))
    except FileNotFoundError:
        train_author_ids = set()

    if not train_author_ids or not os.path.exists(USER_CANDIDATES_INPUT):
        print(
            "\nSkipping train-scoped user_candidates filtering "
            f"(train_author_ids={len(train_author_ids)}, "
            f"user_candidates.csv exists={os.path.exists(USER_CANDIDATES_INPUT)})."
        )
        return

    kept_users = 0
    total_users = 0
    with open(USER_CANDIDATES_INPUT, "r", encoding="utf-8") as fin, open(
        USER_CANDIDATES_TRAIN_OUTPUT, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or ["user_id", "num_tweets", "is_selected"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            uid = row.get("user_id")
            if not uid:
                continue
            total_users += 1
            if str(uid) in train_author_ids:
                writer.writerow(row)
                kept_users += 1

    print(
        f"\nFiltered user_candidates.csv to training period users only: "
        f"{kept_users} of {total_users} users -> {USER_CANDIDATES_TRAIN_OUTPUT}"
    )


if __name__ == "__main__":
    main()

