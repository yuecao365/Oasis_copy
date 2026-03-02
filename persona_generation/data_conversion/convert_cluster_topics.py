#!/usr/bin/env python3
"""
convert_cluster_topics.py

Convert a JSON file shaped as a list of objects with keys {"cluster_id": <int>, "topic": <str>}
into a flat JSON object mapping string cluster_id -> topic, like:
{
  "0": "Customer Support",
  "1": "Billing Issues"
}

Usage:
  python convert_cluster_topics.py --input cluster_topics.json --output topics_flat.json

If --output is omitted, prints the result to stdout.
"""

import argparse
import json
import sys
from typing import Any, Dict, List


def load_input(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: Input file not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Failed to parse JSON in {path}: {e}")

    if not isinstance(data, list):
        sys.exit("Error: Expected a JSON array at the top level (list of objects).")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            sys.exit(f"Error: Element at index {i} is not an object: {type(item)}")
        if 'cluster_id' not in item:
            sys.exit(f"Error: Element at index {i} missing 'cluster_id'.")
        if 'topic' not in item:
            sys.exit(f"Error: Element at index {i} missing 'topic'.")
    return data


def convert(records: List[Dict[str, Any]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for i, item in enumerate(records):
        cid = item['cluster_id']
        topic = item['topic']

        # Normalize cluster_id to string key
        if isinstance(cid, (int, float)) and int(cid) == cid:
            key = str(int(cid))
        else:
            # Allow string cluster_ids, but coerce to string
            key = str(cid)

        if not isinstance(topic, str):
            topic = str(topic)

        # Detect duplicates and warn by appending index
        if key in result:
            # Ensure uniqueness by using a suffix; also print a warning to stderr
            alt_key = f"{key}__{i}"
            print(
                f"Warning: Duplicate cluster_id '{key}' found at index {i}. Using key '{alt_key}'.",
                file=sys.stderr,
            )
            key = alt_key

        result[key] = topic
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert cluster topics JSON to flat mapping.")
    parser.add_argument('-i', '--input', required=True, help='Path to input JSON file (list of objects).')
    parser.add_argument('-o', '--output', help='Path to output JSON file (flat mapping). If omitted, prints to stdout.')
    parser.add_argument('--indent', type=int, default=2, help='Indentation level for output JSON (default: 2).')
    args = parser.parse_args()

    records = load_input(args.input)
    flat = convert(records)

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(flat, f, ensure_ascii=False, indent=args.indent)
        except OSError as e:
            sys.exit(f"Error: Failed to write output file {args.output}: {e}")
        print(f"Wrote {len(flat)} entries to {args.output}")
    else:
        json.dump(flat, sys.stdout, ensure_ascii=False, indent=args.indent)
        print()


if __name__ == '__main__':
    main()