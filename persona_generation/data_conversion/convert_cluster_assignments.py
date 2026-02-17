#!/usr/bin/env python3
import csv
import json
import sys
from argparse import ArgumentParser

# to use: python convert_cluster_assignments.py cluster_assignments.csv -o cluster_assignments.jsonl

def main():
    parser = ArgumentParser(description="Convert cluster assignments CSV to JSON Lines")
    parser.add_argument("input", help="Path to input CSV file (expects headers: tweet_id,cluster_id). Use - for stdin.")
    parser.add_argument("-o", "--output", help="Path to output file. Defaults to stdout.", default="-")
    parser.add_argument("--encoding", help="File encoding", default="utf-8")
    parser.add_argument("--dialect", help="CSV dialect (default 'excel')", default="excel")
    args = parser.parse_args()

    # Open input (file or stdin)
    if args.input == "-":
        infile = sys.stdin
        close_in = False
    else:
        infile = open(args.input, "r", encoding=args.encoding, newline="")
        close_in = True

    # Open output (file or stdout)
    if args.output == "-":
        outfile = sys.stdout
        close_out = False
    else:
        outfile = open(args.output, "w", encoding=args.encoding, newline="\n")
        close_out = True

    try:
        reader = csv.DictReader(infile, dialect=args.dialect)
        # Validate required columns
        required_cols = {"tweet_id", "cluster_id"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Input CSV must contain headers {sorted(required_cols)}; found {reader.fieldnames}"
            )

        count = 0
        for row in reader:
            # Skip empty rows
            if not row or (row.get("tweet_id") is None and row.get("cluster_id") is None):
                continue

            tweet_id_raw = (row.get("tweet_id") or "").strip()
            cluster_id_raw = (row.get("cluster_id") or "").strip()

            if not tweet_id_raw or not cluster_id_raw:
                # Skip or warn; here we skip quietly
                continue

            # Ensure correct types: tweet_id as string, cluster_id as int
            try:
                cluster_id_int = int(cluster_id_raw)
            except ValueError:
                # If parsing fails, skip the row
                continue

            obj = {"tweet_id": tweet_id_raw, "cluster_id": cluster_id_int}
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

        # Optional: write progress to stderr
        print(f"Wrote {count} lines.", file=sys.stderr)

    finally:
        if close_in:
            infile.close()
        if close_out:
            outfile.close()

if __name__ == "__main__":
    main()