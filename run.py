#!/usr/bin/env python3
"""
Full simulation pipeline — one command to run everything.

  python run.py                          # simulation + metrics + dashboard (default config)
  python run.py --config simulation/config.yaml
  python run.py --skip-sim               # skip simulation, just regenerate metrics + dashboard
  python run.py --out my_dashboard.html  # custom output path

Steps:
  1. Delete existing DB (fresh run every time)
  2. Run twitter_simulation.py  (LLM-driven agent simulation → SQLite DB)
  3. Run compare_sim_to_test.py (metrics vs ground-truth test split → JSON)
  4. Run generate_timeline.py   (DB + persona CSV + metrics → HTML dashboard)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _REPO_ROOT / "simulation" / "config.yaml"
_DEFAULT_OUT    = _REPO_ROOT / "visualizations" / "simulation_timeline.html"


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full OASIS simulation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=str(_DEFAULT_CONFIG),
        help="Path to simulation/config.yaml (default: simulation/config.yaml)",
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT),
        help="Output HTML dashboard path (default: visualizations/simulation_timeline.html)",
    )
    parser.add_argument(
        "--skip-sim", action="store_true",
        help="Skip the simulation step and only regenerate metrics + dashboard from an existing DB",
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    if not config_path.exists():
        print(f"[run.py] Error: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from yaml import safe_load
        with open(config_path) as f:
            cfg = safe_load(f)
    except Exception as e:
        print(f"[run.py] Error reading config: {e}", file=sys.stderr)
        sys.exit(1)

    db_path  = _resolve(cfg["data"]["db_path"])
    csv_path = _resolve(cfg["data"]["csv_path"])
    split_info = _resolve(cfg.get("simulation", {}).get("split_info_path", "persona_generation/outputs/split_info.json"))

    metrics_json_path = db_path.with_suffix(".metrics.json")
    metrics_md_path   = db_path.with_suffix(".metrics.md")
    archive_dir       = _REPO_ROOT / "simulation" / "metrics_history"

    # ── Step 1: Clear DB ──────────────────────────────────────────────────────
    if not args.skip_sim:
        if db_path.exists():
            db_path.unlink()
            print(f"[run.py] Cleared existing DB: {db_path.name}")

    # ── Step 2: Simulation ────────────────────────────────────────────────────
    if not args.skip_sim:
        print("=" * 60)
        print("Step 1/3 — Running simulation")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "simulation" / "twitter_simulation.py"),
             "--config_path", str(config_path)],
            cwd=str(_REPO_ROOT),
        )
        if result.returncode != 0:
            print(f"\n[run.py] Simulation exited with code {result.returncode}.", file=sys.stderr)
            sys.exit(result.returncode)
    else:
        print("[run.py] Skipping simulation (--skip-sim)")

    if not db_path.exists():
        print(f"[run.py] Error: DB not found at {db_path}. Run without --skip-sim first.", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Metrics ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 2/3 — Computing metrics")
    print("=" * 60)
    metrics_result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "simulation" / "compare_sim_to_test.py"),
         "--db", str(db_path),
         "--personas_csv", str(csv_path),
         "--split_info", str(split_info),
         "--sim_config", str(config_path),
         "--output_json", str(metrics_json_path),
         "--output_md", str(metrics_md_path),
         "--archive_dir", str(archive_dir)],
        cwd=str(_REPO_ROOT),
    )
    metrics: dict = {}
    if metrics_result.returncode != 0:
        print("[run.py] Warning: metrics step failed — dashboard will show no metrics.", file=sys.stderr)
    elif metrics_json_path.exists():
        with open(metrics_json_path) as f:
            metrics = json.load(f)
        print(f"[run.py] Metrics written to {metrics_json_path.name}")

    # ── Step 4: Dashboard ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 3/3 — Generating dashboard")
    print("=" * 60)

    sys.path.insert(0, str(_REPO_ROOT / "simulation"))
    import generate_timeline  # type: ignore[import-not-found]

    data = generate_timeline.load_data(str(db_path), str(csv_path))
    html = generate_timeline.generate_html(data, str(db_path), metrics=metrics)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print()
    print("=" * 60)
    print("Done.")
    print(f"  Dashboard → {out_path}")
    print(f"  Metrics   → {metrics_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
