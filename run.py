#!/usr/bin/env python3
"""
Full simulation pipeline — one command to run everything.

  python run.py                          # simulation + dashboard (default config)
  python run.py --config simulation/config.yaml
  python run.py --skip-sim               # skip simulation, just regenerate dashboard
  python run.py --out my_dashboard.html  # custom output path

Steps:
  1. Run twitter_simulation.py  (LLM-driven agent simulation → SQLite DB)
  2. Run generate_timeline.py   (DB + persona CSV → two-tab HTML dashboard)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _REPO_ROOT / "simulation" / "config.yaml"
_DEFAULT_OUT    = _REPO_ROOT / "simulation" / "simulation_timeline.html"


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
        help="Output HTML dashboard path (default: simulation/simulation_timeline.html)",
    )
    parser.add_argument(
        "--skip-sim", action="store_true",
        help="Skip the simulation step and only regenerate the dashboard from an existing DB",
    )
    parser.add_argument(
        "--db", default=None,
        help="Override db_path from config (e.g. data/simu_db/evelyn_run.db)",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Override csv_path from config (e.g. persona_generation/outputs/user_personas_evelyn.csv)",
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    if not config_path.exists():
        print(f"[run.py] Error: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config to get db_path and csv_path
    try:
        from yaml import safe_load
        with open(config_path) as f:
            cfg = safe_load(f)
    except Exception as e:
        print(f"[run.py] Error reading config: {e}", file=sys.stderr)
        sys.exit(1)

    db_path  = _resolve(args.db  or cfg["data"]["db_path"])
    csv_path = _resolve(args.csv or cfg["data"]["csv_path"])

    # ── Step 1: Simulation ────────────────────────────────────────────────────
    if not args.skip_sim:
        print("=" * 60)
        print("Step 1/2 — Running simulation")
        print("=" * 60)

        # If CLI overrides db/csv, write a temp config so the simulation picks them up
        import tempfile, copy
        sim_config_path = config_path
        if args.db or args.csv:
            from yaml import safe_load, dump
            patched = copy.deepcopy(cfg)
            patched["data"]["db_path"]  = str(db_path)
            patched["data"]["csv_path"] = str(csv_path)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            dump(patched, tmp)
            tmp.close()
            sim_config_path = Path(tmp.name)

        result = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "simulation" / "twitter_simulation.py"),
             "--config_path", str(sim_config_path)],
            cwd=str(_REPO_ROOT),
        )
        if result.returncode != 0:
            print(f"\n[run.py] Simulation exited with code {result.returncode}.", file=sys.stderr)
            sys.exit(result.returncode)
    else:
        print("[run.py] Skipping simulation (--skip-sim)")

    # ── Step 2: Dashboard ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 2/2 — Generating dashboard")
    print("=" * 60)

    if not db_path.exists():
        print(f"[run.py] Error: DB not found at {db_path}. Run without --skip-sim first.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(_REPO_ROOT / "simulation"))
    import generate_timeline  # noqa: E402

    data = generate_timeline.load_data(str(db_path), str(csv_path))
    html = generate_timeline.generate_html(data, str(db_path))
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print()
    print("=" * 60)
    print("Done.")
    print(f"  Dashboard → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
