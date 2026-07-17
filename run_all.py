from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.experiment import run_experiments, write_manifest
from src.plotting import make_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the numerical DoS scheduling example.")
    parser.add_argument("--config", default="config.json", help="Path to JSON configuration")
    parser.add_argument("--trials", type=int, default=None, help="Override Monte Carlo trial count")
    parser.add_argument("--skip-raw", action="store_true", help="Do not save the full per-trial cost archive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    trials = args.trials or int(config["experiment"]["trials"])
    print(f"[1/4] Starting vectorized Monte Carlo simulation: {trials:,} trials")
    wall_start = time.perf_counter()
    result = run_experiments(config, root, trials_override=args.trials, save_raw=not args.skip_raw)
    print(f"[2/4] Simulation data written in {time.perf_counter() - wall_start:.1f} s")
    make_all_plots(root, result)
    print("[3/4] Figures 2-6 written as PNG and PDF")
    write_manifest(root, result)
    print(f"[4/4] Completed in {time.perf_counter() - wall_start:.1f} s")
    print(f"Output directory: {root}")


if __name__ == "__main__":
    main()

