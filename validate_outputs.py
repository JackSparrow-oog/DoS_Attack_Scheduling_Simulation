from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read_cost_table(path: Path) -> dict[tuple[int, str], float]:
    values = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values[(int(row["T"]), row["schedule"])] = float(row["mean_cost"])
    return values


def main() -> None:
    root = Path(__file__).resolve().parent
    required = [
        root / "data" / "fig2_non_smart_cost.csv",
        root / "data" / "fig3_smart_cost.csv",
        root / "data" / "fig4_expected_covariance.csv",
        root / "data" / "fig5_state_x1.csv",
        root / "data" / "fig6_state_x2.csv",
        root / "data" / "experiment_parameters_and_environment.json",
        root / "data" / "run_manifest.json",
    ]
    required += [root / "figures" / f"Fig{i}_{name}.png" for i, name in (
        (2, "non_smart_cost"), (3, "smart_cost"), (4, "expected_covariance"), (5, "state_x1"), (6, "state_x2")
    )]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing or empty outputs:\n" + "\n".join(missing))

    non_smart = read_cost_table(required[0])
    smart = read_cost_table(required[1])
    for horizon in range(6, 21):
        if non_smart[(horizon, "optimal")] <= non_smart[(horizon, "uniform")]:
            raise SystemExit(f"Expected optimal > uniform for non-smart at T={horizon}")
        if non_smart[(horizon, "optimal")] <= non_smart[(horizon, "last")]:
            raise SystemExit(f"Expected optimal > last for non-smart at T={horizon}")
        if smart[(horizon, "optimal")] <= smart[(horizon, "uniform")]:
            raise SystemExit(f"Expected optimal > uniform for smart at T={horizon}")
        if smart[(horizon, "optimal")] <= smart[(horizon, "last")]:
            raise SystemExit(f"Expected optimal > last for smart at T={horizon}")
    for horizon in range(5, 21):
        for schedule in ("optimal", "uniform", "last"):
            if smart[(horizon, schedule)] >= non_smart[(horizon, schedule)]:
                raise SystemExit(
                    f"Expected smart < non-smart for {schedule} at T={horizon}"
                )

    archive = root / "data" / "monte_carlo_cost_samples.npz"
    if archive.exists():
        with np.load(archive) as data:
            if data["costs"].ndim != 4 or not np.all(np.isfinite(data["costs"])):
                raise SystemExit("Raw Monte Carlo archive is malformed")

    manifest = json.loads((root / "data" / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise SystemExit("Run manifest does not report completion")
    print(f"Validation passed: {len(manifest['files'])} files listed in manifest.")


if __name__ == "__main__":
    main()
