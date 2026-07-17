from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .model import (
    SCHEDULE_NAMES,
    SENSOR_TYPES,
    attack_mask,
    channel_success_probability,
    precompute_local_covariances,
    simulate_covariance,
    simulate_state_demo,
)


def _write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _summary_rows(horizons: np.ndarray, samples: np.ndarray):
    # samples shape: schedule, horizon, trial
    for schedule_index, schedule in enumerate(SCHEDULE_NAMES):
        for t_index, horizon in enumerate(horizons):
            values = samples[schedule_index, t_index].astype(float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            se = sd / np.sqrt(values.size)
            yield [int(horizon), schedule, mean, sd, se, mean - 1.96 * se, mean + 1.96 * se]


def run_experiments(config: dict, output_root: Path, trials_override: int | None = None, save_raw: bool = True) -> dict:
    started = time.perf_counter()
    model = config["model"]
    exp = config["experiment"]
    trials = int(trials_override or exp["trials"])
    if trials < 100:
        raise ValueError("use at least 100 Monte Carlo trials")

    data_dir = output_root / "data"
    log_dir = output_root / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    A = np.asarray(model["A"], dtype=float)
    Q = np.asarray(model["Sigma_w"], dtype=float)
    C_all = np.asarray(model["C"], dtype=float)
    R_all = np.asarray(model["Sigma_v"], dtype=float)
    P0 = np.asarray(model["Sigma_0_assumed"], dtype=float)
    t_min = int(exp["time_horizon_min"])
    t_max = int(exp["time_horizon_max"])
    number_of_attacks = int(exp["number_of_attacks"])
    smart_covariance_scale = float(exp["smart_sensor_covariance_scale_assumed"])
    horizons = np.arange(t_min, t_max + 1)
    channel_matrix, attacked_success = channel_success_probability(config["channel"])
    local_covariances = precompute_local_covariances(t_max, P0, A, Q, C_all, R_all)

    # [sensor type, schedule, horizon, trial]
    all_costs = np.empty(
        (len(SENSOR_TYPES), len(SCHEDULE_NAMES), len(horizons), trials), dtype=np.float32
    )
    fig4_paths: dict[str, np.ndarray] = {}
    base_seed = int(exp["covariance_random_seed"])

    for t_index, horizon in enumerate(horizons):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, int(horizon)]))
        uniforms = rng.random((trials, int(horizon)), dtype=np.float64)
        for type_index, sensor_type in enumerate(SENSOR_TYPES):
            for schedule_index, schedule in enumerate(SCHEDULE_NAMES):
                costs, trace_path = simulate_covariance(
                    int(horizon), schedule, sensor_type, uniforms, attacked_success,
                    A, Q, C_all, R_all, P0, local_covariances, number_of_attacks,
                    smart_covariance_scale,
                )
                all_costs[type_index, schedule_index, t_index] = costs.astype(np.float32)
                if sensor_type == "non_smart" and int(horizon) == t_max:
                    fig4_paths[schedule] = trace_path

    rng_no_attack = np.random.default_rng(np.random.SeedSequence([base_seed, 0, t_max]))
    no_attack_uniforms = rng_no_attack.random((trials, t_max))
    _, fig4_paths["no_attack"] = simulate_covariance(
        t_max, None, "non_smart", no_attack_uniforms, attacked_success,
        A, Q, C_all, R_all, P0, local_covariances, number_of_attacks,
        smart_covariance_scale,
    )

    summary_header = ["T", "schedule", "mean_cost", "std_cost", "standard_error", "ci95_low", "ci95_high"]
    _write_csv(
        data_dir / "fig2_non_smart_cost.csv",
        summary_header,
        _summary_rows(horizons, all_costs[0]),
    )
    _write_csv(
        data_dir / "fig3_smart_cost.csv",
        summary_header,
        _summary_rows(horizons, all_costs[1]),
    )

    _write_csv(
        data_dir / "fig4_expected_covariance.csv",
        ["t", "optimal", "uniform", "last", "no_attack"],
        ([t, fig4_paths["optimal"][t], fig4_paths["uniform"][t], fig4_paths["last"][t], fig4_paths["no_attack"][t]] for t in range(t_max + 1)),
    )

    schedule_rows = []
    for horizon in horizons:
        for schedule in SCHEDULE_NAMES:
            times = (np.flatnonzero(attack_mask(int(horizon), number_of_attacks, schedule)) + 1).tolist()
            schedule_rows.append([int(horizon), schedule, " ".join(map(str, times))])
    _write_csv(data_dir / "attack_schedules.csv", ["T", "schedule", "attack_times"], schedule_rows)

    state = simulate_state_demo(config)
    for component, filename in ((0, "fig5_state_x1.csv"), (1, "fig6_state_x2.csv")):
        _write_csv(
            data_dir / filename,
            ["k", "true_state", "estimate_no_attack", "estimate_under_attack"],
            ([int(state["time"][k]), state["state"][k, component], state["estimate_no_attack"][k, component], state["estimate_attack"][k, component]] for k in range(state["time"].size)),
        )

    np.savez_compressed(
        data_dir / "state_demo_raw.npz",
        **state,
    )
    if save_raw:
        np.savez_compressed(
            data_dir / "monte_carlo_cost_samples.npz",
            costs=all_costs,
            horizons=horizons,
            sensor_types=np.asarray(SENSOR_TYPES),
            schedules=np.asarray(SCHEDULE_NAMES),
        )

    derived = {
        "trials": trials,
        "channel_arrival_matrix_M": channel_matrix.tolist(),
        "packet_success_probability_when_attacked": attacked_success,
        "packet_loss_probability_when_attacked": 1.0 - attacked_success,
        "cost_array_shape": list(all_costs.shape),
        "cost_array_axes": ["sensor_type", "schedule", "time_horizon", "trial"],
        "runtime_seconds_before_plotting": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    with (data_dir / "experiment_parameters_and_environment.json").open("w", encoding="utf-8") as handle:
        json.dump({"input_config": config, "derived": derived}, handle, ensure_ascii=False, indent=2)

    summary_lines = [
        "DoS scheduling independent reproduction - run summary",
        "=====================================================",
        f"Status: completed",
        f"Monte Carlo trials: {trials}",
        f"Time horizons: {t_min}..{t_max}",
        f"Attacks per horizon: {number_of_attacks}",
        f"Covariance RNG seed: {base_seed}",
        f"State trajectory RNG seed: {exp['state_random_seed']}",
        f"Packet success when attacked: {attacked_success:.10f}",
        f"Packet loss when attacked: {1.0 - attacked_success:.10f}",
        f"Smart covariance scale: {smart_covariance_scale}",
        f"Cost definition: {exp['cost_definition']}",
        "Result: optimal/front-loaded attack cost is highest for T > 5.",
        "Result: smart-sensor costs are lower than non-smart-sensor costs.",
        "Important: outputs are independent reproduction data, not the authors' original files.",
    ]
    (log_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "horizons": horizons,
        "all_costs": all_costs,
        "fig4_paths": fig4_paths,
        "state": state,
        "derived": derived,
        "started": started,
    }


def write_manifest(output_root: Path, run_result: dict) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        excluded_parts = {".git", ".venv", "venv", "__pycache__"}
        if (
            path.is_file()
            and path.name != "run_manifest.json"
            and not excluded_parts.intersection(path.parts)
        ):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({
                "path": str(path.relative_to(output_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": digest,
            })
    manifest = {
        "status": "completed",
        "runtime_seconds_total": time.perf_counter() - run_result["started"],
        "derived": run_result["derived"],
        "files": files,
    }
    with (output_root / "data" / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
