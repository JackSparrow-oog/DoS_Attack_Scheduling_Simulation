from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import SCHEDULE_NAMES


COLORS = {"optimal": "#1f5fbf", "uniform": "#e34a33", "last": "#c44e9b", "no_attack": "#222222"}
MARKERS = {"optimal": "*", "uniform": "^", "last": "o", "no_attack": "s"}
LABELS = {"optimal": "Optimal schedule", "uniform": "Uniform schedule", "last": "Last-priority schedule", "no_attack": "No attack"}


def _finish(fig, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_cost(horizons: np.ndarray, samples: np.ndarray, title: str, path_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for idx, schedule in enumerate(SCHEDULE_NAMES):
        means = samples[idx].mean(axis=1)
        ax.plot(horizons, means, color=COLORS[schedule], marker=MARKERS[schedule], markersize=5, linewidth=1.35, label=LABELS[schedule])
    ax.set_xlabel("Time horizon T")
    ax.set_ylabel("Average expected covariance cost")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, fontsize=8)
    ax.set_xticks(horizons[::2])
    _finish(fig, path_stem)


def plot_covariance(paths: dict[str, np.ndarray], path_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for schedule in (*SCHEDULE_NAMES, "no_attack"):
        values = paths[schedule]
        ax.plot(np.arange(values.size), values, color=COLORS[schedule], marker=MARKERS[schedule], markersize=4, linewidth=1.15, label=LABELS[schedule])
    ax.set_xlabel("Time t")
    ax.set_ylabel("Trace of expected error covariance")
    ax.set_title("Non-smart sensor system: covariance evolution")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, fontsize=8)
    _finish(fig, path_stem)


def plot_state(state: dict[str, np.ndarray], component: int, path_stem: Path) -> None:
    time = state["time"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(time, state["state"][:, component], color="#1f5fbf", marker="*", linewidth=1.15, label=f"True state x{component + 1}")
    ax.plot(time, state["estimate_no_attack"][:, component], color="#e34a33", marker="^", linewidth=1.1, label="Estimate without attacks")
    ax.plot(time, state["estimate_attack"][:, component], color="#c44e9b", marker="o", fillstyle="none", linewidth=1.1, label="Estimate under attacks")
    ax.axvspan(1, 5, color="#f4c2c2", alpha=0.25, label="Attack period")
    ax.set_xlabel("Time k")
    ax.set_ylabel("State")
    ax.set_title(f"Evolution of system state x{component + 1}")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, fontsize=8)
    _finish(fig, path_stem)


def make_all_plots(output_root: Path, result: dict) -> None:
    figures = output_root / "figures"
    plot_cost(result["horizons"], result["all_costs"][0], "Non-smart sensor system", figures / "Fig2_non_smart_cost")
    plot_cost(result["horizons"], result["all_costs"][1], "Smart sensor system", figures / "Fig3_smart_cost")
    plot_covariance(result["fig4_paths"], figures / "Fig4_expected_covariance")
    plot_state(result["state"], 0, figures / "Fig5_state_x1")
    plot_state(result["state"], 1, figures / "Fig6_state_x2")

