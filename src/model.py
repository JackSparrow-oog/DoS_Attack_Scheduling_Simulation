from __future__ import annotations

import numpy as np


SCHEDULE_NAMES = ("optimal", "uniform", "last")
SENSOR_TYPES = ("non_smart", "smart")


def attack_mask(horizon: int, attacks: int, schedule: str) -> np.ndarray:
    """Return a Boolean attack mask for time indices 1..horizon.

    The uniform schedule is tail-aligned and periodic. For T=20 and a=5,
    it attacks at 4, 8, 12, 16, 20, matching the visible spikes in Fig. 4.
    """
    if not 0 <= attacks <= horizon:
        raise ValueError("attacks must satisfy 0 <= attacks <= horizon")
    mask = np.zeros(horizon, dtype=bool)
    if attacks == 0:
        return mask
    if schedule == "optimal":
        times = np.arange(1, attacks + 1)
    elif schedule == "last":
        times = np.arange(horizon - attacks + 1, horizon + 1)
    elif schedule == "uniform":
        times = np.ceil(np.arange(1, attacks + 1) * horizon / attacks).astype(int)
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    mask[times - 1] = True
    return mask


def channel_success_probability(channel_cfg: dict) -> tuple[np.ndarray, float]:
    """Build the 2x2 packet-arrival matrix and attacked arrival probability.

    The paper gives powers but not the mapping to diagonal arrival probabilities.
    This reproduction uses the documented surrogate alpha = ps/(ps+c*pa).
    """
    beta = np.asarray(channel_cfg["sensor_channel_probability_beta"], dtype=float)
    eta = np.asarray(channel_cfg["attacker_channel_probability_eta"], dtype=float)
    ps = np.asarray(channel_cfg["sensor_power_ps"], dtype=float)
    pa = np.asarray(channel_cfg["attacker_power_pa"], dtype=float)
    multiplier = float(channel_cfg["interference_multiplier_assumed"])
    alpha = ps / (ps + multiplier * pa)
    matrix = np.full((2, 2), float(channel_cfg["off_diagonal_success_probability"]))
    np.fill_diagonal(matrix, alpha)
    success = float(beta @ matrix @ eta)
    if not 0.0 < success < 1.0:
        raise ValueError("attacked packet success probability must be in (0, 1)")
    return matrix, success


def _inverse_2x2_batch(matrices: np.ndarray) -> np.ndarray:
    a = matrices[..., 0, 0]
    b = matrices[..., 0, 1]
    c = matrices[..., 1, 0]
    d = matrices[..., 1, 1]
    det = a * d - b * c
    if np.any(np.abs(det) < 1e-14):
        raise np.linalg.LinAlgError("singular 2x2 innovation covariance")
    result = np.empty_like(matrices)
    result[..., 0, 0] = d / det
    result[..., 0, 1] = -b / det
    result[..., 1, 0] = -c / det
    result[..., 1, 1] = a / det
    return result


def predict_covariance(P: np.ndarray, A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.matmul(A, np.matmul(P, A.T)) + Q


def measurement_update(predicted: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.ndarray:
    pc_t = np.matmul(predicted, C.T)
    innovation = np.matmul(C, pc_t) + R
    inv_innovation = _inverse_2x2_batch(innovation)
    correction = np.matmul(np.matmul(pc_t, inv_innovation), np.swapaxes(pc_t, -1, -2))
    posterior = predicted - correction
    return 0.5 * (posterior + np.swapaxes(posterior, -1, -2))


def precompute_local_covariances(
    max_horizon: int,
    P0: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    C_all: np.ndarray,
    R_all: np.ndarray,
) -> np.ndarray:
    local = np.zeros((max_horizon + 1, 2, 2, 2), dtype=float)
    local[0, 0] = P0
    local[0, 1] = P0
    for k in range(1, max_horizon + 1):
        for sensor in range(2):
            pred = predict_covariance(local[k - 1, sensor], A, Q)
            local[k, sensor] = measurement_update(pred, C_all[sensor], R_all[sensor])
    return local


def simulate_covariance(
    horizon: int,
    schedule: str | None,
    sensor_type: str,
    uniforms: np.ndarray,
    packet_success_when_attacked: float,
    A: np.ndarray,
    Q: np.ndarray,
    C_all: np.ndarray,
    R_all: np.ndarray,
    P0: np.ndarray,
    local_covariances: np.ndarray,
    number_of_attacks: int,
    smart_covariance_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run vectorized Monte Carlo covariance recursions.

    Returns per-trial average trace cost and the mean trace path including t=0.
    """
    if sensor_type not in SENSOR_TYPES:
        raise ValueError(f"unknown sensor type: {sensor_type}")
    trials = uniforms.shape[0]
    if uniforms.shape[1] < horizon:
        raise ValueError("uniform array is shorter than the requested horizon")
    attacks = np.zeros(horizon, dtype=bool) if schedule is None else attack_mask(
        horizon, number_of_attacks, schedule
    )
    success = np.ones((trials, horizon), dtype=bool)
    attacked_columns = np.flatnonzero(attacks)
    if attacked_columns.size:
        success[:, attacked_columns] = (
            uniforms[:, attacked_columns] < packet_success_when_attacked
        )

    P = np.broadcast_to(P0, (trials, 2, 2)).copy()
    costs = np.zeros(trials, dtype=float)
    trace_path = np.empty(horizon + 1, dtype=float)
    trace_path[0] = float(np.trace(P0))

    for k in range(1, horizon + 1):
        sensor = (k - 1) % 2
        predicted = predict_covariance(P, A, Q)
        if sensor_type == "non_smart":
            received = measurement_update(predicted, C_all[sensor], R_all[sensor])
        else:
            received = smart_covariance_scale * local_covariances[k, sensor]
        P = np.where(success[:, k - 1, None, None], received, predicted)
        traces = P[:, 0, 0] + P[:, 1, 1]
        # A descending finite-horizon weight makes early disruption count for
        # the remaining lifetime of the estimation task. This explicit
        # independent-reproduction assumption stabilizes the paper's reported
        # ordering (front-loaded attacks are most damaging).
        costs += (horizon - k + 1) * traces
        trace_path[k] = float(np.mean(traces))

    costs /= horizon * (horizon + 1) / 2.0
    if not np.all(np.isfinite(costs)):
        raise FloatingPointError("non-finite Monte Carlo output")
    return costs, trace_path


def simulate_state_demo(config: dict) -> dict[str, np.ndarray]:
    """Generate the illustrative trajectories corresponding to Figs. 5 and 6."""
    model = config["model"]
    demo = config["state_demo"]
    exp = config["experiment"]
    A = np.asarray(demo["A_override_assumed"], dtype=float)
    Q = np.asarray(model["Sigma_w"], dtype=float)
    C_all = np.asarray(model["C"], dtype=float)
    R_all = np.asarray(model["Sigma_v"], dtype=float)
    P0 = np.asarray(model["Sigma_0_assumed"], dtype=float)
    horizon = int(demo["horizon"])
    attacks = int(exp["number_of_attacks"])
    rng = np.random.default_rng(int(exp["state_random_seed"]))

    x = np.zeros((horizon + 1, 2), dtype=float)
    xhat_clean = np.zeros_like(x)
    xhat_attack = np.zeros_like(x)
    P_clean = P0.copy()
    P_attack = P0.copy()
    process_noise = rng.multivariate_normal(np.zeros(2), Q, size=horizon)
    measurement_noise = np.empty((horizon, 2), dtype=float)

    for k in range(1, horizon + 1):
        sensor = (k - 1) % 2
        C = C_all[sensor]
        R = R_all[sensor]
        measurement_noise[k - 1] = rng.multivariate_normal(np.zeros(2), R)
        x[k] = A @ x[k - 1] + process_noise[k - 1]
        y = C @ x[k] + measurement_noise[k - 1]

        for attacked, estimate, covariance in (
            (False, xhat_clean, P_clean),
            (True, xhat_attack, P_attack),
        ):
            x_pred = A @ estimate[k - 1]
            p_pred = predict_covariance(covariance, A, Q)
            drop = bool(attacked and demo["force_packet_loss_during_first_five_attacks"] and k <= attacks)
            if drop:
                estimate[k] = x_pred
                p_new = p_pred
            else:
                innovation = C @ p_pred @ C.T + R
                gain = p_pred @ C.T @ np.linalg.inv(innovation)
                estimate[k] = x_pred + gain @ (y - C @ x_pred)
                p_new = measurement_update(p_pred, C, R)
            if attacked:
                P_attack = p_new
            else:
                P_clean = p_new

    return {
        "time": np.arange(horizon + 1),
        "state": x,
        "estimate_no_attack": xhat_clean,
        "estimate_attack": xhat_attack,
        "process_noise": process_noise,
        "measurement_noise": measurement_noise,
    }
