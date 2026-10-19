"""Shared 50-start Case-B benchmark with three fixed stationary obstacles."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_case_b_benchmark import Benchmark, ROOT, run_evaluation


DEFAULT_DATASET = ROOT / "benchmark data" / "case_b_obstacle_benchmark.csv"
NUM_TRIALS = 50
RANDOM_SEED = 0
GOAL = np.array([0.0, 0.0, 0.0, 0.0])
REPRESENTATIVE_INITIAL_STATE = np.array([-5.24, 4.11, 2.72, 0.0])
STATE_NAMES = ("x", "y", "theta", "v")
SAMPLING_BOUNDS = {
    "x": (-8.0, -3.0), "y": (2.0, 7.0),
    "theta": (-np.pi, np.pi), "v": (0.0, 0.0),
}
WORKSPACE_BOUNDS = {"x": (-8.0, 3.0), "y": (-2.0, 8.0)}
OBSTACLES = (
    {"xo": -4.0, "yo": 2.0, "r": 0.5},
    {"xo": -1.5, "yo": 1.5, "r": 0.5},
    {"xo": -2.5, "yo": 3.0, "r": 0.5},
)
ROBOT_RADIUS = 0.25
SAFETY_MARGIN = 0.05
DT, SIMULATION_STEPS, SUCCESS_THRESHOLD = 0.05, 200, 0.6
ACTION_LOW, ACTION_HIGH = (-1.0, -4.0), (1.0, 4.0)

CSV_COLUMNS = [
    "trial", *STATE_NAMES, "dataset_id", "schema_version", "case", "scenario",
    "num_trials", "random_seed", "random_generator", "sample_attempts",
    "rejected_workspace", "rejected_obstacle", "dt", "simulation_steps",
    "success_threshold", "success_rule", "goal_json", "sampling_bounds_json",
    "training_bounds_json", "workspace_bounds_json", "action_low_json",
    "action_high_json", "obstacles_json", "robot_radius", "safety_margin",
    "q_diag_json", "r_diag_json", "terminal_weight", "disturbance_enabled",
    "terminate_on_goal",
]
METADATA_COLUMNS = CSV_COLUMNS[5:]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _dataset_id(states, metadata):
    description = {key: value for key, value in metadata.items() if key != "dataset_id"}
    digest = hashlib.sha256(np.asarray(states, dtype="<f8").tobytes(order="C"))
    digest.update(_json(description).encode("utf-8"))
    return digest.hexdigest()


def _collision_free(position, obstacles=OBSTACLES, clearance=SAFETY_MARGIN):
    for obstacle in obstacles:
        minimum = float(obstacle["r"]) + ROBOT_RADIUS + clearance
        if np.linalg.norm(position - [obstacle["xo"], obstacle["yo"]]) <= minimum:
            return False
    return True


def generate_initial_states(num_trials=NUM_TRIALS, seed=RANDOM_SEED):
    if num_trials < 1:
        raise ValueError("num_trials must be positive.")
    rng = np.random.default_rng(seed)
    # Keep the canonical Case-B state as trial 1, followed by random states.
    accepted, attempts = [REPRESENTATIVE_INITIAL_STATE.copy()], 0
    rejected_workspace = rejected_obstacle = 0
    while len(accepted) < num_trials and attempts < 1000 * num_trials:
        attempts += 1
        state = np.array([
            rng.uniform(*SAMPLING_BOUNDS["x"]),
            rng.uniform(*SAMPLING_BOUNDS["y"]),
            rng.uniform(*SAMPLING_BOUNDS["theta"]), 0.0,
        ])
        if not (WORKSPACE_BOUNDS["x"][0] + ROBOT_RADIUS <= state[0] <= WORKSPACE_BOUNDS["x"][1] - ROBOT_RADIUS
                and WORKSPACE_BOUNDS["y"][0] + ROBOT_RADIUS <= state[1] <= WORKSPACE_BOUNDS["y"][1] - ROBOT_RADIUS):
            rejected_workspace += 1
            continue
        if not _collision_free(state[:2]):
            rejected_obstacle += 1
            continue
        accepted.append(state)
    if len(accepted) != num_trials:
        raise RuntimeError(f"Generated only {len(accepted)} valid states in {attempts} attempts.")
    metadata = {
        "schema_version": 1, "case": "b", "scenario": "three_fixed_obstacles",
        "state_names": list(STATE_NAMES), "num_trials": int(num_trials),
        "random_seed": int(seed), "random_generator": "numpy.random.default_rng/PCG64",
        "sample_attempts": attempts, "rejected_workspace": rejected_workspace,
        "rejected_obstacle": rejected_obstacle, "goal": GOAL.tolist(),
        "sampling_bounds": {k: list(v) for k, v in SAMPLING_BOUNDS.items()},
        "training_bounds": {k: list(v) for k, v in SAMPLING_BOUNDS.items()},
        "workspace_bounds": {k: list(v) for k, v in WORKSPACE_BOUNDS.items()},
        "dt": DT, "simulation_steps": SIMULATION_STEPS,
        "success_threshold": SUCCESS_THRESHOLD,
        "success_rule": "complete rollout, wrapped-angle ACE <= threshold, and no physical obstacle collision",
        "action_low": list(ACTION_LOW), "action_high": list(ACTION_HIGH),
        "obstacles": [dict(item) for item in OBSTACLES],
        "robot_radius": ROBOT_RADIUS, "safety_margin": SAFETY_MARGIN,
        "q_diag": [10.0] * 4, "r_diag": [1.0, 1.0], "terminal_weight": 50.0,
        "disturbance_enabled": False, "terminate_on_goal": False,
    }
    states = np.asarray(accepted)
    metadata["dataset_id"] = _dataset_id(states, metadata)
    return states, metadata


def _row_metadata(metadata):
    direct = ["dataset_id", "schema_version", "case", "scenario", "num_trials",
              "random_seed", "random_generator", "sample_attempts", "rejected_workspace",
              "rejected_obstacle", "dt", "simulation_steps", "success_threshold",
              "success_rule", "robot_radius", "safety_margin", "terminal_weight"]
    row = {key: metadata[key] for key in direct}
    for key in ("goal", "sampling_bounds", "training_bounds", "workspace_bounds",
                "action_low", "action_high", "obstacles", "q_diag", "r_diag"):
        row[f"{key}_json"] = _json(metadata[key])
    row["disturbance_enabled"] = int(metadata["disturbance_enabled"])
    row["terminate_on_goal"] = int(metadata["terminate_on_goal"])
    return row


def _parse_metadata(row):
    integer_keys = ("schema_version", "num_trials", "random_seed", "sample_attempts",
                    "rejected_workspace", "rejected_obstacle")
    float_keys = ("dt", "success_threshold", "robot_radius", "safety_margin", "terminal_weight")
    metadata = {key: int(row[key]) for key in integer_keys}
    metadata.update({key: float(row[key]) for key in float_keys})
    metadata.update({key: row[key] for key in ("dataset_id", "case", "scenario", "random_generator", "success_rule")})
    metadata["simulation_steps"] = int(row["simulation_steps"])
    for key in ("goal", "sampling_bounds", "training_bounds", "workspace_bounds",
                "action_low", "action_high", "obstacles", "q_diag", "r_diag"):
        metadata[key] = json.loads(row[f"{key}_json"])
    metadata["state_names"] = list(STATE_NAMES)
    metadata["disturbance_enabled"] = bool(int(row["disturbance_enabled"]))
    metadata["terminate_on_goal"] = bool(int(row["terminate_on_goal"]))
    return metadata


def save_benchmark(path=DEFAULT_DATASET, num_trials=NUM_TRIALS, seed=RANDOM_SEED, overwrite=False):
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {path}. Use --overwrite to replace it.")
    states, metadata = generate_initial_states(num_trials, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = _row_metadata(metadata)
    with path.open("w" if overwrite else "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for trial, state in enumerate(states, 1):
            writer.writerow({"trial": trial,
                             **dict(zip(STATE_NAMES, (repr(float(value)) for value in state))),
                             **fixed})
    return load_benchmark(path)


def load_benchmark(path=DEFAULT_DATASET):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}. Run generate_case_b_obstacle_benchmark.py.")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("Unexpected Case-B obstacle CSV columns.")
        rows = list(reader)
    if not rows or any({k: r[k] for k in METADATA_COLUMNS} != {k: rows[0][k] for k in METADATA_COLUMNS} for r in rows[1:]):
        raise ValueError("CSV is empty or its settings differ between rows.")
    metadata = _parse_metadata(rows[0])
    states = np.asarray([[float(row[k]) for k in STATE_NAMES] for row in rows])
    trials = np.asarray([int(row["trial"]) for row in rows])
    if metadata["case"] != "b" or metadata["scenario"] != "three_fixed_obstacles":
        raise ValueError("This is not the Case-B three-obstacle benchmark.")
    if metadata["num_trials"] != len(states) or not np.array_equal(trials, np.arange(1, len(states) + 1)):
        raise ValueError("Trial count or ordering is invalid.")
    if not np.array_equal(states[0], REPRESENTATIVE_INITIAL_STATE):
        raise ValueError("Trial 1 must be the representative Case-B initial state.")
    if metadata["obstacles"] != [dict(item) for item in OBSTACLES] or metadata["goal"] != GOAL.tolist():
        raise ValueError("Goal or obstacle geometry differs from Case B.")
    if not np.isfinite(states).all() or np.any(states[:, 3] != 0):
        raise ValueError("Initial states must be finite and stationary.")
    bounds = metadata["sampling_bounds"]
    for index, name in enumerate(STATE_NAMES):
        if np.any(states[:, index] < bounds[name][0]) or np.any(states[:, index] > bounds[name][1]):
            raise ValueError(f"Initial {name} lies outside its sampling bounds.")
    if any(not _collision_free(state[:2]) for state in states):
        raise ValueError("An initial state violates an obstacle safety boundary.")
    if metadata["dataset_id"] != _dataset_id(states, metadata):
        raise ValueError("Dataset ID mismatch: states or settings changed.")
    states.setflags(write=False)
    goal = np.asarray(metadata["goal"]); goal.setflags(write=False)
    return Benchmark(states, goal, metadata, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    benchmark = save_benchmark(args.output, args.trials, args.seed, args.overwrite)
    print(f"Saved {len(benchmark.initial_states)} Case-B obstacle starts to {benchmark.path}")
    print(f"Dataset ID: {benchmark.dataset_id}")
    print(f"Attempts: {benchmark.metadata['sample_attempts']}; workspace rejections: "
          f"{benchmark.metadata['rejected_workspace']}; obstacle rejections: "
          f"{benchmark.metadata['rejected_obstacle']}")


if __name__ == "__main__":
    main()
