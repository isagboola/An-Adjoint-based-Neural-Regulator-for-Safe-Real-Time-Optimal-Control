"""Generate the saved no-obstacle case-C dataset and shared evaluation tools.

Run this file once to generate data. Controller evaluators import its helpers;
importing it never generates or replaces a dataset.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
BENCHMARK_DATA_DIR = ROOT / "benchmark data"
DEFAULT_DATASET = BENCHMARK_DATA_DIR / "case_c_benchmark.csv"

# Edit these settings before generating a new benchmark file.
NUM_TRIALS = 100
RANDOM_SEED = 2026
MIN_DISTANCE_TO_GOAL = 0.5
GOAL = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
REPRESENTATIVE_INITIAL_STATE = np.array([-5.24, 4.11, 2.72, 0.0], dtype=float)
STATE_NAMES = ("x", "y", "theta", "v")
SAMPLING_BOUNDS = {"x": (-6.0, 6.0), "y": (-6.0, 6.0),
                   "theta": (-2.0, 2.0), "v": (0.0, 0.0)}
TRAINING_BOUNDS = {name: (-2.0, 2.0) for name in STATE_NAMES}
DT = 0.05
SIMULATION_STEPS = 200
SUCCESS_THRESHOLD = 0.6
ACTION_LOW = (-1.0, -4.0)
ACTION_HIGH = (1.0, 4.0)

CSV_STATE_COLUMNS = ["trial", *STATE_NAMES]
CSV_METADATA_COLUMNS = [
    "dataset_id", "schema_version", "case", "num_trials", "random_seed",
    "random_generator", "minimum_distance_to_goal", "sample_attempts",
    "rejected_training", "rejected_distance", "dt", "simulation_steps",
    "success_threshold", "exclusion_rule", "success_rule", "obstacle_count",
    "disturbance_enabled", "terminate_on_goal", "terminal_weight",
    *(f"goal_{name}" for name in STATE_NAMES),
    *(f"sampling_{name}_{side}" for name in STATE_NAMES for side in ("min", "max")),
    *(f"training_{name}_{side}" for name in STATE_NAMES for side in ("min", "max")),
    "action_a_min", "action_a_max", "action_omega_min", "action_omega_max",
    *(f"q_{name}" for name in STATE_NAMES), "r_a", "r_omega",
]
CSV_COLUMNS = CSV_STATE_COLUMNS + CSV_METADATA_COLUMNS


@dataclass(frozen=True)
class Benchmark:
    initial_states: np.ndarray
    goal: np.ndarray
    metadata: dict
    path: Path

    @property
    def dt(self):
        return float(self.metadata["dt"])

    @property
    def steps(self):
        return int(self.metadata["simulation_steps"])

    @property
    def success_threshold(self):
        return float(self.metadata["success_threshold"])

    @property
    def action_low(self):
        return np.asarray(self.metadata["action_low"], dtype=float)

    @property
    def action_high(self):
        return np.asarray(self.metadata["action_high"], dtype=float)

    @property
    def dataset_id(self):
        return self.metadata["dataset_id"]


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def state_error(states, goal):
    error = np.asarray(states, dtype=float) - np.asarray(goal, dtype=float)
    error = error.copy()
    error[..., 2] = wrap_to_pi(error[..., 2])
    return error


def _dataset_id(states, metadata):
    description = {key: value for key, value in metadata.items() if key != "dataset_id"}
    digest = hashlib.sha256()
    digest.update(np.asarray(states, dtype="<f8").tobytes(order="C"))
    digest.update(json.dumps(description, sort_keys=True, allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def generate_initial_states(num_trials=NUM_TRIALS, seed=RANDOM_SEED,
                            d_min=MIN_DISTANCE_TO_GOAL):
    if num_trials < 1 or not np.isfinite(d_min) or d_min < 0:
        raise ValueError("num_trials must be positive and d_min finite and nonnegative.")
    # Match the N-trailer sampling convention, including a draw for the fixed v.
    rng = np.random.RandomState(seed)
    # Keep the canonical paper/example state as trial 1, followed by random states.
    accepted = [REPRESENTATIVE_INITIAL_STATE.copy()]
    rejected_training = rejected_distance = attempts = 0
    while len(accepted) < num_trials and attempts < 200 * num_trials:
        attempts += 1
        sample = np.array([rng.uniform(*SAMPLING_BOUNDS[key]) for key in STATE_NAMES])
        inside_position_box = all(
            TRAINING_BOUNDS[key][0] <= sample[index] <= TRAINING_BOUNDS[key][1]
            for index, key in enumerate(STATE_NAMES[:2])
        )
        if inside_position_box:
            rejected_training += 1
            continue
        if np.linalg.norm(sample[:2] - GOAL[:2]) < d_min:
            rejected_distance += 1
            continue
        accepted.append(sample)
    if len(accepted) != num_trials:
        raise ValueError(f"Only {len(accepted)} states accepted in {attempts} attempts; check bounds/d_min.")
    metadata = {
        "schema_version": 1, "case": "c", "state_names": list(STATE_NAMES),
        "num_trials": int(num_trials), "random_seed": int(seed),
        "random_generator": "numpy.random.RandomState/MT19937",
        "goal": GOAL.tolist(), "sampling_bounds": SAMPLING_BOUNDS,
        "training_bounds": TRAINING_BOUNDS,
        "exclusion_rule": "outside closed training position box in physical coordinates",
        "minimum_distance_to_goal": float(d_min), "sample_attempts": attempts,
        "rejected_training": rejected_training, "rejected_distance": rejected_distance,
        "dt": DT, "simulation_steps": SIMULATION_STEPS,
        "success_threshold": SUCCESS_THRESHOLD,
        "success_rule": "complete rollout and sum(abs(final error)), with wrapped theta, <= threshold",
        "action_low": list(ACTION_LOW), "action_high": list(ACTION_HIGH),
        "obstacles": [], "disturbance_enabled": False, "terminate_on_goal": False,
        "q_diag": [10.0] * 4, "r_diag": [1.0, 1.0], "terminal_weight": 50.0,
    }
    states = np.asarray(accepted, dtype=float)
    metadata["dataset_id"] = _dataset_id(states, metadata)
    return states, metadata


def _metadata_to_csv_fields(metadata):
    fields = {
        "dataset_id": metadata["dataset_id"],
        "schema_version": metadata["schema_version"],
        "case": metadata["case"],
        "num_trials": metadata["num_trials"],
        "random_seed": metadata["random_seed"],
        "random_generator": metadata["random_generator"],
        "minimum_distance_to_goal": metadata["minimum_distance_to_goal"],
        "sample_attempts": metadata["sample_attempts"],
        "rejected_training": metadata["rejected_training"],
        "rejected_distance": metadata["rejected_distance"],
        "dt": metadata["dt"],
        "simulation_steps": metadata["simulation_steps"],
        "success_threshold": metadata["success_threshold"],
        "exclusion_rule": metadata["exclusion_rule"],
        "success_rule": metadata["success_rule"],
        "obstacle_count": len(metadata["obstacles"]),
        "disturbance_enabled": int(metadata["disturbance_enabled"]),
        "terminate_on_goal": int(metadata["terminate_on_goal"]),
        "terminal_weight": metadata["terminal_weight"],
        "action_a_min": metadata["action_low"][0],
        "action_a_max": metadata["action_high"][0],
        "action_omega_min": metadata["action_low"][1],
        "action_omega_max": metadata["action_high"][1],
        "r_a": metadata["r_diag"][0],
        "r_omega": metadata["r_diag"][1],
    }
    for index, name in enumerate(STATE_NAMES):
        fields[f"goal_{name}"] = metadata["goal"][index]
        fields[f"sampling_{name}_min"], fields[f"sampling_{name}_max"] = metadata["sampling_bounds"][name]
        fields[f"training_{name}_min"], fields[f"training_{name}_max"] = metadata["training_bounds"][name]
        fields[f"q_{name}"] = metadata["q_diag"][index]
    return fields


def _parse_csv_metadata(row):
    def integer(name):
        value = int(row[name])
        if str(value) != row[name].strip():
            raise ValueError(f"CSV field {name!r} must be an integer.")
        return value

    def number(name):
        value = float(row[name])
        if not np.isfinite(value):
            raise ValueError(f"CSV field {name!r} must be finite.")
        return value

    obstacle_count = integer("obstacle_count")
    if obstacle_count != 0:
        raise ValueError("The case-C initial-state benchmark must be obstacle free.")
    metadata = {
        "dataset_id": row["dataset_id"],
        "schema_version": integer("schema_version"),
        "case": row["case"],
        "state_names": list(STATE_NAMES),
        "num_trials": integer("num_trials"),
        "random_seed": integer("random_seed"),
        "random_generator": row["random_generator"],
        "goal": [number(f"goal_{name}") for name in STATE_NAMES],
        "sampling_bounds": {
            name: [number(f"sampling_{name}_min"), number(f"sampling_{name}_max")]
            for name in STATE_NAMES
        },
        "training_bounds": {
            name: [number(f"training_{name}_min"), number(f"training_{name}_max")]
            for name in STATE_NAMES
        },
        "exclusion_rule": row["exclusion_rule"],
        "minimum_distance_to_goal": number("minimum_distance_to_goal"),
        "sample_attempts": integer("sample_attempts"),
        "rejected_training": integer("rejected_training"),
        "rejected_distance": integer("rejected_distance"),
        "dt": number("dt"),
        "simulation_steps": integer("simulation_steps"),
        "success_threshold": number("success_threshold"),
        "success_rule": row["success_rule"],
        "action_low": [number("action_a_min"), number("action_omega_min")],
        "action_high": [number("action_a_max"), number("action_omega_max")],
        "obstacles": [],
        "disturbance_enabled": bool(integer("disturbance_enabled")),
        "terminate_on_goal": bool(integer("terminate_on_goal")),
        "q_diag": [number(f"q_{name}") for name in STATE_NAMES],
        "r_diag": [number("r_a"), number("r_omega")],
        "terminal_weight": number("terminal_weight"),
    }
    if row["disturbance_enabled"] not in ("0", "1") or row["terminate_on_goal"] not in ("0", "1"):
        raise ValueError("CSV Boolean fields must be 0 or 1.")
    return metadata


def load_benchmark(path=DEFAULT_DATASET):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared dataset: {path}. Run generate_case_c_benchmark.py first.")
    if path.suffix.lower() != ".csv":
        raise ValueError("The shared benchmark dataset must be a .csv file.")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("Unexpected case-C CSV columns or column order.")
        rows = list(reader)
    if not rows:
        raise ValueError("The case-C CSV contains no initial states.")
    first_metadata_fields = {name: rows[0][name] for name in CSV_METADATA_COLUMNS}
    if any({name: row[name] for name in CSV_METADATA_COLUMNS} != first_metadata_fields for row in rows[1:]):
        raise ValueError("Benchmark settings must be identical on every CSV row.")
    metadata = _parse_csv_metadata(rows[0])
    try:
        trial_ids = np.asarray([int(row["trial"]) for row in rows], dtype=int)
        states = np.asarray([[float(row[name]) for name in STATE_NAMES] for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid trial number or initial-state value in the case-C CSV.") from exc
    goal = np.asarray(metadata["goal"], dtype=float)
    if metadata.get("schema_version") != 1 or metadata.get("case") != "c":
        raise ValueError("Expected a version-1 case-C benchmark.")
    if states.ndim != 2 or states.shape[1] != 4 or len(states) < 1 or not np.isfinite(states).all():
        raise ValueError("Initial states must be a finite, nonempty (trials, 4) array.")
    if goal.shape != (4,) or not np.array_equal(goal, GOAL):
        raise ValueError("Case C requires goal [1, 1, 0, 0].")
    if metadata["num_trials"] != len(states) or not np.array_equal(trial_ids, np.arange(1, len(states) + 1)):
        raise ValueError("Trial count/order does not match the saved benchmark.")
    if not np.array_equal(states[0], REPRESENTATIVE_INITIAL_STATE):
        raise ValueError("Trial 1 must be the representative Case-C initial state.")
    if metadata["goal"] != goal.tolist() or metadata["state_names"] != list(STATE_NAMES):
        raise ValueError("State names or goal metadata do not match the dataset.")
    if metadata.get("dataset_id") != _dataset_id(states, metadata):
        raise ValueError("Dataset ID mismatch: the states or metadata have changed.")
    bounds = np.array([metadata["sampling_bounds"][key] for key in STATE_NAMES])
    if bounds.shape != (4, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Invalid sampling bounds.")
    random_states = states[~np.all(states == REPRESENTATIVE_INITIAL_STATE, axis=1)]
    if ((len(random_states) and
         (np.any(random_states < bounds[:, 0]) or np.any(random_states > bounds[:, 1])))
            or np.any(states[:, 3] != 0)):
        raise ValueError("Random states violate case-C sampling bounds or a state violates the v=0 rule.")
    training_xy = np.array([metadata["training_bounds"][key] for key in STATE_NAMES[:2]])
    inside = np.all((states[:, :2] >= training_xy[:, 0]) & (states[:, :2] <= training_xy[:, 1]), axis=1)
    distances = np.linalg.norm(states[:, :2] - goal[:2], axis=1)
    if np.any(inside) or np.any(distances < metadata["minimum_distance_to_goal"]):
        raise ValueError("Dataset violates case-C exclusion or distance rules.")
    if metadata["obstacles"] or metadata["disturbance_enabled"] or metadata["terminate_on_goal"]:
        raise ValueError("This benchmark requires nominal obstacle-free full-duration evaluation.")
    if not np.isfinite(metadata["dt"]) or metadata["dt"] <= 0 or metadata["simulation_steps"] < 2:
        raise ValueError("Invalid simulation interval or length.")
    if int(metadata["simulation_steps"]) != metadata["simulation_steps"]:
        raise ValueError("simulation_steps must be an integer.")
    if not np.isfinite(metadata["success_threshold"]) or metadata["success_threshold"] < 0:
        raise ValueError("Invalid success threshold.")
    low, high = np.asarray(metadata["action_low"]), np.asarray(metadata["action_high"])
    if low.shape != (2,) or high.shape != (2,) or not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("Invalid action bounds.")
    states.setflags(write=False)
    goal.setflags(write=False)
    return Benchmark(states, goal, metadata, path)


def save_benchmark(path, num_trials=NUM_TRIALS, seed=RANDOM_SEED,
                   d_min=MIN_DISTANCE_TO_GOAL, overwrite=False):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".csv":
        raise ValueError("The dataset filename must end with .csv.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {path}. Use another --output or explicit --overwrite.")
    states, metadata = generate_initial_states(num_trials, seed, d_min)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_fields = _metadata_to_csv_fields(metadata)
    with path.open("w" if overwrite else "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for trial, state in enumerate(states, start=1):
            row = {"trial": trial, **{name: repr(float(state[index])) for index, name in enumerate(STATE_NAMES)}}
            row.update(metadata_fields)
            writer.writerow(row)
    return load_benchmark(path)


def compute_rollout_metrics(states, actions, benchmark):
    error = state_error(states, benchmark.goal)
    ace = float(np.abs(error[-1]).sum())
    # Unwrap heading only for derivatives; final tracking error is wrapped above.
    derivative_states = states.copy()
    derivative_states[:, 2] = np.unwrap(derivative_states[:, 2])
    state_msd = float(np.mean(np.gradient(derivative_states, benchmark.dt, axis=0) ** 2)) if len(states) > 1 else 0.0
    control_msd = float(np.mean(np.gradient(actions, benchmark.dt, axis=0) ** 2)) if len(actions) > 1 else 0.0
    cost = None
    if len(actions) == benchmark.steps:
        q = np.asarray(benchmark.metadata["q_diag"])
        r = np.asarray(benchmark.metadata["r_diag"])
        cost = float(np.sum(error[:-1] ** 2 * q) + np.sum(actions ** 2 * r)
                     + benchmark.metadata["terminal_weight"] * np.sum(error[-1] ** 2 * q))
    return {"ace": ace, "avg_state_msd": state_msd, "avg_control_msd": control_msd,
            "quadratic_cost": cost}


def _save_summary_plot(records, benchmark, output_path, controller_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    style = {
        "font.family": "sans-serif",
        "font.weight": "normal",
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 28,
        "text.usetex": False,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 7))
        ordered_records = sorted(
            records,
            key=lambda item: np.array_equal(item["state_traj"][0], REPRESENTATIVE_INITIAL_STATE),
        )
        for record in ordered_records:
            states = record["state_traj"]
            representative = np.array_equal(states[0], REPRESENTATIVE_INITIAL_STATE)
            color = "purple" if representative else ("tab:green" if record["success"] else "tab:red")
            ax.plot(states[:, 0], states[:, 1], color=color,
                    alpha=0.95 if representative else 0.4,
                    linewidth=2.6 if representative else 1.3,
                    zorder=4 if representative else 2)
            ax.scatter(*states[0, :2], color=color, marker="o",
                       s=32 if representative else 22, zorder=4 if representative else 2)
            ax.scatter(*states[-1, :2], color=color, marker="x",
                       s=45 if representative else 35, zorder=4 if representative else 2)
        bounds = benchmark.metadata["training_bounds"]
        xlow, xhigh = bounds["x"]
        ylow, yhigh = bounds["y"]
        ax.add_patch(Rectangle((xlow, ylow), xhigh-xlow, yhigh-ylow,
                               fill=False, edgecolor="tab:blue", linestyle="--", linewidth=2))
        obstacles = benchmark.metadata.get("obstacles", [])
        robot_radius = float(benchmark.metadata.get("robot_radius", 0.25))
        safety_margin = float(benchmark.metadata.get("safety_margin", 0.05))
        for obstacle in obstacles:
            center = (float(obstacle["xo"]), float(obstacle["yo"]))
            radius = float(obstacle["r"])
            ax.add_patch(plt.Circle(center, radius, facecolor="0.35", edgecolor="black",
                                    linewidth=1.5, alpha=0.85, zorder=3))
            ax.add_patch(plt.Circle(center, radius + robot_radius + safety_margin,
                                    fill=False, edgecolor="black", linestyle=":",
                                    linewidth=1.5, alpha=0.8, zorder=3))
        ax.scatter(*benchmark.goal[:2], color="gold", edgecolor="black", marker="*", s=180, zorder=5)
        sample = benchmark.metadata["sampling_bounds"]
        ax.set(
            xlim=tuple(sample["x"]), ylim=tuple(sample["y"]),
            xlabel="x (m)", ylabel="y (m)",
           # title=f"Case {str(benchmark.metadata['case']).upper()} — {controller_name}-PB"
           # title=f"Case {str(benchmark.metadata['case']).upper()} — {controller_name}-QC", #FOR RL when using QC reward(run 4), add -QC to title
           # title=f"Case {str(benchmark.metadata['case']).upper()} — {controller_name}-QC", #FOR RL when using PB reward(run 5), add -PB to title
        )
        ax.set_xticks(np.arange(sample["x"][0], sample["x"][1] + 0.1, 2.0))
        ax.set_yticks(np.arange(sample["y"][0], sample["y"][1] + 0.1, 2.0))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        start_markers = (
            Line2D([], [], marker="o", ls="None", color="tab:green", markersize=9),
            Line2D([], [], marker=r"$/$", ls="None", color="black", markersize=9),
            Line2D([], [], marker="o", ls="None", color="tab:red", markersize=9),
        )
        end_markers = (
            Line2D([], [], marker="x", ls="None", color="tab:green", markersize=9,
                   markeredgewidth=2),
            Line2D([], [], marker=r"$/$", ls="None", color="black", markersize=9),
            Line2D([], [], marker="x", ls="None", color="tab:red", markersize=9,
                   markeredgewidth=2),
        )
        handles = [
            start_markers,
            end_markers,
            Line2D([], [], marker="*", color="gold", markeredgecolor="black",
                   linestyle="None", markersize=15),
            Line2D([], [], color="tab:blue", linestyle="--", linewidth=2),
        ]
        labels = ["Start", "End", "Goal", "Training\ndomain"]
        if obstacles:
            handles.extend([
                Line2D([], [], marker="o", color="black", markerfacecolor="0.35",
                       linestyle="None", markersize=11),
                Line2D([], [], color="black", linestyle=":", linewidth=1.5),
            ])
            labels.extend(["Obstacle", "Safety boundary"])
        # ax.legend(
        #     handles=handles, labels=labels, loc="lower left", frameon=True,
        #     ncol=2 if obstacles else 1,
        #     handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
        # )
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def run_evaluation(benchmark, rollout, *, controller_name, controller_metadata,
                   output_dir, max_trials=None, make_plot=True):
    count = len(benchmark.initial_states) if max_trials is None else max_trials
    if not isinstance(count, (int, np.integer)) or not 1 <= count <= len(benchmark.initial_states):
        raise ValueError("max_trials must select between 1 and the saved number of trials.")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, payload = [], {}
    started = time.perf_counter()
    fieldnames = ["trial", "success", "completed", "steps", "ace", "avg_state_msd",
                  "avg_control_msd", "quadratic_cost", "mean_controller_ms", "failure_reason"]
    fieldnames += [f"initial_{name}" for name in STATE_NAMES] + [f"final_{name}" for name in STATE_NAMES]
    print(f"{controller_name}: case {str(benchmark.metadata['case']).upper()}, "
          f"{count}/{len(benchmark.initial_states)} saved trials")
    print(f"Dataset ID: {benchmark.dataset_id}")
    print(f"dt={benchmark.dt:g}, steps={benchmark.steps}, ACE threshold={benchmark.success_threshold:g}")
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, initial in enumerate(benchmark.initial_states[:count], start=1):
            result = rollout(initial.copy(), benchmark)
            states = np.asarray(result["state_traj"], dtype=float)
            actions = np.asarray(result["action_traj"], dtype=float).reshape(-1, 2)
            timings = np.asarray(result["controller_times"], dtype=float).reshape(-1)
            if states.shape != (len(actions) + 1, 4) or len(actions) > benchmark.steps:
                raise ValueError(f"Trial {index}: invalid trajectory lengths/shapes.")
            if not np.array_equal(states[0], initial):
                raise ValueError(f"Trial {index}: controller did not start from the saved initial state.")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"Trial {index}: nonfinite trajectory; return the valid prefix and failure_reason.")
            if not np.isfinite(timings).all() or np.any(timings < 0):
                raise ValueError(f"Trial {index}: invalid controller timing.")
            failure = result.get("failure_reason") or None
            if np.any(actions < benchmark.action_low - 1e-5) or np.any(actions > benchmark.action_high + 1e-5):
                failure = failure or "Returned control exceeded shared input bounds."
            if len(actions) != benchmark.steps:
                failure = failure or "Rollout ended before the shared simulation duration."
            completed = failure is None and len(actions) == benchmark.steps
            metrics = compute_rollout_metrics(states, actions, benchmark)
            success = bool(completed and metrics["ace"] <= benchmark.success_threshold)
            record = {"trial": index, "initial_state": initial.tolist(), "final_state": states[-1].tolist(),
                      "initial_distance_xy": float(np.linalg.norm(initial[:2] - benchmark.goal[:2])),
                      "success": success, "completed": completed, "steps": len(actions), **metrics,
                      "mean_controller_ms": float(timings.mean() * 1000) if len(timings) else None,
                      "max_controller_ms": float(timings.max() * 1000) if len(timings) else None,
                      "failure_reason": failure, "state_traj": states, "controller_times": timings}
            records.append(record)
            prefix = f"trial_{index:04d}"
            payload[prefix + "_states"] = states
            payload[prefix + "_actions"] = actions
            payload[prefix + "_controller_times"] = timings
            row = {key: record[key] for key in fieldnames if key in record}
            row.update({f"initial_{key}": float(initial[i]) for i, key in enumerate(STATE_NAMES)})
            row.update({f"final_{key}": float(states[-1, i]) for i, key in enumerate(STATE_NAMES)})
            writer.writerow(row)
            stream.flush()
            print(f"Trial {index:03d}/{count}: success={int(success)}, ACE={metrics['ace']:.4f}, "
                  f"steps={len(actions)}" + (f", error={failure}" if failure else ""), flush=True)
    rollout_elapsed = time.perf_counter() - started
    completed_records = [record for record in records if record["completed"]]
    all_times = np.concatenate([record["controller_times"] for record in records])
    aggregate = {
        "num_trials": count, "dataset_trials": len(benchmark.initial_states),
        "partial_benchmark": count != len(benchmark.initial_states),
        "completed_trials": len(completed_records), "failed_rollouts": count - len(completed_records),
        "successes": sum(record["success"] for record in records),
        "success_rate_percent": 100.0 * sum(record["success"] for record in records) / count,
        "mean_ace_completed": float(np.mean([r["ace"] for r in completed_records])) if completed_records else None,
        "mean_state_msd_completed": float(np.mean([r["avg_state_msd"] for r in completed_records])) if completed_records else None,
        "mean_control_msd_completed": float(np.mean([r["avg_control_msd"] for r in completed_records])) if completed_records else None,
        "mean_controller_ms": float(all_times.mean() * 1000) if len(all_times) else None,
        "max_controller_ms": float(all_times.max() * 1000) if len(all_times) else None,
        "rollout_wall_seconds": rollout_elapsed,
    }
    summary = {"controller": controller_name, "controller_metadata": controller_metadata,
               "dataset_path": str(benchmark.path), "dataset_id": benchmark.dataset_id,
               "benchmark_metadata": benchmark.metadata, "aggregate": aggregate,
               "trials": [{key: value for key, value in record.items()
                           if key not in ("state_traj", "controller_times")} for record in records]}
    np.savez_compressed(output_dir / "rollouts.npz", dataset_id=np.array(benchmark.dataset_id),
                        trial_ids=np.arange(1, count + 1), initial_states=benchmark.initial_states[:count], **payload)
    if make_plot:
        _save_summary_plot(records, benchmark, output_dir / "trajectories.png", controller_name)
    total_elapsed = time.perf_counter() - started
    aggregate["total_elapsed_seconds"] = total_elapsed
    aggregate["elapsed_seconds_per_trial"] = total_elapsed / count
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    sampling_bounds = {
        name: tuple(benchmark.metadata["sampling_bounds"][name]) for name in STATE_NAMES
    }
    training_bounds = {
        name: tuple(benchmark.metadata["training_bounds"][name]) for name in STATE_NAMES
    }
    selected_distances = np.linalg.norm(
        benchmark.initial_states[:count, :2] - benchmark.goal[:2], axis=1
    )
    device = controller_metadata.get("device", "cpu")
    checkpoint = controller_metadata.get("checkpoint")
    controller_label = controller_metadata.get("controller", "N/A")
    case_name = str(benchmark.metadata["case"])
    obstacle_count = len(benchmark.metadata.get("obstacles", []))
    obstacle_text = "no obstacles" if obstacle_count == 0 else f"{obstacle_count} obstacles"
    print(f"\n=== Randomized Initial-State {controller_name} Evaluation "
          f"(Case {case_name}, {obstacle_text}) ===")
    print(f"Using device: {device}")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        display_model = (
            f"{checkpoint_path.parent.name}/{checkpoint_path.name}"
            if controller_name == "RL" else checkpoint_path.name
        )
        print(f"model: {display_model}")
    else:
        print(f"controller: {controller_label}")
    print(f"dataset: {benchmark.path}")
    print(f"dataset ID: {benchmark.dataset_id}")
    print(f"random seed: {benchmark.metadata['random_seed']}")
    print(f"trials: {count}")
    print(f"samples eventually used: {count}")
    print(f"total sampling attempts: {benchmark.metadata['sample_attempts']}")
    print(f"goal: {benchmark.goal.tolist()}")
    print(f"sampling bounds: {sampling_bounds}")
    print(f"training bounds excluded: {training_bounds}")
    print(f"minimum initial distance to goal (d_min): {benchmark.metadata['minimum_distance_to_goal']:.3f}")
    print(f"rejected samples inside training bounds: {benchmark.metadata['rejected_training']}")
    print(f"rejected samples due to d_min: {benchmark.metadata['rejected_distance']}")
    if "rejected_obstacle" in benchmark.metadata:
        print(f"rejected samples inside obstacle safety boundaries: "
              f"{benchmark.metadata['rejected_obstacle']}")
    print(f"mean initial distance (selected): {selected_distances.mean():.4f}")
    print(f"success criterion: Absolute convergence error (ACE) <= {benchmark.success_threshold:g}")
    print(f"completed rollouts: {aggregate['completed_trials']}/{count}")
    print(f"success rate: {aggregate['success_rate_percent']:.2f}%")
    print(f"average ACE (completed): {aggregate['mean_ace_completed']:.4f}" if completed_records else "average ACE (completed): N/A")
    print(f"average state MSD (completed): {aggregate['mean_state_msd_completed']:.4f}" if completed_records else "average state MSD (completed): N/A")
    print(f"average control MSD (completed): {aggregate['mean_control_msd_completed']:.4f}" if completed_records else "average control MSD (completed): N/A")
    if aggregate["mean_controller_ms"] is None:
        print(f"Average {controller_name} controller computation time: N/A")
        print(f"Maximum {controller_name} controller computation time: N/A")
    else:
        print(f"Average {controller_name} controller computation time: {aggregate['mean_controller_ms']:.3f} ms")
        print(f"Maximum {controller_name} controller computation time: {aggregate['max_controller_ms']:.3f} ms")
    print(f"total elapsed time: {total_elapsed:.2f}s ({total_elapsed / count:.3f}s/trial)")
    print(f"results saved to: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate fixed no-obstacle case-C states shared by ANR, MPC, and RL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--d-min", type=float, default=MIN_DISTANCE_TO_GOAL)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing dataset.")
    args = parser.parse_args()
    benchmark = save_benchmark(args.output, args.num_trials, args.seed, args.d_min, args.overwrite)
    print(f"Saved {len(benchmark.initial_states)} ordered case-C initial states to {benchmark.path}")
    print(f"Dataset ID: {benchmark.dataset_id}")
    print(f"Attempts: {benchmark.metadata['sample_attempts']}; training-box rejections: "
          f"{benchmark.metadata['rejected_training']}; distance rejections: {benchmark.metadata['rejected_distance']}")
    print("First three initial states [x, y, theta, v]:")
    print(benchmark.initial_states[:3])


if __name__ == "__main__":
    main()
