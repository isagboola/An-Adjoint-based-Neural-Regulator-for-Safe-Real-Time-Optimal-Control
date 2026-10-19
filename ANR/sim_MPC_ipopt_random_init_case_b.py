"""Evaluate the existing IPOPT MPC on the saved case-B initial.

Generate the dataset first with ``python generate_case_b_benchmark.py``.
The prediction horizon below controls the MPC problem; the simulation duration
and selected initial states always come from the shared benchmark dataset.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ANR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANR_DIR.parent
for import_dir in (ANR_DIR, PROJECT_ROOT):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from generate_case_b_benchmark import DEFAULT_DATASET, ROOT, load_benchmark, run_evaluation
from benchmark_disturbance import disturbance_metadata, velocity_disturbance_sequence


# Settings to edit for this evaluator. Shared states and metrics live in the dataset.
DATASET_PATH = DEFAULT_DATASET
PREDICTION_HORIZON = 30
OUTPUT_DIR = ROOT / "benchmark_results" / "case_b" / "mpc_ipopt"
MAX_TRIALS = None  # None runs all saved states; an integer uses the first entries.
MAKE_PLOT = True
NUM_WORKERS = 30
EVALUATOR_MODULE = "ANR.sim_MPC_ipopt_random_init_case_b"


def load_controller(prediction_horizon, benchmark):
    """Use MPC_HOCBF unchanged, with an empty obstacle list at every update."""
    from ANR import config as anr_config
    from ANR.mpc_controller import MPC_HOCBF

    if prediction_horizon < 1:
        raise ValueError("The prediction horizon must be positive.")
    controller = MPC_HOCBF(
        dt=benchmark.dt,
        N=prediction_horizon,
        a_min=float(benchmark.action_low[0]),
        a_max=float(benchmark.action_high[0]),
        w_min=float(benchmark.action_low[1]),
        w_max=float(benchmark.action_high[1]),
        alpha0=anr_config.alpha0,
        alpha1=anr_config.alpha1,
    )

    def rollout(initial_state, benchmark):
        velocity_disturbances = velocity_disturbance_sequence(
            benchmark, initial_state,
            enabled=anr_config.disturbance_enabled,
            standard_deviation=anr_config.disturbance_velocity_std,
            seed=anr_config.disturbance_seed,
        )
        state = np.asarray(initial_state, dtype=float).reshape(4).copy()
        states = [state.copy()]
        actions = []
        controller_times = []
        failure_reason = ""

        for step in range(benchmark.steps):
            try:
                started = time.perf_counter()
                action = np.asarray(
                    controller.get_control_input(state, benchmark.goal, obstacles=[]), dtype=float
                ).reshape(2)
                elapsed = time.perf_counter() - started
                if not np.isfinite(action).all():
                    raise ValueError("Controller returned a nonfinite action.")
                if np.any(action < benchmark.action_low - 1e-6) or np.any(action > benchmark.action_high + 1e-6):
                    raise ValueError("Controller returned an action outside the benchmark bounds.")
                next_state = np.asarray(
                    controller.rk4(
                        state, action, velocity_disturbance=velocity_disturbances[step]
                    ), dtype=float
                ).reshape(4)
                if not np.isfinite(next_state).all():
                    raise ValueError("Plant integration returned a nonfinite state.")
                next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
            except Exception as exc:
                failure_reason = f"Step {step + 1}: {type(exc).__name__}: {exc}"
                break

            state = next_state
            actions.append(action.copy())
            controller_times.append(elapsed)
            states.append(state.copy())

        return {
            "state_traj": np.asarray(states, dtype=float).reshape(-1, 4),
            "action_traj": np.asarray(actions, dtype=float).reshape(-1, 2),
            "controller_times": controller_times,
            "failure_reason": failure_reason,
        }

    metadata = {
        "controller": "ANR.mpc_controller.MPC_HOCBF",
        "solver": "CasADi/IPOPT",
        "solver_status_available": False,
        "prediction_horizon": prediction_horizon,
        "prediction_duration_seconds": prediction_horizon * benchmark.dt,
        "plant_integration": "MPC_HOCBF.rk4",
        "q_diagonal": [10.0, 10.0, 10.0, 10.0],
        "r_diagonal": [1.0, 1.0],
        "terminal_weight": 50.0,
        "obstacles": [],
        "controller_timing": "entire get_control_input call, including NLP construction and solve; excludes plant integration",
        "device": "cpu",
    }
    metadata.update(disturbance_metadata(
        enabled=anr_config.disturbance_enabled,
        standard_deviation=anr_config.disturbance_velocity_std,
        seed=anr_config.disturbance_seed,
    ))
    return rollout, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS, help="Evaluate the first N saved trials, each for the full duration.")
    parser.add_argument("--no-plot", action="store_true", default=not MAKE_PLOT)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args(argv)
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    benchmark = load_benchmark(args.dataset)
    rollout, metadata = load_controller(args.horizon, benchmark)
    count = len(benchmark.initial_states) if args.max_trials is None else args.max_trials
    worker_count = min(args.workers, count)
    metadata["parallel_workers"] = worker_count
    if worker_count == 1:
        return run_evaluation(benchmark, rollout, controller_name="MPC-IPOPT",
            controller_metadata=metadata, output_dir=args.output_dir,
            max_trials=args.max_trials, make_plot=not args.no_plot)
    from ANR.mpc_parallel import initialize_mpc_worker, run_mpc_worker
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count, initializer=initialize_mpc_worker,
            initargs=(EVALUATOR_MODULE, str(Path(args.dataset).resolve()), args.horizon)) as executor:
        results = executor.map(run_mpc_worker,
                               [state.copy() for state in benchmark.initial_states[:count]],
                               chunksize=1)
        return run_evaluation(benchmark, None, controller_name="MPC-IPOPT",
            controller_metadata=metadata, output_dir=args.output_dir,
            max_trials=args.max_trials, make_plot=not args.no_plot,
            rollout_results=results, wall_started=wall_started)


if __name__ == "__main__":
    main()
