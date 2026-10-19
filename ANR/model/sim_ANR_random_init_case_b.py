"""Evaluate the existing ANR controller on the saved case-B initial.

Generate the dataset first with ``python generate_case_b_benchmark.py``.
Edit the settings below or pass command-line options; this script never samples
initial states and runs every selected trial for the dataset's full duration.
"""

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ANR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANR_DIR.parent
for import_dir in (ANR_DIR, PROJECT_ROOT):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from generate_case_b_benchmark import DEFAULT_DATASET, ROOT, load_benchmark, run_evaluation
from benchmark_disturbance import disturbance_metadata, velocity_disturbance_sequence


# Settings to edit for this evaluator. Shared states and metrics live in the dataset.
DATASET_PATH = DEFAULT_DATASET
MODEL_PATH = ANR_DIR / "model" / "anr_N30_h50_seed_0_e50.pth"
PREDICTION_HORIZON = 30
DEVICE = "cpu"  # "auto", "cpu", or "cuda"; match the RL default for timing.
OUTPUT_DIR = ROOT / "benchmark_results" / "case_b" / "anr"
MAX_TRIALS = None  # None runs all saved states; an integer uses the first entries.
MAKE_PLOT = True


def load_controller(model_path, prediction_horizon, device_name, benchmark):
    """Load the same architecture and control/plant helpers as sim_ANR.py."""
    import torch
    import torch.nn as nn

    from ANR import config as anr_config
    from ANR.utils import rk4, solve_qp

    if prediction_horizon < 1:
        raise ValueError("The prediction horizon must be positive.")
    if not np.isclose(benchmark.dt, anr_config.dt, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Dataset dt={benchmark.dt} differs from ANR/config.py dt={anr_config.dt}; "
            "the existing ANR rk4 helper reads dt from that configuration."
        )
    if not (
        np.allclose(benchmark.action_low, [anr_config.a_min, anr_config.w_min], rtol=0.0, atol=1e-12)
        and np.allclose(benchmark.action_high, [anr_config.a_max, anr_config.w_max], rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Dataset action bounds differ from ANR/config.py.")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu.")

    class CoNN(nn.Module):
        def __init__(self, horizon):
            super().__init__()
            self.prediction_horizon = horizon
            self.fc1 = nn.Linear(4, 128)
            self.fc2 = nn.Linear(128, 512)
            self.fc3 = nn.Linear(512, 128)
            self.fc4 = nn.Linear(128, 4 * horizon)

        def forward(self, state):
            state = torch.relu(self.fc1(state))
            state = torch.relu(self.fc2(state))
            state = torch.relu(self.fc3(state))
            return self.fc4(state).view(-1, self.prediction_horizon, 4)

    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ANR checkpoint not found: {checkpoint}")
    model = CoNN(prediction_horizon).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    # Initialize the inference device before measuring controller calls.
    with torch.inference_mode():
        model(torch.zeros((1, 4), dtype=torch.float32, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    def rollout(initial_state, benchmark):
        velocity_disturbances = velocity_disturbance_sequence(
            benchmark, initial_state,
            enabled=anr_config.disturbance_enabled,
            standard_deviation=anr_config.disturbance_velocity_std,
            seed=anr_config.disturbance_seed,
        )
        state = np.asarray(initial_state, dtype=float).reshape(1, 4).copy()
        states = [state.reshape(4).copy()]
        actions = []
        controller_times = []
        failure_reason = ""

        with torch.inference_mode():
            for step in range(benchmark.steps):
                try:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    error = state.reshape(4) - np.asarray(benchmark.goal).reshape(4)
                    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
                    state_tensor = torch.as_tensor(error, dtype=torch.float32, device=device).reshape(1, 4)
                    costate = model(state_tensor)[0, 0].cpu().numpy()
                    a, omega = solve_qp(lambda1=float(costate[0]), lambda2=float(costate[1]),  lambda3=float(costate[2]), lambda4=float(costate[3]),
                        v=float(state[0, 3]), theta=float(state[0, 2]), a_min=float(benchmark.action_low[0]), a_max=float(benchmark.action_high[0]),
                        w_min=float(benchmark.action_low[1]),  w_max=float(benchmark.action_high[1]), mass=anr_config.nominal_mass, nominal_mass=anr_config.nominal_mass,)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    action = np.asarray([a, omega], dtype=float)
                    if not np.isfinite(action).all():
                        raise ValueError("Controller returned a nonfinite action.")
                    if np.any(action < benchmark.action_low - 1e-6) or np.any(action > benchmark.action_high + 1e-6):
                        raise ValueError("Controller returned an action outside the benchmark bounds.")
                    next_state = np.asarray(
                        rk4(state, action,velocity_disturbance=velocity_disturbances[step], mass=anr_config.mass, nominal_mass=anr_config.nominal_mass,),
                        dtype=float,
                    ).reshape(1, 4)
                    if not np.isfinite(next_state).all():
                        raise ValueError("Plant integration returned a nonfinite state.")
                except Exception as exc:
                    failure_reason = f"Step {step + 1}: {type(exc).__name__}: {exc}"
                    break

                state = next_state
                actions.append(action.copy())
                controller_times.append(elapsed)
                states.append(state.reshape(4).copy())

        return { "state_traj": np.asarray(states, dtype=float).reshape(-1, 4),
            "action_traj": np.asarray(actions, dtype=float).reshape(-1, 2), "controller_times": controller_times,"failure_reason": failure_reason,}

    metadata = {
        "checkpoint": str(checkpoint),
        "prediction_horizon": prediction_horizon,
        "device": str(device),
        "architecture": "4-128-512-128-(4*N), ReLU hidden layers",
        "plant_integration": "ANR.utils.rk4",
        "control_function": "ANR.utils.solve_qp",
        "anr_training_mass": anr_config.nominal_mass,
        "deployment_mass": anr_config.mass,
        "controller_mass": anr_config.nominal_mass,
        "plant_mass": anr_config.mass,
        "anr_mass_conditioned": False,
        "qp_mass_aware": bool(np.isclose(anr_config.nominal_mass, anr_config.mass)),
        "mass_model_mismatch": bool(not np.isclose(anr_config.nominal_mass, anr_config.mass)),
        "longitudinal_input": "nominal acceleration command a_cmd",
        "controller_mass_effectiveness": 1.0,
        "plant_mass_effectiveness": anr_config.nominal_mass / anr_config.mass,
        "plant_dynamics": "v_dot = (anr_training_mass / deployment_mass) * a_cmd + d_v",
        "obstacles": [],
        "inference_mode": True,
        "inference_warmup_calls": 1,
        "controller_timing": "error construction, tensor transfer, inference and control calculation; excludes plant integration",
    }
    metadata.update(disturbance_metadata(
        enabled=anr_config.disturbance_enabled,
        standard_deviation=anr_config.disturbance_velocity_std,
        seed=anr_config.disturbance_seed,
    ))
    metadata["disturbance_location"] = "plant velocity derivative: v_dot = (anr_training_mass / deployment_mass) * a_cmd + d_v"
    return rollout, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON, help="Must match the checkpoint's output horizon.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS, help="Evaluate the first N saved trials, each for the full duration.")
    parser.add_argument("--no-plot", action="store_true", default=not MAKE_PLOT)
    args = parser.parse_args(argv)
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")

    benchmark = load_benchmark(args.dataset)
    rollout, metadata = load_controller(args.model, args.horizon, args.device, benchmark)
    return run_evaluation(benchmark, rollout, controller_name="ANR",controller_metadata=metadata, output_dir=args.output_dir,max_trials=args.max_trials, make_plot=not args.no_plot,)


if __name__ == "__main__":
    main()
