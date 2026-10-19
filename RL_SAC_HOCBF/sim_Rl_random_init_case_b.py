"""Evaluate a saved SAC actor on the shared, nominal case-B initial states.

Run ``python RL_SAC_HOCBF/sim_Rl_random_init_case_b.py --help`` for options.
Generate the dataset first with ``generate_case_b_benchmark.py`` at the project
root. This evaluator always runs each trial for the saved simulation duration.
"""

from pathlib import Path
import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np


RL_DIR = Path(__file__).resolve().parent
ROOT = RL_DIR.parent
for import_dir in (ROOT, RL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from generate_case_b_benchmark import DEFAULT_DATASET, load_benchmark, run_evaluation
from benchmark_disturbance import disturbance_metadata, velocity_disturbance_sequence


# Edit these defaults here, or override them on the command line.
DATASET_PATH = DEFAULT_DATASET
RL_RESUME = "2"  # Run number, checkpoint directory, or actor.pkl path.
DEVICE = "cpu"  # "cpu", "cuda", or "auto".
OUTPUT_DIR = ROOT / "benchmark_results" / "case_b" / "rl"
MAX_TRIALS = None  # None evaluates every saved initial state, in order.
MAKE_PLOT = True


def resolve_actor_path(resume):
    """Resolve run numbers and paths independently of the launch directory."""
    resume = str(resume)
    if resume.isdecimal():
        checkpoint = RL_DIR / "output" / f"Unicycle-run{resume}"
    elif resume == "default":
        checkpoint = RL_DIR / "output" / "Unicycle-run0"
    else:
        checkpoint = Path(resume).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = RL_DIR / checkpoint
    if checkpoint.suffix.lower() != ".pkl":
        checkpoint = checkpoint / "actor.pkl"
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAC actor checkpoint does not exist: {checkpoint}")
    return checkpoint


def build_rollout(benchmark, resume=RL_RESUME, device_name=DEVICE):
    """Load the existing actor and return an independent-trial rollout function."""
    import torch

    from ANR import config as anr_config
    from envs.unicycle_env import UnicycleEnv
    from HOCBF_SAC.model import GaussianPolicy

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    device = torch.device(device_name)
    actor_path = resolve_actor_path(resume)

    def make_env(initial_state):
        velocity_disturbances = velocity_disturbance_sequence(
            benchmark, initial_state,
            enabled=anr_config.disturbance_enabled,
            standard_deviation=anr_config.disturbance_velocity_std,
            seed=anr_config.disturbance_seed,
        )
        return UnicycleEnv(
            obs_config="none",
            custom_obstacles=[],
            rand_init=False,
            init_mode="manual",
            manual_init_state=np.asarray(initial_state, dtype=float).reshape(4),
            ref_x=float(benchmark.goal[0]),
            ref_y=float(benchmark.goal[1]),
            ref_theta=float(benchmark.goal[2]),
            ref_v=float(benchmark.goal[3]),
            max_episode_steps=benchmark.steps,
            terminate_on_goal=False,
            disturbance_enabled=anr_config.disturbance_enabled,
            disturbance_velocity_std=anr_config.disturbance_velocity_std,
            disturbance_seed=anr_config.disturbance_seed,
            velocity_disturbances=velocity_disturbances,
        )

    template_env = make_env(benchmark.initial_states[0])
    if not np.isclose(template_env.dt, benchmark.dt, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"RL environment dt={template_env.dt} differs from dataset dt={benchmark.dt}."
        )
    if not (
        np.allclose(template_env.action_space.low, benchmark.action_low, rtol=0.0, atol=1e-8)
        and np.allclose(template_env.action_space.high, benchmark.action_high, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("RL input bounds differ from the shared benchmark bounds.")

    policy = GaussianPolicy(
        template_env.observation_space.shape[0],
        template_env.action_space.shape[0],
        hidden_dim=128,
        action_space=template_env.action_space,
    ).to(device)
    policy.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
    policy.eval()

    def deterministic_action(env):
        observation = torch.as_tensor(
            env.get_obs(), dtype=torch.float32, device=device
        ).unsqueeze(0)
        mean, _ = policy(observation)
        # Identical to the third return value of GaussianPolicy.sample(), which
        # select_action(evaluate=True) uses; no exploration draw is needed here.
        action = torch.tanh(mean) * policy.action_scale + policy.action_bias
        return action.squeeze(0).cpu().numpy().astype(float)

    # Exclude one-time inference initialization from per-control timing.
    with torch.inference_mode():
        deterministic_action(template_env)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    template_env.close()

    def rollout(initial_state, trial_benchmark):
        if trial_benchmark is not benchmark:
            raise ValueError("Build the RL rollout with the benchmark being evaluated.")
        states = [np.asarray(initial_state, dtype=float).reshape(4).copy()]
        actions = []
        controller_times = []
        failure_reason = None
        env = None
        step = 0
        try:
            env = make_env(initial_state)
            with torch.inference_mode():
                for step in range(benchmark.steps):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    action = deterministic_action(env)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    if not np.all(np.isfinite(action)):
                        raise FloatingPointError("The actor returned a non-finite action.")
                    # Preserve the existing RL environment's Euler integration.
                    # Its episode reward is unused by the common evaluation metrics.
                    env.step(action)
                    next_state = np.asarray(env.state, dtype=float).reshape(4).copy()
                    if not np.all(np.isfinite(next_state)):
                        raise FloatingPointError("The RL plant returned a non-finite state.")
                    states.append(next_state)
                    actions.append(action.copy())
                    controller_times.append(float(elapsed))
        except Exception as exc:
            failure_reason = f"Step {step + 1}: {type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                env.close()
        return {
            "state_traj": np.asarray(states, dtype=float),
            "action_traj": np.asarray(actions, dtype=float).reshape(-1, 2),
            "controller_times": controller_times,
            "failure_reason": failure_reason,
        }

    metadata = {
        "checkpoint": str(actor_path),
        "resume": str(resume),
        "policy": "GaussianPolicy deterministic tanh(mean)",
        "device": str(device),
        "plant_integration": "forward Euler (existing RL environment)",
        "cbf_mode": "off",
        "terminate_on_goal": False,
        "timing_scope": "observation, actor inference, action transfer and bounds; excludes plant",
        "inference_warmup_calls": 1,
    }
    metadata.update(disturbance_metadata(
        enabled=anr_config.disturbance_enabled,
        standard_deviation=anr_config.disturbance_velocity_std,
        seed=anr_config.disturbance_seed,
    ))
    return rollout, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default=RL_RESUME, help="Run number, checkpoint directory, or actor.pkl; relative paths start in RL_SAC_HOCBF.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    parser.add_argument("--no-plot", action="store_true", default=not MAKE_PLOT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=DEVICE)
    args = parser.parse_args(argv)

    benchmark = load_benchmark(args.dataset)
    rollout, metadata = build_rollout(benchmark, args.resume, args.device)
    return run_evaluation(
        benchmark,
        rollout,
        controller_name="RL",
        controller_metadata=metadata,
        output_dir=args.output_dir,
        max_trials=args.max_trials,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
