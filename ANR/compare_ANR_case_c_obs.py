"""Overlay paired nominal and model-mismatched/disturbed ANR Case-C rollouts."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


ANR_DIR = Path(__file__).resolve().parent
ROOT = ANR_DIR.parent
for import_dir in (ANR_DIR, ROOT):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from ANR import config as anr_config
from ANR.sim_ANR_case_c_obs import (
    DATASET_PATH,
    DEVICE,
    MODEL_PATH,
    PREDICTION_HORIZON,
    load_controller,
)
from generate_case_c_obstacle_benchmark import load_benchmark


OUTPUT_DIR = ROOT / "figs"
DEPLOYMENT_MASS = 1.4
TRIAL = 1
FIGURE_SIZE = (7, 11)
AXIS_LABEL_SIZE = 25
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 18
THICK_LINE_WIDTH = 2.5
NOMINAL_LINE_WIDTH = 3


def _style_ticks(ax):
    """Match the bold, large tick styling used by sim_ANR.py."""
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("bold")


def _require_complete(name, result, expected_steps):
    if result["failure_reason"]:
        raise RuntimeError(f"{name} rollout failed: {result['failure_reason']}")
    if len(result["action_traj"]) != expected_steps:
        raise RuntimeError(
            f"{name} rollout returned {len(result['action_traj'])} of "
            f"{expected_steps} expected controls."
        )


def _plot_navigation(benchmark, initial_state, nominal, uncertain, output_path,
                     deployment_mass, sigma):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    nominal_states = nominal["state_traj"]
    uncertain_states = uncertain["state_traj"]

    ax.plot(
        nominal_states[:, 0], nominal_states[:, 1], color="black",
        linestyle=":", linewidth=NOMINAL_LINE_WIDTH,
        label="Nominal", zorder=4,
    )
    ax.plot(
        uncertain_states[:, 0], uncertain_states[:, 1], color="tab:blue",
        linestyle="-", linewidth=THICK_LINE_WIDTH,
        label="Model mismatch", zorder=3,
    )

    robot_radius = float(benchmark.metadata["robot_radius"])
    margin = float(benchmark.metadata["safety_margin"])
    obstacle_handle = safety_handle = None
    for obstacle in benchmark.metadata["obstacles"]:
        center = (float(obstacle["xo"]), float(obstacle["yo"]))
        radius = float(obstacle["r"])
        obstacle_patch = Circle(center, radius, color="gray", alpha=0.75, zorder=2)
        ax.add_patch(obstacle_patch)
        safety_patch = Circle(
            center, radius + robot_radius + margin, fill=False,
            edgecolor="tab:red", linestyle="--", linewidth=2.0, zorder=1,
        )
        ax.add_patch(safety_patch)
        obstacle_handle = obstacle_patch
        safety_handle = safety_patch

    ax.scatter(
        initial_state[0], initial_state[1], marker="o", s=110,
        color="tab:green", edgecolor="black", label="_nolegend_", zorder=6,
   )
    ax.scatter(
        benchmark.goal[0], benchmark.goal[1], marker="*", s=230,
        color="gold", edgecolor="black", label="_nolegend_", zorder=6,
    )

    handles, labels = ax.get_legend_handles_labels()
    if obstacle_handle is not None:
        handles.extend([
            Line2D([0], [0], marker="o", linestyle="none", markersize=12,
                   markerfacecolor="gray", markeredgecolor="gray"),
            Line2D([0], [0], color="tab:red", linestyle="--", linewidth=2),
        ])
      #  labels.extend(["Obstacle", "HOCBF safety boundary"])
    ax.legend(handles, labels, fontsize=LEGEND_SIZE, loc="lower left", ncol=1)


    all_x = np.concatenate([
        nominal_states[:, 0], uncertain_states[:, 0],
        [initial_state[0], benchmark.goal[0]],
        [float(item["xo"]) for item in benchmark.metadata["obstacles"]],
    ])
    all_y = np.concatenate([
        nominal_states[:, 1], uncertain_states[:, 1],
        [initial_state[1], benchmark.goal[1]],
        [float(item["yo"]) for item in benchmark.metadata["obstacles"]],
    ])
    padding = 1.0
    ax.set_xlim(float(all_x.min()) - padding, float(all_x.max()) + padding)
    ax.set_ylim(float(all_y.min()) - padding, float(all_y.max()) + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (m)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(r"$y$ (m)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    _style_ticks(ax)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostics(benchmark, nominal, uncertain, output_path,
                      deployment_mass):
    nominal_states = nominal["state_traj"]
    uncertain_states = uncertain["state_traj"]
    nominal_actions = nominal["action_traj"]
    uncertain_actions = uncertain["action_traj"]
    nominal_disturbance = nominal["disturbance_traj"]
    uncertain_disturbance = uncertain["disturbance_traj"]

    state_time = np.arange(len(nominal_states), dtype=float) * benchmark.dt
    control_time = np.arange(len(nominal_actions), dtype=float) * benchmark.dt
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    state_labels = (r"$x$", r"$y$", r"$\theta$", r"$v$")

    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)
    for index, (color, label) in enumerate(zip(colors, state_labels)):
        axes[0].plot(
            state_time, nominal_states[:, index], color=color,
            linestyle=":", linewidth=NOMINAL_LINE_WIDTH,
            label=f"{label} nominal",
        )
        axes[0].plot(
            state_time, uncertain_states[:, index], color=color,
            linestyle="-", linewidth=THICK_LINE_WIDTH,
            label=f"{label} mismatch/disturbed",
        )
    axes[0].set_ylabel("State", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    axes[0].legend(ncol=2, fontsize=LEGEND_SIZE)
    axes[0].grid(True)

    axes[1].plot(
        control_time, nominal_actions[:, 0], color="tab:blue",
        linestyle=":", linewidth=NOMINAL_LINE_WIDTH,
        label=r"$a_{\rm cmd}$ nominal",
    )
    axes[1].plot(
        control_time, uncertain_actions[:, 0], color="tab:blue",
        linestyle="-", linewidth=THICK_LINE_WIDTH,
        label=r"$a_{\rm cmd}$ mismatch/disturbed",
    )
    axes[1].plot(
        control_time, nominal_actions[:, 1], color="tab:orange",
        linestyle=":", linewidth=NOMINAL_LINE_WIDTH,
        label=r"$\omega$ nominal",
    )
    axes[1].plot(
        control_time, uncertain_actions[:, 1], color="tab:orange",
        linestyle="-", linewidth=THICK_LINE_WIDTH,
        label=r"$\omega$ mismatch/disturbed",
    )
    axes[1].set_ylabel("Input", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    axes[1].legend(ncol=2, fontsize=LEGEND_SIZE)
    axes[1].grid(True)

    axes[2].step(
        control_time, nominal_disturbance, where="post", color="black",
        linestyle=":", linewidth=NOMINAL_LINE_WIDTH,
        label=r"Nominal $d_{v,k}=0$",
    )
    axes[2].step(
        control_time, uncertain_disturbance, where="post", color="tab:red",
        linestyle="-", linewidth=THICK_LINE_WIDTH,
        label=r"Applied $d_{v,k}$",
    )
    axes[2].axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    axes[2].set_ylabel("Disturbance", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    axes[2].set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    axes[2].legend(fontsize=LEGEND_SIZE)
    axes[2].grid(True)
    axes[2].set_xlim(0.0, benchmark.steps * benchmark.dt)

    tick_locations = np.linspace(0.0, benchmark.steps * benchmark.dt, 3)
    for ax in axes:
        ax.set_xticks(tick_locations)
        _style_ticks(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--trial", type=int, default=TRIAL, help="One-based saved benchmark trial number.")
    parser.add_argument("--deployment-mass", type=float, default=DEPLOYMENT_MASS)
    parser.add_argument("--sigma", type=float, default=anr_config.disturbance_velocity_std)
    parser.add_argument("--disturbance-seed", type=int, default=anr_config.disturbance_seed)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    benchmark = load_benchmark(args.dataset)
    if not 1 <= args.trial <= len(benchmark.initial_states):
        parser.error(f"--trial must be between 1 and {len(benchmark.initial_states)}")
    if not np.isfinite(args.deployment_mass) or args.deployment_mass <= 0.0:
        parser.error("--deployment-mass must be finite and positive")
    if not np.isfinite(args.sigma) or args.sigma < 0.0:
        parser.error("--sigma must be finite and nonnegative")

    initial_state = benchmark.initial_states[args.trial - 1].copy()
    nominal_rollout, nominal_metadata = load_controller(
        args.model, args.horizon, args.device, benchmark,
        deployment_mass=anr_config.nominal_mass,
        controller_mass=anr_config.nominal_mass,
        plant_mass=anr_config.nominal_mass,
        disturbance_enabled=False,
        disturbance_velocity_std=args.sigma,
        disturbance_seed=args.disturbance_seed,
    )
    uncertain_rollout, uncertain_metadata = load_controller(
        args.model, args.horizon, args.device, benchmark,
        deployment_mass=args.deployment_mass,
        controller_mass=anr_config.nominal_mass,
        plant_mass=args.deployment_mass,
        disturbance_enabled=True,
        disturbance_velocity_std=args.sigma,
        disturbance_seed=args.disturbance_seed,
    )

    nominal = nominal_rollout(initial_state, benchmark)
    uncertain = uncertain_rollout(initial_state, benchmark)
    _require_complete("Nominal", nominal, benchmark.steps)
    _require_complete("Model-mismatched/disturbed", uncertain, benchmark.steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"anr_case_c_obs_trial_{args.trial:03d}_nominal_vs_varied"
    navigation_path = args.output_dir / f"{stem}_navigation.png"
    diagnostics_path = args.output_dir / f"{stem}_diagnostics.png"
    data_path = args.output_dir / f"{stem}.npz"

    _plot_navigation(
        benchmark, initial_state, nominal, uncertain, navigation_path,
        args.deployment_mass, args.sigma,
    )
    _plot_diagnostics(
        benchmark, nominal, uncertain, diagnostics_path, args.deployment_mass,
    )
    np.savez_compressed(
        data_path,
        initial_state=initial_state,
        goal=np.asarray(benchmark.goal, dtype=float),
        dt=float(benchmark.dt),
        nominal_state_traj=nominal["state_traj"],
        nominal_action_traj=nominal["action_traj"],
        nominal_disturbance_traj=nominal["disturbance_traj"],
        varied_state_traj=uncertain["state_traj"],
        varied_action_traj=uncertain["action_traj"],
        varied_disturbance_traj=uncertain["disturbance_traj"],
        nominal_controller_mass=float(nominal_metadata["controller_mass"]),
        nominal_plant_mass=float(nominal_metadata["plant_mass"]),
        mismatched_controller_mass=float(uncertain_metadata["controller_mass"]),
        mismatched_plant_mass=float(uncertain_metadata["plant_mass"]),
        disturbance_sigma=float(args.sigma),
        disturbance_seed=int(args.disturbance_seed),
    )

    print(f"Nominal and model-mismatched/disturbed rollouts completed for trial {args.trial}.")
    print(f"Navigation overlay: {navigation_path.resolve()}")
    print(f"Diagnostics overlay: {diagnostics_path.resolve()}")
    print(f"Paired trajectory data: {data_path.resolve()}")
    return navigation_path, diagnostics_path, data_path


if __name__ == "__main__":
    main()
