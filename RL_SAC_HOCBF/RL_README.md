# RL SAC-HOCBF workflow

During training, each run directory receives `training_rewards.csv` after every
episode and a progressively refreshed `training_reward.png`. The plot shows raw
episode rewards and a 50-episode rolling mean against total environment steps.
By default the image refreshes every 10 episodes; use
`--reward_plot_interval 1` to refresh it after every episode, and adjust the
smoothing with `--reward_rolling_window`.

This folder trains a Soft Actor-Critic (SAC) policy for a unicycle using
nominal, obstacle-free dynamics. The trained policy is then deployed with an
optional HOCBF safety filter and an unmodeled velocity disturbance.

## Supported scope

- Environment: `Unicycle`
- Modes: `train` and `test`
- Training CBF mode: `off`
- Deployment CBF modes: `off` or `baseline`
- Integration: forward Euler with `dt = 0.05 s`

## State, action, and reference

The state and control are

```text
z = [x, y, theta, v]
u = [a, omega]
```

Use `--z0_x`, `--z0_y`, `--z0_theta`, and `--z0_v` to specify a deployment
initial state. Use `--zref_x`, `--zref_y`, `--zref_theta`, and `--zref_v` to
specify the reference. Heading error is wrapped to `[-pi, pi]`.

## Training objective

Training uses nominal dynamics with no obstacles and no plant disturbance:

```text
x_dot     = v cos(theta)
y_dot     = v sin(theta)
theta_dot = omega
v_dot     = a
```

SAC uses `gamma = 0.99` and maximizes the negative quadratic control cost.
For a nonterminal transition,

```text
r_k = -(e_k' Q e_k + u_k' R u_k)
```

where

```text
Q = diag(10, 10, 10, 10)
R = diag(1, 1)
H = 50 Q
```

Training terminates when the existing componentwise goal condition is met or
after 30 steps. On a terminal transition, the reward also includes

```text
-e_(k+1)' H e_(k+1)
```

Both goal and 30-step terminal transitions use a zero replay mask, so the SAC
target does not bootstrap beyond the terminal state. There is no separate
positive goal bonus.

### Train a new policy

Run from the repository root:

```powershell
cd .\RL_SAC_HOCBF

python .\main.py --mode train --env_name Unicycle --cbf_mode off --obs_config none --rand_init True --init_mode grid --init_grid_n 10 --init_x_min -2 --init_x_max 2 --init_y_min -2 --init_y_max 2 --init_theta_min -2 --init_theta_max 2 --init_v_min -2 --init_v_max 2 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --gamma 0.99 --train_horizon_steps 200 --terminal_weight 50 --r_a 1 --r_w 1 --disturbance_enabled False --seed 12345
```

Training samples initial states from the `10 x 10 x 10 x 10` grid. Model files
are written under `output/Unicycle-run*/`. Retrain the policy after changing
the reward or horizon; checkpoints trained with the previous reward represent
a different objective.

## Deployment paradigm

Deployment uses the fixed trained policy. By default, the plant receives an
acceleration disturbance at every step:

```text
d_v,k ~ Normal(0, 0.2^2)
v_dot = a + d_v,k
```

The sampled disturbance is held constant over one Euler step, so

```text
v_(k+1) = v_k + dt (a_k + d_v,k)
```

The SAC policy, nominal dynamics model, and HOCBF filter do not receive the
realized disturbance. They react only through later state observations.

Deployment runs for 200 steps (`10 s`) by default, even if the goal is reached,
so regulation under continued disturbances can be measured. The first
goal-reaching step is still recorded. Add `--terminate_on_goal_eval True` to
stop immediately after reaching the goal.

The relevant deployment options are:

```text
--eval_horizon_steps 200
--disturbance_enabled True
--disturbance_velocity_std 0.2
--disturbance_seed 10
--terminate_on_goal_eval False
```

If `--resume` is a number such as `1`, it resolves to
`output/Unicycle-run1`.

### Case A: no obstacles

```powershell
cd .\RL_SAC_HOCBF

python .\main.py --mode test --env_name Unicycle --resume 5 --cbf_mode off --obs_config none --rand_init False --z0_x -1.9 --z0_y 2 --z0_theta -1.79 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled False --disturbance_velocity_std 0.3 --disturbance_seed 10 --validate_episodes 1 --save_anim --anim_name RL_SAC_no_obs_caseA --save_final_shot --final_shot_name RL_SAC_finalshot_no_obs_caseA --save_traj_plots --traj_plot_name RL_SAC_MSD_no_obs_caseA --traj_plot_dir .\figs\diagnostics --initial_state_option a --seed 12345   
```

### Case A: two obstacles with HOCBF filtering

```powershell
cd .\RL_SAC_HOCBF

python .\main.py --mode test --env_name Unicycle --resume 4 --cbf_mode baseline --obs_config none --rand_init False --obs_circles="-0.5,1.1,0.1; -1.5,1.2,0.1" --z0_x -1.9 --z0_y 2 --z0_theta -1.79 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --robot_r 0.25 --cbf_buffer 0.05 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled False --disturbance_velocity_std 0.2 --disturbance_seed 10 --validate_episodes 1 --save_anim --anim_name RL_SAC_with_2obs_caseA --save_final_shot --final_shot_name RL_SAC_finalshot_with_2obs_caseA --save_traj_plots --traj_plot_name RL_SAC_MSD_with_2obs_caseA --traj_plot_dir .\figs\diagnostics --initial_state_option a --seed 12345
```

### Case B: unseen initial state, no obstacles

```powershell
cd .\RL_SAC_HOCBF

python .\main.py --mode test --env_name Unicycle --resume 5 --cbf_mode off --obs_config none --rand_init False --z0_x -5.24 --z0_y 4.11 --z0_theta 2.72 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled False --disturbance_velocity_std 0.2 --disturbance_seed 10 --validate_episodes 1 --save_anim --anim_name RL_SAC_no_obs_caseB --save_final_shot --final_shot_name RL_SAC_finalshot_no_obs_caseB --save_traj_plots --traj_plot_name RL_SAC_MSD_no_obs_caseB --traj_plot_dir .\figs\diagnostics --initial_state_option b --seed 12345
```

### Case B: unseen initial state, with obstacles

```powershell
cd .\RL_SAC_HOCBF

python .\main.py --mode test --env_name Unicycle --resume 5 --cbf_mode baseline --obs_config none --rand_init False --obs_circles="-4.0,2,0.5; -1.5,1.5,0.5; -2.5,3,0.5" --z0_x -5.24 --z0_y 4.11 --z0_theta 2.72 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled False --disturbance_velocity_std 0.3 --disturbance_seed 10 --validate_episodes 1 --save_anim --anim_name RL-PB_with_obs_caseB --save_final_shot --final_shot_name RL_PB_finalshot_with_obs_caseB --save_traj_plots --traj_plot_name RL_PB_MSD_with_obs_caseB --traj_plot_dir .\figs\diagnostics --initial_state_option b --seed 12345 --qp-solver qpoases 
```

### Case C: unseen initial state, nonzero reference, no obstacles

```powershell
cd .\RL_SAC_HOCBF

python .\main.py --mode test --env_name Unicycle --resume 5 --cbf_mode off --obs_config none --rand_init False --z0_x -5.24 --z0_y 4.11 --z0_theta 2.72 --z0_v 0 --zref_x 1 --zref_y 1 --zref_theta 0 --zref_v 0 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled False --disturbance_velocity_std 0.3 --disturbance_seed 10 --validate_episodes 1 --save_anim --anim_name RL_SAC_no_obs_caseC --save_final_shot --final_shot_name RL_SAC_finalshot_no_obs_caseC --save_traj_plots --traj_plot_name RL_SAC_MSD_no_obs_caseC --traj_plot_dir .\figs\diagnostics --initial_state_option c --seed 12345
```

### Case C: unseen initial state, nonzero reference, three obstacles

```powershell
cd .\RL_SAC_HOCBF

python .\main.py --mode test --env_name Unicycle --resume 5 --cbf_mode baseline --obs_config none --rand_init False --obs_circles="-3.5,2,0.5; -1,1.5,0.5; -2,3,0.5" --z0_x -5.24 --z0_y 4.11 --z0_theta 2.72 --z0_v 0 --zref_x 1 --zref_y 1 --zref_theta 0 --zref_v 0 --robot_r 0.25 --cbf_buffer 0.05 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled False --disturbance_velocity_std 0.3 --disturbance_seed 10 --validate_episodes 1 --save_anim --anim_name RL_PB_with_3obs_caseC --save_final_shot --final_shot_name RL_PB_finalshot_with_3obs_caseC --save_traj_plots --traj_plot_name RL_PB_MSD_with_3obs_caseC --traj_plot_dir .\figs\diagnostics --initial_state_option c --seed 12345 --qp-solver qpoases
```

## Randomized deployment evaluations

These scripts use a distinct deterministic disturbance seed for each trial.
Set `--disturbance_enabled False` to reproduce a nominal evaluation.

### Random initial states in the Case A training domain

```powershell
cd .\RL_SAC_HOCBF

python .\RL_case_a_init_state_randomized.py --resume 3 --num_trials 100 --success_threshold 0.6 --d_min 0.5 --cbf_mode off --seed 0 --init_v_min 0 --init_v_max 0 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled False --disturbance_velocity_std 0.2 --disturbance_seed 10 --max_failed_print 1
```

### Random obstacles with HOCBF filtering

```powershell
cd .\RL_SAC_HOCBF

python .\RL_obstacles_randomized.py --resume 1 --case_option a --num_trials 100 --success_threshold 0.6 --cbf_mode baseline --seed 0 --robot_r 0.25 --cbf_buffer 0.05 --obstacle_min_sep 0.5 --eval_horizon_steps 200 --terminate_on_goal_eval False --disturbance_enabled True --disturbance_velocity_std 0.2 --disturbance_seed 10
```

## HOCBF geometry

For a circular obstacle, the effective safety radius used by the HOCBF and
trajectory plots is

```text
r_eff = obstacle_radius + robot_r + cbf_buffer
```

With the defaults, `robot_r = 0.25 m` and `cbf_buffer = 0.05 m`. The HOCBF is
nominal; the Gaussian deployment forcing is not included in its constraint
prediction.

## Deployment QP solver

Obstacle deployments use the differentiable `qpth` backend by default. To use
the CPU qpOASES backend instead, add `--qp-solver qpoases` to `main.py` or
either shared obstacle-benchmark command. The qpOASES solver is rebuilt at
every control step. It requires a CasADi build that includes the qpOASES plugin
and is intentionally disabled for training.
