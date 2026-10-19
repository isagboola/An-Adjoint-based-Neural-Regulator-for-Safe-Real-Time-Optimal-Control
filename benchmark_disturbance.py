"""Reproducible plant disturbances shared by randomized benchmark evaluators."""

import numpy as np


def trial_number(benchmark, initial_state):
    """Return the one-based CSV trial number for an exact saved initial state."""
    state = np.asarray(initial_state, dtype=float).reshape(4)
    matches = np.flatnonzero(np.all(benchmark.initial_states == state, axis=1))
    if len(matches) != 1:
        raise ValueError(
            "The rollout initial state must match exactly one saved benchmark trial."
        )
    return int(matches[0]) + 1


def velocity_disturbance_sequence(
    benchmark, initial_state, *, enabled, standard_deviation, seed
):
    """Generate the fixed i.i.d. Gaussian sequence assigned to one trial."""
    standard_deviation = float(standard_deviation)
    if standard_deviation < 0.0 or not np.isfinite(standard_deviation):
        raise ValueError("The disturbance standard deviation must be finite and nonnegative.")
    number = trial_number(benchmark, initial_state)
    if not enabled:
        return np.zeros(benchmark.steps, dtype=float)
    # SeedSequence avoids dependence on rollout order or multiprocessing scheduling.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), number]))
    return rng.normal(0.0, standard_deviation, size=benchmark.steps)


def disturbance_metadata(*, enabled, standard_deviation, seed):
    """Describe the common deployment disturbance used by an evaluator."""
    return {
        "disturbance_enabled": bool(enabled),
        "disturbance_velocity_std": float(standard_deviation),
        "disturbance_seed": int(seed),
        "disturbance_distribution": "iid Normal(0, disturbance_velocity_std^2)",
        "disturbance_assignment": "SeedSequence([disturbance_seed, one-based trial number])",
        "disturbance_location": "plant velocity derivative: v_dot = a + d_v",
    }
