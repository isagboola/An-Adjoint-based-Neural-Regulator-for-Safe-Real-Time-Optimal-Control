"""Process-worker initialization for randomized IPOPT MPC benchmarks."""

import importlib


_WORKER_BENCHMARK = None
_WORKER_ROLLOUT = None


def initialize_mpc_worker(evaluator_module, dataset_path, prediction_horizon):
    """Create one independent benchmark and MPC controller in each process."""
    global _WORKER_BENCHMARK, _WORKER_ROLLOUT
    module = importlib.import_module(evaluator_module)
    _WORKER_BENCHMARK = module.load_benchmark(dataset_path)
    _WORKER_ROLLOUT, _ = module.load_controller(
        prediction_horizon, _WORKER_BENCHMARK
    )


def run_mpc_worker(initial_state):
    if _WORKER_ROLLOUT is None or _WORKER_BENCHMARK is None:
        raise RuntimeError("MPC worker was not initialized.")
    return _WORKER_ROLLOUT(initial_state, _WORKER_BENCHMARK)
