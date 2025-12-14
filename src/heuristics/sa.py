"""Simulated Annealing heuristic (time-limited, single-start)."""

import json
import time
from pathlib import Path

import numpy as np
import random
import math
from typing import Tuple, List, Optional, Dict

from ..model.instance import Instance
from ..model.simulator import simulate
from .utils import copy_u, evaluate_u, evaluate_u_incremental
from .neighborhoods import generate_neighbor
from .neighborhoods import (
    generate_problem_aware_neighbor,
    product_rebalance_move,
    block_replan_move,
)

_T0_CONFIG_CACHE: Optional[Dict[str, float]] = None


def _instance_size_key(instance: Instance) -> str:
    """Map an instance to a coarse size category (must match tuner script)."""
    n = instance.num_products
    if n <= 10:
        return "small"
    if n <= 100:
        return "medium"
    return "large"


def _load_sa_t0_config() -> Optional[Dict[str, float]]:
    """Load cached SA T0 configuration if available."""
    global _T0_CONFIG_CACHE
    if _T0_CONFIG_CACHE is not None:
        return _T0_CONFIG_CACHE

    # Config produced by `src/experiments/tune_sa_t0.py`
    config_path = Path(__file__).resolve().parents[2] / "data" / "sa_t0_config.json"
    if not config_path.exists():
        _T0_CONFIG_CACHE = None
        return None

    try:
        with config_path.open("r") as f:
            data = json.load(f)
        # Ensure values are floats
        _T0_CONFIG_CACHE = {k: float(v) for k, v in data.items()}
    except Exception:
        _T0_CONFIG_CACHE = None
    return _T0_CONFIG_CACHE


def _lookup_t0(instance: Instance, base_cost: float) -> Optional[float]:
    """Return an absolute T0 based on tuned multipliers, or None if unavailable."""
    config = _load_sa_t0_config()
    if not config:
        return None
    key = _instance_size_key(instance)
    mult = config.get(key)
    if mult is None:
        return None
    return max(1.0, float(mult) * base_cost)


def simulated_annealing(
    instance: Instance,
    u_init: np.ndarray,
    T0: Optional[float] = None,
    alpha: float = 0.95,
    max_iters: int = 500,
    acceptance_rate: float = 0.9,  # kept for backward-compatibility (ignored)
    max_no_improve: Optional[int] = 100,
    verbose: bool = False,
    seed: Optional[int] = None,
    large_move_prob: float = 0.0,
    time_limit: Optional[float] = None,
    problem_aware_prob: float = 0.8,
) -> Tuple[float, np.ndarray, List[float]]:
    """
    Simulated Annealing over shipment plans (single-start, time-limited).
    
    This version removes the expensive adaptive T0 warm-up and instead:
    - Uses a provided T0 directly if given
    - Otherwise falls back to a simple fraction of the initial cost
    - Stops when either `time_limit` (if provided) or `max_iters` / `max_no_improve`
      are reached.
    
    Args:
        problem_aware_prob: Probability of using problem-aware neighbor generation
            (default: 0.8). Higher values favor moves targeting high-cost/high-demand
            areas; lower values favor random exploration.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    start_time = time.perf_counter()
    
    current_u = copy_u(u_init)
    current_cost, _ = evaluate_u(instance, current_u)
    # Cache simulation result for problem-aware moves
    cached_result = simulate(instance, current_u, check_feasibility=False)
    
    best_u = copy_u(current_u)
    best_cost = current_cost
    
    cost_log = [current_cost]
    
    # T0 selection:
    # 1) if provided explicitly, use it as-is
    # 2) otherwise, look up from tuned config (multiplier * current_cost)
    # 3) fallback: simple percentage of current cost
    if T0 is None:
        tuned_T0 = _lookup_t0(instance, current_cost)
        if tuned_T0 is not None:
            T0 = tuned_T0
            if verbose:
                print(f"T0 not provided; using tuned T0 = {T0:.2f}")
        else:
            T0 = max(1.0, 0.05 * current_cost)
            if verbose:
                print(f"T0 not provided; using heuristic T0 = {T0:.2f}")
    
    T = T0
    no_improve_count = 0
    
    for k in range(max_iters):
        # Time-based stopping
        if time_limit is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit:
                if verbose:
                    print(f"Stopping SA due to time limit ({elapsed:.2f}s >= {time_limit:.2f}s)")
                break
        
        # Occasionally apply a large move to escape local basins
        use_large_move = large_move_prob > 0.0 and random.random() < large_move_prob
        if use_large_move:
            # Large move via product rebalance or block replan, guided by simulation
            T_horizon = instance.T
            N = instance.num_products
            i = random.randrange(N)
            if random.random() < 0.5:
                candidate_u = product_rebalance_move(current_u, instance, i)
            else:
                window = random.randint(2, min(4, T_horizon))
                t_start = random.randrange(0, max(1, T_horizon - window + 1))
                t_end = t_start + window - 1
                candidate_u = block_replan_move(current_u, instance, i, t_start, t_end)
        else:
            # Generate neighbor (with problem-aware moves)
            candidate_u, _ = generate_neighbor(
                instance,
                current_u,
                problem_aware_prob=problem_aware_prob,
                simulation_result=cached_result,
            )
        # Use incremental evaluation for faster neighbor evaluation
        candidate_cost, _, candidate_result = evaluate_u_incremental(
            instance, candidate_u, current_u, cached_result
        )
        
        delta = candidate_cost - current_cost
        
        # Accept or reject
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_u = candidate_u
            current_cost = candidate_cost
            # Update cached result for next iteration (already computed by incremental evaluation)
            cached_result = candidate_result
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_u = copy_u(current_u)
                no_improve_count = 0  # Reset counter on improvement
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        cost_log.append(best_cost)
        
        # Early stopping if no improvement
        if max_no_improve is not None and no_improve_count >= max_no_improve:
            if verbose:
                print(f"Stopping early: no improvement for {max_no_improve} iterations")
            break
        
        # Cool temperature
        T *= alpha
        if T < 1e-6:
            break
        
        # Verbose output
        if verbose and (k + 1) % 50 == 0:
            print(
                f"Iteration {k+1}/{max_iters}: best_cost={best_cost:.2f}, "
                f"current_cost={current_cost:.2f}, T={T:.2f}"
            )
    
    return best_cost, best_u, cost_log

