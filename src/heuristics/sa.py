"""Simulated Annealing heuristic."""

import numpy as np
import random
import math
from typing import Tuple, List, Optional

from ..model.instance import Instance
from ..model.simulator import simulate
from .utils import copy_u, evaluate_u
from .neighborhoods import generate_neighbor


def simulated_annealing(
    instance: Instance,
    u_init: np.ndarray,
    T0: Optional[float] = None,
    alpha: float = 0.95,
    max_iters: int = 500,
    neighbors_per_iter: int = 1,
    acceptance_rate: float = 0.9,
    max_no_improve: Optional[int] = 100,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[float, np.ndarray, List[float]]:
    """
    Simulated Annealing over shipment plans.
    
    Args:
        instance: Problem instance
        u_init: Initial shipment plan, shape (T, N, J)
        T0: Initial temperature (if None, computed adaptively to achieve acceptance_rate)
        alpha: Cooling rate (temperature *= alpha each iteration)
        max_iters: Maximum number of iterations
        neighbors_per_iter: Number of neighbors to try per iteration
        acceptance_rate: Target acceptance rate for non-improving moves (used if T0 is None)
        max_no_improve: Maximum iterations without improvement before early stopping (None to disable)
        verbose: If True, print progress every 50 iterations
        seed: Random seed (optional)
    
    Returns:
        Tuple of (best_cost, best_u, cost_log)
        cost_log is list of costs over iterations
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    current_u = copy_u(u_init)
    current_cost, _ = evaluate_u(instance, current_u)
    # Cache simulation result for problem-aware moves
    cached_result = simulate(instance, current_u, check_feasibility=False)
    
    best_u = copy_u(current_u)
    best_cost = current_cost
    
    cost_log = [current_cost]
    
    # Adaptive T0 calculation if not provided
    if T0 is None:
        # Warm-up phase: sample neighbors to estimate cost differences
        warmup_samples = 50
        deltas = []
        
        for _ in range(warmup_samples):
            candidate_u, _ = generate_neighbor(instance, current_u)
            candidate_cost, _ = evaluate_u(instance, candidate_u)
            delta = candidate_cost - current_cost
            if delta > 0:  # Only consider non-improving moves
                deltas.append(delta)
        
        if deltas:
            # Calculate T0 to achieve target acceptance rate
            # We want: exp(-delta_avg / T0) = acceptance_rate
            # So: T0 = -delta_avg / ln(acceptance_rate)
            delta_avg = np.mean(deltas)
            T0 = -delta_avg / math.log(acceptance_rate)
            # Ensure T0 is positive and reasonable
            T0 = max(1.0, T0)
            if verbose:
                print(f"Adaptive T0 = {T0:.2f} (from {len(deltas)} non-improving samples)")
        else:
            # Fallback: use percentage of current cost
            T0 = current_cost * 0.1
            if verbose:
                print(f"Adaptive T0 = {T0:.2f} (fallback: 10% of initial cost)")
    
    T = T0
    no_improve_count = 0
    
    for k in range(max_iters):
        # Generate neighbor (with problem-aware moves)
        candidate_u, _ = generate_neighbor(
            instance, current_u,
            problem_aware_prob=0.3,
            simulation_result=cached_result
        )
        candidate_cost, _ = evaluate_u(instance, candidate_u)
        
        delta = candidate_cost - current_cost
        
        # Accept or reject
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_u = candidate_u
            current_cost = candidate_cost
            # Update cached result for next iteration
            cached_result = simulate(instance, current_u, check_feasibility=False)
            
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
            print(f"Iteration {k+1}/{max_iters}: best_cost={best_cost:.2f}, "
                  f"current_cost={current_cost:.2f}, T={T:.2f}")
    
    return best_cost, best_u, cost_log

