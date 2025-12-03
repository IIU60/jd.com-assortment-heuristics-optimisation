"""Simulated Annealing heuristic."""

import numpy as np
import random
import math
from typing import Tuple, List, Optional

from ..model.instance import Instance
from .utils import copy_u, evaluate_u
from .neighborhoods import generate_neighbor


def simulated_annealing(
    instance: Instance,
    u_init: np.ndarray,
    T0: float = 50000.0,
    alpha: float = 0.95,
    max_iters: int = 500,
    neighbors_per_iter: int = 1,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[float, np.ndarray, List[float]]:
    """
    Simulated Annealing over shipment plans.
    
    Args:
        instance: Problem instance
        u_init: Initial shipment plan, shape (T, N, J)
        T0: Initial temperature
        alpha: Cooling rate (temperature *= alpha each iteration)
        max_iters: Maximum number of iterations
        neighbors_per_iter: Number of neighbors to try per iteration
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
    best_u = copy_u(current_u)
    best_cost = current_cost
    
    cost_log = [current_cost]
    T = T0
    
    for k in range(max_iters):
        # Generate neighbor
        candidate_u, _ = generate_neighbor(instance, current_u)
        candidate_cost, _ = evaluate_u(instance, candidate_u)
        
        delta = candidate_cost - current_cost
        
        # Accept or reject
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_u = candidate_u
            current_cost = candidate_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_u = copy_u(current_u)
        
        cost_log.append(best_cost)
        
        # Cool temperature
        T *= alpha
        if T < 1e-6:
            break
        
        # Verbose output
        if verbose and (k + 1) % 50 == 0:
            print(f"Iteration {k+1}/{max_iters}: best_cost={best_cost:.2f}, "
                  f"current_cost={current_cost:.2f}, T={T:.2f}")
    
    return best_cost, best_u, cost_log

