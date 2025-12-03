"""Tabu Search heuristic."""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional

from ..model.instance import Instance
from .utils import copy_u, evaluate_u
from .neighborhoods import generate_neighbor


def tabu_search(
    instance: Instance,
    u_init: np.ndarray,
    tabu_tenure: int = 5,
    max_iters: int = 200,
    neighborhood_size: int = 30,
    max_no_improve: Optional[int] = 50,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[float, np.ndarray, List[float]]:
    """
    Tabu Search over shipment plans.
    
    Uses random sampled neighborhood and a tabu list on move keys.
    
    Args:
        instance: Problem instance
        u_init: Initial shipment plan, shape (T, N, J)
        tabu_tenure: Tabu list tenure (number of iterations to keep move tabu)
        max_iters: Maximum number of iterations
        neighborhood_size: Number of neighbors to sample per iteration
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
    best_u = copy_u(current_u)
    best_cost = current_cost
    
    cost_log = [current_cost]
    tabu_list: Dict[Tuple, int] = {}  # move_key -> remaining tenure
    no_improve_count = 0
    
    for it in range(max_iters):
        # Sample neighborhood
        candidates = []
        for _ in range(neighborhood_size):
            new_u, move_key = generate_neighbor(instance, current_u)
            cost, _ = evaluate_u(instance, new_u)
            candidates.append((cost, new_u, move_key))
        
        # Update tabu tenures
        to_delete = []
        for key in tabu_list:
            tabu_list[key] -= 1
            if tabu_list[key] <= 0:
                to_delete.append(key)
        for key in to_delete:
            del tabu_list[key]
        
        # Sort candidates by cost
        candidates.sort(key=lambda x: x[0])
        
        # Choose best non-tabu candidate (or best if all tabu and doesn't improve)
        chosen_cost, chosen_u, chosen_key = None, None, None
        
        for cost, u_cand, key in candidates:
            # Aspiration criterion: allow tabu if improves global best
            if key in tabu_list and cost >= best_cost:
                # Tabu and doesn't improve: skip
                continue
            
            chosen_cost, chosen_u, chosen_key = cost, u_cand, key
            break
        
        # If all candidates are tabu, take the best one anyway
        if chosen_u is None:
            chosen_cost, chosen_u, chosen_key = candidates[0]
        
        # Update current solution
        current_u = chosen_u
        current_cost = chosen_cost
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_u = copy_u(current_u)
            no_improve_count = 0  # Reset counter on improvement
        else:
            no_improve_count += 1
        
        # Add to tabu list
        tabu_list[chosen_key] = tabu_tenure
        
        cost_log.append(best_cost)
        
        # Early stopping if no improvement
        if max_no_improve is not None and no_improve_count >= max_no_improve:
            if verbose:
                print(f"Stopping early: no improvement for {max_no_improve} iterations")
            break
        
        # Verbose output
        if verbose and (it + 1) % 50 == 0:
            print(f"Iteration {it+1}/{max_iters}: best_cost={best_cost:.2f}, "
                  f"current_cost={current_cost:.2f}, tabu_size={len(tabu_list)}")
    
    return best_cost, best_u, cost_log

