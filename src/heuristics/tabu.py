"""Tabu Search heuristic (time-limited, single-start)."""

import time
import numpy as np
import random
from typing import Tuple, List, Dict, Optional

from ..model.instance import Instance
from ..model.result import SimulationResult
from .utils import copy_u, evaluate_u, evaluate_u_incremental
from .neighborhoods import generate_neighbor
from .neighborhoods import (
    product_rebalance_move,
    block_replan_move,
)


def tabu_search(
    instance: Instance,
    u_init: np.ndarray,
    tabu_tenure: int = 5,
    max_iters: int = 200,
    neighborhood_size: int = 8,
    max_no_improve: Optional[int] = 50,
    verbose: bool = False,
    seed: Optional[int] = None,
    large_move_prob: float = 0.0,
    time_limit: Optional[float] = None,
    problem_aware_prob: float = 0.8,
    tabu_acceptance_threshold: float = 0.01,
) -> Tuple[float, np.ndarray, List[float]]:
    """
    Tabu Search over shipment plans (single-start, time-limited).
    
    Uses a sampled neighborhood and a simple tabu list on move keys.
    Stops when either `time_limit` (if provided) or `max_iters` / `max_no_improve`
    are reached.
    
    Args:
        problem_aware_prob: Probability of using problem-aware neighbor generation
            (default: 0.8). Higher values favor moves targeting high-cost/high-demand
            areas; lower values favor random exploration.
        tabu_acceptance_threshold: Maximum relative cost increase allowed for tabu moves
            (default: 0.01 = 1%). Tabu moves that don't improve are only accepted if
            within this threshold of the best cost.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    start_time = time.perf_counter()
    
    current_u = copy_u(u_init)
    current_cost, _ = evaluate_u(instance, current_u)
    # Cache the full simulation result for incremental evaluation
    from ..model.simulator import simulate
    cached_result = simulate(instance, current_u, check_feasibility=False)
    
    best_u = copy_u(current_u)
    best_cost = current_cost
    
    cost_log = [current_cost]
    tabu_list: Dict[Tuple, int] = {}  # move_key -> remaining tenure
    no_improve_count = 0
    
    for it in range(max_iters):
        # Time-based stopping
        if time_limit is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit:
                if verbose:
                    print(f"Stopping Tabu due to time limit ({elapsed:.2f}s >= {time_limit:.2f}s)")
                break
        
        # Sample neighborhood
        candidates = []
        for _ in range(neighborhood_size):
            # Occasionally apply large moves to diversify neighborhood
            use_large_move = large_move_prob > 0.0 and random.random() < large_move_prob
            if use_large_move:
                T_horizon = instance.T
                N = instance.num_products
                i = random.randrange(N)
                if random.random() < 0.5:
                    new_u = product_rebalance_move(current_u, instance, i)
                    move_key = ("product_rebalance", i)
                else:
                    window = random.randint(2, min(4, T_horizon))
                    t_start = random.randrange(0, max(1, T_horizon - window + 1))
                    t_end = t_start + window - 1
                    new_u = block_replan_move(current_u, instance, i, t_start, t_end)
                    move_key = ("block_replan", i, t_start, t_end)
            else:
                # Pass simulation result for problem-aware moves
                new_u, move_key = generate_neighbor(
                    instance,
                    current_u,
                    problem_aware_prob=problem_aware_prob,
                    simulation_result=cached_result,
                )
            # Use incremental evaluation
            cost, _, new_result = evaluate_u_incremental(
                instance, new_u, current_u, cached_result
            )
            candidates.append((cost, new_u, move_key, new_result))
        
        if not candidates:
            break
        
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
        chosen_cost, chosen_u, chosen_key, chosen_result = None, None, None, None
        
        for cost, u_cand, key, result in candidates:
            # Aspiration criterion: allow tabu if improves global best
            if key in tabu_list and cost >= best_cost:
                # Tabu and doesn't improve: skip
                continue
            
            chosen_cost, chosen_u, chosen_key, chosen_result = cost, u_cand, key, result
            break
        
        # If all candidates are tabu, only accept if within threshold
        if chosen_u is None:
            best_candidate = candidates[0]
            candidate_cost = best_candidate[0]
            # Only accept if it improves or is within threshold
            if candidate_cost < best_cost * (1.0 + tabu_acceptance_threshold):
                chosen_cost, chosen_u, chosen_key, chosen_result = best_candidate
            else:
                # Reject: keep current solution (no move)
                no_improve_count += 1
                cost_log.append(best_cost)
                continue
        
        # Update current solution and cache
        current_u = chosen_u
        current_cost = chosen_cost
        cached_result = chosen_result
        
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
            print(
                f"Iteration {it+1}/{max_iters}: best_cost={best_cost:.2f}, "
                f"current_cost={current_cost:.2f}, tabu_size={len(tabu_list)}"
            )
    
    return best_cost, best_u, cost_log

