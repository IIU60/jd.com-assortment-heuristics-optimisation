"""Construction heuristics: greedy and GRASP."""

import numpy as np
import random
from typing import Optional, List, Dict, Any

from ..model.instance import Instance
from ..baselines import myopic_greedy
from .utils import clamp_u_to_feasibility, evaluate_u
from .neighborhoods import generate_neighbor


def greedy_constructor(instance: Instance) -> np.ndarray:
    """
    Greedy constructor using future cumulative demand for priority scores.
    
    For each product i and FDC j, compute priority score based on:
    - Future cumulative demand
    - Transfer cost
    
    Allocate shipments to FDCs by descending score, respecting constraints.
    
    Args:
        instance: Problem instance
    
    Returns:
        Feasible shipment plan, shape (T, N, J)
    """
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    shipments = np.zeros((T, N, J), dtype=float)
    
    # Compute future cumulative demand for each (i, j)
    # future_demand[i, j] = sum of demand from period t onwards
    future_demand = np.zeros((T, N, J), dtype=float)
    for t in range(T):
        future_demand[t, :, :] = np.sum(instance.demand_fdc[t:, :, :], axis=0)
    
    # Track inventories
    inventory_rdc = instance.initial_inventory_rdc.copy()
    
    for t in range(T):
        # Add replenishment
        inventory_rdc += instance.replenishment[t, :]
        
        # Remaining outbound capacity
        remaining_outbound = instance.outbound_capacity[t]
        
        # Track FDC total inventory
        if t == 0:
            fdc_total = np.sum(instance.initial_inventory_fdc, axis=0)
        else:
            fdc_total = np.zeros(J)
            for j in range(J):
                fdc_total[j] = np.sum(instance.initial_inventory_fdc[:, j]) + \
                              np.sum(shipments[:t, :, j])
        
        # Compute priority scores for this period
        # Score = future_demand / (transfer_cost + 1) to favor high demand, low cost
        scores = np.zeros((N, J), dtype=float)
        for i in range(N):
            for j in range(J):
                if future_demand[t, i, j] > 0:
                    scores[i, j] = future_demand[t, i, j] / (instance.transfer_cost[i, j] + 1.0)
        
        # Allocate in order of descending score
        # Flatten and sort
        flat_indices = np.argsort(scores.flatten())[::-1]  # Descending
        
        for idx in flat_indices:
            i = idx // J
            j = idx % J
            
            if scores[i, j] <= 0:
                continue
            if inventory_rdc[i] <= 0:
                continue
            if remaining_outbound <= 0:
                break
            
            # Compute desired allocation based on future demand
            desired = min(
                future_demand[t, i, j] * 0.5,  # Allocate 50% of future demand
                inventory_rdc[i],
                remaining_outbound
            )
            
            # FDC capacity constraint
            if instance.fdc_capacity is not None:
                cap_fdc = instance.fdc_capacity[j] - fdc_total[j]
            else:
                cap_fdc = np.inf
            
            feasible = min(desired, cap_fdc)
            
            shipments[t, i, j] = feasible
            inventory_rdc[i] -= feasible
            fdc_total[j] += feasible
            remaining_outbound -= feasible
    
    # Clamp to ensure feasibility
    shipments = clamp_u_to_feasibility(instance, shipments)
    
    return shipments


def grasp_constructor(
    instance: Instance,
    alpha: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    GRASP constructor: greedy with randomized selection from RCL.
    
    At each allocation choice:
    1. Compute scores for all (i, j) pairs
    2. Build Restricted Candidate List (RCL) of good FDCs
    3. Randomly select from RCL
    
    Args:
        instance: Problem instance
        alpha: GRASP parameter (0 = pure random, 1 = pure greedy)
            Controls size of RCL: alpha=0.5 means RCL contains top 50% of candidates
        seed: Random seed (optional)
    
    Returns:
        Feasible shipment plan, shape (T, N, J)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    shipments = np.zeros((T, N, J), dtype=float)
    
    # Compute future cumulative demand
    future_demand = np.zeros((T, N, J), dtype=float)
    for t in range(T):
        future_demand[t, :, :] = np.sum(instance.demand_fdc[t:, :, :], axis=0)
    
    # Track inventories
    inventory_rdc = instance.initial_inventory_rdc.copy()
    
    for t in range(T):
        # Add replenishment
        inventory_rdc += instance.replenishment[t, :]
        
        # Remaining outbound capacity
        remaining_outbound = instance.outbound_capacity[t]
        
        # Track FDC total inventory
        if t == 0:
            fdc_total = np.sum(instance.initial_inventory_fdc, axis=0)
        else:
            fdc_total = np.zeros(J)
            for j in range(J):
                fdc_total[j] = np.sum(instance.initial_inventory_fdc[:, j]) + \
                              np.sum(shipments[:t, :, j])
        
        # Compute scores
        scores = np.zeros((N, J), dtype=float)
        for i in range(N):
            for j in range(J):
                if future_demand[t, i, j] > 0:
                    scores[i, j] = future_demand[t, i, j] / (instance.transfer_cost[i, j] + 1.0)
        
        # Build RCL and allocate
        # For each product, select FDC from RCL
        for i in range(N):
            if inventory_rdc[i] <= 0 or remaining_outbound <= 0:
                continue
            
            # Get scores for this product
            product_scores = scores[i, :].copy()
            
            # Build RCL: top (1-alpha) fraction of candidates
            if alpha < 1.0:
                # Sort by score descending
                sorted_indices = np.argsort(product_scores)[::-1]
                n_rcl = max(1, int((1 - alpha) * J))
                rcl_indices = sorted_indices[:n_rcl]
            else:
                # Pure greedy: all candidates
                rcl_indices = np.arange(J)
            
            # Filter RCL to only feasible FDCs
            feasible_rcl = []
            for j in rcl_indices:
                if product_scores[j] <= 0:
                    continue
                
                # Check FDC capacity
                if instance.fdc_capacity is not None:
                    cap_fdc = instance.fdc_capacity[j] - fdc_total[j]
                else:
                    cap_fdc = np.inf
                
                if cap_fdc > 0:
                    feasible_rcl.append(j)
            
            if not feasible_rcl:
                continue
            
            # Randomly select from RCL
            selected_j = random.choice(feasible_rcl)
            
            # Allocate
            desired = min(
                future_demand[t, i, selected_j] * 0.5,
                inventory_rdc[i],
                remaining_outbound
            )
            
            if instance.fdc_capacity is not None:
                cap_fdc = instance.fdc_capacity[selected_j] - fdc_total[selected_j]
            else:
                cap_fdc = np.inf
            
            feasible = min(desired, cap_fdc)
            
            shipments[t, i, selected_j] = feasible
            inventory_rdc[i] -= feasible
            fdc_total[selected_j] += feasible
            remaining_outbound -= feasible
    
    # Clamp to ensure feasibility
    shipments = clamp_u_to_feasibility(instance, shipments)
    
    return shipments


def build_starting_pool(
    instance: Instance,
    max_starts: int = 4,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Build a diversified pool of high-quality starting solutions.
    
    Includes:
    - Myopic greedy baseline
    - Greedy constructor
    - GRASP with different alphas / seeds
    - A lightly perturbed variant of myopic (stochastic smoothing)
    
    Args:
        instance: Problem instance
        max_starts: Maximum number of starting solutions to return
        seed: Base random seed for reproducibility
    
    Returns:
        List of dictionaries with keys: 'name', 'cost', 'u'
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)
    
    starts: List[Dict[str, Any]] = []
    
    # 1) Myopic greedy
    cost_myopic, u_myopic = myopic_greedy(instance)
    starts.append({'name': 'myopic', 'cost': float(cost_myopic), 'u': u_myopic})
    
    # 2) Greedy constructor
    u_greedy = greedy_constructor(instance)
    cost_greedy, _ = evaluate_u(instance, u_greedy)
    starts.append({'name': 'greedy', 'cost': float(cost_greedy), 'u': u_greedy})
    
    # 3) GRASP with different alphas / seeds
    grasp_configs = [
        (0.3, seed + 1),
        (0.7, seed + 2),
    ]
    for alpha, s in grasp_configs:
        u_grasp = grasp_constructor(instance, alpha=alpha, seed=s)
        cost_grasp, _ = evaluate_u(instance, u_grasp)
        starts.append({
            'name': f'grasp_{alpha:.1f}',
            'cost': float(cost_grasp),
            'u': u_grasp
        })
    
    # 4) Stochastic smoothing of myopic: apply a few neighbor moves
    #    to introduce structural diversity around a strong baseline.
    try:
        u_smooth = np.copy(u_myopic)
        # Apply a small number of problem-aware neighbor moves
        from ..model.simulator import simulate
        cached_result = simulate(instance, u_smooth, check_feasibility=False)
        for _ in range(5):
            u_candidate, _ = generate_neighbor(
                instance,
                u_smooth,
                problem_aware_prob=0.5,
                simulation_result=cached_result,
            )
            # Accept if it improves or with small probability to diversify
            cost_current, _ = evaluate_u(instance, u_smooth)
            cost_candidate, _ = evaluate_u(instance, u_candidate)
            if cost_candidate <= cost_current or rng.rand() < 0.2:
                u_smooth = u_candidate
        cost_smooth, _ = evaluate_u(instance, u_smooth)
        starts.append({'name': 'myopic_smooth', 'cost': float(cost_smooth), 'u': u_smooth})
    except Exception:
        # In case anything goes wrong, just skip the smoothed variant
        pass
    
    # Sort by cost (best first) and truncate
    starts.sort(key=lambda s: s['cost'])
    return starts[:max_starts]

