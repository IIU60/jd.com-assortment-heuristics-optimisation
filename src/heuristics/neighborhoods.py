"""Neighborhood move operators for local search."""

import numpy as np
import random
from typing import Tuple, Dict, Optional

from ..model.instance import Instance
from ..model.result import SimulationResult
from .utils import copy_u, clamp_u_to_feasibility, get_feasible_delta, compute_available_capacity


def time_shift_move(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j: int,
    t_from: int,
    t_to: int,
    delta: float
) -> np.ndarray:
    """
    Time-shift move: reduce shipment at t_from, increase at t_to.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j: FDC index
        t_from: Period to reduce shipment
        t_to: Period to increase shipment
        delta: Amount to shift
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Reduce at t_from
    new_u[t_from, i, j] = max(0.0, new_u[t_from, i, j] - delta)
    
    # Increase at t_to
    new_u[t_to, i, j] += delta
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def fdc_swap_move(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j_from: int,
    j_to: int,
    t: int,
    delta: float
) -> np.ndarray:
    """
    FDC-swap move: reduce shipment to j_from, increase to j_to.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j_from: FDC to reduce shipment
        j_to: FDC to increase shipment
        t: Period
        delta: Amount to swap
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Reduce at j_from
    new_u[t, i, j_from] = max(0.0, new_u[t, i, j_from] - delta)
    
    # Increase at j_to
    new_u[t, i, j_to] += delta
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def magnitude_tweak(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j: int,
    t: int,
    delta: float
) -> np.ndarray:
    """
    Magnitude tweak: add or subtract delta from shipment.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j: FDC index
        t: Period
        delta: Amount to add (can be negative to subtract)
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Apply delta
    new_u[t, i, j] = max(0.0, new_u[t, i, j] + delta)
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def magnitude_tweak_feasible(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j: int,
    t: int,
    delta: float,
    cached_result: Optional[SimulationResult] = None
) -> np.ndarray:
    """
    Constraint-aware magnitude tweak: add or subtract delta from shipment.
    
    Checks constraints before applying delta to minimize clipping.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j: FDC index
        t: Period
        delta: Desired amount to add (can be negative to subtract)
        cached_result: Optional cached simulation result for constraint checking
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Get feasible delta
    feasible_delta = get_feasible_delta(instance, u, cached_result, i, j, t, delta)
    
    # Apply feasible delta
    new_u[t, i, j] = max(0.0, new_u[t, i, j] + feasible_delta)
    
    # Clamp to feasibility (should be minimal now)
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def fdc_swap_move_feasible(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j_from: int,
    j_to: int,
    t: int,
    delta: float,
    cached_result: Optional[SimulationResult] = None
) -> np.ndarray:
    """
    Constraint-aware FDC-swap move: reduce shipment to j_from, increase to j_to.
    
    Checks constraints at both source and destination.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j_from: FDC to reduce shipment
        j_to: FDC to increase shipment
        t: Period
        delta: Desired amount to swap
        cached_result: Optional cached simulation result for constraint checking
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Can always reduce from j_from (up to current value)
    max_reduce = min(delta, u[t, i, j_from])
    
    # Check capacity at j_to for increase
    if cached_result is not None:
        rdc_avail, fdc_slack, outbound_remaining = compute_available_capacity(instance, cached_result, t)
        avail_rdc = rdc_avail[i]
        avail_fdc_to = fdc_slack[j_to]
        avail_outbound = outbound_remaining
        
        # Maximum feasible increase at j_to
        max_increase = min(avail_rdc, avail_fdc_to, avail_outbound)
        
        # Actual swap amount is min of what we can reduce and what we can increase
        feasible_delta = min(max_reduce, max_increase)
    else:
        # Conservative estimate
        feasible_delta = min(max_reduce, instance.outbound_capacity[t] * 0.1)
    
    # Apply swap
    new_u[t, i, j_from] = max(0.0, new_u[t, i, j_from] - feasible_delta)
    new_u[t, i, j_to] += feasible_delta
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def time_shift_move_feasible(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j: int,
    t_from: int,
    t_to: int,
    delta: float,
    cached_result: Optional[SimulationResult] = None
) -> np.ndarray:
    """
    Constraint-aware time-shift move: reduce shipment at t_from, increase at t_to.
    
    Checks constraints at both periods.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j: FDC index
        t_from: Period to reduce shipment
        t_to: Period to increase shipment
        delta: Desired amount to shift
        cached_result: Optional cached simulation result for constraint checking
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Can always reduce from t_from (up to current value)
    max_reduce = min(delta, u[t_from, i, j])
    
    # Check capacity at t_to for increase
    if cached_result is not None:
        rdc_avail, fdc_slack, outbound_remaining = compute_available_capacity(instance, cached_result, t_to)
        avail_rdc = rdc_avail[i]
        avail_fdc = fdc_slack[j]
        avail_outbound = outbound_remaining
        
        # Maximum feasible increase at t_to
        max_increase = min(avail_rdc, avail_fdc, avail_outbound)
        
        # Actual shift amount is min of what we can reduce and what we can increase
        feasible_delta = min(max_reduce, max_increase)
    else:
        # Conservative estimate
        feasible_delta = min(max_reduce, instance.outbound_capacity[t_to] * 0.1)
    
    # Apply shift
    new_u[t_from, i, j] = max(0.0, new_u[t_from, i, j] - feasible_delta)
    new_u[t_to, i, j] += feasible_delta
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def product_rebalance_move(
    u: np.ndarray,
    instance: Instance,
    i: int
) -> np.ndarray:
    """
    Large move: rebalance shipments for product i across all FDCs
    over the entire horizon, keeping per-period total shipments similar.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index to rebalance
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    T = instance.T
    J = instance.num_fdcs
    
    # Compute weights per FDC based on average demand and transfer cost
    demand_ij = np.mean(instance.demand_fdc[:, i, :], axis=0)  # shape (J,)
    cost_ij = instance.transfer_cost[i, :]  # shape (J,)
    # Higher demand, lower cost preferred
    weights = (demand_ij + 1.0) / (cost_ij + 1.0)
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    
    for t in range(T):
        total_t = float(np.sum(new_u[t, i, :]))
        if total_t <= 0:
            continue
        # Redistribute total shipments at period t according to weights
        new_u[t, i, :] = total_t * weights
    
    new_u = clamp_u_to_feasibility(instance, new_u)
    return new_u


def block_replan_move(
    u: np.ndarray,
    instance: Instance,
    i: int,
    t_start: int,
    t_end: int
) -> np.ndarray:
    """
    Large move: replan shipments for a product i over a time window.
    
    Uses future demand and transfer cost to greedily allocate shipments
    in the window, then clamps to feasibility.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        t_start: Start period (inclusive)
        t_end: End period (inclusive)
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    T = instance.T
    J = instance.num_fdcs
    
    t_start = max(0, min(t_start, T - 1))
    t_end = max(t_start, min(t_end, T - 1))
    
    # Precompute future demand for this product
    future_demand = np.zeros((T, J), dtype=float)
    for t in range(T):
        future_demand[t, :] = np.sum(instance.demand_fdc[t:, i, :], axis=0)
    
    for t in range(t_start, t_end + 1):
        # Zero out current shipments in the window for this product
        new_u[t, i, :] = 0.0
        
        # Use a fraction of future demand as a target volume
        scores = np.zeros(J, dtype=float)
        for j in range(J):
            if future_demand[t, j] > 0:
                scores[j] = future_demand[t, j] / (instance.transfer_cost[i, j] + 1.0)
        if np.all(scores <= 0):
            continue
        
        # Normalize scores to get proportions
        total_score = np.sum(scores)
        proportions = scores / total_score
        
        # Target volume: 40% of cumulative future demand for this period
        target_volume = 0.4 * np.sum(future_demand[t, :])
        if target_volume <= 0:
            continue
        
        for j in range(J):
            new_u[t, i, j] = target_volume * proportions[j]
    
    new_u = clamp_u_to_feasibility(instance, new_u)
    return new_u


def generate_neighbor(
    instance: Instance,
    u: np.ndarray,
    move_probs: Optional[Dict[str, float]] = None,
    delta_choices: Optional[Tuple[float, ...]] = None,
    problem_aware_prob: float = 0.8,
    simulation_result: Optional[SimulationResult] = None
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Generate a neighbor using hybrid approach: mix of random and problem-aware moves.
    
    Args:
        instance: Problem instance
        u: Current shipment plan, shape (T, N, J)
        move_probs: Probabilities for each move type
            Keys: 'time_shift', 'fdc_swap', 'magnitude_tweak'
            If None, uses equal probabilities
        delta_choices: Possible delta values for moves
            If None, computes adaptive deltas based on problem scale
        problem_aware_prob: Probability of using problem-aware move (default: 0.8)
            Remaining probability uses random move
        simulation_result: Optional simulation result for problem-aware moves
            If None, problem-aware moves use only instance data (cost, demand)
    
    Returns:
        Tuple of (new_u, move_key)
        move_key is (move_type, indices...) for tabu list
    """
    # Hybrid approach: problem_aware_prob chance of problem-aware, else random
    if random.random() < problem_aware_prob:
        return generate_problem_aware_neighbor(
            instance, u, simulation_result, move_probs, delta_choices
        )
    
    # Random move (original implementation)
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    if move_probs is None:
        move_probs = {
            'time_shift': 0.25,
            'fdc_swap': 0.25,
            'magnitude_tweak': 0.30,
            'product_rebalance': 0.10,
            'block_replan': 0.10,
        }
    
    # Compute adaptive deltas if not provided
    if delta_choices is None:
        # Calculate average demand to scale deltas appropriately
        avg_demand = np.mean(instance.demand_fdc[instance.demand_fdc > 0])
        if avg_demand > 0:
            # Use 10%, 20%, 30% of average demand as deltas
            # Apply minimum thresholds to avoid tiny values
            delta_choices = (
                max(5.0, avg_demand * 0.10),
                max(10.0, avg_demand * 0.20),
                max(20.0, avg_demand * 0.30)
            )
        else:
            # Fallback to fixed values if no demand
            delta_choices = (5.0, 10.0, 20.0)
    
    # Select move type
    r = random.random()
    if r < move_probs.get('magnitude_tweak', 0.30):
        # Magnitude tweak (constraint-aware)
        t = random.randrange(T)
        i = random.randrange(N)
        j = random.randrange(J)
        delta = random.choice(delta_choices)
        if random.random() < 0.5:
            delta = -delta
        
        new_u = magnitude_tweak_feasible(u, instance, i, j, t, delta, simulation_result)
        move_key = ('magnitude_tweak', t, i, j)
    
    elif r < move_probs.get('magnitude_tweak', 0.30) + move_probs.get('fdc_swap', 0.25):
        # FDC swap (constraint-aware)
        t = random.randrange(T)
        i = random.randrange(N)
        j_from = random.randrange(J)
        j_to = random.randrange(J)
        while j_to == j_from:
            j_to = random.randrange(J)
        delta = random.choice(delta_choices)
        
        new_u = fdc_swap_move_feasible(u, instance, i, j_from, j_to, t, delta, simulation_result)
        move_key = ('fdc_swap', t, i, j_from, j_to)
    
    elif r < (
        move_probs.get('magnitude_tweak', 0.30)
        + move_probs.get('fdc_swap', 0.25)
        + move_probs.get('time_shift', 0.25)
    ):
        # Time shift (constraint-aware)
        t_from = random.randrange(T)
        t_to = random.randrange(T)
        while t_to == t_from:
            t_to = random.randrange(T)
        i = random.randrange(N)
        j = random.randrange(J)
        delta = random.choice(delta_choices)
        
        new_u = time_shift_move_feasible(u, instance, i, j, t_from, t_to, delta, simulation_result)
        move_key = ('time_shift', i, j, t_from, t_to)
    elif r < (
        move_probs.get('magnitude_tweak', 0.30)
        + move_probs.get('fdc_swap', 0.25)
        + move_probs.get('time_shift', 0.25)
        + move_probs.get('product_rebalance', 0.10)
    ):
        # Product rebalance (large move)
        i = random.randrange(N)
        new_u = product_rebalance_move(u, instance, i)
        move_key = ('product_rebalance', i)
    else:
        # Block replan (large move)
        i = random.randrange(N)
        # Window length between 2 and 4 periods
        window = random.randint(2, min(4, T))
        t_start = random.randrange(0, max(1, T - window + 1))
        t_end = t_start + window - 1
        new_u = block_replan_move(u, instance, i, t_start, t_end)
        move_key = ('block_replan', i, t_start, t_end)
    
    return new_u, move_key


def _select_weighted_index(weights: np.ndarray) -> int:
    """
    Select an index with probability proportional to weights.
    
    Args:
        weights: Array of weights (will be normalized)
    
    Returns:
        Selected index
    """
    if np.sum(weights) <= 0:
        # Fallback to uniform if all weights are zero
        return random.randrange(len(weights))
    
    # Normalize weights
    probs = weights / np.sum(weights)
    
    # Sample
    return np.random.choice(len(weights), p=probs)


def compute_capacity_slack(
    instance: Instance,
    cached_result: Optional[SimulationResult]
) -> np.ndarray:
    """
    Compute capacity slack weights, shape (N, J).
    
    Higher slack = more likely to be selected for moves.
    Combines RDC inventory availability, FDC capacity slack, and outbound capacity.
    
    Args:
        instance: Problem instance
        cached_result: Optional simulation result
    
    Returns:
        Capacity slack weights, shape (N, J)
    """
    N = instance.num_products
    J = instance.num_fdcs
    T = instance.T
    
    if cached_result is None:
        # No simulation result: return uniform weights
        return np.ones((N, J), dtype=float)
    
    # Average capacity slack across all periods
    slack_weights = np.zeros((N, J), dtype=float)
    
    for t in range(T):
        rdc_avail, fdc_slack, outbound_remaining = compute_available_capacity(instance, cached_result, t)
        
        # For each (i, j), compute available capacity
        for i in range(N):
            for j in range(J):
                # Available is min of RDC inventory, FDC capacity, outbound capacity
                avail = min(rdc_avail[i], fdc_slack[j], outbound_remaining)
                slack_weights[i, j] += max(0.0, avail)
    
    # Average across periods
    slack_weights /= T
    
    # Normalize to make all positive
    slack_weights = slack_weights - np.min(slack_weights) + 1.0
    
    return slack_weights


def generate_problem_aware_neighbor(
    instance: Instance,
    u: np.ndarray,
    simulation_result: Optional[SimulationResult] = None,
    move_probs: Optional[Dict[str, float]] = None,
    delta_choices: Optional[Tuple[float, ...]] = None
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Generate a problem-aware neighbor by targeting high-cost or high-demand areas.
    
    Uses:
    - High transfer cost routes (transfer_cost[i, j])
    - High demand areas (demand_fdc[t, i, j])
    - Poor service areas (high lost sales or cross-fulfillment from simulation)
    
    Args:
        instance: Problem instance
        u: Current shipment plan, shape (T, N, J)
        simulation_result: Optional simulation result to identify poor service areas
        move_probs: Probabilities for each move type
        delta_choices: Possible delta values for moves
    
    Returns:
        Tuple of (new_u, move_key)
    """
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    if move_probs is None:
        move_probs = {
            'time_shift': 0.25,
            'fdc_swap': 0.25,
            'magnitude_tweak': 0.30,
            'product_rebalance': 0.10,
            'block_replan': 0.10,
        }
    
    # Compute adaptive deltas if not provided
    if delta_choices is None:
        avg_demand = np.mean(instance.demand_fdc[instance.demand_fdc > 0])
        if avg_demand > 0:
            delta_choices = (
                max(5.0, avg_demand * 0.10),
                max(10.0, avg_demand * 0.20),
                max(20.0, avg_demand * 0.30)
            )
        else:
            delta_choices = (5.0, 10.0, 20.0)
    
    # Build problem-aware selection weights
    # 1. Transfer cost weights (higher cost = more likely to be selected)
    transfer_weights = instance.transfer_cost.copy()  # shape (N, J)
    transfer_weights = transfer_weights - np.min(transfer_weights) + 1.0  # Make all positive
    
    # 2. Demand weights (higher demand = more likely to be selected)
    # Average demand across all periods for each (i, j)
    demand_weights = np.mean(instance.demand_fdc, axis=0)  # shape (N, J)
    demand_weights = demand_weights - np.min(demand_weights) + 1.0
    
    # 3. Service quality weights (if simulation result available)
    service_weights = np.ones((N, J))
    if simulation_result is not None:
        # Target areas with high lost sales or cross-fulfillment
        lost_sales = np.sum(simulation_result.lost_fdc, axis=0)  # shape (N, J)
        cross_fulfill = np.sum(simulation_result.fdc_from_rdc_fulfilled, axis=0)  # shape (N, J)
        # Combine: high lost sales or cross-fulfillment = high weight
        service_weights = lost_sales + cross_fulfill * 0.5
        service_weights = service_weights - np.min(service_weights) + 1.0
    
    # 4. Capacity slack weights (constraint-aware)
    capacity_slack_weights = compute_capacity_slack(instance, simulation_result)
    
    # Combine weights (weighted average)
    # Prefer high-demand areas WITH available capacity
    combined_weights = (0.3 * transfer_weights + 0.3 * demand_weights + 
                        0.2 * service_weights + 0.2 * capacity_slack_weights)
    
    # Select move type
    r = random.random()
    if r < move_probs.get('magnitude_tweak', 0.30):
        # Magnitude tweak - select (i, j) based on weights, random period (constraint-aware)
        t = random.randrange(T)
        # Flatten weights for selection
        flat_weights = combined_weights.flatten()
        flat_idx = _select_weighted_index(flat_weights)
        i = flat_idx // J
        j = flat_idx % J
        
        delta = random.choice(delta_choices)
        # Bias toward increasing shipments in high-demand/high-cost areas
        if random.random() < 0.7:  # 70% chance to increase
            delta = abs(delta)
        else:
            delta = -abs(delta)
        
        new_u = magnitude_tweak_feasible(u, instance, i, j, t, delta, simulation_result)
        move_key = ('magnitude_tweak', t, i, j)
    
    elif r < move_probs.get('magnitude_tweak', 0.30) + move_probs.get('fdc_swap', 0.25):
        # FDC swap - select (i, t) based on weights, swap to lower-cost FDC (constraint-aware)
        t = random.randrange(T)
        # Select product based on average weights across FDCs
        product_weights = np.mean(combined_weights, axis=1)  # shape (N,)
        i = _select_weighted_index(product_weights)
        
        # Select j_from (high cost/demand) and j_to (lower cost, prefer high demand + capacity)
        j_from_weights = combined_weights[i, :]
        j_from = _select_weighted_index(j_from_weights)
        
        # For j_to, prefer FDCs with lower transfer cost, high demand, AND available capacity
        cost_weights = 1.0 / (instance.transfer_cost[i, :] + 1.0)  # Inverse cost
        demand_weights_j = instance.demand_fdc[:, i, :].mean(axis=0)  # Average demand for this product
        # Include capacity slack in selection
        if simulation_result is not None:
            rdc_avail, fdc_slack, outbound_remaining = compute_available_capacity(instance, simulation_result, t)
            capacity_weights_j = np.minimum(rdc_avail[i], fdc_slack)  # Available at each FDC
            capacity_weights_j = capacity_weights_j / (np.max(capacity_weights_j) + 1e-9)  # Normalize
        else:
            capacity_weights_j = np.ones(J)
        j_to_weights = cost_weights * (demand_weights_j + 1.0) * (capacity_weights_j + 1.0)
        j_to = _select_weighted_index(j_to_weights)
        
        while j_to == j_from:
            j_to = _select_weighted_index(j_to_weights)
        
        delta = random.choice(delta_choices)
        new_u = fdc_swap_move_feasible(u, instance, i, j_from, j_to, t, delta, simulation_result)
        move_key = ('fdc_swap', t, i, j_from, j_to)
    
    elif r < (
        move_probs.get('magnitude_tweak', 0.30)
        + move_probs.get('fdc_swap', 0.25)
        + move_probs.get('time_shift', 0.25)
    ):
        # Time shift - select (i, j) based on weights, shift to earlier period if high demand
        # Select (i, j) based on weights
        flat_weights = combined_weights.flatten()
        flat_idx = _select_weighted_index(flat_weights)
        i = flat_idx // J
        j = flat_idx % J
        
        # For time shift, prefer shifting to earlier periods if demand is high later
        # Select t_from (later period) and t_to (earlier period)
        demand_by_period = instance.demand_fdc[:, i, j]  # shape (T,)
        if np.sum(demand_by_period) > 0:
            # Prefer t_from from later periods, t_to from earlier periods
            period_weights_later = np.arange(T, 0, -1, dtype=float)  # Later periods have higher weight
            period_weights_earlier = np.arange(1, T + 1, dtype=float)  # Earlier periods have higher weight
            
            t_from = _select_weighted_index(period_weights_later)
            t_to = _select_weighted_index(period_weights_earlier)
        else:
            t_from = random.randrange(T)
            t_to = random.randrange(T)
        
        while t_to == t_from:
            if t_from < T - 1:
                t_to = random.randrange(t_from + 1, T)
            else:
                t_to = random.randrange(T - 1)
        
        delta = random.choice(delta_choices)
        new_u = time_shift_move_feasible(u, instance, i, j, t_from, t_to, delta, simulation_result)
        move_key = ('time_shift', i, j, t_from, t_to)
    elif r < (
        move_probs.get('magnitude_tweak', 0.30)
        + move_probs.get('fdc_swap', 0.25)
        + move_probs.get('time_shift', 0.25)
        + move_probs.get('product_rebalance', 0.10)
    ):
        # Product rebalance focused on high-weight product
        product_weights = np.mean(combined_weights, axis=1)  # shape (N,)
        i = _select_weighted_index(product_weights)
        new_u = product_rebalance_move(u, instance, i)
        move_key = ('product_rebalance', i)
    else:
        # Block replan focused on high-weight product
        product_weights = np.mean(combined_weights, axis=1)  # shape (N,)
        i = _select_weighted_index(product_weights)
        window = max(2, min(4, T))
        t_start = _select_weighted_index(np.arange(1, T + 1, dtype=float))
        t_start = min(t_start, T - 1)
        t_end = min(T - 1, t_start + window - 1)
        new_u = block_replan_move(u, instance, i, t_start, t_end)
        move_key = ('block_replan', i, t_start, t_end)
    
    return new_u, move_key

