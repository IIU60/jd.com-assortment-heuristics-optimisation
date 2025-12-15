"""Solution representation helpers for heuristics."""

import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..model.instance import Instance
from ..model.simulator import simulate
from ..model.result import SimulationResult


def compute_available_capacity(
    instance: Instance,
    cached_result: SimulationResult,
    t: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute available capacities at period t.
    
    Args:
        instance: Problem instance
        cached_result: Cached simulation result with current state
        t: Period index
    
    Returns:
        Tuple of (rdc_inventory_avail, fdc_capacity_slack, outbound_capacity_remaining)
        - rdc_inventory_avail: shape (N,), available RDC inventory per product
        - fdc_capacity_slack: shape (J,), available FDC capacity per FDC
        - outbound_capacity_remaining: scalar, remaining outbound capacity
    """
    # RDC inventory available (after replenishment, before shipments)
    rdc_inventory_avail = cached_result.inventory_rdc[t, :].copy()
    rdc_inventory_avail += instance.replenishment[t, :]
    
    # FDC capacity slack
    fdc_total_inventory = np.sum(cached_result.inventory_fdc[t, :, :], axis=0)  # shape (J,)
    if instance.fdc_capacity is not None:
        fdc_capacity_slack = instance.fdc_capacity - fdc_total_inventory
        fdc_capacity_slack = np.maximum(0.0, fdc_capacity_slack)  # Non-negative
    else:
        fdc_capacity_slack = np.full(instance.num_fdcs, np.inf)
    
    # Outbound capacity (full capacity for period t)
    outbound_capacity_remaining = instance.outbound_capacity[t]
    
    return rdc_inventory_avail, fdc_capacity_slack, outbound_capacity_remaining


def get_feasible_delta(
    instance: Instance,
    u: np.ndarray,
    cached_result: Optional[SimulationResult],
    i: int,
    j: int,
    t: int,
    desired_delta: float
) -> float:
    """
    Compute maximum feasible delta for a move at (t, i, j).
    
    Checks RDC inventory, FDC capacity, and outbound capacity constraints.
    
    Args:
        instance: Problem instance
        u: Current shipment plan, shape (T, N, J)
        cached_result: Optional cached simulation result (if None, uses approximate checks)
        i: Product index
        j: FDC index
        t: Period index
        desired_delta: Desired change amount (can be positive or negative)
    
    Returns:
        Maximum feasible delta that can be applied without clipping.
        If desired_delta is negative (reducing), returns desired_delta (can always reduce).
        If desired_delta is positive (increasing), returns min(desired_delta, available_capacity).
    """
    if desired_delta < 0:
        # Can always reduce (up to current value)
        return max(desired_delta, -u[t, i, j])
    
    # For increases, check available capacity
    if cached_result is not None:
        rdc_avail, fdc_slack, outbound_remaining = compute_available_capacity(instance, cached_result, t)
        
        # Available RDC inventory for product i
        avail_rdc = rdc_avail[i]
        
        # Available FDC capacity at j
        avail_fdc = fdc_slack[j]
        
        # Available outbound capacity
        avail_outbound = outbound_remaining
        
        # Maximum feasible increase
        max_delta = min(avail_rdc, avail_fdc, avail_outbound)
        
        return min(desired_delta, max_delta)
    else:
        # Approximate: use outbound capacity as conservative estimate
        return min(desired_delta, instance.outbound_capacity[t] * 0.1)  # Conservative


def is_move_feasible(
    instance: Instance,
    cached_result: SimulationResult,
    i: int,
    j: int,
    t: int,
    delta: float
) -> Tuple[bool, float]:
    """
    Check if a move is feasible and return maximum feasible delta.
    
    Args:
        instance: Problem instance
        cached_result: Cached simulation result
        i: Product index
        j: FDC index
        t: Period index
        delta: Desired delta
    
    Returns:
        Tuple of (is_feasible, max_feasible_delta)
    """
    max_feasible = get_feasible_delta(instance, None, cached_result, i, j, t, delta)
    is_feasible = abs(max_feasible - delta) < 1e-9
    return is_feasible, max_feasible


def copy_u(u: np.ndarray) -> np.ndarray:
    """
    Deep copy of shipment plan.
    
    Args:
        u: Shipment plan, shape (T, N, J)
    
    Returns:
        Copy of u
    """
    return np.copy(u)


def random_feasible_u(instance: Instance, seed: int = None) -> np.ndarray:
    """
    Generate a random feasible shipment plan.
    
    Uses a simple random allocation that respects constraints.
    This is a basic implementation - may not be fully optimal.
    
    Args:
        instance: Problem instance
        seed: Random seed (optional)
    
    Returns:
        Random feasible shipment plan, shape (T, N, J)
    """
    if seed is not None:
        np.random.seed(seed)
    
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    u = np.zeros((T, N, J), dtype=float)
    
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
                              np.sum(u[:t, :, j])
        
        # Random allocation for each product
        for i in range(N):
            if inventory_rdc[i] <= 0 or remaining_outbound <= 0:
                continue
            
            # Random amount to allocate (10-50% of available)
            fraction = np.random.uniform(0.1, 0.5)
            total_allocate = min(fraction * inventory_rdc[i], remaining_outbound)
            
            # Random proportions across FDCs
            proportions = np.random.dirichlet(np.ones(J))
            
            for j in range(J):
                if proportions[j] <= 0:
                    continue
                
                request = proportions[j] * total_allocate
                
                # FDC capacity (ensure non-negative slack)
                if instance.fdc_capacity is not None:
                    cap_fdc = max(0.0, instance.fdc_capacity[j] - fdc_total[j])
                else:
                    cap_fdc = np.inf
                
                # Clip to feasible and enforce non-negativity
                feasible = max(0.0, min(request, inventory_rdc[i], cap_fdc, remaining_outbound))
                u[t, i, j] = feasible
                inventory_rdc[i] -= feasible
                fdc_total[j] += feasible
                remaining_outbound -= feasible
                
                if remaining_outbound <= 0:
                    break
    
    # Defensive check (can be relaxed later if desired)
    assert np.all(u >= 0), "random_feasible_u generated negative shipments"
    return u


def clamp_u_to_feasibility(instance: Instance, u: np.ndarray) -> np.ndarray:
    """
    Clamp shipment plan to ensure feasibility.
    
    Enforces:
    - Non-negativity
    - Outbound capacity constraints
    - FDC capacity constraints (approximate)
    - RDC inventory availability (approximate)
    
    Args:
        instance: Problem instance
        u: Shipment plan, shape (T, N, J)
    
    Returns:
        Clamped shipment plan
    """
    u_clamped = np.maximum(0, u.copy())  # Non-negativity
    
    # Clamp outbound capacity
    for t in range(instance.T):
        total = np.sum(u_clamped[t, :, :])
        if total > instance.outbound_capacity[t]:
            scale = instance.outbound_capacity[t] / total
            u_clamped[t, :, :] *= scale
    
    # Note: FDC capacity and RDC inventory are harder to enforce without simulation
    # This is a simplified version - full feasibility is checked in simulator
    
    return u_clamped


def evaluate_u(instance: Instance, u: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a shipment plan and return cost and metrics.
    
    Args:
        instance: Problem instance
        u: Shipment plan, shape (T, N, J)
    
    Returns:
        Tuple of (total_cost, metrics_dict)
        metrics_dict contains: cost_transfer, cost_cross, cost_lost, clipped_shipments
    """
    result = simulate(instance, u, check_feasibility=False)
    
    metrics = {
        'cost_transfer': result.cost_transfer,
        'cost_cross': result.cost_cross,
        'cost_lost': result.cost_lost,
        'cost_clipped': result.cost_clipped,
        'clipped_shipments': result.clipped_shipments
    }
    
    return result.total_cost, metrics


def evaluate_u_incremental(
    instance: Instance,
    u_new: np.ndarray,
    u_old: np.ndarray,
    cached_result: Optional[SimulationResult] = None
) -> Tuple[float, Dict[str, Any], SimulationResult]:
    """
    Incrementally evaluate a neighbor solution by only re-simulating from the first changed period.
    
    This is much faster than full simulation when only small changes are made.
    
    Args:
        instance: Problem instance
        u_new: New shipment plan to evaluate, shape (T, N, J)
        u_old: Previous shipment plan, shape (T, N, J)
        cached_result: Cached simulation result from u_old (if None, does full simulation)
    
    Returns:
        Tuple of (total_cost, metrics_dict, new_result)
        metrics_dict contains: cost_transfer, cost_cross, cost_lost, clipped_shipments
        new_result is the full SimulationResult for caching
    """
    # If no cache, do full simulation
    if cached_result is None:
        result = simulate(instance, u_new, check_feasibility=False)
        metrics = {
            'cost_transfer': result.cost_transfer,
            'cost_cross': result.cost_cross,
            'cost_lost': result.cost_lost,
            'cost_clipped': result.cost_clipped,
            'clipped_shipments': result.clipped_shipments
        }
        return result.total_cost, metrics, result
    
    # Find the earliest period where shipments differ
    diff = np.abs(u_new - u_old)
    periods_changed = np.any(diff > 1e-9, axis=(1, 2))  # shape (T,)
    first_changed_period = np.argmax(periods_changed) if np.any(periods_changed) else instance.T
    
    # If nothing changed, return cached result
    if first_changed_period >= instance.T:
        metrics = {
            'cost_transfer': cached_result.cost_transfer,
            'cost_cross': cached_result.cost_cross,
            'cost_lost': cached_result.cost_lost,
            'cost_clipped': cached_result.cost_clipped,
            'clipped_shipments': cached_result.clipped_shipments
        }
        return cached_result.total_cost, metrics, cached_result
    
    # Re-simulate from first_changed_period onwards
    # We need to start from the inventory state at the beginning of first_changed_period
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    # Initialize from cached state
    inventory_rdc = cached_result.inventory_rdc.copy()
    inventory_fdc = cached_result.inventory_fdc.copy()
    
    # Initialize flow arrays (copy old, will overwrite from first_changed_period)
    actual_shipments = cached_result.shipments.copy()
    fdc_local_fulfilled = cached_result.fdc_local_fulfilled.copy()
    fdc_from_rdc_fulfilled = cached_result.fdc_from_rdc_fulfilled.copy()
    lost_fdc = cached_result.lost_fdc.copy()
    rdc_fulfilled = cached_result.rdc_fulfilled.copy()
    lost_rdc = cached_result.lost_rdc.copy()
    
    # Subtract old clipping for periods being re-simulated
    old_clipped_in_periods = np.sum(cached_result.clipped_per_period[first_changed_period:])
    total_clipped = cached_result.clipped_shipments - old_clipped_in_periods
    
    # Compute old cost_clipped contribution for re-simulated periods
    # This is the cost of clipped shipments in the old solution for these periods
    old_cost_clipped_contribution = 0.0
    for t_old in range(first_changed_period, T):
        # Compare old requests (u_old) with old actual shipments to find clipped amounts
        old_requests = u_old[t_old, :, :]
        old_actual = cached_result.shipments[t_old, :, :]
        clipped_mask = old_requests > old_actual + 1e-9  # Account for floating point
        if np.any(clipped_mask):
            clipped_amounts = old_requests - old_actual
            clipped_amounts[~clipped_mask] = 0.0
            # Cost is transfer_cost * clipped_amount for each (i,j)
            old_cost_clipped_contribution += np.sum(instance.transfer_cost * clipped_amounts)
    
    # Start with full old cost_clipped, subtract contribution from re-simulated periods
    cost_clipped = cached_result.cost_clipped - old_cost_clipped_contribution
    
    # Track per-period clipping for re-simulated periods (will be reset and recomputed)
    clipped_per_period = cached_result.clipped_per_period.copy()
    
    # Reset clipping for re-simulated periods
    clipped_per_period[first_changed_period:] = 0.0
    
    # Re-simulate from first_changed_period to T
    for t in range(first_changed_period, T):
        # 1) Replenishment at RDC (arrives at start of period t)
        inventory_rdc[t, :] += instance.replenishment[t, :]
        
        # 2) Shipments from RDC to FDCs
        remaining_outbound_capacity = instance.outbound_capacity[t]
        fdc_total_inventory = np.sum(inventory_fdc[t, :, :], axis=0)  # shape (J,)
        
        for i in range(N):
            for j in range(J):
                request = u_new[t, i, j]
                if request <= 0:
                    continue
                
                avail_rdc = inventory_rdc[t, i]
                
                if instance.fdc_capacity is not None:
                    cap_left_fdc = instance.fdc_capacity[j] - fdc_total_inventory[j]
                else:
                    cap_left_fdc = np.inf
                
                cap_left_outbound = remaining_outbound_capacity
                
                feasible_ship = min(request, avail_rdc, cap_left_fdc, cap_left_outbound)
                
                if feasible_ship < request:
                    clipped_amount = request - feasible_ship
                    total_clipped += clipped_amount
                    clipped_per_period[t] += clipped_amount  # Accumulate within period
                    # Cost penalty: proportional to transfer cost of clipped amount
                    cost_clipped += instance.transfer_cost[i, j] * clipped_amount
                
                actual_shipments[t, i, j] = feasible_ship
                inventory_rdc[t, i] -= feasible_ship
                fdc_total_inventory[j] += feasible_ship
                remaining_outbound_capacity -= feasible_ship
        
        # 3) FDC local fulfillment
        for i in range(N):
            for j in range(J):
                # Shipments ordered in period (t - lead_time) arrive in period t
                if t >= instance.lead_time:
                    arrivals = actual_shipments[t - instance.lead_time, i, j]
                else:
                    arrivals = 0.0
                avail_fdc = inventory_fdc[t, i, j] + arrivals
                demand = instance.demand_fdc[t, i, j]
                
                fulfilled_local = min(avail_fdc, demand)
                fdc_local_fulfilled[t, i, j] = fulfilled_local
                
                remaining_inventory = avail_fdc - fulfilled_local
                remaining_demand = demand - fulfilled_local
                
                inventory_fdc[t+1, i, j] = remaining_inventory
                
                if remaining_demand > 0:
                    avail_rdc = inventory_rdc[t, i]
                    cross_fulfilled = min(remaining_demand, avail_rdc)
                    fdc_from_rdc_fulfilled[t, i, j] = cross_fulfilled
                    inventory_rdc[t, i] -= cross_fulfilled
                    remaining_demand -= cross_fulfilled
                    
                    lost_fdc[t, i, j] = remaining_demand
                else:
                    fdc_from_rdc_fulfilled[t, i, j] = 0.0
                    lost_fdc[t, i, j] = 0.0
        
        # 6) RDC own demand fulfillment
        for i in range(N):
            demand_rdc = instance.demand_rdc[t, i]
            if demand_rdc <= 0:
                inventory_rdc[t+1, i] = inventory_rdc[t, i]
                continue
            
            avail_rdc = inventory_rdc[t, i]
            fulfilled = min(avail_rdc, demand_rdc)
            rdc_fulfilled[t, i] = fulfilled
            inventory_rdc[t, i] -= fulfilled
            lost_rdc[t, i] = demand_rdc - fulfilled
            inventory_rdc[t+1, i] = inventory_rdc[t, i]
    
    # Compute total costs (actual_shipments now contains old values for unchanged periods
    # and new values for changed periods, so we can sum over all periods)
    new_cost_transfer = np.sum(instance.transfer_cost * actual_shipments)
    new_cost_cross = instance.rdc_fulfillment_cost * np.sum(fdc_from_rdc_fulfilled)
    new_cost_lost = instance.lost_sale_cost * (
        np.sum(lost_fdc) + np.sum(lost_rdc)
    )
    
    # cost_clipped now has the correct value (old cost minus old contribution, plus new contribution)
    
    total_cost = new_cost_transfer + new_cost_cross + new_cost_lost + cost_clipped
    
    result = SimulationResult(
        total_cost=total_cost,
        cost_transfer=new_cost_transfer,
        cost_cross=new_cost_cross,
        cost_lost=new_cost_lost,
        cost_clipped=cost_clipped,
        inventory_rdc=inventory_rdc,
        inventory_fdc=inventory_fdc,
        shipments=actual_shipments,
        fdc_local_fulfilled=fdc_local_fulfilled,
        fdc_from_rdc_fulfilled=fdc_from_rdc_fulfilled,
        lost_fdc=lost_fdc,
        rdc_fulfilled=rdc_fulfilled,
        lost_rdc=lost_rdc,
        clipped_shipments=total_clipped,
        clipped_per_period=clipped_per_period
    )
    
    metrics = {
        'cost_transfer': new_cost_transfer,
        'cost_cross': new_cost_cross,
        'cost_lost': new_cost_lost,
        'cost_clipped': cost_clipped,
        'clipped_shipments': total_clipped
    }
    
    return total_cost, metrics, result

