"""Solution representation helpers for heuristics."""

import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..model.instance import Instance
from ..model.simulator import simulate
from ..model.result import SimulationResult


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
                
                # FDC capacity
                if instance.fdc_capacity is not None:
                    cap_fdc = instance.fdc_capacity[j] - fdc_total[j]
                else:
                    cap_fdc = np.inf
                
                feasible = min(request, inventory_rdc[i], cap_fdc, remaining_outbound)
                u[t, i, j] = feasible
                inventory_rdc[i] -= feasible
                fdc_total[j] += feasible
                remaining_outbound -= feasible
                
                if remaining_outbound <= 0:
                    break
    
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
    
    total_clipped = cached_result.clipped_shipments
    
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
                    total_clipped += (request - feasible_ship)
                
                actual_shipments[t, i, j] = feasible_ship
                inventory_rdc[t, i] -= feasible_ship
                fdc_total_inventory[j] += feasible_ship
                remaining_outbound_capacity -= feasible_ship
        
        # 3) FDC local fulfillment
        for i in range(N):
            for j in range(J):
                avail_fdc = inventory_fdc[t, i, j] + actual_shipments[t, i, j]
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
    
    total_cost = new_cost_transfer + new_cost_cross + new_cost_lost
    
    result = SimulationResult(
        total_cost=total_cost,
        cost_transfer=new_cost_transfer,
        cost_cross=new_cost_cross,
        cost_lost=new_cost_lost,
        inventory_rdc=inventory_rdc,
        inventory_fdc=inventory_fdc,
        shipments=actual_shipments,
        fdc_local_fulfilled=fdc_local_fulfilled,
        fdc_from_rdc_fulfilled=fdc_from_rdc_fulfilled,
        lost_fdc=lost_fdc,
        rdc_fulfilled=rdc_fulfilled,
        lost_rdc=lost_rdc,
        clipped_shipments=total_clipped
    )
    
    metrics = {
        'cost_transfer': new_cost_transfer,
        'cost_cross': new_cost_cross,
        'cost_lost': new_cost_lost,
        'clipped_shipments': total_clipped
    }
    
    return total_cost, metrics, result

