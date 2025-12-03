"""Solution representation helpers for heuristics."""

import numpy as np
from typing import Tuple, Dict, Any

from ..model.instance import Instance
from ..model.simulator import simulate


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

