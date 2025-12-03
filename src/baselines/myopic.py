"""Myopic greedy baseline algorithm."""

import numpy as np
from typing import Tuple

from ..model.instance import Instance
from ..model.simulator import simulate


def myopic_greedy(instance: Instance) -> Tuple[float, np.ndarray]:
    """
    Myopic greedy baseline: allocate shipments based on current period demand.
    
    For each period t:
    - Look at today's FDC demand D[t, i, j]
    - Compute required shipments to cover D[t, i, j] - current_fdc_inventory
    - If RDC inventory insufficient, allocate proportional to demand
    - Respect outbound_capacity[t] and fdc_capacity[j]
    
    Args:
        instance: Problem instance
    
    Returns:
        Tuple of (total_cost, shipment_plan)
        shipment_plan shape (T, N, J)
    """
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    shipments = np.zeros((T, N, J), dtype=float)
    
    # Track inventories as we go
    inventory_rdc = instance.initial_inventory_rdc.copy()
    inventory_fdc = instance.initial_inventory_fdc.copy()
    
    for t in range(T):
        # Add replenishment
        inventory_rdc += instance.replenishment[t, :]
        
        # Remaining outbound capacity
        remaining_outbound = instance.outbound_capacity[t]
        
        # Track FDC total inventory for capacity
        fdc_total = np.sum(inventory_fdc, axis=0)  # shape (J,)
        
        # For each product, allocate to FDCs based on demand
        for i in range(N):
            # Compute demand shortfall at each FDC
            shortfall = np.maximum(0, instance.demand_fdc[t, i, :] - inventory_fdc[i, :])
            total_shortfall = np.sum(shortfall)
            
            if total_shortfall <= 0:
                continue
            
            # Available RDC inventory for this product
            avail_rdc = inventory_rdc[i]
            
            if avail_rdc <= 0:
                continue
            
            # Allocate proportionally to shortfall
            if total_shortfall > 0:
                proportions = shortfall / total_shortfall
            else:
                proportions = np.ones(J) / J
            
            # Allocate to each FDC
            for j in range(J):
                if shortfall[j] <= 0:
                    continue
                
                # Requested amount
                request = proportions[j] * min(avail_rdc, total_shortfall, remaining_outbound)
                
                # FDC capacity constraint
                if instance.fdc_capacity is not None:
                    cap_left_fdc = instance.fdc_capacity[j] - fdc_total[j]
                else:
                    cap_left_fdc = np.inf
                
                # Clip to feasible
                feasible = min(request, avail_rdc, cap_left_fdc, remaining_outbound)
                
                shipments[t, i, j] = feasible
                inventory_rdc[i] -= feasible
                fdc_total[j] += feasible
                remaining_outbound -= feasible
                
                if remaining_outbound <= 0:
                    break
            
            if remaining_outbound <= 0:
                break
        
        # Update FDC inventories after shipments (before demand)
        inventory_fdc += shipments[t, :, :]
        
        # Fulfill demand (simplified - just track what's used)
        for i in range(N):
            for j in range(J):
                fulfilled = min(inventory_fdc[i, j], instance.demand_fdc[t, i, j])
                inventory_fdc[i, j] -= fulfilled
    
    # Evaluate the plan
    result = simulate(instance, shipments, check_feasibility=False)
    return result.total_cost, shipments

