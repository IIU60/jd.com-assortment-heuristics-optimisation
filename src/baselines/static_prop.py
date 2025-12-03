"""Static proportional baseline algorithm."""

import numpy as np
from typing import Tuple

from ..model.instance import Instance
from ..model.simulator import simulate


def static_proportional(instance: Instance) -> Tuple[float, np.ndarray]:
    """
    Static proportional baseline: pre-compute fixed proportions from total demand.
    
    Before horizon:
    - Compute total_D_ij = sum over t of D[t, i, j] for each (i, j)
    - For each product i, compute fixed proportions p_ij = total_D_ij / total_D_i
    
    For each period t:
    - For each product i, allocate RDC inventory to FDCs according to p_ij
    - Capped by outbound_capacity[t] and fdc_capacity[j]
    
    Args:
        instance: Problem instance
    
    Returns:
        Tuple of (total_cost, shipment_plan)
        shipment_plan shape (T, N, J)
    """
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    # Pre-compute proportions from total demand
    total_demand_fdc = np.sum(instance.demand_fdc, axis=0)  # shape (N, J)
    total_demand_per_product = np.sum(total_demand_fdc, axis=1)  # shape (N,)
    
    # Proportions: p[i, j] = total demand at FDC j for product i / total demand for product i
    proportions = np.zeros((N, J), dtype=float)
    for i in range(N):
        if total_demand_per_product[i] > 0:
            proportions[i, :] = total_demand_fdc[i, :] / total_demand_per_product[i]
        else:
            proportions[i, :] = 1.0 / J  # Equal if no demand
    
    shipments = np.zeros((T, N, J), dtype=float)
    
    # Track inventories
    inventory_rdc = instance.initial_inventory_rdc.copy()
    
    for t in range(T):
        # Add replenishment
        inventory_rdc += instance.replenishment[t, :]
        
        # Remaining outbound capacity
        remaining_outbound = instance.outbound_capacity[t]
        
        # Track FDC total inventory for capacity
        # We need to simulate what inventory will be after shipments
        # For simplicity, use current period's starting inventory
        if t == 0:
            fdc_total = np.sum(instance.initial_inventory_fdc, axis=0)
        else:
            # Approximate: use previous period's ending inventory
            # This is simplified - in reality we'd need to track it properly
            fdc_total = np.zeros(J)
            for j in range(J):
                # Rough estimate: sum of all products' initial + shipments so far
                fdc_total[j] = np.sum(instance.initial_inventory_fdc[:, j]) + \
                              np.sum(shipments[:t, :, j])
        
        # Allocate for each product
        for i in range(N):
            avail_rdc = inventory_rdc[i]
            
            if avail_rdc <= 0:
                continue
            
            # Allocate proportionally
            for j in range(J):
                if proportions[i, j] <= 0:
                    continue
                
                # Requested amount based on proportion
                request = proportions[i, j] * min(avail_rdc, remaining_outbound)
                
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
    
    # Evaluate the plan
    result = simulate(instance, shipments, check_feasibility=False)
    return result.total_cost, shipments

