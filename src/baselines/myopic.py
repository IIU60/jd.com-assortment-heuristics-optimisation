"""Myopic greedy baseline algorithm."""

import numpy as np
from typing import Tuple

from ..model.instance import Instance
from ..model.simulator import simulate


def myopic_greedy(instance: Instance) -> Tuple[float, np.ndarray]:
    """
    Myopic greedy baseline: allocate shipments based on future period demand.
    
    For each period t:
    - Look ahead by lead_time periods: look at demand D[t + lead_time, i, j]
    - Compute required shipments to cover D[t + lead_time, i, j] - current_fdc_inventory
    - If RDC inventory insufficient, allocate proportional to demand
    - Respect outbound_capacity[t] and fdc_capacity[j]
    - In last lead_time periods, falls back to current period demand
    
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
        
        # Add shipments that arrive in this period (ordered at t - lead_time)
        if t >= instance.lead_time:
            arrivals = shipments[t - instance.lead_time, :, :]
            inventory_fdc += arrivals
        
        # Remaining outbound capacity
        remaining_outbound = instance.outbound_capacity[t]
        
        # Track FDC total inventory for capacity (including arrivals)
        fdc_total = np.sum(inventory_fdc, axis=0)  # shape (J,)
        
        # For each product, allocate to FDCs based on demand
        for i in range(N):
            # Look ahead by lead_time periods
            target_period = t + instance.lead_time
            if target_period < instance.T:
                # Compute expected inventory at target period
                # This includes current inventory plus shipments in transit that will arrive by then
                expected_inventory = inventory_fdc[i, :].copy()
                # Add shipments that are in transit (ordered but not yet arrived)
                # Shipments ordered at time tau arrive at tau + lead_time
                # We only count shipments ordered at times where tau + lead_time > t (not yet arrived)
                # and tau + lead_time <= target_period (will arrive by target)
                for tau in range(max(0, t - instance.lead_time + 1), t):
                    arrival_time = tau + instance.lead_time
                    if arrival_time > t and arrival_time <= target_period:
                        expected_inventory += shipments[tau, i, :]
                
                shortfall = np.maximum(0, instance.demand_fdc[target_period, i, :] - expected_inventory)
            else:
                # Last lead_time periods: can't see full future, use current period as fallback
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
                # Use current capacity as conservative estimate
                # (demand will consume inventory in the meantime, making this safe)
                if instance.fdc_capacity is not None:
                    cap_left_fdc = instance.fdc_capacity[j] - fdc_total[j]
                else:
                    cap_left_fdc = np.inf
                
                # Clip to feasible
                feasible = min(request, avail_rdc, cap_left_fdc, remaining_outbound)
                
                shipments[t, i, j] = feasible
                inventory_rdc[i] -= feasible
                # Note: Don't add to fdc_total yet - shipment arrives at t + lead_time
                remaining_outbound -= feasible
                
                if remaining_outbound <= 0:
                    break
            
            if remaining_outbound <= 0:
                break
        
        # Fulfill demand (simplified - just track what's used)
        for i in range(N):
            for j in range(J):
                fulfilled = min(inventory_fdc[i, j], instance.demand_fdc[t, i, j])
                inventory_fdc[i, j] -= fulfilled
    
    # Evaluate the plan
    result = simulate(instance, shipments, check_feasibility=False)
    return result.total_cost, shipments

