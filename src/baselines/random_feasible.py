"""Random feasible baseline algorithm."""

import numpy as np
import random
from typing import Tuple

from ..model.instance import Instance
from ..model.simulator import simulate


def random_feasible(instance: Instance, seed: int = 42) -> Tuple[float, np.ndarray]:
    """
    Random feasible baseline: generate random feasible shipment plans.
    
    For each period t:
    - Randomly draw proportions using Dirichlet-like distribution
    - Scale to respect inventory and capacity constraints
    - Ensure feasibility
    
    Args:
        instance: Problem instance
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (total_cost, shipment_plan)
        shipment_plan shape (T, N, J)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    shipments = np.zeros((T, N, J), dtype=float)
    
    # Track inventories
    inventory_rdc = instance.initial_inventory_rdc.copy()
    
    for t in range(T):
        # Add replenishment
        inventory_rdc += instance.replenishment[t, :]
        
        # Remaining outbound capacity
        remaining_outbound = instance.outbound_capacity[t]
        
        # Track FDC total inventory for capacity
        if t == 0:
            fdc_total = np.sum(instance.initial_inventory_fdc, axis=0)
        else:
            fdc_total = np.zeros(J)
            for j in range(J):
                fdc_total[j] = np.sum(instance.initial_inventory_fdc[:, j]) + \
                              np.sum(shipments[:t, :, j])
        
        # For each product, allocate randomly
        for i in range(N):
            avail_rdc = inventory_rdc[i]
            
            if avail_rdc <= 0 or remaining_outbound <= 0:
                continue
            
            # Random proportions for this product across FDCs
            # Use Dirichlet distribution (alpha=1 gives uniform)
            proportions = np.random.dirichlet(np.ones(J))
            
            # Amount to allocate (random fraction of available)
            fraction = random.uniform(0.1, 0.9)  # Use 10-90% of available
            total_to_allocate = min(
                fraction * avail_rdc,
                remaining_outbound
            )
            
            # Allocate to each FDC
            for j in range(J):
                if proportions[j] <= 0:
                    continue
                
                # Requested amount
                request = proportions[j] * total_to_allocate
                
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
    
    # Evaluate the plan
    result = simulate(instance, shipments, check_feasibility=False)
    return result.total_cost, shipments

