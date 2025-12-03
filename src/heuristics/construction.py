"""Construction heuristics: greedy and GRASP."""

import numpy as np
import random
from typing import Optional

from ..model.instance import Instance
from .utils import clamp_u_to_feasibility


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

