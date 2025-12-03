"""Core simulator for evaluating shipment plans."""

import numpy as np
from typing import Optional

from .instance import Instance
from .result import SimulationResult


def simulate(instance: Instance,
             shipments: np.ndarray,
             check_feasibility: bool = True) -> SimulationResult:
    """
    Simulate a shipment plan over the entire horizon.
    
    Given a shipment plan u[t, i, j] (units of product i shipped from RDC to FDC j
    at the start of period t), this function simulates inventory flows and computes
    total cost.
    
    Shipments ordered in period t arrive at FDCs in period t + lead_time.
    If lead_time = 1, shipments ordered in period t arrive in period t+1.
    
    The simulator enforces constraints by clipping shipments:
    - RDC inventory availability
    - FDC capacity limits
    - RDC outbound capacity
    
    Args:
        instance: Problem instance
        shipments: Shipment plan, shape (T, N, J)
            shipments[t, i, j] = planned units of product i to FDC j in period t
        check_feasibility: If True, assert that constraints are satisfied after clipping
    
    Returns:
        SimulationResult with costs, inventories, and flows
    """
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    # Validate input shape
    assert shipments.shape == (T, N, J), \
        f"shipments shape {shipments.shape} != ({T}, {N}, {J})"
    assert np.all(shipments >= 0), "shipments must be non-negative"
    
    # Initialize inventories (T+1 periods: 0 to T)
    inventory_rdc = np.zeros((T + 1, N), dtype=float)
    inventory_fdc = np.zeros((T + 1, N, J), dtype=float)
    
    # Set initial inventories
    inventory_rdc[0, :] = instance.initial_inventory_rdc.copy()
    inventory_fdc[0, :, :] = instance.initial_inventory_fdc.copy()
    
    # Initialize flow arrays
    actual_shipments = np.zeros((T, N, J), dtype=float)
    fdc_local_fulfilled = np.zeros((T, N, J), dtype=float)
    fdc_from_rdc_fulfilled = np.zeros((T, N, J), dtype=float)
    lost_fdc = np.zeros((T, N, J), dtype=float)
    rdc_fulfilled = np.zeros((T, N), dtype=float)
    lost_rdc = np.zeros((T, N), dtype=float)
    
    total_clipped = 0.0
    
    for t in range(T):
        # 1) Replenishment at RDC (arrives at start of period t)
        inventory_rdc[t, :] += instance.replenishment[t, :]
        
        # 2) Shipments from RDC to FDCs
        # Process shipments respecting constraints
        remaining_outbound_capacity = instance.outbound_capacity[t]
        
        # Track FDC total inventory for capacity checks
        fdc_total_inventory = np.sum(inventory_fdc[t, :, :], axis=0)  # shape (J,)
        
        for i in range(N):
            for j in range(J):
                request = shipments[t, i, j]
                if request <= 0:
                    continue
                
                # Available RDC inventory
                avail_rdc = inventory_rdc[t, i]
                
                # FDC capacity constraint
                if instance.fdc_capacity is not None:
                    cap_left_fdc = instance.fdc_capacity[j] - fdc_total_inventory[j]
                else:
                    cap_left_fdc = np.inf
                
                # Outbound capacity constraint
                cap_left_outbound = remaining_outbound_capacity
                
                # Clip to feasible amount
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
                # Available FDC inventory (after shipments arrive)
                # Shipments ordered in period (t - lead_time) arrive in period t
                if t >= instance.lead_time:
                    arrivals = actual_shipments[t - instance.lead_time, i, j]
                else:
                    arrivals = 0.0  # No shipments before period 0
                avail_fdc = inventory_fdc[t, i, j] + arrivals
                demand = instance.demand_fdc[t, i, j]
                
                # Fulfill from local inventory
                fulfilled_local = min(avail_fdc, demand)
                fdc_local_fulfilled[t, i, j] = fulfilled_local
                
                # Remaining inventory and demand
                remaining_inventory = avail_fdc - fulfilled_local
                remaining_demand = demand - fulfilled_local
                
                # Carry inventory to next period
                inventory_fdc[t+1, i, j] = remaining_inventory
                
                # 4) Cross-fulfillment from RDC (if demand not fully satisfied)
                if remaining_demand > 0:
                    avail_rdc = inventory_rdc[t, i]
                    cross_fulfilled = min(remaining_demand, avail_rdc)
                    fdc_from_rdc_fulfilled[t, i, j] = cross_fulfilled
                    inventory_rdc[t, i] -= cross_fulfilled
                    remaining_demand -= cross_fulfilled
                    
                    # 5) Lost sales at FDC
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
    
    # 7) Compute costs
    # Transfer cost
    cost_transfer = np.sum(instance.transfer_cost * actual_shipments)
    
    # Cross-fulfillment cost
    cost_cross = instance.rdc_fulfillment_cost * np.sum(fdc_from_rdc_fulfilled)
    
    # Lost sales cost
    cost_lost = instance.lost_sale_cost * (
        np.sum(lost_fdc) + np.sum(lost_rdc)
    )
    
    total_cost = cost_transfer + cost_cross + cost_lost
    
    # Feasibility checks
    if check_feasibility:
        # Check non-negativity
        assert np.all(inventory_rdc >= 0), "Negative RDC inventory detected"
        assert np.all(inventory_fdc >= 0), "Negative FDC inventory detected"
        assert np.all(actual_shipments >= 0), "Negative shipments detected"
        assert np.all(fdc_local_fulfilled >= 0), "Negative FDC local fulfillment"
        assert np.all(fdc_from_rdc_fulfilled >= 0), "Negative cross-fulfillment"
        assert np.all(lost_fdc >= 0), "Negative FDC lost sales"
        assert np.all(rdc_fulfilled >= 0), "Negative RDC fulfillment"
        assert np.all(lost_rdc >= 0), "Negative RDC lost sales"
        
        # Check outbound capacity
        for t in range(T):
            total_shipped = np.sum(actual_shipments[t, :, :])
            assert total_shipped <= instance.outbound_capacity[t] + 1e-6, \
                f"Outbound capacity violated in period {t}: {total_shipped} > {instance.outbound_capacity[t]}"
        
        # Check FDC capacity
        if instance.fdc_capacity is not None:
            for t in range(T + 1):
                for j in range(J):
                    total_fdc_inv = np.sum(inventory_fdc[t, :, j])
                    assert total_fdc_inv <= instance.fdc_capacity[j] + 1e-6, \
                        f"FDC capacity violated at FDC {j} in period {t}: {total_fdc_inv} > {instance.fdc_capacity[j]}"
    
    return SimulationResult(
        total_cost=total_cost,
        cost_transfer=cost_transfer,
        cost_cross=cost_cross,
        cost_lost=cost_lost,
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

