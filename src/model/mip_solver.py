"""Exact MIP solver for small instances."""

import numpy as np
from typing import Tuple, Optional
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from .instance import Instance


def solve_exact(instance: Instance, time_limit: Optional[float] = None) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Solve instance to optimality using MIP (only for small instances).
    
    Only runs on instances with N <= 5, J <= 3, T <= 4.
    Uses PuLP (or Gurobi if available) to solve the full MIP formulation.
    
    Args:
        instance: Problem instance (must be small)
        time_limit: Time limit in seconds (None = no limit)
    
    Returns:
        Tuple of (optimal_cost, optimal_shipments)
        Returns (None, None) if instance is too large or solver unavailable
    """
    if not PULP_AVAILABLE:
        print("Warning: PuLP not available. Install with: pip install pulp")
        return None, None
    
    # Check instance size
    if instance.num_products > 5 or instance.num_fdcs > 3 or instance.T > 4:
        print(f"Warning: Instance too large for exact solver "
              f"(N={instance.num_products}, J={instance.num_fdcs}, T={instance.T}). "
              f"Max: N=5, J=3, T=4")
        return None, None
    
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    # Create problem
    prob = pulp.LpProblem("MultiEchelonDistribution", pulp.LpMinimize)
    
    # Decision variables
    # Shipments: u[t, i, j]
    u = {}
    for t in range(T):
        for i in range(N):
            for j in range(J):
                u[t, i, j] = pulp.LpVariable(f"u_{t}_{i}_{j}", lowBound=0, cat='Continuous')
    
    # Inventories: I_rdc[t_inv, i], I_fdc[t_inv, i, j]
    I_rdc = {}
    for t_inv in range(T + 1):
        for i in range(N):
            I_rdc[t_inv, i] = pulp.LpVariable(f"I_rdc_{t_inv}_{i}", lowBound=0, cat='Continuous')
    
    I_fdc = {}
    for t_inv in range(T + 1):
        for i in range(N):
            for j in range(J):
                I_fdc[t_inv, i, j] = pulp.LpVariable(f"I_fdc_{t_inv}_{i}_{j}", lowBound=0, cat='Continuous')
    
    # Fulfillment variables
    # x[t, i, j] = fdc_local_fulfilled
    x = {}
    for t in range(T):
        for i in range(N):
            for j in range(J):
                x[t, i, j] = pulp.LpVariable(f"x_{t}_{i}_{j}", lowBound=0, cat='Continuous')
    
    # y_fdc[t, i, j] = fdc_from_rdc_fulfilled
    y_fdc = {}
    for t in range(T):
        for i in range(N):
            for j in range(J):
                y_fdc[t, i, j] = pulp.LpVariable(f"y_fdc_{t}_{i}_{j}", lowBound=0, cat='Continuous')
    
    # z_fdc[t, i, j] = lost_fdc
    z_fdc = {}
    for t in range(T):
        for i in range(N):
            for j in range(J):
                z_fdc[t, i, j] = pulp.LpVariable(f"z_fdc_{t}_{i}_{j}", lowBound=0, cat='Continuous')
    
    # y_rdc[t, i] = rdc_fulfilled
    y_rdc = {}
    for t in range(T):
        for i in range(N):
            y_rdc[t, i] = pulp.LpVariable(f"y_rdc_{t}_{i}", lowBound=0, cat='Continuous')
    
    # z_rdc[t, i] = lost_rdc
    z_rdc = {}
    for t in range(T):
        for i in range(N):
            z_rdc[t, i] = pulp.LpVariable(f"z_rdc_{t}_{i}", lowBound=0, cat='Continuous')
    
    # Objective: minimize total cost
    obj = (
        pulp.lpSum([instance.transfer_cost[i, j] * u[t, i, j]
                    for t in range(T) for i in range(N) for j in range(J)]) +
        instance.rdc_fulfillment_cost * pulp.lpSum([y_fdc[t, i, j]
                                                   for t in range(T) for i in range(N) for j in range(J)]) +
        instance.lost_sale_cost * (
            pulp.lpSum([z_fdc[t, i, j] for t in range(T) for i in range(N) for j in range(J)]) +
            pulp.lpSum([z_rdc[t, i] for t in range(T) for i in range(N)])
        )
    )
    prob += obj
    
    # Initial inventories
    for i in range(N):
        prob += I_rdc[0, i] == instance.initial_inventory_rdc[i]
        for j in range(J):
            prob += I_fdc[0, i, j] == instance.initial_inventory_fdc[i, j]
    
    # RDC inventory balance
    for t in range(T):
        for i in range(N):
            prob += (
                I_rdc[t+1, i] ==
                I_rdc[t, i] +
                instance.replenishment[t, i] -
                pulp.lpSum([u[t, i, j] for j in range(J)]) -
                pulp.lpSum([y_fdc[t, i, j] for j in range(J)]) -
                y_rdc[t, i]
            )
    
    # FDC inventory balance
    for t in range(T):
        for i in range(N):
            for j in range(J):
                prob += (
                    I_fdc[t+1, i, j] ==
                    I_fdc[t, i, j] + u[t, i, j] - x[t, i, j]
                )
    
    # FDC demand satisfaction
    for t in range(T):
        for i in range(N):
            for j in range(J):
                prob += (
                    x[t, i, j] + y_fdc[t, i, j] + z_fdc[t, i, j] ==
                    instance.demand_fdc[t, i, j]
                )
    
    # RDC demand satisfaction
    for t in range(T):
        for i in range(N):
            prob += (
                y_rdc[t, i] + z_rdc[t, i] == instance.demand_rdc[t, i]
            )
    
    # Outbound capacity
    for t in range(T):
        prob += (
            pulp.lpSum([u[t, i, j] for i in range(N) for j in range(J)]) <=
            instance.outbound_capacity[t]
        )
    
    # FDC capacity
    if instance.fdc_capacity is not None:
        for t in range(T + 1):
            for j in range(J):
                prob += (
                    pulp.lpSum([I_fdc[t, i, j] for i in range(N)]) <=
                    instance.fdc_capacity[j]
                )
    
    # Solve
    if time_limit is not None:
        prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status != pulp.LpStatusOptimal:
        print(f"Warning: MIP solver status: {pulp.LpStatus[prob.status]}")
        return None, None
    
    # Extract solution
    optimal_cost = pulp.value(prob.objective)
    optimal_shipments = np.zeros((T, N, J), dtype=float)
    
    for t in range(T):
        for i in range(N):
            for j in range(J):
                optimal_shipments[t, i, j] = pulp.value(u[t, i, j])
    
    return optimal_cost, optimal_shipments

