"""Exact MIP solver for small instances."""

import numpy as np
from typing import Tuple, Optional
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from .instance import Instance


def solve_exact(instance: Instance, time_limit: Optional[float] = 60.0) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Solve instance to optimality using MIP.
    
    Uses PuLP (or Gurobi if available) to solve the full MIP formulation.
    If time limit is exceeded, returns None to indicate the instance was too difficult.
    
    Args:
        instance: Problem instance
        time_limit: Time limit in seconds (default: 60 seconds)
    
    Returns:
        Tuple of (optimal_cost, optimal_shipments)
        Returns (None, None) if solver unavailable, time limit exceeded, or solver failed
    """
    if not PULP_AVAILABLE:
        print("Warning: PuLP not available. Install with: pip install pulp")
        return None, None
    
    if time_limit is None:
        time_limit = 60.0  # Default 60 seconds
    
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
                # Shipments ordered in period (t - lead_time) arrive in period t
                if t >= instance.lead_time:
                    # Shipments that arrive in period t were ordered in t - lead_time
                    arrivals = u[t - instance.lead_time, i, j]
                else:
                    # No shipments arrive in periods before lead_time
                    arrivals = 0.0
                
                prob += (
                    I_fdc[t+1, i, j] ==
                    I_fdc[t, i, j] + arrivals - x[t, i, j]
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
    
    # Solve with time limit
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
    
    # Check solver status
    if prob.status == pulp.LpStatusOptimal:
        # Optimal solution found
        pass
    elif prob.status == pulp.LpStatusNotSolved:
        # Time limit exceeded or other issue
        print(f"Warning: MIP solver did not solve within time limit ({time_limit}s) "
              f"for instance (N={N}, J={J}, T={T})")
        return None, None
    else:
        # Other solver status (infeasible, unbounded, etc.)
        print(f"Warning: MIP solver status: {pulp.LpStatus[prob.status]} "
              f"for instance (N={N}, J={J}, T={T})")
        return None, None
    
    # Extract solution
    optimal_cost = pulp.value(prob.objective)
    if optimal_cost is None:
        print(f"Warning: MIP solver returned None objective value")
        return None, None
    
    optimal_shipments = np.zeros((T, N, J), dtype=float)
    
    for t in range(T):
        for i in range(N):
            for j in range(J):
                val = pulp.value(u[t, i, j])
                if val is None:
                    print(f"Warning: MIP variable u[{t},{i},{j}] is None")
                    return None, None
                optimal_shipments[t, i, j] = max(0.0, float(val))  # Ensure non-negative
    
    # Final validation
    if np.any(optimal_shipments < 0) or np.any(np.isnan(optimal_shipments)):
        print(f"Warning: MIP solution contains invalid values")
        return None, None
    
    return optimal_cost, optimal_shipments

