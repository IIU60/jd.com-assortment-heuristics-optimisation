"""Neighborhood move operators for local search."""

import numpy as np
import random
from typing import Tuple, Dict, Optional

from ..model.instance import Instance
from .utils import copy_u, clamp_u_to_feasibility


def time_shift_move(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j: int,
    t_from: int,
    t_to: int,
    delta: float
) -> np.ndarray:
    """
    Time-shift move: reduce shipment at t_from, increase at t_to.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j: FDC index
        t_from: Period to reduce shipment
        t_to: Period to increase shipment
        delta: Amount to shift
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Reduce at t_from
    new_u[t_from, i, j] = max(0.0, new_u[t_from, i, j] - delta)
    
    # Increase at t_to
    new_u[t_to, i, j] += delta
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def fdc_swap_move(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j_from: int,
    j_to: int,
    t: int,
    delta: float
) -> np.ndarray:
    """
    FDC-swap move: reduce shipment to j_from, increase to j_to.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j_from: FDC to reduce shipment
        j_to: FDC to increase shipment
        t: Period
        delta: Amount to swap
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Reduce at j_from
    new_u[t, i, j_from] = max(0.0, new_u[t, i, j_from] - delta)
    
    # Increase at j_to
    new_u[t, i, j_to] += delta
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def magnitude_tweak(
    u: np.ndarray,
    instance: Instance,
    i: int,
    j: int,
    t: int,
    delta: float
) -> np.ndarray:
    """
    Magnitude tweak: add or subtract delta from shipment.
    
    Args:
        u: Current shipment plan, shape (T, N, J)
        instance: Problem instance
        i: Product index
        j: FDC index
        t: Period
        delta: Amount to add (can be negative to subtract)
    
    Returns:
        New shipment plan (copy)
    """
    new_u = copy_u(u)
    
    # Apply delta
    new_u[t, i, j] = max(0.0, new_u[t, i, j] + delta)
    
    # Clamp to feasibility
    new_u = clamp_u_to_feasibility(instance, new_u)
    
    return new_u


def generate_neighbor(
    instance: Instance,
    u: np.ndarray,
    move_probs: Optional[Dict[str, float]] = None,
    delta_choices: Optional[Tuple[float, ...]] = None
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Generate a random neighbor using one of the move operators.
    
    Args:
        instance: Problem instance
        u: Current shipment plan, shape (T, N, J)
        move_probs: Probabilities for each move type
            Keys: 'time_shift', 'fdc_swap', 'magnitude_tweak'
            If None, uses equal probabilities
        delta_choices: Possible delta values for moves
            If None, computes adaptive deltas based on problem scale
    
    Returns:
        Tuple of (new_u, move_key)
        move_key is (move_type, indices...) for tabu list
    """
    T = instance.T
    N = instance.num_products
    J = instance.num_fdcs
    
    if move_probs is None:
        move_probs = {
            'time_shift': 0.33,
            'fdc_swap': 0.33,
            'magnitude_tweak': 0.34
        }
    
    # Compute adaptive deltas if not provided
    if delta_choices is None:
        # Calculate average demand to scale deltas appropriately
        avg_demand = np.mean(instance.demand_fdc[instance.demand_fdc > 0])
        if avg_demand > 0:
            # Use 10%, 20%, 30% of average demand as deltas
            # Apply minimum thresholds to avoid tiny values
            delta_choices = (
                max(5.0, avg_demand * 0.10),
                max(10.0, avg_demand * 0.20),
                max(20.0, avg_demand * 0.30)
            )
        else:
            # Fallback to fixed values if no demand
            delta_choices = (5.0, 10.0, 20.0)
    
    # Select move type
    r = random.random()
    if r < move_probs.get('magnitude_tweak', 0.34):
        # Magnitude tweak
        t = random.randrange(T)
        i = random.randrange(N)
        j = random.randrange(J)
        delta = random.choice(delta_choices)
        if random.random() < 0.5:
            delta = -delta
        
        new_u = magnitude_tweak(u, instance, i, j, t, delta)
        move_key = ('magnitude_tweak', t, i, j)
    
    elif r < move_probs.get('magnitude_tweak', 0.34) + move_probs.get('fdc_swap', 0.33):
        # FDC swap
        t = random.randrange(T)
        i = random.randrange(N)
        j_from = random.randrange(J)
        j_to = random.randrange(J)
        while j_to == j_from:
            j_to = random.randrange(J)
        delta = random.choice(delta_choices)
        
        new_u = fdc_swap_move(u, instance, i, j_from, j_to, t, delta)
        move_key = ('fdc_swap', t, i, j_from, j_to)
    
    else:
        # Time shift
        t_from = random.randrange(T)
        t_to = random.randrange(T)
        while t_to == t_from:
            t_to = random.randrange(T)
        i = random.randrange(N)
        j = random.randrange(J)
        delta = random.choice(delta_choices)
        
        new_u = time_shift_move(u, instance, i, j, t_from, t_to, delta)
        move_key = ('time_shift', i, j, t_from, t_to)
    
    return new_u, move_key

