"""Instance generator for creating synthetic problem instances."""

import numpy as np
import random
import math
import json
from pathlib import Path
from typing import Tuple, Literal

from .instance import Instance


def generate_instance(
    n_products: int = 200,
    n_fdcs: int = 50,
    T: int = 10,
    demand_low: int = 5,
    demand_high: int = 30,
    seed: int = 42,
    level: Literal['small', 'medium'] = 'small'
) -> Tuple[Instance, np.ndarray]:
    """
    Generate a synthetic multi-product, multi-FDC, single-RDC instance.
    
    Args:
        n_products: Number of products/SKUs
        n_fdcs: Number of FDCs
        T: Number of time periods
        demand_low: Lower bound for base demand
        demand_high: Upper bound for base demand
        seed: Random seed for reproducibility
        level: Instance size level ('small' or 'medium')
    
    Returns:
        Tuple of (Instance, initial_shipment_plan)
        initial_shipment_plan is all zeros, shape (T, N, J)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize arrays
    demand_fdc = np.zeros((T, n_products, n_fdcs), dtype=float)
    demand_rdc = np.zeros((T, n_products), dtype=float)
    replenishment = np.zeros((T, n_products), dtype=float)
    initial_inventory_rdc = np.zeros(n_products, dtype=float)
    initial_inventory_fdc = np.zeros((n_products, n_fdcs), dtype=float)
    
    # Track totals for capacity calculations
    total_demand_per_product = np.zeros(n_products, dtype=float)
    total_demand_per_period = np.zeros(T, dtype=float)
    
    # 1) Generate FDC demands
    for t in range(T):
        for i in range(n_products):
            for j in range(n_fdcs):
                base = random.randint(demand_low, demand_high)
                # Small temporal fluctuation
                fluct = 1.0 + 0.2 * math.sin(2 * math.pi * (t + 1) / T)
                val = max(0, int(base * fluct))
                demand_fdc[t, i, j] = float(val)
                total_demand_per_product[i] += val
                total_demand_per_period[t] += val
    
    # 2) Initial RDC inventory ~70% of horizon demand per product
    initial_inventory_rdc = 0.7 * total_demand_per_product
    
    # 3) No replenishment during horizon (all stock upfront)
    # replenishment already zeros
    
    # 4) Outbound capacity Q[t] ~60% of total demand that period
    outbound_capacity = 0.6 * total_demand_per_period
    
    # 5) FDC capacities ~80% of their total horizon demand
    fdc_capacity = np.zeros(n_fdcs, dtype=float)
    for j in range(n_fdcs):
        total_demand_j = np.sum(demand_fdc[:, :, j])
        fdc_capacity[j] = 0.8 * total_demand_j
    
    # 6) Costs
    transfer_cost = np.ones((n_products, n_fdcs), dtype=float)  # unit transfer cost
    rdc_fulfillment_cost = 3.0  # RDC cross-fulfillment penalty
    lost_sale_cost = 10.0  # lost sales penalty
    
    # Create instance
    instance = Instance(
        num_products=n_products,
        num_fdcs=n_fdcs,
        T=T,
        demand_fdc=demand_fdc,
        demand_rdc=demand_rdc,
        initial_inventory_rdc=initial_inventory_rdc,
        initial_inventory_fdc=initial_inventory_fdc,
        replenishment=replenishment,
        outbound_capacity=outbound_capacity,
        fdc_capacity=fdc_capacity,
        transfer_cost=transfer_cost,
        rdc_fulfillment_cost=rdc_fulfillment_cost,
        lost_sale_cost=lost_sale_cost
    )
    
    # Validate instance
    instance.validate()
    
    # Initial shipment plan: all zeros
    initial_shipments = np.zeros((T, n_products, n_fdcs), dtype=float)
    
    return instance, initial_shipments


def save_instance(instance: Instance, filepath: str) -> None:
    """
    Save instance to file as JSON (converting numpy arrays to lists).
    
    Args:
        instance: Instance to save
        filepath: Path to save file
    """
    data = {
        'num_products': instance.num_products,
        'num_fdcs': instance.num_fdcs,
        'T': instance.T,
        'demand_fdc': instance.demand_fdc.tolist(),
        'demand_rdc': instance.demand_rdc.tolist(),
        'initial_inventory_rdc': instance.initial_inventory_rdc.tolist(),
        'initial_inventory_fdc': instance.initial_inventory_fdc.tolist(),
        'replenishment': instance.replenishment.tolist(),
        'outbound_capacity': instance.outbound_capacity.tolist(),
        'fdc_capacity': instance.fdc_capacity.tolist() if instance.fdc_capacity is not None else None,
        'transfer_cost': instance.transfer_cost.tolist(),
        'rdc_fulfillment_cost': instance.rdc_fulfillment_cost,
        'lost_sale_cost': instance.lost_sale_cost
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_instance(filepath: str) -> Instance:
    """
    Load instance from JSON file.
    
    Args:
        filepath: Path to instance file
    
    Returns:
        Loaded Instance
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    instance = Instance(
        num_products=data['num_products'],
        num_fdcs=data['num_fdcs'],
        T=data['T'],
        demand_fdc=np.array(data['demand_fdc']),
        demand_rdc=np.array(data['demand_rdc']),
        initial_inventory_rdc=np.array(data['initial_inventory_rdc']),
        initial_inventory_fdc=np.array(data['initial_inventory_fdc']),
        replenishment=np.array(data['replenishment']),
        outbound_capacity=np.array(data['outbound_capacity']),
        fdc_capacity=np.array(data['fdc_capacity']) if data['fdc_capacity'] is not None else None,
        transfer_cost=np.array(data['transfer_cost']),
        rdc_fulfillment_cost=data['rdc_fulfillment_cost'],
        lost_sale_cost=data['lost_sale_cost']
    )
    
    instance.validate()
    return instance


def generate_instance_set(
    output_dir: str = 'instances',
    n_small: int = 10,
    n_medium: int = 20
) -> None:
    """
    Generate a set of small and medium instances and save them.
    
    Args:
        output_dir: Directory to save instances
        n_small: Number of small instances to generate
        n_medium: Number of medium instances to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate small instances
    print(f"Generating {n_small} small instances...")
    for i in range(n_small):
        instance, _ = generate_instance(
            n_products=10,
            n_fdcs=3,
            T=5,
            seed=100 + i,
            level='small'
        )
        filepath = output_path / f'small_{i:02d}.json'
        save_instance(instance, str(filepath))
        print(f"  Saved {filepath}")
    
    # Generate medium instances
    print(f"Generating {n_medium} medium instances...")
    for i in range(n_medium):
        instance, _ = generate_instance(
            n_products=50,
            n_fdcs=10,
            T=10,
            seed=200 + i,
            level='medium'
        )
        filepath = output_path / f'medium_{i:02d}.json'
        save_instance(instance, str(filepath))
        print(f"  Saved {filepath}")
    
    print(f"\nGenerated {n_small} small and {n_medium} medium instances in {output_dir}/")

