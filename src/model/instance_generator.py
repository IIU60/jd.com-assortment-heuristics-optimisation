"""Instance generator for creating synthetic problem instances."""

import numpy as np
import random
import json
from pathlib import Path
from typing import Tuple, Literal

from .instance import Instance


def generate_instance(
    n_products: int = 50,
    n_fdcs: int = 10,
    T: int = 14,
    seed: int = 42,
    level: Literal['small', 'medium'] = 'medium',
    lead_time: int = 1,
    # Realistic cost parameters (JD.com calibrated: s:c:r ≈ 50:5:1)
    transfer_cost_range: Tuple[float, float] = (0.2, 2.0),
    rdc_fulfillment_cost: float = 5.0,
    lost_sale_cost: float = 50.0,
    # Capacity parameters
    initial_inventory_ratio: float = 0.7,
    outbound_capacity_ratio: float = 0.8,
    fdc_capacity_per_sku_range: Tuple[float, float] = (20.0, 200.0),
    rdc_capacity_per_sku_range: Tuple[float, float] = (500.0, 2000.0),
    # Replenishment parameters
    replenishment_mean_range: Tuple[float, float] = (50.0, 200.0),
    replenishment_cv: float = 0.15,
) -> Tuple[Instance, np.ndarray]:
    """
    Generate a realistic multi-product, multi-FDC, single-RDC instance using Poisson demand.
    
    Args:
        n_products: Number of products/SKUs (N: 20-100)
        n_fdcs: Number of FDCs
        T: Number of time periods (horizon: 7-30 days)
        seed: Random seed for reproducibility
        level: Instance size level ('small' or 'medium')
        lead_time: Transfer lead time in periods (1-2 days)
        transfer_cost_range: Range for transfer cost r_ij (0.2-2.0)
        rdc_fulfillment_cost: RDC fulfillment cost c (2-10, default 5)
        lost_sale_cost: Lost sale penalty s (20-100, default 50)
        initial_inventory_ratio: Fraction of horizon demand for initial inventory
        outbound_capacity_ratio: Fraction of period demand for outbound capacity
        fdc_capacity_per_sku_range: FDC capacity per SKU range (20-200)
        rdc_capacity_per_sku_range: RDC capacity per SKU range (500-2000)
        replenishment_mean_range: Replenishment mean range (50-200)
        replenishment_cv: Coefficient of variation for replenishment noise (15%)
    
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
    
    # 1) Generate FDC demands using Poisson distribution (lambda: 1-8)
    for t in range(T):
        for i in range(n_products):
            for j in range(n_fdcs):
                # Poisson demand: lambda varies uniformly in range 1-8
                lambda_ij = np.random.uniform(1.0, 8.0)
                demand_fdc[t, i, j] = float(np.random.poisson(lambda_ij))
                total_demand_per_product[i] += demand_fdc[t, i, j]
                total_demand_per_period[t] += demand_fdc[t, i, j]
    
    # 2) Generate RDC demands using Poisson distribution (lambda: 3-15)
    for t in range(T):
        for i in range(n_products):
            # Poisson demand: lambda varies uniformly in range 3-15
            lambda_i = np.random.uniform(3.0, 15.0)
            demand_rdc[t, i] = float(np.random.poisson(lambda_i))
    
    # 3) Generate replenishment (based on demand + noise, as per JD.com)
    # Replenishment should be roughly 0.85-0.95x of average demand per product per period (more constrained)
    for i in range(n_products):
        # Average demand per period for this product
        avg_demand_per_period = (total_demand_per_product[i] / T) if T > 0 else 0
        # Mean replenishment: 0.85-0.95x of average demand (creates supply constraints)
        replenishment_multiplier = np.random.uniform(0.85, 0.95)
        mu_i = avg_demand_per_period * replenishment_multiplier
        sigma_i = mu_i * replenishment_cv  # Coefficient of variation
        
        for t in range(T):
            # Normal distribution with clipping to non-negative
            replenishment[t, i] = max(0, np.random.normal(mu_i, sigma_i))
    
    # 4) Initial RDC inventory (fraction of horizon demand, capped by RDC capacity)
    # Use lower ratio (0.5-0.6) to create more constraints
    initial_inventory_ratio_constrained = initial_inventory_ratio * 0.75  # Reduce initial inventory
    initial_inventory_rdc = initial_inventory_ratio_constrained * total_demand_per_product
    # Cap by per-SKU RDC capacity
    rdc_per_sku_capacity = np.random.uniform(
        rdc_capacity_per_sku_range[0],
        rdc_capacity_per_sku_range[1]
    )
    initial_inventory_rdc = np.minimum(initial_inventory_rdc, rdc_per_sku_capacity)
    
    # 5) Outbound capacity (fraction of period demand, make it tighter)
    # Reduce to 0.7x to create more capacity constraints
    outbound_capacity = 0.7 * total_demand_per_period
    
    # 6) FDC capacities (per-SKU capacity: 20-200, total = per_sku * n_products)
    fdc_capacity = np.zeros(n_fdcs, dtype=float)
    for j in range(n_fdcs):
        # Per-SKU capacity varies by FDC
        per_sku_capacity = np.random.uniform(
            fdc_capacity_per_sku_range[0],
            fdc_capacity_per_sku_range[1]
        )
        # Total FDC capacity = per-SKU capacity * number of products
        fdc_capacity[j] = per_sku_capacity * n_products
    
    # 7) Transfer costs (vary by product and FDC, realistic range: 0.2-2.0)
    transfer_cost = np.zeros((n_products, n_fdcs), dtype=float)
    for i in range(n_products):
        for j in range(n_fdcs):
            # Base cost with variation
            base_cost = np.random.uniform(transfer_cost_range[0], transfer_cost_range[1])
            transfer_cost[i, j] = base_cost
    
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
        lost_sale_cost=lost_sale_cost,
        lead_time=lead_time
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
        'lost_sale_cost': instance.lost_sale_cost,
        'lead_time': instance.lead_time
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
        lost_sale_cost=data['lost_sale_cost'],
        lead_time=data.get('lead_time', 1)  # Default to 1 for backward compatibility
    )
    
    instance.validate()
    return instance


def generate_instance_set(
    output_dir: str = 'instances',
    n_small: int = 10,
    n_medium: int = 20,
    lead_time: int = 1
) -> None:
    """
    Generate a set of small and medium instances and save them.
    
    Args:
        output_dir: Directory to save instances
        n_small: Number of small instances to generate
        n_medium: Number of medium instances to generate
        lead_time: Transfer lead time in periods (1-2 days)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate small instances (N: 20, T: 7 days)
    print(f"Generating {n_small} small instances...")
    for i in range(n_small):
        instance, _ = generate_instance(
            n_products=20,
            n_fdcs=5,
            T=7,
            seed=100 + i,
            level='small',
            lead_time=lead_time
        )
        filepath = output_path / f'small_{i:02d}.json'
        save_instance(instance, str(filepath))
        print(f"  Saved {filepath}")
    
    # Generate medium instances (N: 50, T: 14 days)
    print(f"Generating {n_medium} medium instances...")
    for i in range(n_medium):
        instance, _ = generate_instance(
            n_products=50,
            n_fdcs=10,
            T=14,
            seed=200 + i,
            level='medium',
            lead_time=lead_time
        )
        filepath = output_path / f'medium_{i:02d}.json'
        save_instance(instance, str(filepath))
        print(f"  Saved {filepath}")
    
    print(f"\nGenerated {n_small} small and {n_medium} medium instances in {output_dir}/")

