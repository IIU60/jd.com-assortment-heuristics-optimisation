"""Parameter tuning scripts for SA and Tabu Search."""

import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from ..model.instance import Instance
from ..model.instance_generator import load_instance
from ..heuristics.sa import simulated_annealing
from ..heuristics.tabu import tabu_search
from ..heuristics.utils import random_feasible_u


def tune_sa(
    instances: List[Instance],
    T0_values: List[float] = None,
    alpha_values: List[float] = None,
    n_runs: int = 5,
    max_iters: int = 500,
    output_dir: str = 'results'
) -> pd.DataFrame:
    """
    Tune Simulated Annealing parameters.
    
    Args:
        instances: List of instances to tune on
        T0_values: List of initial temperatures to try
        alpha_values: List of cooling rates to try
        n_runs: Number of runs per configuration
        max_iters: Maximum iterations per SA run
        output_dir: Directory to save results
    
    Returns:
        DataFrame with tuning results
    """
    if T0_values is None:
        T0_values = [50000.0, 100000.0, 200000.0]
    if alpha_values is None:
        alpha_values = [0.90, 0.95, 0.98]
    
    results = []
    
    print("\n=== Simulated Annealing Parameter Tuning ===")
    
    for instance in instances:
        print(f"\nInstance: {instance.num_products} products, "
              f"{instance.num_fdcs} FDCs, {instance.T} periods")
        
        # Generate initial solution
        u0 = random_feasible_u(instance, seed=42)
        
        for T0 in T0_values:
            for alpha in alpha_values:
                cfg = {'T0': T0, 'alpha': alpha}
                costs = []
                runtimes = []
                
                start_time = time.time()
                for run in range(n_runs):
                    seed = 1000 + run
                    best_cost, _, _ = simulated_annealing(
                        instance, u0,
                        T0=T0,
                        alpha=alpha,
                        max_iters=max_iters,
                        seed=seed
                    )
                    costs.append(best_cost)
                
                elapsed = time.time() - start_time
                runtimes.append(elapsed / n_runs)
                
                avg_cost = np.mean(costs)
                best_cost = np.min(costs)
                std_cost = np.std(costs)
                
                results.append({
                    'instance_size': f"{instance.num_products}_{instance.num_fdcs}_{instance.T}",
                    'T0': T0,
                    'alpha': alpha,
                    'avg_cost': avg_cost,
                    'best_cost': best_cost,
                    'std_cost': std_cost,
                    'avg_runtime': elapsed / n_runs,
                    'n_runs': n_runs
                })
                
                print(f"  SA cfg {cfg} -> avg={avg_cost:.2f}, best={best_cost:.2f}, "
                      f"std={std_cost:.2f}, time={elapsed/n_runs:.1f}s")
    
    df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    df.to_csv(output_path / 'sa_tuning_results.csv', index=False)
    
    # Find best config
    best_idx = df.groupby('instance_size')['avg_cost'].idxmin()
    best_configs = df.loc[best_idx]
    print("\n=== Best SA Configurations ===")
    print(best_configs[['instance_size', 'T0', 'alpha', 'avg_cost', 'best_cost']])
    
    return df


def tune_tabu(
    instances: List[Instance],
    tabu_tenures: List[int] = None,
    neighborhood_sizes: List[int] = None,
    n_runs: int = 5,
    max_iters: int = 200,
    output_dir: str = 'results'
) -> pd.DataFrame:
    """
    Tune Tabu Search parameters.
    
    Args:
        instances: List of instances to tune on
        tabu_tenures: List of tabu tenures to try
        neighborhood_sizes: List of neighborhood sizes to try
        n_runs: Number of runs per configuration
        max_iters: Maximum iterations per TS run
        output_dir: Directory to save results
    
    Returns:
        DataFrame with tuning results
    """
    if tabu_tenures is None:
        tabu_tenures = [3, 5, 8, 12]
    if neighborhood_sizes is None:
        neighborhood_sizes = [20, 40, 80]
    
    results = []
    
    print("\n=== Tabu Search Parameter Tuning ===")
    
    for instance in instances:
        print(f"\nInstance: {instance.num_products} products, "
              f"{instance.num_fdcs} FDCs, {instance.T} periods")
        
        # Generate initial solution
        u0 = random_feasible_u(instance, seed=42)
        
        for tenure in tabu_tenures:
            for nh_size in neighborhood_sizes:
                cfg = {'tenure': tenure, 'nh_size': nh_size}
                costs = []
                runtimes = []
                
                start_time = time.time()
                for run in range(n_runs):
                    seed = 2000 + run
                    best_cost, _, _ = tabu_search(
                        instance, u0,
                        tabu_tenure=tenure,
                        max_iters=max_iters,
                        neighborhood_size=nh_size,
                        seed=seed
                    )
                    costs.append(best_cost)
                
                elapsed = time.time() - start_time
                runtimes.append(elapsed / n_runs)
                
                avg_cost = np.mean(costs)
                best_cost = np.min(costs)
                std_cost = np.std(costs)
                
                results.append({
                    'instance_size': f"{instance.num_products}_{instance.num_fdcs}_{instance.T}",
                    'tabu_tenure': tenure,
                    'neighborhood_size': nh_size,
                    'avg_cost': avg_cost,
                    'best_cost': best_cost,
                    'std_cost': std_cost,
                    'avg_runtime': elapsed / n_runs,
                    'n_runs': n_runs
                })
                
                print(f"  TS cfg {cfg} -> avg={avg_cost:.2f}, best={best_cost:.2f}, "
                      f"std={std_cost:.2f}, time={elapsed/n_runs:.1f}s")
    
    df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    df.to_csv(output_path / 'tabu_tuning_results.csv', index=False)
    
    # Find best config
    best_idx = df.groupby('instance_size')['avg_cost'].idxmin()
    best_configs = df.loc[best_idx]
    print("\n=== Best TS Configurations ===")
    print(best_configs[['instance_size', 'tabu_tenure', 'neighborhood_size', 
                        'avg_cost', 'best_cost']])
    
    return df


def load_instances_for_tuning(instances_dir: str = 'instances', n_instances: int = 3) -> List[Instance]:
    """
    Load instances for parameter tuning.
    
    Args:
        instances_dir: Directory containing instance files
        n_instances: Number of instances to load
    
    Returns:
        List of instances
    """
    instances_path = Path(instances_dir)
    instance_files = sorted(list(instances_path.glob('*.json')))[:n_instances]
    
    instances = []
    for filepath in instance_files:
        try:
            instance = load_instance(str(filepath))
            instances.append(instance)
            print(f"Loaded {filepath.name}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return instances

