"""Experimental harness for running all algorithms on instances."""

import time
import json
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, Optional
import numpy as np

from ..model.instance import Instance
from ..model.simulator import simulate
from ..baselines import myopic_greedy, static_proportional, random_feasible
from ..heuristics.utils import evaluate_u


def run_baselines_on_instance(instance: Instance, seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Run all baseline algorithms on an instance.
    
    Args:
        instance: Problem instance
        seed: Random seed for random baseline
    
    Returns:
        Dictionary mapping algorithm name to results
        Each result contains: cost, runtime, shipment_plan, metrics
    """
    results = {}
    
    # Myopic greedy
    start = time.perf_counter()
    cost, shipments = myopic_greedy(instance)
    runtime = time.perf_counter() - start
    result = simulate(instance, shipments, check_feasibility=False)
    results['myopic'] = {
        'cost': cost,
        'runtime': runtime,
        'shipments': shipments.tolist(),
        'cost_transfer': result.cost_transfer,
        'cost_cross': result.cost_cross,
        'cost_lost': result.cost_lost,
        'clipped_shipments': result.clipped_shipments
    }
    
    # Static proportional
    start = time.perf_counter()
    cost, shipments = static_proportional(instance)
    runtime = time.perf_counter() - start
    result = simulate(instance, shipments, check_feasibility=False)
    results['static_prop'] = {
        'cost': cost,
        'runtime': runtime,
        'shipments': shipments.tolist(),
        'cost_transfer': result.cost_transfer,
        'cost_cross': result.cost_cross,
        'cost_lost': result.cost_lost,
        'clipped_shipments': result.clipped_shipments
    }
    
    # Random feasible
    start = time.perf_counter()
    cost, shipments = random_feasible(instance, seed=seed)
    runtime = time.perf_counter() - start
    result = simulate(instance, shipments, check_feasibility=False)
    results['random'] = {
        'cost': cost,
        'runtime': runtime,
        'shipments': shipments.tolist(),
        'cost_transfer': result.cost_transfer,
        'cost_cross': result.cost_cross,
        'cost_lost': result.cost_lost,
        'clipped_shipments': result.clipped_shipments
    }
    
    return results


def run_all_algorithms_on_instance(
    instance: Instance,
    alg_dict: Dict[str, Callable[[Instance], Tuple[float, np.ndarray]]],
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Run all algorithms (baselines + heuristics) on an instance.
    
    Args:
        instance: Problem instance
        alg_dict: Dictionary mapping algorithm name to function
            Each function takes (instance) and returns (cost, shipments)
        seed: Random seed (passed to algorithms that need it)
    
    Returns:
        Dictionary mapping algorithm name to results
        Each result contains: cost, runtime, shipment_plan, metrics, parameters
    """
    results = {}
    
    for alg_name, alg_func in alg_dict.items():
        try:
            start = time.perf_counter()
            
            # Call algorithm
            if 'seed' in alg_func.__code__.co_varnames:
                cost, shipments = alg_func(instance, seed=seed)
            else:
                cost, shipments = alg_func(instance)
            
            runtime = time.perf_counter() - start
            
            # Handle None returns (e.g., MIP solver timed out)
            if cost is None or shipments is None:
                results[alg_name] = {
                    'error': 'Solver timed out or failed',
                    'cost': None,
                    'runtime': runtime if runtime else None,
                    'skipped': True
                }
                continue
            
            # Evaluate
            result = simulate(instance, shipments, check_feasibility=False)
            
            results[alg_name] = {
                'cost': float(cost),
                'runtime': runtime,
                'shipments': shipments.tolist(),
                'cost_transfer': float(result.cost_transfer),
                'cost_cross': float(result.cost_cross),
                'cost_lost': float(result.cost_lost),
                'clipped_shipments': float(result.clipped_shipments),
                'seed': seed
            }
        except Exception as e:
            print(f"Error running {alg_name}: {e}")
            results[alg_name] = {
                'error': str(e),
                'cost': None,
                'runtime': None
            }
    
    return results


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to results file
    
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def run_all_experiments(
    instances_dir: str = 'instances',
    output_dir: str = 'results',
    sa_params: Dict[str, Any] = None,
    tabu_params: Dict[str, Any] = None,
    mip_max_time: float = 120.0
) -> None:
    """
    Run all algorithms on all instances and save results.
    
    Args:
        instances_dir: Directory containing instance files
        output_dir: Directory to save results
        sa_params: SA parameters (T0, alpha, max_iters)
        tabu_params: Tabu Search parameters (tabu_tenure, max_iters, neighborhood_size)
        mip_max_time: Maximum time limit for MIP solver in seconds (default: 120)
    """
    from pathlib import Path
    from ..model.instance_generator import load_instance
    from ..baselines import myopic_greedy, static_proportional, random_feasible
    from ..heuristics.sa import simulated_annealing
    from ..heuristics.tabu import tabu_search
    from ..heuristics.construction import greedy_constructor, grasp_constructor
    from ..heuristics.utils import random_feasible_u, evaluate_u
    from .plots import create_all_plots
    import pandas as pd
    
    if sa_params is None:
        sa_params = {'alpha': 0.95, 'max_iters': 1000, 'max_no_improve': 200}  # More iterations, relaxed early stopping
    if tabu_params is None:
        tabu_params = {'tabu_tenure': 5, 'max_iters': 500, 'neighborhood_size': 15, 'max_no_improve': 100}  # More iterations, larger neighborhood
    
    # Load instances
    instances_path = Path(instances_dir)
    instance_files = sorted(list(instances_path.glob('*.json')))
    
    if not instance_files:
        print(f"No instance files found in {instances_dir}")
        return
    
    all_results = []
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Running experiments on {len(instance_files)} instances...")
    
    for instance_file in instance_files:
        print(f"\nProcessing {instance_file.name}...")
        instance = load_instance(str(instance_file))
        instance_type = 'small' if instance.num_products <= 10 else 'medium'
        
        # Prepare algorithm dictionary
        def sa_wrapper(inst):
            # Start from second-best solution to give room for improvement
            # This allows SA to explore and potentially find better solutions
            candidates = []
            # Myopic
            _, u_myopic = myopic_greedy(inst)
            cost_myopic, _ = evaluate_u(inst, u_myopic)
            candidates.append((cost_myopic, u_myopic, 'myopic'))
            
            # Greedy construction
            u_greedy = greedy_constructor(inst)
            cost_greedy, _ = evaluate_u(inst, u_greedy)
            candidates.append((cost_greedy, u_greedy, 'greedy'))
            
            # GRASP
            u_grasp = grasp_constructor(inst, alpha=0.5, seed=42)
            cost_grasp, _ = evaluate_u(inst, u_grasp)
            candidates.append((cost_grasp, u_grasp, 'grasp'))
            
            # Sort by cost - start from myopic (best) but with more exploration
            candidates.sort(key=lambda x: x[0])
            # Start from myopic (best) - algorithms should be able to improve or at least match it
            u0 = candidates[0][1]  # Best (myopic)
            
            cost, best_u, _ = simulated_annealing(inst, u0, seed=42, **sa_params)
            return cost, best_u
        
        def tabu_wrapper(inst):
            # Start from second-best solution to give room for improvement
            # This allows Tabu to explore and potentially find better solutions
            candidates = []
            # Myopic
            _, u_myopic = myopic_greedy(inst)
            cost_myopic, _ = evaluate_u(inst, u_myopic)
            candidates.append((cost_myopic, u_myopic, 'myopic'))
            
            # Greedy construction
            u_greedy = greedy_constructor(inst)
            cost_greedy, _ = evaluate_u(inst, u_greedy)
            candidates.append((cost_greedy, u_greedy, 'greedy'))
            
            # GRASP
            u_grasp = grasp_constructor(inst, alpha=0.5, seed=42)
            cost_grasp, _ = evaluate_u(inst, u_grasp)
            candidates.append((cost_grasp, u_grasp, 'grasp'))
            
            # Sort by cost - start from myopic (best) but with more exploration
            candidates.sort(key=lambda x: x[0])
            # Start from myopic (best) - algorithms should be able to improve or at least match it
            u0 = candidates[0][1]  # Best (myopic)
            
            cost, best_u, _ = tabu_search(inst, u0, seed=42, **tabu_params)
            return cost, best_u
        
        def greedy_wrapper(inst):
            u = greedy_constructor(inst)
            cost, _ = evaluate_u(inst, u)
            return cost, u
        
        def grasp_wrapper(inst):
            u = grasp_constructor(inst, alpha=0.5, seed=42)
            cost, _ = evaluate_u(inst, u)
            return cost, u
        
        def mip_wrapper(inst):
            from ..model.mip_solver import solve_exact
            import numpy as np
            # Use reasonable time limit based on instance size
            # Larger instances get more time, but cap at mip_max_time
            time_limit = min(30.0 + (inst.num_products * inst.num_fdcs * inst.T) * 0.1, mip_max_time)
            optimal_cost, optimal_u = solve_exact(inst, time_limit=time_limit)
            if optimal_cost is None or optimal_u is None:
                # Solver failed or timed out - return a sentinel value
                # The runner will handle None gracefully
                return None, None
            return optimal_cost, optimal_u
        
        alg_dict = {
            'myopic': myopic_greedy,
            'static_prop': static_proportional,
            'random': lambda inst: random_feasible(inst, seed=42),
            'greedy': greedy_wrapper,
            'grasp': grasp_wrapper,
            'sa': sa_wrapper,
            'tabu': tabu_wrapper,
            'mip': mip_wrapper
        }
        
        # Run all algorithms
        results = run_all_algorithms_on_instance(instance, alg_dict, seed=42)
        
        # Add instance info
        for alg_name in results:
            results[alg_name]['instance'] = instance_file.stem
            results[alg_name]['instance_type'] = instance_type
            results[alg_name]['algorithm'] = alg_name
        
        # Save individual instance results
        save_results(results, str(output_path / f"{instance_file.stem}_results.json"))
        
        # Collect for summary (skip algorithms that failed or were skipped)
        for alg_name, alg_results in results.items():
            if 'cost' in alg_results and alg_results['cost'] is not None and not alg_results.get('skipped', False):
                all_results.append({
                    'instance': instance_file.stem,
                    'instance_type': instance_type,
                    'algorithm': alg_name,
                    'cost': alg_results['cost'],
                    'runtime': alg_results.get('runtime', None),
                    'cost_transfer': alg_results.get('cost_transfer', None),
                    'cost_cross': alg_results.get('cost_cross', None),
                    'cost_lost': alg_results.get('cost_lost', None)
                })
    
    # Create summary DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_path / 'all_results.csv', index=False)
        print(f"\nSaved summary to {output_path / 'all_results.csv'}")
        
        # Create plots
        print("\nCreating plots...")
        create_all_plots({}, df, str(output_path))
    
    print(f"\nExperiments complete! Results saved to {output_dir}/")

