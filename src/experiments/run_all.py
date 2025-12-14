"""Experimental harness for running all algorithms on instances."""

import time
import json
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, Optional, List
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
        'cost_clipped': result.cost_clipped,
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
        'cost_clipped': result.cost_clipped,
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
                'cost_clipped': float(result.cost_clipped),
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
    instances_dir: str,
    output_dir: str,
    sa_params: Dict[str, Any] = None,
    tabu_params: Dict[str, Any] = None,
    mip_max_time: float = 120.0,
    baseline_names: Optional[List[str]] = None,
) -> None:
    """
    Run all algorithms on all instances and save results.
    
    This function assumes the caller has already created a dedicated
    experiment directory under ``experiments/`` with the usual layout::
    
        experiments/<experiment_name>/
            instances/
            results/
    
    Args:
        instances_dir: Directory containing instance files for *this* experiment
        output_dir: Directory to save results for *this* experiment
        sa_params: SA parameters (T0, alpha, max_iters)
        tabu_params: Tabu Search parameters (tabu_tenure, max_iters, neighborhood_size)
        mip_max_time: Maximum time limit for MIP solver in seconds (default: 120)
        baseline_names: Optional list of baseline algorithm names to run from
            {'myopic', 'static_prop', 'random'}. If None, defaults to ['random'].
    """
    from pathlib import Path
    from ..model.instance_generator import load_instance
    from ..baselines import myopic_greedy, static_proportional, random_feasible
    from ..heuristics.sa import simulated_annealing
    from ..heuristics.tabu import tabu_search
    from ..heuristics.construction import greedy_constructor, grasp_constructor, build_starting_pool
    from ..heuristics.utils import random_feasible_u, evaluate_u
    from .plots import create_all_plots
    import pandas as pd
    
    # Baselines to include (default: myopic and random)
    if baseline_names is None:
        baseline_names = ['myopic', 'random']
    baseline_set = set(baseline_names)

    if sa_params is None:
        sa_params = {
            'alpha': 0.95,
            'max_iters': 500,
            'max_no_improve': 200,
            'large_move_prob': 0.1,
            'time_limit': None, # optional override (seconds) per instance for SA
        }  # Single-start SA with occasional large moves
    if tabu_params is None:
        tabu_params = {
            'tabu_tenure': 5,
            'max_iters': 200,
            'neighborhood_size': 8,
            'max_no_improve': 100,
            'large_move_prob': 0.1,
            'time_limit': None # optional override (seconds) per instance for Tabu
        }  # Single-start Tabu with occasional large moves
    
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
        
        # Time budgets for heuristics (seconds), scaled by instance size
        base_scale = instance.num_products * instance.num_fdcs * instance.T
        # Keep budgets modest to ensure heuristics are much cheaper than MIP
        sa_time_limit = float(sa_params.get(
            'time_limit',
            min(5.0, 0.002 * base_scale + 1.0),
        ))
        tabu_time_limit = float(tabu_params.get(
            'time_limit',
            min(5.0, 0.002 * base_scale + 1.5),
        ))
        sa_large_move_prob = float(sa_params.get('large_move_prob', 0.0))
        tabu_large_move_prob = float(tabu_params.get('large_move_prob', 0.0))
        
        # Prepare algorithm dictionary (single-start heuristics)
        def sa_wrapper(inst):
            # Start from myopic baseline
            _, u0 = myopic_greedy(inst)
            cost, u_sa, _ = simulated_annealing(
                inst,
                u0,
                T0=sa_params.get('T0', None),
                alpha=sa_params.get('alpha', 0.95),
                max_iters=sa_params.get('max_iters', 500),
                max_no_improve=sa_params.get('max_no_improve', 200),
                seed=42,
                large_move_prob=sa_large_move_prob,
                time_limit=sa_time_limit,
            )
            return cost, u_sa
        
        def tabu_wrapper(inst):
            # Start from myopic baseline
            _, u0 = myopic_greedy(inst)
            cost, u_tabu, _ = tabu_search(
                inst,
                u0,
                tabu_tenure=tabu_params.get('tabu_tenure', 5),
                max_iters=tabu_params.get('max_iters', 200),
                neighborhood_size=tabu_params.get('neighborhood_size', 8),
                max_no_improve=tabu_params.get('max_no_improve', 100),
                seed=43,
                large_move_prob=tabu_large_move_prob,
                time_limit=tabu_time_limit,
            )
            return cost, u_tabu
        
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
        
        alg_dict: Dict[str, Callable[[Instance], Tuple[float, np.ndarray]]] = {}

        # Select baselines according to requested metrics
        if 'myopic' in baseline_set:
            alg_dict['myopic'] = myopic_greedy
        if 'static_prop' in baseline_set:
            alg_dict['static_prop'] = static_proportional
        if 'random' in baseline_set:
            alg_dict['random'] = lambda inst: random_feasible(inst, seed=42)

        # Heuristic constructors and MIP are always included
        alg_dict.update({
            'greedy': greedy_wrapper,
            'grasp': grasp_wrapper,
            'sa': sa_wrapper,
            'tabu': tabu_wrapper,
            'mip': mip_wrapper,
        })
        
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
                    'cost_lost': alg_results.get('cost_lost', None),
                    'cost_clipped': alg_results.get('cost_clipped', None),
                    'clipped_shipments': alg_results.get('clipped_shipments', None)
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

