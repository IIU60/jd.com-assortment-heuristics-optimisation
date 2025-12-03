"""CLI experiment runner with organized directory structure."""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime

from .run_all import run_all_experiments


def setup_experiment(
    experiment_name: Optional[str] = None,
) -> tuple[Path, Path]:
    """
    Set up a *new* experiment directory under ``experiments/``.
    
    This follows the new standard layout:
    
        experiments/<experiment_name>/
            instances/
            results/
    
    The function ensures that the base experiment directory does not already
    exist to avoid accidentally overwriting previous runs. If no explicit
    ``experiment_name`` is provided, a timestamped name is generated.
    
    Args:
        experiment_name: Optional explicit experiment name. If omitted, a
            descriptive timestamp-based name is used.
    
    Returns:
        Tuple of (instances_dir, output_dir)
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    base_dir = Path('experiments') / experiment_name
    if base_dir.exists():
        raise FileExistsError(
            f"Experiment directory '{base_dir}' already exists. "
            f"Please choose a different experiment name."
        )
    
    instances_dir = base_dir / 'instances'
    output_dir = base_dir / 'results'
    
    instances_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    
    return instances_dir, output_dir


def main():
    """CLI entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description='Run optimization experiments on instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on large instances only
  python -m src.experiments.run_experiment instances/large_*.json --name large_test

  # Run on all instances in a directory
  python -m src.experiments.run_experiment instances/medium_test_*.json --name medium

  # Run with custom parameters
  python -m src.experiments.run_experiment instances/ --name test \\
    --sa-t0 200000 --sa-alpha 0.96 --sa-iters 500 \\
    --tabu-tenure 8 --tabu-iters 300 --tabu-nh-size 50 \\
    --mip-time 300
        """
    )
    
    # SA parameters
    parser.add_argument(
        '--sa-t0',
        type=float,
        default=None,
        help='SA initial temperature (default: None, uses adaptive T0 to achieve 90%% acceptance rate)'
    )
    
    parser.add_argument(
        '--sa-alpha',
        type=float,
        default=0.95,
        help='SA cooling rate (default: 0.95)'
    )
    
    parser.add_argument(
        '--sa-iters',
        type=int,
        default=500,
        help='SA max iterations (default: 500)'
    )
    
    parser.add_argument(
        '--sa-num-starts',
        type=int,
        default=3,
        help='SA number of diversified starting solutions (default: 3)'
    )
    
    parser.add_argument(
        '--sa-large-move-prob',
        type=float,
        default=0.1,
        help='SA probability of applying a large destroy-repair move (default: 0.1)'
    )
    parser.add_argument(
        '--sa-time-limit',
        type=float,
        default=None,
        help='SA time limit in seconds per instance (default: None, uses size-based heuristic)'
    )
    
    # Tabu parameters
    parser.add_argument(
        '--tabu-tenure',
        type=int,
        default=5,
        help='Tabu tenure (default: 5)'
    )
    
    parser.add_argument(
        '--tabu-iters',
        type=int,
        default=200,
        help='Tabu max iterations (default: 200)'
    )
    
    parser.add_argument(
        '--tabu-nh-size',
        type=int,
        default=10,
        help='Tabu neighborhood size (default: 10)'
    )
    
    parser.add_argument(
        '--tabu-num-starts',
        type=int,
        default=2,
        help='Tabu number of diversified starting solutions (default: 2)'
    )
    
    parser.add_argument(
        '--tabu-large-move-prob',
        type=float,
        default=0.1,
        help='Tabu probability of applying a large destroy-repair move (default: 0.1)'
    )
    parser.add_argument(
        '--tabu-time-limit',
        type=float,
        default=None,
        help='Tabu time limit in seconds per instance (default: None, uses size-based heuristic)'
    )
    
    # MIP parameters
    parser.add_argument(
        '--mip-time',
        type=float,
        default=120.0,
        help='MIP max time limit in seconds (default: 120.0)'
    )
    
    # Output
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Set up experiment directories
    print(f"\n{'='*70}")
    print(f"Setting up experiment directory")
    print(f"{'='*70}")
    
    # The new standard is that experiment-specific scripts (e.g., run_small,
    # run_medium, run_large) are responsible for generating instances inside
    # the experiment directory. This CLI now only creates an empty experiment
    # folder and reports its locations so that the user or other tooling can
    # populate instances as needed.
    instances_dir, output_dir = setup_experiment(experiment_name=None)
    
    print(f"Instances directory: {instances_dir}")
    print(f"Results directory: {output_dir}")
    
    # Prepare parameters
    sa_params = {
        'alpha': args.sa_alpha,
        'max_iters': args.sa_iters,
        'max_no_improve': 100,
        'large_move_prob': args.sa_large_move_prob,
    }
    if args.sa_t0 is not None:
        sa_params['T0'] = args.sa_t0
    if args.sa_time_limit is not None:
        sa_params['time_limit'] = args.sa_time_limit
    
    tabu_params = {
        'tabu_tenure': args.tabu_tenure,
        'max_iters': args.tabu_iters,
        'neighborhood_size': args.tabu_nh_size,
        'large_move_prob': args.tabu_large_move_prob,
    }
    if args.tabu_time_limit is not None:
        tabu_params['time_limit'] = args.tabu_time_limit
    
    print(f"\nAlgorithm parameters:")
    sa_t0_str = f"T0={sa_params.get('T0', 'adaptive')}"
    sa_time_str = sa_params.get('time_limit', 'auto')
    tabu_time_str = tabu_params.get('time_limit', 'auto')
    print(f"  SA: {sa_t0_str}, alpha={sa_params['alpha']}, iters={sa_params['max_iters']}, time_limit={sa_time_str}s")
    print(f"  Tabu: tenure={tabu_params['tabu_tenure']}, iters={tabu_params['max_iters']}, nh_size={tabu_params['neighborhood_size']}, time_limit={tabu_time_str}s")
    print(f"  MIP: max_time={args.mip_time}s")
    
    # Run experiments
    print(f"\n{'='*70}")
    print("Running experiments...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    run_all_experiments(
        instances_dir=str(instances_dir),
        output_dir=str(output_dir),
        sa_params=sa_params,
        tabu_params=tabu_params,
        mip_max_time=args.mip_time
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Experiment complete!")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

