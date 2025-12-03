"""CLI experiment runner with organized directory structure."""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import time

from .run_all import run_all_experiments


def setup_experiment(
    instances_source: str,
    experiment_name: Optional[str] = None,
    copy_instances: bool = True
) -> tuple[Path, Path]:
    """
    Set up experiment directories.
    
    Args:
        instances_source: Path to source instances (directory, glob pattern, or single file)
        experiment_name: Name for experiment (default: based on source or timestamp)
        copy_instances: If True, copy instances to experiment directory
    
    Returns:
        Tuple of (instances_dir, output_dir)
    """
    import glob
    
    # Find instance files
    source_path = Path(instances_source)
    
    if source_path.is_file() and source_path.suffix == '.json':
        # Single file
        instance_files = [str(source_path)]
        base_name = source_path.stem
    elif source_path.is_dir():
        # Directory - get all JSON files
        pattern = str(source_path / '*.json')
        instance_files = glob.glob(pattern)
        base_name = source_path.name
    elif '*' in instances_source:
        # Glob pattern
        instance_files = glob.glob(instances_source)
        # Extract meaningful name from pattern
        base_name = instances_source.replace('*', '').replace('/', '_').replace('\\', '_').strip('_')
        if not base_name or base_name == '_':
            base_name = 'filtered'
    else:
        raise ValueError(f"Invalid instances source: {instances_source}")
    
    if not instance_files:
        raise ValueError(f"No instance files found matching: {instances_source}")
    
    # Determine experiment name
    if experiment_name is None:
        experiment_name = base_name
    
    # Create experiment directories
    base_dir = Path('experiments') / experiment_name
    instances_dir = base_dir / 'instances'
    output_dir = base_dir / 'results'
    
    instances_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy instances if requested
    if copy_instances:
        for src_file in instance_files:
            dst_file = instances_dir / Path(src_file).name
            shutil.copy2(src_file, dst_file)
        print(f"Copied {len(instance_files)} instances to {instances_dir}")
    else:
        # Use source directory directly (only works for directory input)
        if source_path.is_dir():
            instances_dir = source_path
        else:
            # For glob patterns, we need to copy
            for src_file in instance_files:
                dst_file = instances_dir / Path(src_file).name
                shutil.copy2(src_file, dst_file)
            print(f"Copied {len(instance_files)} instances to {instances_dir}")
    
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
    
    # Instance selection
    parser.add_argument(
        'instances',
        type=str,
        help='Path to instances: directory, glob pattern (e.g., "instances/large_*.json"), or single file'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (default: based on source or timestamp)'
    )
    
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Do not copy instances (use source directory directly)'
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
    print(f"Setting up experiment: {args.name or 'unnamed'}")
    print(f"{'='*70}")
    
    instances_dir, output_dir = setup_experiment(
        args.instances,
        experiment_name=args.name,
        copy_instances=not args.no_copy
    )
    
    print(f"Instances directory: {instances_dir}")
    print(f"Results directory: {output_dir}")
    
    # Prepare parameters
    sa_params = {
        'alpha': args.sa_alpha,
        'max_iters': args.sa_iters,
        'max_no_improve': 100
    }
    if args.sa_t0 is not None:
        sa_params['T0'] = args.sa_t0
    
    tabu_params = {
        'tabu_tenure': args.tabu_tenure,
        'max_iters': args.tabu_iters,
        'neighborhood_size': args.tabu_nh_size
    }
    
    print(f"\nAlgorithm parameters:")
    sa_t0_str = f"T0={sa_params.get('T0', 'adaptive')}"
    print(f"  SA: {sa_t0_str}, alpha={sa_params['alpha']}, iters={sa_params['max_iters']}")
    print(f"  Tabu: tenure={tabu_params['tabu_tenure']}, iters={tabu_params['max_iters']}, nh_size={tabu_params['neighborhood_size']}")
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

