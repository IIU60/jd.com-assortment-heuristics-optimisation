"""Run a *small* experiment using the new experiments/<name> layout.

This script:
- Creates a fresh experiment directory under ``experiments/`` with a
  descriptive name encoding the size parameters.
- Generates a single synthetic instance with 5 FDCs and 20 SKUs.
- Saves the instance under ``instances/`` inside the experiment folder.
- Runs all algorithms and stores results under ``results/``.
"""

from pathlib import Path
from datetime import datetime
import argparse

from .run_all import run_all_experiments
from .run_experiment import setup_experiment
from ..model.instance_generator import generate_instance, save_instance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small experiment (5 FDC, 20 SKU) in a fresh experiments/ folder."
    )
    parser.add_argument(
        '--baseline',
        dest='baselines',
        action='append',
        choices=['myopic', 'static_prop', 'random'],
        help='Baseline algorithms to run (can be given multiple times). '
             'Default: random only.',
    )
    args = parser.parse_args()

    # Base simulation parameters for the "small" scenario
    n_fdcs = 5
    n_products = 20
    T = 14
    size_label = "small"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{size_label}_{n_fdcs}fdc_{n_products}sku_T{T}_{timestamp}"

    # Create experiment directory structure under experiments/
    instances_dir, results_dir = setup_experiment(experiment_name=experiment_name)

    # Generate a single synthetic instance for this experiment
    instance, _ = generate_instance(
        n_products=n_products,
        n_fdcs=n_fdcs,
        T=T,
        seed=42,
        level='small',
    )
    instance_path = Path(instances_dir) / f"{size_label}_00.json"
    save_instance(instance, str(instance_path))

    # Run all algorithms on the generated instance
    run_all_experiments(
        instances_dir=str(instances_dir),
        output_dir=str(results_dir),
        sa_params={
            'T0': 50000.0,
            'alpha': 0.95,
            'max_iters': 300,
            # Hard cap of 2 minutes per instance for SA (may stop earlier on non-improvement)
            'time_limit': 120.0,
        },
        tabu_params={
            'tabu_tenure': 5,
            'max_iters': 150,
            'neighborhood_size': 20,
            # Hard cap of 2 minutes per instance for Tabu (may stop earlier on non-improvement)
            'time_limit': 120.0,
        },
        # Hard cap of 2 minutes per instance for the exact MIP solver
        mip_max_time=120.0,
        baseline_names=args.baselines or ['myopic', 'random'],
    )


if __name__ == '__main__':
    main()

