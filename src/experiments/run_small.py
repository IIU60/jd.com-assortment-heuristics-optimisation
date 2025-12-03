"""Run experiments on small instances only."""

from .run_all import run_all_experiments


if __name__ == '__main__':
    # Run on small instances only
    run_all_experiments(
        instances_dir='instances',
        output_dir='results/small',
        sa_params={'T0': 50000.0, 'alpha': 0.95, 'max_iters': 300},
        tabu_params={'tabu_tenure': 5, 'max_iters': 150, 'neighborhood_size': 20}
    )

