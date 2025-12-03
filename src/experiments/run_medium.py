"""Run experiments on medium-sized instances with all algorithms including MIP."""

from .run_all import run_all_experiments


if __name__ == '__main__':
    # Run on medium instances with comprehensive settings
    run_all_experiments(
        instances_dir='instances',
        output_dir='results/medium_full',
        sa_params={
            'T0': 100000.0,
            'alpha': 0.95,
            'max_iters': 300
        },
        tabu_params={
            'tabu_tenure': 5,
            'max_iters': 150,
            'neighborhood_size': 25
        },
        mip_max_time=180.0  # 3 minutes max for MIP on medium instances
    )

