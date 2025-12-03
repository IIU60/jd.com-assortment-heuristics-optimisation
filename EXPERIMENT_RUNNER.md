# Experiment Runner Guide

## New CLI Experiment Runner

The `run_experiment.py` script provides a flexible way to run experiments with organized directory structure.

## Features

- **Organized directories**: Creates `experiments/{name}/instances/` and `experiments/{name}/results/`
- **Instance filtering**: Use glob patterns to select specific instances
- **Automatic copying**: Copies selected instances to experiment directory
- **Full parameter control**: All algorithm parameters via CLI
- **Clean organization**: Each experiment is self-contained

## Basic Usage

### Run on specific instances (glob pattern)

```bash
# Run only on large instances
python -m src.experiments.run_experiment "instances/large_*.json" --name large_test

# Run only on medium test instances
python -m src.experiments.run_experiment "instances/medium_test_*.json" --name medium_test
```

### Run on all instances in a directory

```bash
python -m src.experiments.run_experiment instances/ --name full_run
```

### Run on a single instance

```bash
python -m src.experiments.run_experiment instances/large_00.json --name single_test
```

## Parameter Customization

### Simulated Annealing

```bash
python -m src.experiments.run_experiment instances/ \
  --sa-t0 200000 \
  --sa-alpha 0.96 \
  --sa-iters 500
```

### Tabu Search

```bash
python -m src.experiments.run_experiment instances/ \
  --tabu-tenure 8 \
  --tabu-iters 300 \
  --tabu-nh-size 50
```

### MIP Solver

```bash
python -m src.experiments.run_experiment instances/ \
  --mip-time 300  # 5 minutes max
```

## Complete Example: Large Instance Run

```bash
python -m src.experiments.run_experiment "instances/large_*.json" \
  --name large_full \
  --sa-t0 200000 \
  --sa-alpha 0.96 \
  --sa-iters 500 \
  --tabu-tenure 8 \
  --tabu-iters 300 \
  --tabu-nh-size 50 \
  --mip-time 300
```

## Output Structure

```
experiments/
└── {experiment_name}/
    ├── instances/          # Copied instance files
    │   ├── large_00.json
    │   ├── large_01.json
    │   └── ...
    └── results/            # All results
        ├── all_results.csv
        ├── cost_comparison.png
        ├── service_level.png
        ├── runtime_comparison.png
        ├── cost_by_type.png
        └── {instance}_results.json  # Per-instance results
```

## Default Parameters

- **SA**: T0=100000, alpha=0.95, iters=500
- **Tabu**: tenure=5, iters=200, nh_size=30
- **MIP**: max_time=120s

## Tips

1. **Large instances**: MIP will likely timeout - that's expected. Focus on heuristics.
2. **Quick tests**: Use `--sa-iters 50 --tabu-iters 50` for faster runs
3. **No copying**: Use `--no-copy` to use source directory directly (faster but less organized)
4. **Experiment names**: Use descriptive names like `large_200x50` or `medium_20x5`

## Viewing Results

```python
import pandas as pd

df = pd.read_csv('experiments/{name}/results/all_results.csv')
print(df.groupby('algorithm')['cost'].mean().sort_values())
```

