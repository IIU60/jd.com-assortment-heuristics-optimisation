# Experiment Runner Guide

## Experiment Workflow

The new workflow is built around *size-specific* runners that each:

- Create a fresh experiment directory under `experiments/{name}/`
- Create `instances/` and `results/` subfolders
- Generate synthetic instances with the desired size
- Run all algorithms via `run_all_experiments`

## Size-Specific Runners

```bash
# Small: 5 FDC, 20 SKU
python -m src.experiments.run_small

# Medium: 20 FDC, 100 SKU
python -m src.experiments.run_medium

# Large: 80 FDC, 320 SKU
python -m src.experiments.run_large

# Massive: 100 FDC, 1000 SKU
python -m src.experiments.run_massive
```

Each script creates a directory such as:

```
experiments/
└── small_5fdc_20sku_T14_20250101-120000/
    ├── instances/
    │   └── small_00.json
    └── results/
        ├── all_results.csv
        ├── cost_comparison.png
        ├── service_level.png
        ├── runtime_comparison.png
        ├── cost_by_type.png
        └── {instance}_results.json
```

## Baseline Selection

All runners accept a `--baseline` flag (repeatable) to control which baselines
are evaluated:

```bash
# Run only random baseline (default)
python -m src.experiments.run_medium

# Run myopic + random
python -m src.experiments.run_medium --baseline myopic --baseline random

# Run all three baselines
python -m src.experiments.run_large --baseline myopic --baseline static_prop --baseline random
```

If no `--baseline` flags are given, only the **random** baseline is run
(`myopic` and `static_prop` are disabled by default).

## Viewing Results

```python
import pandas as pd

df = pd.read_csv('experiments/{name}/results/all_results.csv')
print(df.groupby('algorithm')['cost'].mean().sort_values())
```

