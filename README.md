# JD.com Assortment Heuristics Optimization

Multi-echelon distribution optimization project for JD.com's supply chain, implementing various heuristic algorithms (Simulated Annealing, Tabu Search, GRASP) and baseline methods.

## Project Structure

```
jd.com-assortment-heuristics-optimisation/
├── instances/          # Generated instance files (JSON)
├── src/
│   ├── model/         # Core model components
│   │   ├── instance.py           # Instance data structure
│   │   ├── result.py             # Simulation result structure
│   │   ├── simulator.py          # Core simulator
│   │   ├── instance_generator.py # Instance generation
│   │   └── mip_solver.py        # Exact MIP solver (small instances)
│   ├── heuristics/    # Heuristic algorithms
│   │   ├── utils.py             # Solution representation helpers
│   │   ├── neighborhoods.py     # Neighborhood move operators
│   │   ├── construction.py      # Greedy and GRASP constructors
│   │   ├── sa.py                # Simulated Annealing
│   │   └── tabu.py              # Tabu Search
│   ├── baselines/     # Baseline algorithms
│   │   ├── myopic.py           # Myopic greedy
│   │   ├── static_prop.py      # Static proportional
│   │   └── random_feasible.py  # Random feasible
│   └── experiments/   # Experiment orchestration
│       ├── run_all.py          # Main experiment runner
│       ├── run_small.py        # Small instance experiments
│       ├── tune_params.py      # Parameter tuning
│       └── plots.py            # Plotting functions
├── results/           # Experiment results (JSON/CSV + plots)
├── latex/             # Report files
├── requirements.txt   # Dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate Instances

```python
from src.model import generate_instance_set

generate_instance_set(n_small=10, n_medium=20)
```

### Run Experiments

```python
from src.experiments.run_all import run_all_experiments

run_all_experiments(
    instances_dir='instances',
    output_dir='results'
)
```

Or use the command line:

```bash
python -m src.experiments.run_small
```

### Run Parameter Tuning

```python
from src.experiments.tune_params import tune_sa, tune_tabu, load_instances_for_tuning

instances = load_instances_for_tuning('instances', n_instances=3)
tune_sa(instances)
tune_tabu(instances)
```

## Model Formulation

The problem optimizes shipment decisions `u[t, i, j]` (units of product `i` shipped from RDC to FDC `j` in period `t`) to minimize total cost:

- **Transfer cost**: `r[i, j] * u[t, i, j]`
- **Cross-fulfillment cost**: `c * y_fdc[t, i, j]` (RDC directly serving FDC demand)
- **Lost sales cost**: `s * (z_fdc[t, i, j] + z_rdc[t, i])`

Subject to:
- RDC inventory balance
- FDC inventory balance
- Outbound capacity constraints
- FDC capacity constraints
- Demand satisfaction

## Algorithms

### Baselines
- **Myopic Greedy**: Allocates based on current period demand
- **Static Proportional**: Pre-computes fixed proportions from total demand
- **Random Feasible**: Random feasible shipment plans

### Heuristics
- **Greedy Constructor**: Uses future cumulative demand for priority scores
- **GRASP**: Greedy Randomized Adaptive Search Procedure
- **Simulated Annealing**: Temperature-based acceptance
- **Tabu Search**: Memory-based local search

### Exact Solver
- **MIP Solver**: Exact solution using PuLP (only for small instances: N≤5, J≤3, T≤4)

## Usage Examples

### Create and Evaluate an Instance

```python
from src.model import generate_instance, simulate
import numpy as np

instance, u0 = generate_instance(n_products=10, n_fdcs=3, T=5, seed=42)
result = simulate(instance, u0, check_feasibility=True)
print(f"Total cost: {result.total_cost}")
```

### Run a Heuristic

```python
from src.heuristics import simulated_annealing
from src.heuristics.utils import random_feasible_u

u0 = random_feasible_u(instance, seed=42)
best_cost, best_u, log = simulated_annealing(
    instance, u0,
    T0=100000.0,
    alpha=0.95,
    max_iters=500
)
```

## Testing

Run basic integration test:

```bash
python test_basic.py
```

## License

This project is for academic/research purposes.
