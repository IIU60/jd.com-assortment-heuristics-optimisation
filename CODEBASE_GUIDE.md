# Complete Codebase Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module-by-Module Explanation](#module-by-module-explanation)
4. [How Components Interact](#how-components-interact)
5. [How to Run Everything](#how-to-run-everything)
6. [Example Workflows](#example-workflows)

---

## Overview

This codebase solves a **multi-echelon distribution optimization problem** for JD.com's supply chain. The problem is:

- **1 RDC (Regional Distribution Center)** - central warehouse
- **Multiple FDCs (Fulfillment Distribution Centers)** - local warehouses  
- **Multiple products** - different SKUs
- **Multiple time periods** - planning horizon

**Goal**: Decide how much of each product to ship from RDC to each FDC in each period (`u[t, i, j]`) to minimize total cost.

**Costs**:
- Transfer cost: shipping from RDC to FDC
- Cross-fulfillment cost: RDC directly serving FDC demand (penalty)
- Lost sales cost: unmet demand

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Problem Instance                          │
│  (demand, inventory, capacities, costs)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Shipment Plan u[t, i, j]                       │
│  (decision variables: how much to ship where and when)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Simulator                                 │
│  - Enforces constraints (clips infeasible shipments)       │
│  - Simulates inventory flows                                │
│  - Computes total cost                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SimulationResult                                │
│  (costs, inventories, fulfillment, lost sales)             │
└─────────────────────────────────────────────────────────────┘
```

**Algorithms** generate shipment plans → **Simulator** evaluates them → **Best plan** wins!

---

## Module-by-Module Explanation

### 📦 `src/model/` - Core Model Components

#### `instance.py` - Problem Data Structure

**What it does**: Defines the `Instance` dataclass that holds all problem parameters.

**Key components**:
```python
Instance(
    num_products=10,           # N products
    num_fdcs=3,                # J FDCs  
    T=5,                       # T time periods
    demand_fdc=...,            # shape (T, N, J) - demand at each FDC
    demand_rdc=...,            # shape (T, N) - demand at RDC
    initial_inventory_rdc=..., # shape (N,) - starting RDC stock
    initial_inventory_fdc=..., # shape (N, J) - starting FDC stock
    replenishment=...,         # shape (T, N) - new stock arriving
    outbound_capacity=...,     # shape (T,) - max shipments per period
    fdc_capacity=...,         # shape (J,) - max storage per FDC
    transfer_cost=...,        # shape (N, J) - shipping costs
    rdc_fulfillment_cost=3.0, # penalty for cross-fulfillment
    lost_sale_cost=10.0      # penalty for lost sales
)
```

**How it works**: 
- Uses numpy arrays for efficient computation
- Has `validate()` method to check all shapes and non-negativity
- All field names match the mathematical model exactly

#### `simulator.py` - Core Evaluation Engine

**What it does**: Takes a shipment plan and simulates what happens, computing total cost.

**How it works** (step by step for each period `t`):

1. **Replenishment**: Add new stock to RDC
   ```python
   inventory_rdc[t, :] += replenishment[t, :]
   ```

2. **Shipments**: Try to ship from RDC to FDCs
   - Clips to available RDC inventory
   - Clips to FDC capacity limits
   - Clips to outbound capacity
   - Records how much was actually shipped vs requested

3. **FDC Fulfillment**: Each FDC tries to meet its demand
   - First uses local FDC inventory
   - Remaining demand → cross-fulfillment from RDC (penalty cost)
   - Still remaining → lost sales (big penalty)

4. **RDC Fulfillment**: RDC serves its own demand

5. **Inventory Update**: Carry remaining inventory to next period

6. **Cost Calculation**:
   ```python
   total_cost = transfer_cost + cross_fulfillment_cost + lost_sales_cost
   ```

**Key feature**: The simulator **clips** infeasible shipments automatically, so algorithms can propose any plan and it will be made feasible.

#### `result.py` - Simulation Output

**What it does**: Stores all results from a simulation run.

**Contains**:
- `total_cost`, `cost_transfer`, `cost_cross`, `cost_lost`
- `inventory_rdc`, `inventory_fdc` (over time)
- `shipments` (actual executed shipments)
- `fdc_local_fulfilled`, `fdc_from_rdc_fulfilled`, `lost_fdc`
- `rdc_fulfilled`, `lost_rdc`
- `clipped_shipments` (how much was clipped)

#### `instance_generator.py` - Synthetic Data Generation

**What it does**: Creates realistic problem instances for testing.

**Functions**:
- `generate_instance()`: Creates one instance with specified size
- `save_instance()` / `load_instance()`: Save/load to JSON
- `generate_instance_set()`: Generate many instances (small + medium)

**How it generates**:
- Random demand with temporal fluctuations
- Initial inventory = 70% of total demand
- Outbound capacity = 60% of period demand (creates scarcity)
- FDC capacity = 80% of total demand (creates constraints)
- Fixed costs: transfer=1, cross-fulfillment=3, lost sales=10

#### `mip_solver.py` - Exact Optimal Solution

**What it does**: Solves small instances to optimality using Mixed Integer Programming.

**How it works**:
- Uses PuLP library
- Defines all decision variables (shipments, inventories, fulfillments)
- Adds all constraints from the model
- Solves to get optimal cost and shipment plan

**Limitation**: Only runs on very small instances (N≤5, J≤3, T≤4) due to computational complexity.

---

### 🎯 `src/baselines/` - Simple Comparison Algorithms

These are simple algorithms used as baselines to compare against the heuristics.

#### `myopic.py` - Myopic Greedy

**What it does**: Looks only at current period demand, allocates shipments to cover it.

**How it works**:
- For each period `t`:
  - Compute demand shortfall at each FDC: `demand[t] - current_inventory`
  - Allocate RDC inventory proportionally to shortfall
  - Respect capacity constraints

**Pros**: Fast, simple
**Cons**: Doesn't look ahead, may create future problems

#### `static_prop.py` - Static Proportional

**What it does**: Pre-computes fixed proportions from total demand, allocates proportionally each period.

**How it works**:
- Before horizon: compute `proportion[i, j] = total_demand[i, j] / total_demand[i]`
- Each period: allocate RDC inventory according to these fixed proportions

**Pros**: Very simple, consistent
**Cons**: Ignores temporal dynamics

#### `random_feasible.py` - Random Feasible

**What it does**: Generates random feasible shipment plans.

**How it works**:
- Random proportions using Dirichlet distribution
- Random allocation amounts
- Respects constraints

**Pros**: Provides random baseline
**Cons**: Not optimized at all

---

### 🧠 `src/heuristics/` - Advanced Search Algorithms

#### `utils.py` - Helper Functions

**Functions**:
- `copy_u(u)`: Deep copy of shipment plan
- `random_feasible_u(instance)`: Generate random feasible plan
- `clamp_u_to_feasibility(instance, u)`: Enforce constraints
- `evaluate_u(instance, u)`: Wrapper around simulator, returns (cost, metrics)

#### `neighborhoods.py` - Move Operators

**What it does**: Defines how to modify a shipment plan to create a "neighbor" solution.

**Move types**:

1. **Time-shift move**: `time_shift_move(u, i, j, t_from, t_to, delta)`
   - Reduce shipment at period `t_from`, increase at `t_to`
   - Example: Ship 10 units earlier/later

2. **FDC-swap move**: `fdc_swap_move(u, i, j_from, j_to, t, delta)`
   - Reduce shipment to FDC `j_from`, increase to FDC `j_to`
   - Example: Reallocate from one FDC to another

3. **Magnitude tweak**: `magnitude_tweak(u, i, j, t, delta)`
   - Add/subtract `delta` from a shipment
   - Example: Increase/decrease shipment amount

4. **`generate_neighbor()`**: Randomly selects a move type and applies it
   - Used by SA and Tabu Search

#### `construction.py` - Initial Solution Builders

**What it does**: Creates good starting solutions for local search.

**Functions**:

1. **`greedy_constructor()`**:
   - Computes future cumulative demand for each (product, FDC)
   - Priority score = future_demand / (transfer_cost + 1)
   - Allocates shipments in order of descending score
   - Respects constraints

2. **`grasp_constructor()`** - GRASP (Greedy Randomized Adaptive Search Procedure):
   - Similar to greedy, but:
   - Builds Restricted Candidate List (RCL) of top candidates
   - Randomly selects from RCL (controlled by `alpha` parameter)
   - `alpha=0` = pure random, `alpha=1` = pure greedy

#### `sa.py` - Simulated Annealing

**What it does**: Temperature-based probabilistic local search.

**How it works**:
1. Start with initial solution `u0`
2. For each iteration:
   - Generate neighbor `u'`
   - Compute cost difference: `Δ = cost(u') - cost(u)`
   - Accept if `Δ ≤ 0` (improvement) OR with probability `exp(-Δ/T)` (worse solution)
   - Temperature `T` decreases: `T *= alpha`
3. Return best solution found

**Parameters**:
- `T0`: Initial temperature (high = accept many bad moves)
- `alpha`: Cooling rate (0.9-0.98 typical)
- `max_iters`: Number of iterations

**Why it works**: High temperature allows exploration, low temperature focuses on exploitation.

#### `tabu.py` - Tabu Search

**What it does**: Memory-based local search that avoids revisiting recent solutions.

**How it works**:
1. Start with initial solution
2. For each iteration:
   - Sample `neighborhood_size` random neighbors
   - Evaluate all neighbors
   - Choose best non-tabu neighbor (or best if all tabu but improves global best)
   - Add chosen move to tabu list (for `tabu_tenure` iterations)
3. Return best solution found

**Parameters**:
- `tabu_tenure`: How long moves stay tabu (5-15 typical)
- `neighborhood_size`: How many neighbors to sample (20-80 typical)
- `max_iters`: Number of iterations

**Why it works**: Tabu list prevents cycling and forces exploration of new regions.

---

### 🧪 `src/experiments/` - Experiment Framework

#### `run_all.py` - Main Experiment Runner

**What it does**: Orchestrates running all algorithms on all instances.

**Key functions**:

1. **`run_baselines_on_instance()`**: Runs all 3 baselines on one instance
2. **`run_all_algorithms_on_instance()`**: Runs any set of algorithms
3. **`run_all_experiments()`**: Main function that:
   - Loads all instances from directory
   - Runs all algorithms (baselines + heuristics)
   - Saves results to JSON/CSV
   - Creates plots

**Output format**: Standardized JSON with:
```json
{
  "algorithm_name": {
    "cost": 1234.56,
    "runtime": 0.5,
    "cost_transfer": 100.0,
    "cost_cross": 200.0,
    "cost_lost": 934.56,
    "shipments": [...],
    ...
  }
}
```

#### `tune_params.py` - Parameter Tuning

**What it does**: Systematically tests different parameter combinations to find best settings.

**Functions**:
- `tune_sa()`: Tests different `T0` and `alpha` values
- `tune_tabu()`: Tests different `tabu_tenure` and `neighborhood_size` values
- Uses pandas to collect and summarize results
- Saves to CSV

#### `plots.py` - Visualization

**What it does**: Creates plots comparing algorithms.

**Plot types**:
- Cost comparison (boxplots)
- Cost by instance type (bar charts)
- Service level comparison
- Runtime comparison

#### `run_small.py` - Quick Test Script

**What it does**: Convenience script to run experiments on small instances only.

---

## How Components Interact

### Typical Flow:

```
1. Generate Instance
   instance_generator.generate_instance()
   ↓
2. Create Initial Solution
   - Option A: random_feasible_u()
   - Option B: greedy_constructor()
   - Option C: grasp_constructor()
   ↓
3. Run Algorithm
   - simulated_annealing(instance, u0, ...)
   - tabu_search(instance, u0, ...)
   - OR baseline: myopic_greedy(instance)
   ↓
4. Evaluate Solution
   simulate(instance, best_u)
   ↓
5. Get Results
   SimulationResult with costs, metrics
```

### Algorithm → Simulator Interaction:

```python
# Algorithm proposes a plan
u = np.zeros((T, N, J))
u[0, 0, 0] = 100.0  # Ship 100 units

# Simulator evaluates it
result = simulate(instance, u)

# Simulator automatically:
# - Clips if u exceeds RDC inventory
# - Clips if u exceeds FDC capacity  
# - Clips if u exceeds outbound capacity
# - Simulates demand fulfillment
# - Computes total cost

# Algorithm gets feedback
cost = result.total_cost
# Algorithm modifies u and tries again
```

---

## How to Run Everything

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Instances

```python
from src.model import generate_instance_set

# Generate 10 small + 20 medium instances
generate_instance_set(n_small=10, n_medium=20)
# Files saved to instances/small_00.json, instances/medium_00.json, etc.
```

Or generate a single instance:

```python
from src.model import generate_instance, save_instance

instance, u0 = generate_instance(
    n_products=10,
    n_fdcs=3,
    T=5,
    seed=42
)
save_instance(instance, 'instances/my_instance.json')
```

### 3. Run Basic Test

```bash
python test_basic.py
```

This tests:
- Instance creation and validation
- Simulator with zero shipments
- All baselines
- Greedy constructor

### 4. Run a Single Algorithm

```python
from src.model import load_instance, simulate
from src.heuristics import simulated_annealing
from src.heuristics.utils import random_feasible_u

# Load instance
instance = load_instance('instances/small_00.json')

# Create initial solution
u0 = random_feasible_u(instance, seed=42)

# Run Simulated Annealing
best_cost, best_u, cost_log = simulated_annealing(
    instance, u0,
    T0=100000.0,
    alpha=0.95,
    max_iters=500,
    verbose=True  # Print progress
)

print(f"Best cost: {best_cost:.2f}")

# Evaluate final solution
result = simulate(instance, best_u, check_feasibility=True)
print(f"Transfer cost: {result.cost_transfer:.2f}")
print(f"Cross-fulfillment: {result.cost_cross:.2f}")
print(f"Lost sales: {result.cost_lost:.2f}")
```

### 5. Run All Algorithms on All Instances

```python
from src.experiments.run_all import run_all_experiments

run_all_experiments(
    instances_dir='instances',
    output_dir='results',
    sa_params={'T0': 100000.0, 'alpha': 0.95, 'max_iters': 500},
    tabu_params={'tabu_tenure': 5, 'max_iters': 200, 'neighborhood_size': 30}
)
```

Or use command line:

```bash
python -m src.experiments.run_small
```

This will:
- Load all instances
- Run all baselines + heuristics
- Save results to `results/`
- Create comparison plots

### 6. Parameter Tuning

```python
from src.experiments.tune_params import tune_sa, tune_tabu, load_instances_for_tuning

# Load a few instances
instances = load_instances_for_tuning('instances', n_instances=3)

# Tune SA parameters
sa_results = tune_sa(
    instances,
    T0_values=[50000.0, 100000.0, 200000.0],
    alpha_values=[0.90, 0.95, 0.98],
    n_runs=5,
    max_iters=500
)

# Tune Tabu parameters
tabu_results = tune_tabu(
    instances,
    tabu_tenures=[3, 5, 8, 12],
    neighborhood_sizes=[20, 40, 80],
    n_runs=5,
    max_iters=200
)
```

Results saved to `results/sa_tuning_results.csv` and `results/tabu_tuning_results.csv`.

### 7. Compare Algorithms

```python
from src.experiments.run_all import run_baselines_on_instance, load_results
from src.model import load_instance
from src.experiments.plots import plot_cost_comparison

# Load instance
instance = load_instance('instances/small_00.json')

# Run baselines
results = run_baselines_on_instance(instance)

# Create plot
plot_cost_comparison(results, 'results/cost_comparison.png')
```

---

## Example Workflows

### Workflow 1: Quick Test on Small Instance

```python
from src.model import generate_instance, simulate
from src.baselines import myopic_greedy

# Generate small test instance
instance, u0 = generate_instance(n_products=5, n_fdcs=2, T=3, seed=42)

# Run myopic baseline
cost, shipments = myopic_greedy(instance)
print(f"Myopic cost: {cost:.2f}")

# Evaluate
result = simulate(instance, shipments)
print(f"Lost sales cost: {result.cost_lost:.2f}")
```

### Workflow 2: Compare Construction Heuristics

```python
from src.model import load_instance
from src.heuristics import greedy_constructor, grasp_constructor
from src.heuristics.utils import evaluate_u

instance = load_instance('instances/small_00.json')

# Greedy
u_greedy = greedy_constructor(instance)
cost_greedy, _ = evaluate_u(instance, u_greedy)

# GRASP with different alpha values
for alpha in [0.3, 0.5, 0.7, 1.0]:
    u_grasp = grasp_constructor(instance, alpha=alpha, seed=42)
    cost_grasp, _ = evaluate_u(instance, u_grasp)
    print(f"GRASP alpha={alpha}: {cost_grasp:.2f}")

print(f"Greedy: {cost_greedy:.2f}")
```

### Workflow 3: Full Experiment Pipeline

```python
# 1. Generate instances
from src.model import generate_instance_set
generate_instance_set(n_small=10, n_medium=20)

# 2. Tune parameters
from src.experiments.tune_params import tune_sa, load_instances_for_tuning
instances = load_instances_for_tuning('instances', n_instances=3)
tune_sa(instances)

# 3. Run full experiments
from src.experiments.run_all import run_all_experiments
run_all_experiments(
    instances_dir='instances',
    output_dir='results',
    sa_params={'T0': 100000.0, 'alpha': 0.95, 'max_iters': 500},
    tabu_params={'tabu_tenure': 5, 'max_iters': 200, 'neighborhood_size': 30}
)

# 4. Analyze results
import pandas as pd
df = pd.read_csv('results/all_results.csv')
print(df.groupby('algorithm')['cost'].mean())
```

### Workflow 4: Exact Solution for Small Instance

```python
from src.model import generate_instance
from src.model.mip_solver import solve_exact

# Generate very small instance
instance, _ = generate_instance(n_products=3, n_fdcs=2, T=3, seed=42)

# Solve to optimality
optimal_cost, optimal_u = solve_exact(instance, time_limit=60)

if optimal_cost is not None:
    print(f"Optimal cost: {optimal_cost:.2f}")
    print(f"Optimal shipments:\n{optimal_u}")
else:
    print("Instance too large or solver failed")
```

---

## Key Design Decisions

1. **Numpy arrays everywhere**: Efficient computation, clear shapes
2. **Simulator clips infeasible plans**: Algorithms can propose anything
3. **Modular structure**: Easy to add new algorithms
4. **Standardized output**: All algorithms return same format
5. **Comprehensive validation**: Instance validation, feasibility checks

---

## Tips for Using the Codebase

1. **Start small**: Test with `n_products=5, n_fdcs=2, T=3` first
2. **Use verbose mode**: Set `verbose=True` in SA/Tabu to see progress
3. **Check feasibility**: Always use `check_feasibility=True` in simulator for debugging
4. **Save intermediate results**: Use `save_results()` to avoid re-running
5. **Visualize**: Use plots to understand algorithm performance
6. **Tune parameters**: Don't use default parameters blindly - tune them!

---

## Troubleshooting

**Import errors**: Make sure you're in the project root and `src/` is in Python path:
```python
import sys
sys.path.append('.')
from src.model import ...
```

**Memory issues**: Reduce instance size or use smaller `neighborhood_size` in Tabu Search

**Slow performance**: 
- Reduce `max_iters` in SA/Tabu
- Use smaller instances for testing
- Consider using exact solver only for very small instances

**Infeasible solutions**: Check that instance parameters are reasonable (capacities not too tight)

