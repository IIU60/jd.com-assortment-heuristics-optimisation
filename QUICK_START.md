# Quick Start Guide

## 🚀 Fastest Way to Get Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Test Instances
```python
from src.model import generate_instance_set
generate_instance_set(n_small=5, n_medium=10)
```

### 3. Run Basic Test
```bash
python test_basic.py
```

### 4. Run Full Experiments
```bash
python -m src.experiments.run_small
python -m src.experiments.run_medium
python -m src.experiments.run_large
python -m src.experiments.run_massive  # 100 FDC, 1000 SKU stress test
```

---

## 📝 Common Code Snippets

### Create and Evaluate an Instance
```python
from src.model import generate_instance, simulate
import numpy as np

# Create instance
instance, u0 = generate_instance(n_products=10, n_fdcs=3, T=5, seed=42)

# Zero shipments (worst case)
result = simulate(instance, u0, check_feasibility=True)
print(f"Cost with no shipments: {result.total_cost:.2f}")
```

### Run a Baseline
```python
from src.baselines import myopic_greedy

cost, shipments = myopic_greedy(instance)
print(f"Myopic cost: {cost:.2f}")
```

### Run Simulated Annealing
```python
from src.heuristics import simulated_annealing
from src.heuristics.utils import random_feasible_u

u0 = random_feasible_u(instance, seed=42)
best_cost, best_u, log = simulated_annealing(
    instance, u0,
    T0=100000.0,
    alpha=0.95,
    max_iters=500,
    verbose=True
)
print(f"Best cost: {best_cost:.2f}")
```

### Run Tabu Search
```python
from src.heuristics import tabu_search

u0 = random_feasible_u(instance, seed=42)
best_cost, best_u, log = tabu_search(
    instance, u0,
    tabu_tenure=5,
    max_iters=200,
    neighborhood_size=30,
    verbose=True
)
```

### Compare All Algorithms
```python
from src.experiments.run_all import run_all_experiments

run_all_experiments(
    instances_dir='instances',
    output_dir='results'
)
```

---

## 📊 Understanding Output

### SimulationResult Fields
- `total_cost`: Total cost (what to minimize)
- `cost_transfer`: Shipping costs
- `cost_cross`: Cross-fulfillment penalty
- `cost_lost`: Lost sales penalty
- `clipped_shipments`: How much was clipped (should be 0 for good solutions)

### Good vs Bad Solutions
- **Good**: Low `cost_lost` (few lost sales), low `clipped_shipments` (plan was feasible)
- **Bad**: High `cost_lost` (many lost sales), high `clipped_shipments` (infeasible plan)

---

## 🎯 Algorithm Selection Guide

| Algorithm | When to Use | Pros | Cons |
|-----------|------------|------|------|
| **Myopic** | Quick baseline | Fast, simple | No lookahead |
| **Static Prop** | Simple baseline | Very fast | Ignores dynamics |
| **Greedy** | Good starting point | Fast, reasonable | Not optimal |
| **GRASP** | Diverse starts | Multiple solutions | Need to tune alpha |
| **SA** | Medium instances | Good exploration | Need to tune T0, alpha |
| **Tabu** | Medium instances | Good intensification | Need to tune tenure |
| **MIP** | Very small only | Optimal solution | Only N≤5, J≤3, T≤4 |

---

## 🔧 Parameter Tuning Quick Reference

### Simulated Annealing
- `T0`: Start high (50000-200000), lower for faster convergence
- `alpha`: 0.90-0.98, higher = slower cooling
- `max_iters`: 300-1000, more = better but slower

### Tabu Search
- `tabu_tenure`: 5-15, higher = more diversification
- `neighborhood_size`: 20-80, more = better but slower
- `max_iters`: 200-500

### GRASP
- `alpha`: 0.0-1.0, 0.5 is good starting point

---

## 📁 File Locations

- **Instances**: `instances/*.json`
- **Results**: `results/*.json`, `results/*.csv`
- **Plots**: `results/*.png`
- **Code**: `src/`

---

## ⚠️ Common Mistakes

1. **Forgetting to generate instances first**
   ```python
   # Do this first!
   generate_instance_set()
   ```

2. **Using wrong array shapes**
   - Shipments must be shape `(T, N, J)`
   - Check with `shipments.shape`

3. **Not checking feasibility**
   ```python
   # Always check!
   result = simulate(instance, u, check_feasibility=True)
   ```

4. **Using default parameters blindly**
   - Always tune parameters for your instances!

---

## 🆘 Quick Debugging

```python
# Check instance is valid
instance.validate()

# Check shipment plan shape
assert shipments.shape == (instance.T, instance.num_products, instance.num_fdcs)

# Check costs make sense
result = simulate(instance, shipments)
print(f"Transfer: {result.cost_transfer}")
print(f"Cross: {result.cost_cross}")
print(f"Lost: {result.cost_lost}")
print(f"Clipped: {result.clipped_shipments}")  # Should be ~0
```

---

For detailed explanations, see `CODEBASE_GUIDE.md`

