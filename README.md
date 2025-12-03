## JD.com Assortment & Multi‑Echelon Distribution Heuristics

This project implements and evaluates heuristic algorithms for a **single‑RDC, multi‑FDC, multi‑period inventory distribution problem**, inspired by JD.com’s e‑commerce fulfillment network.

At a high level, we:
- Generate synthetic problem instances that mimic JD.com‑style demand and capacity patterns.
- Model shipment decisions from a **Regional Distribution Center (RDC)** to multiple **Fulfillment Distribution Centers (FDCs)** over a time horizon.
- Evaluate shipment plans via a forward **simulator** and an **exact MIP** formulation (for small instances).
- Compare **construction heuristics, metaheuristics (SA, Tabu), and baselines** on cost, service level, and runtime.

---

## 1. Problem Overview

We consider:
- One **RDC**, multiple **FDCs** (\(J\)), multiple **products/SKUs** (\(N\)), over \(T\) time periods.
- At each period \(t\), we decide **how many units of each product to ship from the RDC to each FDC**.
- Demand occurs both at:
  - FDCs: `demand_fdc[t, i, j]`
  - RDC: `demand_rdc[t, i]`

Key elements:
- **Inventories**
  - Initial RDC inventory `initial_inventory_rdc[i]`
  - Initial FDC inventory `initial_inventory_fdc[i, j]`
  - External replenishment into the RDC `replenishment[t, i]`
- **Capacities**
  - RDC outbound capacity per period `outbound_capacity[t]`
  - Total FDC storage capacity `fdc_capacity[j]` (sum of all SKUs at FDC \(j\))
  - Lead time `lead_time`: shipments ordered at \(t\) arrive at \(t + \text{lead\_time}\)
- **Costs**
  - Transfer cost `transfer_cost[i, j]` per unit shipped from RDC to FDC \(j\)
  - Cross‑fulfillment cost `rdc_fulfillment_cost` for serving FDC demand directly from RDC
  - Lost sale penalty `lost_sale_cost` for unmet demand (at both FDC and RDC)

**Decision variable (solution):**
- A shipment plan \(u\) with shape `(T, N, J)`:
  - `u[t, i, j]`: units of product `i` shipped from RDC to FDC `j` at the start of period `t`.

**Objective:**
- Minimize total cost:
  - Transfer cost + cross‑fulfillment cost + lost sales cost,
  - Subject to capacity and inventory feasibility.

---

## 2. Repository Structure

Relevant directories:

- `src/model/`
  - `instance.py`: defines the `Instance` dataclass (all parameters of a problem instance).
  - `simulator.py`: forward simulator to evaluate a shipment plan and compute total cost & metrics.
  - `result.py`: `SimulationResult` dataclass bundling inventories, flows, and cost components.
  - `instance_generator.py`: synthetic instance generator (plus save/load helpers and batch generation).
  - `mip_solver.py`: exact MIP formulation using PuLP, for small instances / ground truth.

- `src/heuristics/`
  - `utils.py`: helper functions for:
    - copying shipment plans,
    - generating random feasible plans,
    - approximate feasibility clamping,
    - fast full and incremental evaluation of neighbors.
  - `neighborhoods.py`: neighborhood move operators (time shifts, FDC swaps, magnitude tweaks, product rebalances, block re‑planning) and neighbor generation (random + problem‑aware).
  - `construction.py`: greedy and GRASP constructors, plus a diversified starting‑solution pool.
  - `sa.py`: simulated annealing (time‑limited, single‑start) over shipment plans.
  - `tabu.py`: tabu search (time‑limited, single‑start) over shipment plans.

- `src/baselines/`
  - `myopic.py`: myopic greedy baseline using short‑term demand look‑ahead.
  - `static_prop.py`: static proportional baseline derived from total demand shares.
  - `random_feasible.py`: random feasible baseline respecting constraints.

- `src/experiments/`
  - `run_experiment.py`: CLI to create a new `experiments/<name>/` folder and run experiments.
  - `run_all.py`: core experimental harness that:
    - loads instances,
    - runs baselines, heuristics, and MIP,
    - saves per‑instance JSON results and an aggregated `all_results.csv`,
    - generates plots via `plots.py`.
  - `run_small.py`, `run_medium.py`, `run_large.py`, `run_massive.py`:
    - create size‑specific experiment subfolders;
    - generate synthetic instances of the requested size;
    - run all algorithms with scenario‑appropriate parameters.
  - `plots.py`: creates cost, runtime, and service‑level plots from result CSVs.
  - `tune_sa_t0.py`: offline tuning of SA’s initial temperature `T0` per instance size category.
  - `tune_params.py`: general parameter tuning utilities for SA and Tabu.

- `experiments/`
  - Contains **completed experiments**, each in its own subdirectory:
    - `experiments/<experiment_name>/instances/`: JSON instances used.
    - `experiments/<experiment_name>/results/`: per‑instance JSONs, `all_results.csv`, and plots.

---

## 3. Model & Simulation Details

### 3.1 Instance representation

The core dataclass `Instance` (in `src/model/instance.py`) encapsulates:
- Dimensions: `num_products`, `num_fdcs`, `T`
- Demand:
  - `demand_fdc[t, i, j]`
  - `demand_rdc[t, i]`
- Inventories and supply:
  - `initial_inventory_rdc[i]`
  - `initial_inventory_fdc[i, j]`
  - `replenishment[t, i]` at RDC
- Capacity:
  - `outbound_capacity[t]`
  - `fdc_capacity[j]` (or `None` for unlimited)
- Costs:
  - `transfer_cost[i, j]`
  - `rdc_fulfillment_cost`
  - `lost_sale_cost`
- Lead time:
  - `lead_time` (typically 1)

`Instance.validate()` checks array shapes, non‑negativity, and a valid lead time.

### 3.2 Simulation (`simulate`)

The function `simulate(instance, shipments)` in `src/model/simulator.py`:
- Takes an `Instance` and a shipment tensor `shipments` with shape `(T, N, J)`.
- Simulates period by period:
  1. **RDC replenishment** at the start of each period.
  2. **Shipments** from RDC to FDCs, enforcing:
     - RDC inventory availability,
     - FDC capacity limits (total inventory),
     - RDC outbound capacity per period.
     Infeasible requests are **clipped**, and the total amount clipped is recorded.
  3. **FDC demand fulfillment**:
     - Arrivals after `lead_time`,
     - Local fulfillment from FDC inventory,
     - Cross‑fulfillment from RDC if FDC stock is insufficient,
     - Remaining unmet demand becomes lost sales at FDC.
  4. **RDC demand fulfillment**:
     - Served from remaining RDC inventory, or lost.
  5. **Cost computation**:
     - Transfer cost from all `actual_shipments`,
     - Cross‑fulfillment cost from `fdc_from_rdc_fulfilled`,
     - Lost sales cost from lost demand at FDC and RDC.
- Returns a `SimulationResult` object with:
  - `total_cost`, `cost_transfer`, `cost_cross`, `cost_lost`,
  - time‑series inventories at RDC and FDCs,
  - all flows (shipments, fulfilled demand, lost sales),
  - `clipped_shipments`.

For small instances, `mip_solver.solve_exact` builds and solves an equivalent MIP using PuLP, providing an optimal benchmark when the solver/time limit allows.

---

## 4. Heuristics & Baselines

### 4.1 Solution representation

All heuristics and baselines operate on a shipment plan:
- `u.shape == (T, N, J)`
- Non‑negative entries (any negative values are clamped to 0 before simulation).

Evaluation is done via:
- `heuristics.utils.evaluate_u(instance, u)` → `(total_cost, metrics)`
- `heuristics.utils.evaluate_u_incremental(...)` → reuses cached simulation to speed up neighbor evaluation.

### 4.2 Baselines

Located in `src/baselines/`:

- **Myopic greedy (`myopic.py`)**
  - For each period, looks `lead_time` steps ahead (or current period for the tail).
  - Computes shortage at each FDC given current FDC inventory.
  - Allocates available RDC inventory proportionally to these shortfalls, respecting:
    - outbound capacity and FDC capacity.
  - Uses the simulator to evaluate the resulting plan.

- **Static proportional (`static_prop.py`)**
  - Pre‑computes total demand per product and FDC over the whole horizon.
  - Sets fixed proportions per (i, j) based on total demand shares.
  - Each period, allocates RDC inventory according to these static proportions (bounded by capacities).

- **Random feasible (`random_feasible.py`)**
  - For each period and product, draws Dirichlet proportions over FDCs.
  - Allocates a random fraction of available RDC inventory to FDCs, respecting:
    - outbound capacity and FDC capacity.

All baselines return `(total_cost, shipments)` and then are re‑evaluated in a standardized way inside the experiment harness.

### 4.3 Construction heuristics (`construction.py`)

- **Greedy constructor**
  - Uses future cumulative demand and transfer cost to build a priority score for each (product, FDC, time).
  - Allocates shipments in descending score order, within inventory and capacity limits.
- **GRASP constructor**
  - Similar to greedy but uses a **Restricted Candidate List (RCL)**:
    - selects among high‑score candidates at random (controlled by `alpha`).
  - Produces diverse, relatively strong starting solutions.
- **Starting pool**
  - Combines:
    - myopic baseline,
    - greedy constructor,
    - multiple GRASP variants,
    - a “smoothed” myopic variant perturbed by a few problem‑aware moves.

### 4.4 Neighborhoods & local search (`neighborhoods.py`)

Key move types:
- `time_shift_move`: move volume for a product & FDC between two periods.
- `fdc_swap_move`: reallocate volume between two FDCs for a product at a fixed time.
- `magnitude_tweak`: locally increase or decrease shipment quantity at `(t, i, j)`.
- `product_rebalance_move`: rebalance all shipments of a product across FDCs according to demand/cost‑based weights.
- `block_replan_move`: destroy‑and‑repair within a time window for one product, guided by future demand and transfer costs.

`generate_neighbor`:
- With some probability, uses **problem‑aware neighbor generation**:
  - focuses on high‑cost, high‑demand, or poorly served (lost sales / high cross‑fulfillment) regions.
- Otherwise, selects random move types and indices with adaptive step sizes.

### 4.5 Simulated annealing (`sa.py`)

Implements time‑limited, single‑start SA:
- Starts from a given `u_init` (usually GRASP output).
- Selects neighbors using `generate_neighbor` plus occasional large moves (`product_rebalance_move`, `block_replan_move`).
- Evaluates neighbors via `evaluate_u_incremental`.
- Accepts moves using the standard Metropolis criterion with:
  - tuned or heuristic initial temperature `T0`,
  - cooling rate `alpha`,
  - early stopping based on no‑improvement counter and/or wall‑clock `time_limit`.

### 4.6 Tabu search (`tabu.py`)

Implements time‑limited, single‑start Tabu Search:
- Maintains a **tabu list** of recent move keys with a given tenure.
- At each iteration:
  - samples a neighborhood of candidate moves (including occasional large moves),
  - uses incremental evaluation,
  - picks the best **non‑tabu** candidate, with aspiration allowing tabu moves if they improve the global best.
- Stops based on iteration count, no‑improvement threshold, or `time_limit`.

---

## 5. Instance Generation

Instance generation lives in `src/model/instance_generator.py`.

### 5.1 Single instance

`generate_instance(...)`:
- Uses Poisson demand:
  - FDC demand: λ uniformly in [1, 8].
  - RDC demand: λ uniformly in [3, 15].
- Sets replenishment so that supply is slightly **below** average demand per period, with noise.
- Derives initial RDC inventory as a fraction of horizon demand, capped by per‑SKU RDC capacity.
- Sets **tight RDC outbound capacity** per period (e.g. 70% of total period demand).
- Samples FDC capacities and transfer costs from configurable ranges.
- Returns:
  - `Instance`,
  - an initial all‑zero shipment plan of shape `(T, N, J)`.

### 5.2 Saving & loading

- `save_instance(instance, filepath)`: writes JSON with all arrays converted to lists.
- `load_instance(filepath)`: reconstructs an `Instance` and validates it.

### 5.3 Multiple instances

`generate_instance_set(...)`:
- Generates multiple “small” and “medium” instances into a given directory (e.g. for tuning).

---

## 6. Experiments & How to Run

### 6.1 Quickstart (scenarios)

From the project root:

```bash
# Small scenario (5 FDC, 20 SKUs, single instance)
python -m src.experiments.run_small

# Medium scenario (20 FDC, 100 SKUs)
python -m src.experiments.run_medium

# Large scenario (80 FDC, 320 SKUs)
python -m src.experiments.run_large

# Massive scenario (100 FDC, 1000 SKUs)
python -m src.experiments.run_massive
```

Each command:
- Creates a new subdirectory under `experiments/` with a timestamped name.
- Generates a single synthetic JSON instance in `instances/`.
- Runs:
  - selected baselines (`random` by default; others optional),
  - greedy & GRASP constructors,
  - simulated annealing,
  - tabu search,
  - exact MIP (if feasible within time).
- Writes:
  - `<instance_name>_results.json` per instance under `results/`,
  - `all_results.csv` summary,
  - plots: `cost_comparison.png`, `service_level.png`, `runtime_comparison.png`, `cost_by_type.png`.

### 6.2 Running on custom instances

1. **Generate or prepare instances**
   - Use `generate_instance_set` or your own script to create JSON instances.
2. **Place them in an experiment directory**
   - Create `experiments/my_experiment/instances/` and put your JSON files there.
   - Ensure there is a `results/` subdirectory (or let `run_all_experiments` create it).
3. **Run all algorithms**
   - Either via `run_experiment.py` (which sets up a fresh experiment folder) or via a custom script calling:

     ```python
     from src.experiments.run_all import run_all_experiments
     run_all_experiments(
         instances_dir="experiments/my_experiment/instances",
         output_dir="experiments/my_experiment/results",
         sa_params={...},    # optional overrides
         tabu_params={...},  # optional overrides
         mip_max_time=120.0,
         baseline_names=['random', 'myopic', 'static_prop'],
     )
     ```

### 6.3 Tuning

- **SA initial temperature tuning**
  - `src/experiments/tune_sa_t0.py` runs short SA runs on a set of instances, chooses good `T0` multipliers per size category, and saves them to `data/sa_t0_config.json`.
- **General SA/Tabu parameter tuning**
  - `src/experiments/tune_params.py` provides:
    - `tune_sa(...)`
    - `tune_tabu(...)`
  - These loop over parameter grids, measure average/best costs and runtimes, and save CSV tuning logs.

---

## 7. Extending the Project

To **add a new heuristic**:
1. Implement a function with signature:
   ```python
   def my_heuristic(instance: Instance) -> tuple[float, np.ndarray]:
       ...
       return best_cost, best_u
   ```
2. Use:
   - `generate_instance` / `load_instance` to get an `Instance`.
   - `evaluate_u` / `evaluate_u_incremental` for cost evaluation.
   - The move operators in `neighborhoods.py`, or add new ones following the same pattern.
3. Register your heuristic in `run_all_experiments` by adding a wrapper to `alg_dict`.

To **change the model or costs**:
- Update `Instance` and `simulate` consistently (and optionally `mip_solver`).
- Extend `SimulationResult` if you add new metrics or flows.

To **modify experimental configurations**:
- Adjust SA/Tabu defaults in:
  - `src/experiments/run_all.py`, or
  - the scenario scripts (`run_small/medium/large/massive.py`) for specific size regimes.

This structure is designed so that:
- **Model & evaluation** are centralized in `src/model`.
- **Search logic** (heuristics) lives in `src/heuristics`.
- **Baselines & experiments** can be swapped, tuned, or extended without touching the core simulator.



