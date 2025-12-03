"""Offline tuning of SA initial temperature (T0) for different instance sizes.

This script runs short, time-limited SA runs on a set of instances to
estimate good T0 values per size category (e.g. small / medium / large),
and saves the mapping to a JSON config that the SA heuristic can query.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..model.instance_generator import load_instance
from ..model.instance import Instance
from ..heuristics.construction import grasp_constructor
from ..heuristics.sa import simulated_annealing
from ..heuristics.utils import evaluate_u


def _instance_size_key(instance: Instance) -> str:
    """Map an instance to a coarse size category."""
    n = instance.num_products
    if n <= 10:
        return "small"
    if n <= 100:
        return "medium"
    return "large"


def tune_sa_t0(
    instances_dir: str = "old_experiments/instances",
    output_path: str = "data/sa_t0_config.json",
    time_limit_per_run: float = 1.5,
    t0_multipliers: Tuple[float, ...] = (0.02, 0.05, 0.1),
    max_instances_per_size: int = 5,
    seeds: Tuple[int, ...] = (42, 43),
) -> Dict[str, float]:
    """
    Tune T0 values for SA for different instance size categories.

    For each size category (small/medium/large), this:
    - Samples up to `max_instances_per_size` instances from `instances_dir`
    - For each candidate T0 multiplier and seed, runs SA with a short
      time limit and measures improvement over the starting construction.
    - Chooses the multiplier with the best average improvement and
      records the corresponding T0 (multiplier * starting cost).
    """
    instances_path = Path(instances_dir)
    instance_files = sorted(p for p in instances_path.glob("*.json"))

    if not instance_files:
        print(f"No instances found in {instances_dir}; nothing to tune.")
        return {}

    # Group instances by size key
    size_to_files: Dict[str, List[Path]] = {"small": [], "medium": [], "large": []}
    for f in instance_files:
        inst = load_instance(str(f))
        key = _instance_size_key(inst)
        if len(size_to_files[key]) < max_instances_per_size:
            size_to_files[key].append(f)

    size_to_t0: Dict[str, float] = {}

    for size_key, files in size_to_files.items():
        if not files:
            continue

        print(f"Tuning T0 for size '{size_key}' on {len(files)} instances...")

        multiplier_stats: Dict[float, List[float]] = {m: [] for m in t0_multipliers}

        for inst_path in files:
            instance = load_instance(str(inst_path))

            for seed in seeds:
                # Build a strong starting solution (GRASP)
                u0 = grasp_constructor(instance, alpha=0.5, seed=seed)
                start_cost, _ = evaluate_u(instance, u0)
                if start_cost <= 0:
                    continue

                for m in t0_multipliers:
                    T0 = m * start_cost
                    start = time.perf_counter()
                    best_cost, _, _ = simulated_annealing(
                        instance,
                        u0,
                        T0=T0,
                        alpha=0.95,
                        max_iters=1000,
                        max_no_improve=100,
                        verbose=False,
                        seed=seed,
                        large_move_prob=0.1,
                        time_limit=time_limit_per_run,
                    )
                    elapsed = time.perf_counter() - start
                    improvement = max(0.0, (start_cost - best_cost) / start_cost)
                    # Weight improvement lightly by actual runtime, but stay simple:
                    score = improvement
                    multiplier_stats[m].append(score)
                    print(
                        f"[{size_key}] {inst_path.name}, seed={seed}, "
                        f"mult={m:.3f}, T0={T0:.1f}, "
                        f"impr={improvement:.4f}, time={elapsed:.2f}s"
                    )

        # Choose the multiplier with the best average score
        best_mult = None
        best_score = -np.inf
        for m, scores in multiplier_stats.items():
            if not scores:
                continue
            avg_score = float(np.mean(scores))
            if avg_score > best_score:
                best_score = avg_score
                best_mult = m

        if best_mult is not None:
            # Use a representative starting cost (e.g. 1.0) – we store the multiplier
            # but encode it as an absolute T0 factor to keep SA's lookup simple.
            size_to_t0[size_key] = float(best_mult)
            print(
                f"Chosen T0 multiplier for size '{size_key}': {best_mult:.4f} "
                f"(avg score={best_score:.4f})"
            )

    # Persist as JSON (mapping size_key -> T0 multiplier)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(size_to_t0, f, indent=2)
    print(f"Saved SA T0 configuration to {out_path}")

    return size_to_t0


if __name__ == "__main__":
    tune_sa_t0()


