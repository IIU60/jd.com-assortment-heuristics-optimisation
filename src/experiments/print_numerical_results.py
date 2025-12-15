"""
Utility script to print numerical results for *small*, *medium*, *large*,
and *massive* experiments in a compact table format, similar to the
example in the project report.

It scans the top-level ``experiments/`` directory, finds the most recent
experiment folder for each size class (by modification time), loads the
corresponding ``results/all_results.csv`` file, and prints a summary
table with:

- MIP status and best MIP cost
- Per–algorithm cost and percentage gap to the MIP optimum

Usage (from the project root):

    python -m src.experiments.print_numerical_results

Optional flags allow you to override the automatically selected
experiment folders:

    python -m src.experiments.print_numerical_results \\
        --small small_5fdc_20sku_T14_20251203-121639 \\
        --medium medium_20fdc_100sku_T14_20251203-121701 \\
        --large large_80fdc_320sku_T14_20251203-121837 \\
        --massive massive_100fdc_1000sku_T14_20251203-122454
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EXPERIMENT_ROOT_NAME = "experiments"
RESULTS_SUBDIR = "results"
SUMMARY_FILENAME = "all_results.csv"


@dataclass
class AlgorithmSummary:
    name: str
    cost: float
    gap_pct: Optional[float]
    runtime: float


def _project_root() -> Path:
    """Return the project root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _experiments_root() -> Path:
    return _project_root() / EXPERIMENT_ROOT_NAME


def find_experiment_dir(
    size_label: str,
    override_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Find the experiment directory for a given size label.

    If ``override_name`` is provided, that directory name is used
    directly (under ``experiments/``). Otherwise, the most recently
    modified directory whose name starts with ``f"{size_label}_"`` and
    that contains ``results/all_results.csv`` is returned.
    """
    experiments_root = _experiments_root()

    if override_name:
        candidate = experiments_root / override_name
        if (candidate / RESULTS_SUBDIR / SUMMARY_FILENAME).is_file():
            return candidate
        return None

    if not experiments_root.is_dir():
        return None

    candidates: List[Tuple[float, Path]] = []
    for child in experiments_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith(f"{size_label}_"):
            continue
        summary_path = child / RESULTS_SUBDIR / SUMMARY_FILENAME
        if summary_path.is_file():
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, child))

    if not candidates:
        return None

    # Pick the most recently modified candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_summary_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load the summary CSV as a list of dict rows."""
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_algorithm_costs(
    rows: Iterable[Dict[str, str]],
) -> Dict[str, float]:
    """
    Compute average cost per algorithm across all instances.

    Returns a mapping ``algorithm -> mean(cost)``.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for row in rows:
        alg = row.get("algorithm")
        cost_str = row.get("cost")
        if not alg or cost_str in (None, "", "None"):
            continue
        try:
            cost_val = float(cost_str)
        except (TypeError, ValueError):
            continue

        sums[alg] = sums.get(alg, 0.0) + cost_val
        counts[alg] = counts.get(alg, 0) + 1

    return {alg: sums[alg] / counts[alg] for alg in sums if counts.get(alg, 0) > 0}


def compute_algorithm_runtimes(
    rows: Iterable[Dict[str, str]],
) -> Dict[str, float]:
    """
    Compute average runtime per algorithm across all instances.

    Returns a mapping ``algorithm -> mean(runtime)``.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for row in rows:
        alg = row.get("algorithm")
        runtime_str = row.get("runtime")
        if not alg or runtime_str in (None, "", "None"):
            continue
        try:
            runtime_val = float(runtime_str)
        except (TypeError, ValueError):
            continue

        sums[alg] = sums.get(alg, 0.0) + runtime_val
        counts[alg] = counts.get(alg, 0) + 1

    return {alg: sums[alg] / counts[alg] for alg in sums if counts.get(alg, 0) > 0}


def build_summary(
    rows: List[Dict[str, str]],
) -> Tuple[Optional[float], List[AlgorithmSummary]]:
    """
    Build per–algorithm summaries and extract the MIP cost.

    Returns
    -------
    mip_cost
        Average MIP cost across instances (if available).
    summaries
        List of ``AlgorithmSummary`` for all non-MIP algorithms.
    """
    costs_by_alg = compute_algorithm_costs(rows)
    runtimes_by_alg = compute_algorithm_runtimes(rows)
    mip_cost = costs_by_alg.get("mip")

    algo_names = sorted(a for a in costs_by_alg.keys() if a != "mip")

    summaries: List[AlgorithmSummary] = []
    for alg in algo_names:
        cost = costs_by_alg[alg]
        runtime = runtimes_by_alg.get(alg, 0.0)
        gap_pct = None
        if mip_cost is not None and mip_cost > 0.0:
            gap_pct = (cost - mip_cost) / mip_cost * 100.0
        summaries.append(AlgorithmSummary(name=alg, cost=cost, gap_pct=gap_pct, runtime=runtime))

    return mip_cost, summaries


def _pretty_name(alg_name: str) -> str:
    """Human-friendly algorithm name for printing."""
    mapping = {
        "random": "Random",
        "greedy": "Greedy",
        "grasp": "GRASP",
        "sa": "SA",
        "tabu": "Tabu",
        "mip": "MIP",
        "myopic": "Myopic",
        "static_prop": "StaticProp",
    }
    return mapping.get(alg_name, alg_name)


def print_table(
    label: str,
    exp_dir: Path,
    mip_cost: Optional[float],
    summaries: List[AlgorithmSummary],
) -> None:
    """Print a nicely formatted table for a single experiment."""
    print()
    print(f"{label} experiment: {exp_dir.name}")
    print("-" * (len(label) + len(" experiment: ") + len(exp_dir.name)))

    if mip_cost is None:
        print("MIP status: No MIP results available")
    else:
        print("MIP status: Optimal (feasible solution found)")
        print(f"Optimal cost (MIP): {mip_cost:,.2f}")

    print()
    print(f"{'Method':10s} {'Cost':>15s} {'Gap (%)':>10s} {'Runtime (s)':>12s}")
    print("-" * 52)

    if not summaries:
        print("(no algorithm results)")
        return

    for s in summaries:
        gap_str = "-"
        if s.gap_pct is not None:
            gap_str = f"{s.gap_pct:6.2f}"
        print(f"{_pretty_name(s.name):10s} {s.cost:15,.2f} {gap_str:>10s} {s.runtime:12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print numerical results for small, medium, large, and massive "
            "experiments from experiments/<name>/results/all_results.csv."
        )
    )
    parser.add_argument(
        "--small",
        type=str,
        default=None,
        help=(
            "Name of the small experiment directory under experiments/. "
            "If omitted, the latest 'small_*' experiment with a summary is used."
        ),
    )
    parser.add_argument(
        "--medium",
        type=str,
        default=None,
        help=(
            "Name of the medium experiment directory under experiments/. "
            "If omitted, the latest 'medium_*' experiment with a summary is used."
        ),
    )
    parser.add_argument(
        "--large",
        type=str,
        default=None,
        help=(
            "Name of the large experiment directory under experiments/. "
            "If omitted, the latest 'large_*' experiment with a summary is used."
        ),
    )
    parser.add_argument(
        "--massive",
        type=str,
        default=None,
        help=(
            "Name of the massive experiment directory under experiments/. "
            "If omitted, the latest 'massive_*' experiment with a summary is used."
        ),
    )

    args = parser.parse_args()

    for size_label, override in (
        ("small", args.small),
        ("medium", args.medium),
        ("large", args.large),
        ("massive", args.massive),
    ):
        exp_dir = find_experiment_dir(size_label=size_label, override_name=override)
        if exp_dir is None:
            print(f"\nNo {size_label} experiment with '{SUMMARY_FILENAME}' found.")
            continue

        csv_path = exp_dir / RESULTS_SUBDIR / SUMMARY_FILENAME
        rows = load_summary_csv(csv_path)
        mip_cost, summaries = build_summary(rows)

        pretty_label = size_label.capitalize()
        print_table(pretty_label, exp_dir, mip_cost, summaries)


if __name__ == "__main__":
    main()






