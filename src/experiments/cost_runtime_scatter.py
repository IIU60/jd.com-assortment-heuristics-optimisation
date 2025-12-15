"""
Create a Cost vs. Runtime scatter plot (log-scale runtime) from the latest experiments.

One point per algorithm per instance size.

Usage (from the project root):
    python -m src.experiments.cost_runtime_scatter
"""

from __future__ import annotations

import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EXPERIMENT_ROOT_NAME = "experiments"
RESULTS_SUBDIR = "results"
SUMMARY_FILENAME = "all_results.csv"


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


def create_scatter_plot(
    data: Dict[str, Dict[str, Tuple[float, float]]],
    output_path: Optional[Path] = None,
) -> None:
    """
    Create a Cost vs. Runtime scatter plot with log-scale runtime.
    
    Args:
        data: Dictionary mapping instance_size -> algorithm -> (cost, runtime)
        output_path: Path to save the plot. If None, saves to project root.
    """
    # Set up the plot with large, clear styling
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 24,
        'axes.titlesize': 28,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 30,
        'lines.markersize': 15,
        'lines.linewidth': 2.5,
    })
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Define distinct, vibrant colors for instance sizes
    size_colors = {
        'small': '#0066CC',      # Bright blue
        'medium': '#00AA44',     # Bright green
        'large': '#FF6600',      # Bright orange
        'massive': '#CC0000',    # Bright red
    }
    
    # Define markers for different algorithms
    alg_markers = {
        'myopic': 'o',      # Circle
        'random': 's',      # Square
        'greedy': '^',      # Triangle up
        'grasp': 'D',       # Diamond
        'sa': 'v',          # Triangle down
        'tabu': 'p',        # Pentagon
        'mip': '*',         # Star
        'static_prop': 'X', # X marker
    }
    
    size_labels_pretty = {
        'small': 'Small',
        'medium': 'Medium',
        'large': 'Large',
        'massive': 'Massive',
    }
    
    # Collect all points
    all_points: List[Tuple[str, str, float, float]] = []  # (alg, size, cost, runtime)
    
    for size_label, alg_data in data.items():
        for alg_name, (cost, runtime) in alg_data.items():
            if alg_name == 'mip':  # Skip MIP for cleaner plot
                continue
            all_points.append((alg_name, size_label, cost, runtime))
    
    # Plot all points without labels
    for alg_name, size_label, cost, runtime in all_points:
        color = size_colors.get(size_label, '#000000')
        marker = alg_markers.get(alg_name, 'o')
        
        edge_color = 'white'
        edge_width = 3
        
        ax.scatter(
            runtime,
            cost,
            c=color,
            marker=marker,
            s=800,  # Very large markers for high visibility
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=0.9,
            zorder=3,
        )
    
    # Set log scale for x-axis (runtime)
    ax.set_xscale('log')
    
    # Labels and title
    ax.set_xlabel('Runtime (seconds, log scale)', fontsize=26, fontweight='bold', labelpad=15)
    ax.set_ylabel('Cost', fontsize=26, fontweight='bold', labelpad=15)
    ax.set_title('Cost vs. Runtime Comparison', 
                 fontsize=30, fontweight='bold', pad=25)
    
    # Grid for better readability
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=2)
    ax.set_axisbelow(True)
    
    # Format axes with thicker ticks
    ax.tick_params(axis='both', which='major', width=2.5, length=10, pad=8)
    ax.tick_params(axis='both', which='minor', width=2, length=6)
    
    # Format y-axis to show numbers with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Create separate legends for instance sizes (colors) and algorithms (markers)
    # Legend 1: Instance sizes (colors) - use circular patches
    size_legend_elements = []
    for size_label in sorted(data.keys()):
        size_pretty = size_labels_pretty.get(size_label, size_label.capitalize())
        color = size_colors.get(size_label, '#000000')
        # Use a larger patch - create a circle-like patch
        patch = Patch(facecolor=color, edgecolor='white', linewidth=3, label=size_pretty)
        size_legend_elements.append(patch)
    
    # Legend 2: Algorithms (markers) - use Line2D with markers
    alg_legend_elements = []
    all_algorithms = set()
    for size_label, alg_data in data.items():
        all_algorithms.update(alg_data.keys())
    all_algorithms.discard('mip')  # Skip MIP
    
    for alg_name in sorted(all_algorithms):
        pretty_name = _pretty_name(alg_name)
        marker = alg_markers.get(alg_name, 'o')
        # Create a line with marker for the legend
        alg_legend_elements.append(
            Line2D([0], [0], marker=marker, color='black', label=pretty_name,
                   markersize=22, linestyle='None', markeredgewidth=3, 
                   markeredgecolor='black', markerfacecolor='gray', alpha=0.9)
        )
    
    # Create combined legend - larger and in top left
    all_legend_elements = size_legend_elements + alg_legend_elements
    legend = ax.legend(
        handles=all_legend_elements,
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        ncol=2,
        columnspacing=3.0,
        handletextpad=2.0,
        borderpad=2.0,
        fontsize=20,  # Larger font size
        prop={'size': 20},
    )
    legend.get_frame().set_linewidth(3)
    legend.get_frame().set_edgecolor('black')
    
    # Make legend markers even larger
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_markersize'):
            handle.set_markersize(22)
        elif hasattr(handle, 'get_facecolor'):
            # For patches, we can't easily change size, but we can ensure visibility
            pass
    
    # Tight layout and save
    plt.tight_layout()
    
    if output_path is None:
        output_path = _project_root() / 'cost_runtime_scatter.png'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved scatter plot to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Cost vs. Runtime scatter plot from the latest experiments. "
            "One point per algorithm per instance size."
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot. Default: cost_runtime_scatter.png in project root.",
    )

    args = parser.parse_args()

    # Collect data from all instance sizes
    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    
    for size_label, override in (
        ("small", args.small),
        ("medium", args.medium),
        ("large", args.large),
        ("massive", args.massive),
    ):
        exp_dir = find_experiment_dir(size_label=size_label, override_name=override)
        if exp_dir is None:
            print(f"Warning: No {size_label} experiment with '{SUMMARY_FILENAME}' found. Skipping.")
            continue

        csv_path = exp_dir / RESULTS_SUBDIR / SUMMARY_FILENAME
        rows = load_summary_csv(csv_path)
        
        costs_by_alg = compute_algorithm_costs(rows)
        runtimes_by_alg = compute_algorithm_runtimes(rows)
        
        # Store (cost, runtime) tuples for each algorithm
        data[size_label] = {}
        for alg_name in costs_by_alg.keys():
            cost = costs_by_alg[alg_name]
            runtime = runtimes_by_alg.get(alg_name, 0.0)
            data[size_label][alg_name] = (cost, runtime)
        
        print(f"Loaded {size_label} experiment: {exp_dir.name}")
        print(f"  Algorithms: {sorted(costs_by_alg.keys())}")

    if not data:
        print("Error: No experiment data found. Cannot create plot.")
        return

    # Create the scatter plot
    output_path = Path(args.output) if args.output else None
    create_scatter_plot(data, output_path)


if __name__ == "__main__":
    main()

