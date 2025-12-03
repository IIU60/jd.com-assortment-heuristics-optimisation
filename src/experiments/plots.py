"""Plotting functions for experiment results."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


def plot_cost_comparison(results: Dict[str, Dict[str, Any]], output_path: str = 'results/cost_comparison.png'):
    """
    Create boxplot comparing costs across algorithms.
    
    Args:
        results: Dictionary mapping algorithm name to results
        output_path: Path to save plot
    """
    algorithms = []
    costs = []
    
    for alg_name, alg_results in results.items():
        if 'cost' in alg_results and alg_results['cost'] is not None:
            algorithms.append(alg_name)
            costs.append(alg_results['cost'])
    
    if not costs:
        print("No cost data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(costs, labels=algorithms)
    plt.ylabel('Total Cost')
    plt.title('Cost Comparison Across Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_cost_by_instance_type(results_df: pd.DataFrame, output_path: str = 'results/cost_by_type.png'):
    """
    Create bar chart showing average cost per algorithm vs instance type.
    
    Args:
        results_df: DataFrame with columns: algorithm, instance_type, cost
        output_path: Path to save plot
    """
    if results_df.empty:
        print("No data to plot")
        return
    
    pivot = results_df.pivot_table(values='cost', index='instance_type', columns='algorithm', aggfunc='mean')
    
    pivot.plot(kind='bar', figsize=(12, 6))
    plt.ylabel('Average Cost')
    plt.xlabel('Instance Type')
    plt.title('Average Cost by Algorithm and Instance Type')
    plt.legend(title='Algorithm')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_service_level_comparison(results: Dict[str, Dict[str, Any]], output_path: str = 'results/service_level.png'):
    """
    Compare service levels (fulfillment rates) across algorithms.
    
    Args:
        results: Dictionary mapping algorithm name to results
        output_path: Path to save plot
    """
    algorithms = []
    service_levels = []
    
    for alg_name, alg_results in results.items():
        if 'cost_lost' in alg_results and 'cost' in alg_results:
            # Service level = 1 - (lost_cost / total_cost) (approximate)
            total_cost = alg_results.get('cost', 0)
            lost_cost = alg_results.get('cost_lost', 0)
            if total_cost > 0:
                service_level = 1 - (lost_cost / total_cost)
                algorithms.append(alg_name)
                service_levels.append(service_level)
    
    if not service_levels:
        print("No service level data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, service_levels)
    plt.ylabel('Service Level (1 - Lost Sales Ratio)')
    plt.title('Service Level Comparison')
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_runtime_comparison(results: Dict[str, Dict[str, Any]], output_path: str = 'results/runtime_comparison.png'):
    """
    Compare runtime across algorithms.
    
    Args:
        results: Dictionary mapping algorithm name to results
        output_path: Path to save plot
    """
    algorithms = []
    runtimes = []
    
    for alg_name, alg_results in results.items():
        if 'runtime' in alg_results and alg_results['runtime'] is not None:
            algorithms.append(alg_name)
            runtimes.append(alg_results['runtime'])
    
    if not runtimes:
        print("No runtime data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, runtimes)
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.xticks(rotation=45)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def create_all_plots(results: Dict[str, Dict[str, Any]] = None, results_df: pd.DataFrame = None, 
                     output_dir: str = 'results'):
    """
    Create all plots from results.
    
    Args:
        results: Dictionary mapping algorithm name to results (optional)
        results_df: DataFrame with structured results (preferred)
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # If we have a DataFrame, use it for all plots
    if results_df is not None and not results_df.empty:
        # Cost comparison from DataFrame
        if 'cost' in results_df.columns:
            plt.figure(figsize=(10, 6))
            results_df.boxplot(column='cost', by='algorithm', ax=plt.gca())
            plt.ylabel('Total Cost')
            plt.title('Cost Comparison Across Algorithms')
            plt.suptitle('')  # Remove default title
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cost_comparison.png')
            plt.close()
            print(f"Saved plot to {output_dir}/cost_comparison.png")
        
        # Service level from DataFrame
        if 'cost_lost' in results_df.columns and 'cost' in results_df.columns:
            results_df['service_level'] = 1 - (results_df['cost_lost'] / results_df['cost'])
            plt.figure(figsize=(10, 6))
            results_df.groupby('algorithm')['service_level'].mean().plot(kind='bar')
            plt.ylabel('Service Level (1 - Lost Sales Ratio)')
            plt.title('Service Level Comparison')
            plt.xticks(rotation=45)
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(f'{output_dir}/service_level.png')
            plt.close()
            print(f"Saved plot to {output_dir}/service_level.png")
        
        # Runtime from DataFrame
        if 'runtime' in results_df.columns:
            plt.figure(figsize=(10, 6))
            results_df.groupby('algorithm')['runtime'].mean().plot(kind='bar')
            plt.ylabel('Runtime (seconds)')
            plt.title('Runtime Comparison')
            plt.xticks(rotation=45)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/runtime_comparison.png')
            plt.close()
            print(f"Saved plot to {output_dir}/runtime_comparison.png")
        
        # Cost by instance type
        plot_cost_by_instance_type(results_df, f'{output_dir}/cost_by_type.png')
    
    # Fallback to dict-based plots if provided
    elif results is not None:
        plot_cost_comparison(results, f'{output_dir}/cost_comparison.png')
        plot_service_level_comparison(results, f'{output_dir}/service_level.png')
        plot_runtime_comparison(results, f'{output_dir}/runtime_comparison.png')

