"""Baseline algorithms for comparison"""

from .myopic import myopic_greedy
from .static_prop import static_proportional
from .random_feasible import random_feasible

__all__ = ['myopic_greedy', 'static_proportional', 'random_feasible']
