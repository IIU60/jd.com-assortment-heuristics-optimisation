"""Basic integration test to verify all components work together."""

import numpy as np
from src.model import Instance, generate_instance, simulate
from src.baselines import myopic_greedy, static_proportional, random_feasible
from src.heuristics import simulated_annealing, tabu_search, greedy_constructor


def test_small_instance():
    """Test with a very small instance (1 SKU, 1 FDC, 2 periods)."""
    print("Testing with small instance (1 SKU, 1 FDC, 2 periods)...")
    
    # Create minimal instance manually
    instance = Instance(
        num_products=1,
        num_fdcs=1,
        T=2,
        demand_fdc=np.array([[[10.0]], [[15.0]]]),  # shape (2, 1, 1)
        demand_rdc=np.array([[0.0], [0.0]]),  # shape (2, 1)
        initial_inventory_rdc=np.array([20.0]),  # shape (1,)
        initial_inventory_fdc=np.array([[0.0]]),  # shape (1, 1)
        replenishment=np.array([[0.0], [0.0]]),  # shape (2, 1)
        outbound_capacity=np.array([15.0, 15.0]),  # shape (2,)
        fdc_capacity=np.array([30.0]),  # shape (1,)
        transfer_cost=np.array([[1.0]]),  # shape (1, 1)
        rdc_fulfillment_cost=3.0,
        lost_sale_cost=10.0
    )
    
    # Validate
    instance.validate()
    print("  Instance validated successfully")
    
    # Test simulator with zero shipments
    shipments = np.zeros((2, 1, 1))
    result = simulate(instance, shipments, check_feasibility=True)
    print(f"  Simulator test: total_cost = {result.total_cost:.2f}")
    assert result.total_cost > 0, "Cost should be positive (lost sales)"
    print("  Simulator test passed")
    
    # Test baselines
    print("\nTesting baselines...")
    cost1, u1 = myopic_greedy(instance)
    print(f"  Myopic greedy: cost = {cost1:.2f}")
    
    cost2, u2 = static_proportional(instance)
    print(f"  Static proportional: cost = {cost2:.2f}")
    
    cost3, u3 = random_feasible(instance, seed=42)
    print(f"  Random feasible: cost = {cost3:.2f}")
    
    # Test heuristics (quick test)
    print("\nTesting heuristics...")
    u_greedy = greedy_constructor(instance)
    result_greedy = simulate(instance, u_greedy, check_feasibility=False)
    print(f"  Greedy constructor: cost = {result_greedy.total_cost:.2f}")
    
    print("\nAll basic tests passed!")


if __name__ == '__main__':
    test_small_instance()

