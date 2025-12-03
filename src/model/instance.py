"""Instance data structure for the multi-echelon distribution problem."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Instance:
    """
    Instance parameters for the multi-echelon distribution optimization problem.
    
    This represents a single RDC (Regional Distribution Center) serving multiple
    FDCs (Fulfillment Distribution Centers) over a time horizon.
    
    Attributes:
        num_products: Number of products/SKUs (N)
        num_fdcs: Number of FDCs (J)
        T: Number of time periods
        
        demand_fdc: Demand at FDCs, shape (T, N, J)
            demand_fdc[t, i, j] = demand for product i at FDC j in period t
        
        demand_rdc: Demand at RDC, shape (T, N)
            demand_rdc[t, i] = demand for product i at RDC in period t
            Can be all zeros if not modeling RDC demand
        
        initial_inventory_rdc: Initial RDC inventory, shape (N,)
            initial_inventory_rdc[i] = initial inventory of product i at RDC
        
        initial_inventory_fdc: Initial FDC inventory, shape (N, J)
            initial_inventory_fdc[i, j] = initial inventory of product i at FDC j
        
        replenishment: External supply arriving at RDC, shape (T, N)
            replenishment[t, i] = units of product i arriving at RDC at start of period t
        
        outbound_capacity: RDC outbound capacity, shape (T,)
            outbound_capacity[t] = max total units that can be shipped from RDC in period t
        
        fdc_capacity: FDC storage capacity, shape (J,) or None
            fdc_capacity[j] = max total inventory units that FDC j can store
            None means no capacity limit
        
        transfer_cost: Cost per unit shipped from RDC to FDC, shape (N, J)
            transfer_cost[i, j] = cost per unit of product i shipped to FDC j
        
        rdc_fulfillment_cost: Cost per unit when RDC fulfills FDC demand directly
            This is a penalty for cross-fulfillment (slower, more expensive)
        
        lost_sale_cost: Penalty per unit of lost demand
            Applied to both FDC and RDC lost sales
    """
    num_products: int
    num_fdcs: int
    T: int
    
    # Demand arrays
    demand_fdc: np.ndarray  # shape (T, N, J)
    demand_rdc: np.ndarray  # shape (T, N)
    
    # Initial inventories
    initial_inventory_rdc: np.ndarray  # shape (N,)
    initial_inventory_fdc: np.ndarray  # shape (N, J)
    
    # Replenishment
    replenishment: np.ndarray  # shape (T, N)
    
    # Capacities
    outbound_capacity: np.ndarray  # shape (T,)
    fdc_capacity: Optional[np.ndarray]  # shape (J,) or None
    
    # Costs
    transfer_cost: np.ndarray  # shape (N, J)
    rdc_fulfillment_cost: float
    lost_sale_cost: float
    
    def validate(self) -> None:
        """
        Validate that all arrays have correct shapes and non-negative values.
        Raises ValueError if validation fails.
        """
        # Check shapes
        assert self.demand_fdc.shape == (self.T, self.num_products, self.num_fdcs), \
            f"demand_fdc shape {self.demand_fdc.shape} != ({self.T}, {self.num_products}, {self.num_fdcs})"
        assert self.demand_rdc.shape == (self.T, self.num_products), \
            f"demand_rdc shape {self.demand_rdc.shape} != ({self.T}, {self.num_products})"
        assert self.initial_inventory_rdc.shape == (self.num_products,), \
            f"initial_inventory_rdc shape {self.initial_inventory_rdc.shape} != ({self.num_products},)"
        assert self.initial_inventory_fdc.shape == (self.num_products, self.num_fdcs), \
            f"initial_inventory_fdc shape {self.initial_inventory_fdc.shape} != ({self.num_products}, {self.num_fdcs})"
        assert self.replenishment.shape == (self.T, self.num_products), \
            f"replenishment shape {self.replenishment.shape} != ({self.T}, {self.num_products})"
        assert self.outbound_capacity.shape == (self.T,), \
            f"outbound_capacity shape {self.outbound_capacity.shape} != ({self.T},)"
        assert self.transfer_cost.shape == (self.num_products, self.num_fdcs), \
            f"transfer_cost shape {self.transfer_cost.shape} != ({self.num_products}, {self.num_fdcs})"
        
        if self.fdc_capacity is not None:
            assert self.fdc_capacity.shape == (self.num_fdcs,), \
                f"fdc_capacity shape {self.fdc_capacity.shape} != ({self.num_fdcs},)"
        
        # Check non-negativity
        assert np.all(self.demand_fdc >= 0), "demand_fdc must be non-negative"
        assert np.all(self.demand_rdc >= 0), "demand_rdc must be non-negative"
        assert np.all(self.initial_inventory_rdc >= 0), "initial_inventory_rdc must be non-negative"
        assert np.all(self.initial_inventory_fdc >= 0), "initial_inventory_fdc must be non-negative"
        assert np.all(self.replenishment >= 0), "replenishment must be non-negative"
        assert np.all(self.outbound_capacity >= 0), "outbound_capacity must be non-negative"
        assert np.all(self.transfer_cost >= 0), "transfer_cost must be non-negative"
        assert self.rdc_fulfillment_cost >= 0, "rdc_fulfillment_cost must be non-negative"
        assert self.lost_sale_cost >= 0, "lost_sale_cost must be non-negative"
        
        if self.fdc_capacity is not None:
            assert np.all(self.fdc_capacity >= 0), "fdc_capacity must be non-negative"

