"""Simulation result data structure."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    """
    Results from simulating a shipment plan.
    
    Attributes:
        total_cost: Total cost (transfer + cross-fulfillment + lost sales)
        cost_transfer: Total transfer cost (sum of r[i,j] * shipments[t,i,j])
        cost_cross: Total cross-fulfillment cost (rdc_fulfillment_cost * sum of y_fdc)
        cost_lost: Total lost sales cost (lost_sale_cost * sum of lost demand)
        
        inventory_rdc: RDC inventory over time, shape (T+1, N)
            inventory_rdc[t, i] = inventory of product i at RDC at start of period t
        inventory_fdc: FDC inventory over time, shape (T+1, N, J)
            inventory_fdc[t, i, j] = inventory of product i at FDC j at start of period t
        
        shipments: Actual shipments executed, shape (T, N, J)
            shipments[t, i, j] = units of product i shipped to FDC j in period t
        fdc_local_fulfilled: FDC demand fulfilled from local inventory, shape (T, N, J)
        fdc_from_rdc_fulfilled: FDC demand fulfilled from RDC (cross-fulfillment), shape (T, N, J)
        lost_fdc: Lost sales at FDCs, shape (T, N, J)
        
        rdc_fulfilled: RDC demand fulfilled, shape (T, N)
        lost_rdc: Lost sales at RDC, shape (T, N)
        
        clipped_shipments: Total amount of shipments that were clipped due to constraints
    """
    total_cost: float
    cost_transfer: float
    cost_cross: float
    cost_lost: float
    
    inventory_rdc: np.ndarray  # shape (T+1, N)
    inventory_fdc: np.ndarray  # shape (T+1, N, J)
    
    shipments: np.ndarray  # shape (T, N, J)
    fdc_local_fulfilled: np.ndarray  # shape (T, N, J)
    fdc_from_rdc_fulfilled: np.ndarray  # shape (T, N, J)
    lost_fdc: np.ndarray  # shape (T, N, J)
    
    rdc_fulfilled: np.ndarray  # shape (T, N)
    lost_rdc: np.ndarray  # shape (T, N)
    
    clipped_shipments: float

