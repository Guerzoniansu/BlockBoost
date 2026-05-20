"""
Schema for experimental configurations to ensure rigor and serialization.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Phase0Config:
    n_samples: int = 3000
    n_features: int = 10
    ar_coeffs: List[float] = None
    noise_std: float = 1.0
    random_state: int = 42

    def __post_init__(self):
        if self.ar_coeffs is None:
            # Default to an AR(5) that is stationary (sum < 1)
            self.ar_coeffs = [0.4, 0.2, 0.1, 0.05, 0.05]

@dataclass
class Phase1Config:
    n_samples: int = 3000
    n_features: int = 10
    ar_coeffs: List[float] = None
    noise_std: float = 1.0
    base_seed: int = 42
    n_seeds: int = 10
    M: int = 500  # Number of boosting rounds
    
    # Phase 1.1 parameters
    etas: List[float] = None  # Base noise rates
    alpha_markov: float = 0.5 # Default persistence
    block_sizes: List[int] = None # Will default to [a_T*]
    
    def __post_init__(self):
        if self.ar_coeffs is None:
            self.ar_coeffs = [0.4, 0.2, 0.1, 0.05, 0.05]
        if self.etas is None:
            self.etas = [0.0, 0.05, 0.10, 0.20, 0.30]

