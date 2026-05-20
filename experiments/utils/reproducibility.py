# experiments/utils/reproducibility.py

import os
import random
import numpy as np

def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and specific environment variables 
    to ensure reproducible results across runs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_seeds(n_seeds: int, base_seed: int = 42) -> list[int]:
    """
    Generate a fixed list of seeds for multiple independent runs.
    """
    set_seed(base_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(n_seeds)]
