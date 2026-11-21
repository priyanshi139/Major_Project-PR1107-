import random
import numpy as np


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    return seed
