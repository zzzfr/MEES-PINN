import random
import numpy as np
from .real import Real

real = Real(32)

random_seed = None
def set_random_seed(seed: int):
    global random_seed
    random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
