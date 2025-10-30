""" base pinn network definition"""

import jax
import jax.numpy as jnp
from typing import Dict, Callable, Tuple
from flax import linen as nn
from abc import abstractmethod

# nodes default value
nodes = 8


class BaseNN(nn.Module):
    input_dim: int  
    output_dim: int = 1  
    width: int = nodes
    depth: int = 4

    @nn.compact
    def __call__(self, inputs):
        """
        inputs: shape (N, input_dim)
        return: shape (N, output_dim)
        """
        h = inputs
        for _ in range(self.depth - 1):
            h = nn.tanh(nn.Dense(self.width, kernel_init=jax.nn.initializers.glorot_uniform())(h))
        return nn.Dense(self.output_dim)(h)

    @abstractmethod
    def derivatives(self, params, X: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
       
        """

