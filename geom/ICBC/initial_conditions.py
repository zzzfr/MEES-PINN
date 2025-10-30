"""Initial conditions."""

__all__ = ["IC"]

import numpy as np
import jax
import jax.numpy as jnp
from .boundary_conditions import BC


class IC(BC):
    """Initial condition: enforce u(x, t0) = func(x, t0)."""

    def __init__(self, geom, func, on_initial, component=0):
        self.geom = geom
        self.func = func
        self.on_initial = jax.vmap(on_initial, in_axes=(0, 0))
        self.component = component

    def filter(self, X):
        """
           X: array-like (N, D)
           return: jnp.ndarray of shape (N,), dtype=bool
        """
        X_np = jnp.array(X)
        geom_mask = self.geom.on_initial(X_np)
        mask = self.on_initial(X_np, geom_mask)
        return jnp.array(mask, dtype=jnp.bool_)

    def collocation_points(self, X):
        mask = self.filter(X)
        return X[mask]

    def error(self, pred_bc, X_bc):
        # u_pred - u_true
        u_pred = pred_bc[:, self.component:self.component + 1]
        u_true = self.func(X_bc)
        return u_pred - u_true
