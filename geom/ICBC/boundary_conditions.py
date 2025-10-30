"""Boundary conditions."""

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from functools import wraps


def return_jax(func):
    """
    Decorator: wrap a Python/numpy function so that its output is
    converted to a jax.numpy array of default floating type.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return jnp.array(out, dtype=jnp.float32)
    return wrapper


class BC(ABC):

    def __init__(self, geom, on_boundary, function=None, component=0):
        self.geom = geom
        self.on_boundary = on_boundary
        self.function = function
        self.component = component

    def filter(self, X):
        X_np = jnp.array(X)
        geom_mask = self.geom.on_boundary(X_np)
        mask = self.on_boundary(X_np, geom_mask)
        return jnp.array(mask, dtype=jnp.bool_)

    def collocation_points(self, X):
        mask = self.filter(X)
        return X[mask]

    @abstractmethod
    def error(self, pred_bc: jnp.ndarray, X_bc: jnp.ndarray) -> jnp.ndarray:
        """
        """


class DirichletBC(BC):
    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = return_jax(func)

    def error(self, pred_bc, X_bc):
        # u_pred - u_true
        u_pred = pred_bc[:, self.component:self.component+1]
        u_true = self.func(X_bc)
        return u_pred - u_true


class NeumannBC(BC):
    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = return_jax(func)

    def error(self, pred_bc, X_bc):
        normals = self.geom.boundary_normal(X_bc)

        start = self.component + 1
        stop = start + normals.shape[1] 
        u_grad = pred_bc[:, start:stop]  # (M,D)

        nd_pred = jnp.sum(u_grad * normals, axis=1, keepdims=True)

        q_true = self.func(X_bc)

        return nd_pred - q_true


class RobinBC(BC):

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, pred_bc, X_bc):
        normals = self.geom.boundary_normal(X_bc)

        start = self.component + 1
        stop = start + normals.shape[1]
        u_grad = pred_bc[:, start:stop]

        nd_pred = jnp.sum(u_grad * normals, axis=1, keepdims=True)
        u_pred = pred_bc[:, self.component:self.component + 1]

        rhs = self.func(X_bc, u_pred)
        return nd_pred - rhs


class PeriodicBC(BC):

    def __init__(self, geom, component_x, on_boundary,
                 derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order

    def collocation_points(self, X):
        X_left = self.filter(X)
        X_right = self.geom.periodic_point(X_left, self.component_x)
        return np.vstack((X_left, X_right))

    def _slice_value(self, pred):
        if self.derivative_order == 0:
            return pred[:, self.component:self.component + 1]
        elif self.derivative_order == 1:
            idx = self.component + 1 + self.component_x
            return pred[:, idx:idx + 1]
        else:
            raise NotImplementedError(
                f"PeriodicBC received {self.derivative_order}"
            )

    def error(self, pred_bc, X_bc):
        M = X_bc.shape[0] // 2
        left_val = self._slice_value(pred_bc[:M, :])
        right_val = self._slice_value(pred_bc[M:, :])
        return left_val - right_val


class OperatorBC(BC):

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, pred_bc, X_bc):
        return self.func(pred_bc, X_bc)


class PointSetBC:
    """
    Dirichlet BC for a fixed set of points.
    Enforce u(x) = values at given discrete points.

    Attributes:
      points: ndarray (M, D) of coordinates
      values: jnp.ndarray (M, 1) of target values
      component: int, which output channel to apply
    """

    def __init__(self, points, values, component=0):
        # points: numpy array or jnp array of shape (M, D)
        self.points = np.array(points)
        # values: array-like of shape (M,) or (M,1)
        arr = np.array(values)
        # ensure shape (M, 1)
        arr = arr.reshape(-1, 1)
        self.values = jnp.array(arr)
        self.component = component

    def collocation_points(self, X=None):
        """
        Return the fixed set of points for this BC.
        Ignores X, because points are predefined.
        """
        return self.points

    def error(self, pred_bc, X_bc=None):
        # pred_bc = model_fn(self.points)
        u_pred = pred_bc[:, self.component:self.component+1]
        return u_pred - self.values


class PointSetOperatorBC:
    """
    Operator BC for a set of points.
    Enforce func(X, u_pred) = values at given discrete points.

    Attributes:
      points: ndarray (M, D) of coordinates
      values: jnp.ndarray (M, 1) of target operator values
      func: Callable[[X, u_pred], jnp.ndarray] -> (M,1)
    """

    def __init__(self, points, values, func, component=0):
        self.points = np.array(points)
        arr = np.array(values)
        arr = arr.reshape(-1, 1)
        self.values = jnp.array(arr)
        self.func = func
        self.component = component

    def collocation_points(self, X=None):
        return self.points

    def error(self, pred_bc, X_bc=None):
        # pred_bc = model_fn(self.points)
        u_pred = pred_bc[:, self.component:self.component+1]
        return u_pred - self.values
