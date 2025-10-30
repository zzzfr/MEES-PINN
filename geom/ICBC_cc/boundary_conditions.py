"""Boundary conditions."""

from abc import ABC, abstractmethod
import numpy as np
import jax
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

        X_np = np.array(X)
        geom_mask = self.geom.on_boundary(X_np)
        mask = self.on_boundary(X_np, geom_mask)
        return jnp.array(mask, dtype=jnp.bool_)

    def collocation_points(self, X):
        mask = self.filter(X)
        return X[mask]

    def normal_derivative(self, model_fn, X_bc):

        def predict_comp(x):
            return model_fn(x[None, :])[0, self.component]

        grads = jax.vmap(jax.grad(predict_comp))(X_bc)

        normals = self.geom.boundary_normal(X_bc)

        return jnp.sum(grads * normals, axis=1, keepdims=True)

    @abstractmethod
    def error(self, model_fn, X_bc):

        ...
class DirichletBC(BC):
    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = return_jax(func)

    def error(self, model_fn, X_bc):
        u_pred = model_fn(X_bc)[:, self.component:self.component+1]
        u_true = self.func(X_bc)
        return u_pred - u_true


class NeumannBC(BC):
    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = return_jax(func)

    def error(self, model_fn, X_bc):
        nd = self.normal_derivative(model_fn, X_bc)
        q_true = self.func(X_bc)
        return nd - q_true


class RobinBC(BC):
    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, model_fn, X_bc):
        nd = self.normal_derivative(model_fn, X_bc)
        u_pred = model_fn(X_bc)[:, self.component:self.component+1]
        rhs = self.func(X_bc, u_pred)
        return nd - rhs


class PeriodicBC(BC):
    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, model_fn, X_bc):
        out = model_fn(X_bc)[:, self.component:self.component+1]
        M = out.shape[0] // 2
        left, right = out[:M], out[M:]
        return left - right


class OperatorBC(BC):


    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, model_fn, X_bc):

        X_jax = jnp.array(X_bc)
        outputs = model_fn(X_jax)  # (M, output_dim)

        err = self.func(X_jax, outputs, X_jax)  # (M,1)

        return err


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

    def error(self, model_fn, X_bc):
        """
        Compute u_pred(X_bc) - u_true for each point.

        model_fn: Callable[[array], jnp.ndarray] -> (N, output_dim)
        X_bc: ndarray of BC points of shape (M, D)
        """
        # predict
        preds = model_fn(jnp.array(X_bc))  # shape (M, output_dim)
        u_pred = preds[:, self.component:self.component + 1]  # (M,1)
        # error vector
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

    def error(self, model_fn, X_bc):
        """
        Compute func(X_bc, u_pred) - values.

        model_fn: Callable[[array], jnp.ndarray]
        X_bc: ndarray of shape (M, D)
        """
        preds = model_fn(jnp.array(X_bc))
        u_pred = preds[:, self.component:self.component + 1]
        op_val = self.func(X_bc, u_pred)  # should return (M,1)
        return op_val - self.values
