"""Internal utilities."""
import inspect
import sys
import timeit
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from matplotlib import animation

from .external import apply


def timing(func):
    """Decorator for measuring the execution time of methods."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = timeit.default_timer()
        result = func(*args, **kwargs)
        te = timeit.default_timer()
        print(f"'{func.__name__}' took {te - ts:.6f} s")
        sys.stdout.flush()
        return result
    return wrapper


def run_if_all_none(*attrs):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            vals = [getattr(self, a) for a in attrs]
            if all(v is None for v in vals):
                return func(self, *args, **kwargs)
            return vals if len(vals) > 1 else vals[0]
        return wrapper
    return decorator


def run_if_any_none(*attrs):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            vals = [getattr(self, a) for a in attrs]
            if any(v is None for v in vals):
                return func(self, *args, **kwargs)
            return vals if len(vals) > 1 else vals[0]
        return wrapper
    return decorator


def vectorize(**vectorize_kwargs):
    """numpy.vectorize wrapper that works with instance methods.

    References:

    - https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    - https://stackoverflow.com/questions/48981501/is-it-possible-to-numpy-vectorize-an-instance-method
    - https://github.com/numpy/numpy/issues/9477
    """

    def decorator(fn):
        vectorized_fn = np.vectorize(fn, **vectorize_kwargs)
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return vectorized_fn(*args, **kwargs)
        return wrapper
    return decorator


def return_tensor(func):
    """Convert the output to a Tensor."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return jnp.array(out, dtype=jnp.float32)
    return wrapper


def to_numpy(tensors):
    """Create numpy ndarrays that shares the same underlying storage, if possible.

    Args:
        tensors. A Tensor or a list of Tensor.

    Returns:
        A numpy ndarray or a list of numpy ndarray.
    """

    if isinstance(tensors, (list, tuple)):
        return [np.array(t) for t in tensors]
    return np.array(tensors)


def make_dict(keys, values):
    """Convert two lists or two variables into a dictionary."""

    if isinstance(keys, (list, tuple)):
        if len(keys) != len(values):
            raise ValueError("keys and values length mismatch.")
        return dict(zip(keys, values))
    return {keys: values}


def save_animation(filename, xdata, ydata, y_reference=None, logy=False):
    apply(
        _save_animation,
        args=(filename, xdata, ydata),
        kwds={"y_reference": y_reference, "logy": logy},
    )


def _save_animation(filename, xdata, ydata, y_reference=None, logy=False):
    """The animation figure window cannot be closed automatically.

    References:

    - https://stackoverflow.com/questions/43776528/python-animation-figure-window-cannot-be-closed-automatically
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if logy:
        ax.set_yscale("log")
    ln, = ax.plot([], [], "b-")
    if y_reference is not None:
        ax.plot(xdata, y_reference, "r--")

    def init():
        ax.set_xlim(min(xdata), max(xdata))
        # 计算 y 范围
        if isinstance(ydata[0], (list, tuple, np.ndarray)):
            y_min = min(min(y) for y in ydata)
            y_max = max(max(y) for y in ydata)
        else:
            y_min, y_max = min(ydata), max(ydata)
        ax.set_ylim(y_min, y_max)
        ln.set_data([], [])
        return ln,

    def update(frame):
        y = ydata[frame]
        ln.set_data(xdata, y)
        return ln,

    anim = animation.FuncAnimation(
        fig, update, frames=len(ydata), init_func=init, blit=True
    )
    anim.save(filename, writer="ffmpeg")
    plt.close(fig)


def list_to_str(nums, precision=2):
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))


def get_num_args(func):
    """Get the number of arguments of a Python function.

        References:

        - https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function
    """
    sig = inspect.signature(func)
    params = sig.parameters
    count = len(params)
    if "self" in params:
        count -= 1
    return count