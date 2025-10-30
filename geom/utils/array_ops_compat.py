import numpy as np
import jax.numpy as jnp

def istensorlist(values):
    return any(isinstance(v, jnp.ndarray) for v in values)

def convert_to_array(value):
    if isinstance(value, (list, tuple)) and istensorlist(value):
        return jnp.stack(value, axis=0)
    return jnp.array(value)

def hstack(tup):
    if isinstance(tup[0], jnp.ndarray):
        return jnp.hstack(tup)
    return np.hstack(tup)

def roll(a, shift, axis):
    if isinstance(a, jnp.ndarray):
        return jnp.roll(a, shift, axis)
    return np.roll(a, shift, axis=axis)

def zero_padding(array, pad_width):
    if isinstance(array, (list, tuple)) and len(array) == 3:
        indices, values, dense_shape = array
        indices = [(i + pad_width[0][0], j + pad_width[1][0]) for i, j in indices]
        dense_shape = (
            dense_shape[0] + sum(pad_width[0]),
            dense_shape[1] + sum(pad_width[1]),
        )
        return indices, values, dense_shape
    if isinstance(array, jnp.ndarray):
        return jnp.pad(array, pad_width)
    return np.pad(array, pad_width)