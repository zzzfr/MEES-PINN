import jax.numpy as jnp
from typing import List


def stack_outputs(outs: dict, layout: List[str]):
    """根据 layout 顺序把各张量 hstack，并检查缺失键。"""
    pieces = []
    for key in layout:
        if key not in outs:
            raise KeyError(f'[stack_outputs] 缺少输出: {key}')
        pieces.append(outs[key])
    return jnp.hstack(pieces)          # (N, len(layout))
