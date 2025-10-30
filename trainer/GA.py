import time
import numpy as np
from jax import numpy as jnp
from evojax.algo import SimpleGA

from flax.struct import dataclass
from typing import List


@dataclass
class Result:
    best_w: jnp.ndarray
    best_fit: float
    evals: int
    iter_time_ls: List[float]
    loss_ls: List[float]
    various_loss_ls: List[float]


def train(get_fitness, policy, sim_mgr, pop_size=50, max_iters=5000, seed=0):
    solver = SimpleGA(
        pop_size=pop_size,
        param_size=policy.num_params,
        seed=seed,
    )

    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    runtime = 0.0
    train_iters = 0

    best_loss = np.inf
    best_flat_params = None


    while train_iters < max_iters:
        t0 = time.time()
        params = solver.ask()
        losses, scores = get_fitness(sim_mgr, params)
        solver.tell(fitness=scores)


        avg_loss = np.mean(np.array(scores, copy=False))
        various_loss = np.mean(np.array(losses, copy=False), axis=0)

        loss_ls.append(-avg_loss)
        various_loss_ls.append(various_loss)

        idx_best = int(np.argmax(scores))
        cur_best_loss = float(-scores[idx_best])  # fitness = -loss

        if cur_best_loss < best_loss:
            best_loss = cur_best_loss
            best_flat_params = np.array(params[idx_best], copy=True)

        elapsed = time.time() - t0
        iter_time_ls.append(elapsed)
        runtime += elapsed
        train_iters += 1

        print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss(avg)={loss_ls[-1]:.2e}  pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")

    print(f"\nFinished at iter={train_iters}, last loss(avg)={loss_ls[-1]:.2e}, best loss={best_loss:.2e}")

    return Result(best_w=best_flat_params, best_fit=best_loss, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)
