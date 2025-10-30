import numpy as np
import jax.numpy as jnp
from jax import random
import time
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


class PSOAdapter:
    def __init__(self, f, sim_mgr, dim, pop_size, seed,
                 w=0.3, c1=1.5, c2=1.5, v_max=0.5,
                 low=-2.0, high=2.0):
        self.f = f
        self.dim = dim
        self.pop_size = pop_size
        self.w, self.c1, self.c2 = w, c1, c2
        self.v_max = v_max
        self.low, self.high = low, high

        self._key = random.PRNGKey(seed)
        self._key, k1, k2 = random.split(self._key, 3)
   
        self.p = random.uniform(k1, shape=(pop_size, dim), minval=low, maxval=high)
        self.v = random.normal(k2, shape=(pop_size, dim)) * 0.5
        self.pb = self.p.copy()

        _, self.pbs = self.f(sim_mgr, self.pb)  
        best_idx = jnp.argmax(self.pbs)
        self.gb = self.pb[best_idx]
        self.gbs = self.pbs[best_idx]

    def ask(self):
        return self.p

    def tell(self, fitness):
        better = fitness > self.pbs
        self.pb = jnp.where(better[:, None], self.p, self.pb)
        self.pbs = jnp.where(better, fitness, self.pbs)

        best_idx = jnp.argmax(self.pbs)
        self.gb = self.pb[best_idx]
        self.gbs = self.pbs[best_idx]

        self._key, k1, k2 = random.split(self._key, 3)
        r1 = random.normal(k1, shape=self.p.shape)
        r2 = random.normal(k2, shape=self.p.shape)

        self.v = (self.w * self.v
                  + self.c1 * r1 * (self.pb - self.p)
                  + self.c2 * r2 * (self.gb - self.p))
        self.v = jnp.clip(self.v, -self.v_max, self.v_max)
        self.p = self.p + self.v
        self.p = jnp.clip(self.p, self.low, self.high)

    @property
    def best_params(self):
        return np.array(self.gb)


def train(get_fitness, policy, sim_mgr, pop_size=50, max_iters=5000, seed=0, w=0.3, c1=1.5, c2=1.5, v_max=0.5,
          bound_down=-2.0, bound_up=2.0):

    solver = PSOAdapter(
        f=get_fitness, sim_mgr=sim_mgr,
        dim=policy.num_params,
        pop_size=pop_size,
        seed=seed,
        w=w, c1=c1, c2=c2, v_max=v_max,
        low=bound_down, high=bound_up
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

        print(
            f"iter={train_iters:5d}  time={runtime:6.2f}s  loss(avg)={loss_ls[-1]:.2e}  pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")

    print(f"\nFinished at iter={train_iters}, last loss(avg)={loss_ls[-1]:.2e}, best loss={best_loss:.2e}")

    return Result(best_w=best_flat_params, best_fit=best_loss, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)
