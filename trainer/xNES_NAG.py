import time
import numpy as np
from jax import random, numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
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

local_seed = 1


def train(f, policy, sim_mgr, *, bs=100, lr=1e-2, sigma=1e-3, max_iters=5000,
        momentum_coeff=0.9, verbose=True):

    key, rng = random.split(random.PRNGKey(local_seed))
    center = jnp.zeros(policy.num_params)
    dim = int(center.shape[0])
    I = jnp.eye(dim)
    A = I * sigma
    momentum = jnp.zeros(dim)
    m_A = jnp.zeros_like(A)
    beta = 0.0

    bestFitness = -jnp.inf
    bestFound = None

    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    runtime = 0.0

    @jit
    def project_sample(center, A, momentum, rng):
        center_proj = center + momentum_coeff * momentum
        key, rng = random.split(rng)
        samples = random.normal(key, (bs, dim))
        samples_o = samples @ A + center_proj
        return samples, samples_o, rng

    @jit
    def compute_utilities(fitnesses):
        order = jnp.argsort(fitnesses)
        ranks = jnp.argsort(order).astype(jnp.float32)
        L = fitnesses.size
        u_raw = jnp.log(L / 2.0 + 1.0) - jnp.log(L - ranks)
        utilities = jnp.maximum(0.0, u_raw)
        utilities = utilities / jnp.sum(utilities)
        utilities = utilities - 1.0 / L
        return utilities

    @jit
    def update_parameters(center, A, momentum, m_g, utilities, samples):
        update_center = A @ (utilities @ samples) + momentum_coeff * momentum
        momentum = update_center
        center = center + update_center
        covGrad = jnp.sum(
            utilities[:, None, None] *
            (samples[:, :, None] * samples[:, None, :] - I),
            axis=0
        )
        m_g = (1 - beta) * (0.5 * lr * covGrad) + beta * m_g
        A = A @ expm(m_g)
        return center, A, momentum, m_g

    numEvals = 0
    for it in range(max_iters):
        t0 = time.time()

        samples, samples_o, rng = project_sample(center, A, momentum, rng)
        losses, fitnesses = f(sim_mgr, samples_o)

        avg_fit = float(jnp.mean(fitnesses))
        loss_iter = -avg_fit
        various_loss = np.mean(np.array(losses, copy=False), axis=0)

        loss_ls.append(loss_iter)
        various_loss_ls.append(various_loss)

        # 保存最优
        cur_best = float(jnp.max(fitnesses))
        if cur_best > float(bestFitness):
            idx = int(jnp.argmax(fitnesses))
            bestFitness = cur_best
            bestFound = np.array(samples_o[idx])

        numEvals += bs
        if verbose:
            runtime += (time.time() - t0)
            print(f"iter={it+1:5d}  time={runtime:6.2f}s  loss={loss_iter:.2e} pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")

        utilities = compute_utilities(fitnesses)
        center, A, momentum, m_A = update_parameters(center, A, momentum, m_A, utilities, samples)

        iter_time_ls.append(float(time.time() - t0))

    print(f"\nFinished at iter={max_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")

    return Result(best_w=bestFound, best_fit=bestFitness, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)

