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


def train(f, policy, sim_mgr, *, bs=100, lr=2e-2, lr2=0.003, sigma=1e-3, max_iters=10_000,
        beta1=0.9, beta2=0.999, eps=1e-8, verbose=True):
    key, rng = random.split(random.PRNGKey(local_seed))

    w0 = jnp.zeros(policy.num_params)
    center = w0.copy()
    dim = int(center.shape[0])
    I = jnp.eye(dim)
    A = I * sigma

    # Adam moments
    m = jnp.zeros(dim)
    v = jnp.zeros(dim)
    t_adam = 0

    bestFitness = -jnp.inf
    bestFound = None

    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    runtime = 0.0

    @jit
    def project_sample(center, A, rng):
        key, rng = random.split(rng)
        samples = random.normal(key, (bs, dim))
        samples_o = samples @ A + center
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
    def update(center, A, m, v, t_adam, utilities, samples):
        
        grad_c = A @ (utilities @ samples)

        
        m = beta1 * m + (1.0 - beta1) * grad_c
        v = beta2 * v + (1.0 - beta2) * (grad_c ** 2)
        t_adam += 1
        m_hat = m / (1.0 - beta1 ** t_adam)
        v_hat = v / (1.0 - beta2 ** t_adam)
        center = center + lr2 * m_hat / (jnp.sqrt(v_hat) + eps)

       
        covGrad = jnp.sum(
            utilities[:, None, None] *
            (samples[:, :, None] * samples[:, None, :] - I),
            axis=0
        )
        A = A @ expm(0.5 * lr * covGrad)
        return center, A, m, v, t_adam

    numEvals = 0
    for it in range(max_iters):
        t0 = time.time()

        samples, samples_o, rng = project_sample(center, A, rng)
        losses, fitnesses = f(sim_mgr, samples_o)  # shape [bs]

        avg_fit = float(jnp.mean(fitnesses))
        loss_iter = -avg_fit
        various_loss = np.mean(np.array(losses, copy=False), axis=0)

        loss_ls.append(loss_iter)
        various_loss_ls.append(various_loss)

        cur_best = float(jnp.max(fitnesses))
        if cur_best > float(bestFitness):
            idx = int(jnp.argmax(fitnesses))
            bestFitness = cur_best
            bestFound = np.array(samples_o[idx])

        numEvals += bs

        if verbose:
            runtime += (time.time() - t0)
            print(f"iter={it+1:5d}  time={runtime:6.2f}s  loss={loss_ls[-1]:.2e} pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")


        utilities = compute_utilities(fitnesses)
        center, A, m, v, t_adam = update(center, A, m, v, t_adam, utilities, samples)

        iter_time_ls.append(float(time.time() - t0))

    print(f"\nFinished at iter={max_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")

    return Result(best_w=bestFound, best_fit=bestFitness, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)


