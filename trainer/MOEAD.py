import time
import numpy as np
import jax
import jax.numpy as jnp
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


def sbx_crossover(key, p1, p2, pc=0.9, eta_c=15.0, lower=0.0, upper=1.0):
    B, L = p1.shape
    key, kprob, ku = jax.random.split(key, 3)
    do_cx = jax.random.bernoulli(kprob, pc, (B, 1))
    u = jax.random.uniform(ku, (B, L))
    beta = jnp.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta_c + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
    )
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    c1 = jnp.where(do_cx, c1, p1)
    c2 = jnp.where(do_cx, c2, p2)
    return key, jnp.clip(c1, lower, upper), jnp.clip(c2, lower, upper)


def polynomial_mutation(key, X, pm=0.01, eta_m=20.0, lower=0.0, upper=1.0):
    N, L = X.shape
    key, kmask, kr = jax.random.split(key, 3)
    mask = jax.random.bernoulli(kmask, pm, X.shape)
    r = jax.random.uniform(kr, X.shape)
    xl = jnp.full((1, L), lower) if np.isscalar(lower) else jnp.array(lower)[None, :]
    xu = jnp.full((1, L), upper) if np.isscalar(upper) else jnp.array(upper)[None, :]
    width = xu - xl + 1e-12
    delta1 = (X - xl) / width
    delta2 = (xu - X) / width
    mut_pow = 1.0 / (eta_m + 1.0)
    r_less = (r < 0.5)
    xy1 = 1.0 - delta1
    val1 = 2.0 * r + (1.0 - 2.0 * r) * (xy1 ** (eta_m + 1.0))
    deltaq1 = (val1 ** mut_pow) - 1.0
    xy2 = 1.0 - delta2
    val2 = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy2 ** (eta_m + 1.0))
    deltaq2 = 1.0 - (val2 ** mut_pow)
    deltaq = jnp.where(r_less, deltaq1, deltaq2)
    X_mut = X + jnp.where(mask, deltaq * width, 0.0)
    return key, jnp.clip(X_mut, xl, xu)


def _tchebycheff(F, lamb, z):
    return jnp.max(lamb[None, :] * jnp.abs(F - z[None, :]), axis=-1)


def train(get_fitness, sim_mgr, pop_size=50, params_num=260, max_iters=2000, seed=0,
          lower=0.0, upper=1.0, pc=0.9, pm=None, eta_c=15.0, eta_m=20.0,
          T=None, delta=0.9, nr=2, verbose=True):

    if pm is None: pm = 1.0 / params_num
    if T is None:  T = max(5, int(0.1 * pop_size))
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    pop = jax.random.uniform(k0, (pop_size, params_num), minval=lower, maxval=upper)

    objs = - get_fitness(sim_mgr, pop)
    if objs.ndim == 1:
        objs = objs[:, None]
    k = int(objs.shape[1])

    key, kw = jax.random.split(key)
    lambdas = jax.random.dirichlet(kw, jnp.ones((k,)), (pop_size,))
    dists = jnp.linalg.norm(lambdas[:, None, :] - lambdas[None, :, :], axis=-1)
    neighbors = jnp.argsort(dists, axis=1)[:, :T]
    z = jnp.min(objs, axis=0)

    best_loss = float("inf")
    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    best_w = None

    for iter in range(1, max_iters + 1):
        t0 = time.time()
        key, kperm = jax.random.split(key)
        order = jax.random.permutation(kperm, pop_size)

        for i in order.tolist():
            Bi = neighbors[i]
            key, kprob = jax.random.split(key)
            use_nb = jax.random.bernoulli(kprob, delta)
            if use_nb:
                key, ksel = jax.random.split(key)
                parents = jax.random.choice(ksel, Bi, shape=(2,), replace=False)
            else:
                key, ksel = jax.random.split(key)
                parents = jax.random.randint(ksel, (2,), 0, pop_size)

            p1, p2 = pop[parents[0]][None, :], pop[parents[1]][None, :]
            key, c1, c2 = sbx_crossover(key, p1, p2, pc, eta_c, lower, upper)
            key, kpick = jax.random.split(key)
            child = jnp.where(jax.random.bernoulli(kpick, 0.5), c1, c2)
            key, child = polynomial_mutation(key, child, pm, eta_m, lower, upper)
            child = child[0]

            child_obj = - get_fitness(sim_mgr, child[None, :])[0]
            if np.ndim(child_obj) == 0:
                child_obj = np.array([child_obj])

            z = jnp.minimum(z, child_obj)

            key, knei = jax.random.split(key)
            Bi_perm = jax.random.permutation(knei, Bi.shape[0])
            replaced = 0
            for jj in Bi[Bi_perm].tolist():
                f_old = _tchebycheff(objs[jj:jj+1, :], lambdas[jj], z)[0]
                f_new = _tchebycheff(child_obj[None, :], lambdas[jj], z)[0]
                if f_new <= f_old:
                    pop = pop.at[jj].set(child)
                    objs = objs.at[jj].set(child_obj)
                    replaced += 1
                    if replaced >= nr: break

        sum_losses = jnp.sum(objs, axis=1)

        best_idx = int(jnp.argmin(sum_losses))
        best_losses = objs[best_idx]
        best_total = float(jnp.sum(best_losses))
        various_loss = np.mean(np.array(objs, copy=False), axis=0)

        loss_ls.append(best_total)
        various_loss_ls.append(various_loss)

        iter_time_ls.append(time.time() - t0)

        if best_total < best_loss:
            best_loss = best_total
            best_w = np.array(pop[best_idx])

        if verbose:
            print(f"iter {iter}: best_total={best_total:.6f}, losses={best_losses.tolist()}  pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")

    print(f"\nFinished at iter={max_iters}, last loss(avg)={loss_ls[-1]:.2e}, best loss={best_loss:.2e}")

    return Result(best_w=best_w, best_fit=best_loss, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)
