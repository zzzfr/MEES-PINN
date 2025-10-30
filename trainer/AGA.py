import numpy as np
import jax
import jax.numpy as jnp
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


def train(get_fitness_fn, sim_mgr, popsize=50, params_num=260, elite_rate=0.05, seed=1, max_iters=5000, lower=-10.0, upper=10.0, ):

    cx_min, cx_max = 0.60, 0.95
    pm_min, pm_max = 0.001, 0.02
    t_min, t_max = 2, 4
    mut_sigma_base = 0.10 * (upper - lower)

    key = jax.random.PRNGKey(seed)
    key, k1 = jax.random.split(key)
    pop = jax.random.uniform(k1, shape=(popsize, params_num), minval=lower, maxval=upper)

    elite = max(1, int(popsize * elite_rate))

    no_improve = 0
    iter_time_ls = []
    loss_ls = []
    various_loss_ls = []

   
    losses, fitness = get_fitness_fn(sim_mgr, pop)
    best_idx = int(jnp.argmax(fitness))
    best_fit = float(fitness[best_idx])
    best_vec_indiv = np.array(pop[best_idx])

    best_loss = np.inf
    saved_vec = None

    for it in range(1, max_iters + 1):
        t0 = time.time()

       
        std = jnp.std(pop, axis=0)
        D = jnp.clip(std / (upper - lower + 1e-9), 0.0, 1.0).mean()
        cx_rate = float(cx_min + (cx_max - cx_min) * (1.0 - D))
        pm_base = float(pm_min + (pm_max - pm_min) * (1.0 - D))
        t = int(jnp.floor(t_min + (1.0 - D) * (t_max - t_min)))
        t = max(t_min, min(t, t_max))
        mut_sigma = mut_sigma_base

        if no_improve >= 10:
            pm_base = min(pm_base * 1.5, pm_max)
            no_improve = 0


        elite_idx = jnp.argsort(-fitness)[:elite]
        elites = pop[elite_idx]


        def tournament_select(rng, pop, fitness, ksize):
            idx = jax.random.randint(rng, (ksize,), 0, pop.shape[0])
            return idx[jnp.argmax(fitness[idx])]

        children, parent_fits = [], []
        num_children = popsize - elite
        for _ in range(num_children):
            key, k1 = jax.random.split(key)
            key, k2 = jax.random.split(key)
            p1 = tournament_select(k1, pop, fitness, t)
            p2 = tournament_select(k2, pop, fitness, t)

            key, k3 = jax.random.split(key)
            key, k4 = jax.random.split(key)
            do_cx = jax.random.bernoulli(k3, cx_rate)
            mask = jax.random.bernoulli(k4, 0.5, shape=(params_num,))
            child = jnp.where(do_cx & mask, pop[p1], pop[p2])

            children.append(child)
            parent_fits.append((fitness[p1] + fitness[p2]) / 2.0)

        children = jnp.stack(children, axis=0)
        parent_fits = jnp.stack(parent_fits, axis=0)

        
        f_avg = float(jnp.mean(fitness))
        f_max = float(jnp.max(fitness))
        denom = (f_max - f_avg + 1e-8)
        pm_i = jnp.where(
            parent_fits >= f_avg,
            pm_min + (pm_max - pm_min) * (f_max - parent_fits) / denom,
            pm_max,
        )
        pm_i = jnp.maximum(pm_i, pm_base)

        key, km1 = jax.random.split(key)
        key, km2 = jax.random.split(key)
        mut_mask = jax.random.bernoulli(km1, pm_i[:, None], shape=children.shape)
        noise = jax.random.normal(km2, shape=children.shape) * mut_sigma
        children = jnp.clip(children + mut_mask * noise, lower, upper)

        
        pop = jnp.vstack([elites, children])

        估
        losses, fitness = get_fitness_fn(sim_mgr, pop)
        curr_idx = int(jnp.argmax(fitness))
        curr_fit = float(fitness[curr_idx])
        if curr_fit > best_fit + 1e-12:
            best_fit = curr_fit
            best_vec_indiv = np.array(pop[curr_idx])
            no_improve = 0
        else:
            no_improve += 1

       
        loss_iter = float(-jnp.mean(fitness))
        various_loss = np.mean(np.array(losses, copy=False), axis=0)

        loss_ls.append(loss_iter)
        various_loss_ls.append(various_loss)

        if loss_iter < best_loss:
            best_loss = loss_iter
            elite_mean = np.asarray(jnp.mean(elites, axis=0))
            saved_vec = elite_mean.copy()

        elapsed = time.time() - t0
        iter_time_ls.append(elapsed)

        print(f"iter={it:5d}  time={np.sum(iter_time_ls):6.2f}s  loss(avg)={loss_iter:.2e}  pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")

    if saved_vec is None:
        saved_vec = np.array(best_vec_indiv, copy=True)

    return Result(best_w=saved_vec, best_fit=best_fit, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)
