import time
import jax
import jax.numpy as jnp
import numpy as np
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


def fast_non_dominated_ranks(objs: jnp.ndarray) -> jnp.ndarray:
    N = objs.shape[0]
    objs_p = objs[:, None, :]   # (N,1,M)
    objs_q = objs[None, :, :]   # (1,N,M)
    less_equal = jnp.all(objs_p <= objs_q, axis=2)     # (N,N)
    strictly_less = jnp.any(objs_p < objs_q, axis=2)   # (N,N)
    dominates = jnp.logical_and(less_equal, strictly_less)

    dom_count = jnp.sum(dominates, axis=0)
    ranks = -jnp.ones((N,), dtype=jnp.int32)

    current = jnp.where(dom_count == 0)[0]
    r = 0
    while current.size > 0:
        ranks = ranks.at[current].set(r)
        reduces = jnp.sum(dominates[current, :], axis=0)
        dom_count = dom_count - reduces
        mask_unranked = (ranks < 0)
        current = jnp.where(jnp.logical_and(dom_count == 0, mask_unranked))[0]
        r += 1
    return ranks


def crowding_distance_for_front(objs: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    F = idx.size
    if F == 0:
        return jnp.array([], dtype=objs.dtype)
    if F == 1:
        return jnp.array([jnp.inf], dtype=objs.dtype)

    M = objs.shape[1]
    cd = jnp.zeros((F,), dtype=objs.dtype)
    sub = objs[idx, :]  # (F, M)

    for m in range(M):
        order = jnp.argsort(sub[:, m])
        sorted_vals = sub[order, m]
        denom = sorted_vals[-1] - sorted_vals[0]
        contrib = jnp.zeros((F,)).at[jnp.array([0, F-1])].set(jnp.inf)
        if F > 2:
            diff = (sorted_vals[2:] - sorted_vals[:-2]) / (denom + 1e-12)
            contrib = contrib.at[1:-1].add(diff)
        inv_order = jnp.empty_like(order).at[order].set(jnp.arange(F))
        cd = cd + contrib[inv_order]
    return cd


def crowding_distance_all(objs: jnp.ndarray, ranks: jnp.ndarray) -> jnp.ndarray:
    N = objs.shape[0]
    cd = jnp.zeros((N,), dtype=objs.dtype)
    max_rank = int(jnp.max(ranks)) if N > 0 else -1
    for r in range(max_rank + 1):
        idx = jnp.where(ranks == r)[0]
        if idx.size == 0:
            continue
        cd_r = crowding_distance_for_front(objs, idx)
        cd = cd.at[idx].set(cd_r)
    return cd


def binary_tournament(key, ranks: jnp.ndarray, crowd: jnp.ndarray, num: int):
    N = ranks.shape[0]
    key, k = jax.random.split(key)
    cand = jax.random.randint(k, (num, 2), 0, N)  # (num,2)
    a, b = cand[:, 0], cand[:, 1]
    ra, rb = ranks[a], ranks[b]
    ca, cb = crowd[a], crowd[b]
    better_a = jnp.logical_or(ra < rb, jnp.logical_and(ra == rb, ca > cb))
    chosen = jnp.where(better_a, a, b)
    return key, chosen


def sbx_crossover(key, p1: jnp.ndarray, p2: jnp.ndarray,
                  pc=0.9, eta_c=15.0, lower=0.0, upper=1.0):
    B, L = p1.shape
    key, kprob, ku = jax.random.split(key, 3)
    do_cx = jax.random.bernoulli(kprob, pc, (B, 1))
    u = jax.random.uniform(ku, (B, L), minval=0.0, maxval=1.0)
    beta = jnp.where(u <= 0.5, (2.0*u)**(1.0/(eta_c+1.0)),
                              (1.0/(2.0*(1.0-u)))**(1.0/(eta_c+1.0)))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    c1 = jnp.where(do_cx, c1, p1)
    c2 = jnp.where(do_cx, c2, p2)
    c1 = jnp.clip(c1, lower, upper)
    c2 = jnp.clip(c2, lower, upper)
    return key, c1, c2


def polynomial_mutation(key, X, pm=0.01, eta_m=20.0,
                        lower=-5.0, upper=5.0):
    N, L = X.shape
    key, kmask, kr = jax.random.split(key, 3)
    mask = jax.random.bernoulli(kmask, pm, X.shape)
    r = jax.random.uniform(kr, X.shape, minval=0.0, maxval=1.0)

    xl, xu = lower, upper
    delta1 = (X - xl) / (xu - xl + 1e-12)
    delta2 = (xu - X) / (xu - xl + 1e-12)
    mut_pow = 1.0 / (eta_m + 1.0)

    r_less = (r < 0.5)
    xy1 = 1.0 - delta1
    val1 = 2.0*r + (1.0 - 2.0*r) * (xy1 ** (eta_m + 1.0))
    deltaq1 = (val1 ** mut_pow) - 1.0

    xy2 = 1.0 - delta2
    val2 = 2.0*(1.0 - r) + 2.0*(r - 0.5) * (xy2 ** (eta_m + 1.0))
    deltaq2 = 1.0 - (val2 ** mut_pow)

    deltaq = jnp.where(r_less, deltaq1, deltaq2)
    X_mut = X + jnp.where(mask, deltaq*(xu - xl), 0.0)
    X_mut = jnp.clip(X_mut, xl, xu)
    return key, X_mut


def environmental_selection(pop, objs, N):
    ranks = fast_non_dominated_ranks(objs)  # (2N,)
    selected_mask = jnp.zeros((pop.shape[0],), dtype=bool)
    selected_cnt = 0
    max_rank = int(jnp.max(ranks))

    for r in range(max_rank + 1):
        idx = jnp.where(ranks == r)[0]
        sz = int(idx.size)
        if selected_cnt + sz <= N:
            selected_mask = selected_mask.at[idx].set(True)
            selected_cnt += sz
        else:
            K = N - selected_cnt
            if K > 0:
                cd = crowding_distance_for_front(objs, idx)
                _, top_pos = jax.lax.top_k(cd, K)
                chosen = idx[top_pos]
                selected_mask = selected_mask.at[chosen].set(True)
                selected_cnt += K
            break

    next_pop = pop[selected_mask]
    next_objs = objs[selected_mask]
    next_ranks = ranks[selected_mask]
    return next_pop, next_objs, next_ranks



def train(get_fitness, sim_mgr, pop_size=50, params_num=260, max_iters=2000, seed=0,
          lower=-5.0, upper=5.0,
          pc=0.9, pm=None, eta_c=15.0, eta_m=20.0):
    if pm is None:
        pm = 1.0 / params_num

    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    pop = jax.random.uniform(k0, (pop_size, params_num), minval=lower, maxval=upper)

    # objs: shape (n, M) where M = number of objectives returned by get_fitness
    # assume get_fitness accepts batch pop (n, m) and returns (n, M)
    objs = -(get_fitness(sim_mgr, pop))
    ranks = fast_non_dominated_ranks(objs)
    crowd = crowding_distance_all(objs, ranks)

    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    bestFitness = jnp.inf
    bestParams = None
    runtime = 0.0

    for iter in range(1, max_iters+1):
        t0 = time.time()

        key, parent_idx = binary_tournament(key, ranks, crowd, pop_size)
        parents = pop[parent_idx]

        key, kperm = jax.random.split(key)
        perm = jax.random.permutation(kperm, pop_size)
        p1 = parents[perm[0::2]]
        p2 = parents[perm[1::2]]

        key, c1, c2 = sbx_crossover(key, p1, p2, pc=pc, eta_c=eta_c, lower=lower, upper=upper)
        children = jnp.vstack([c1, c2])
        key, children = polynomial_mutation(key, children, pm=pm, eta_m=eta_m, lower=lower, upper=upper)

        objs_children = -(get_fitness(sim_mgr, children))

        pop_comb = jnp.vstack([pop, children])
        objs_comb = jnp.vstack([objs, objs_children])
        pop, objs, _ = environmental_selection(pop_comb, objs_comb, pop_size)

        ranks = fast_non_dominated_ranks(objs)
        crowd = crowding_distance_all(objs, ranks)

        sum_losses = jnp.sum(objs, axis=1)

        best_idx = int(jnp.argmin(sum_losses))
        best_losses = objs[best_idx]
        best_total = float(jnp.sum(best_losses))
        various_loss = np.mean(np.array(objs, copy=False), axis=0)

        loss_ls.append(best_total)
        various_loss_ls.append(various_loss)

        took = time.time() - t0
        iter_time_ls.append(took)
        runtime += took

        if best_total < float(bestFitness):
            bestFitness = best_total
            bestParams = pop[best_idx]
            print(f"iter {iter}: best_total={best_total:.6f}, losses={best_losses.tolist()}  pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")

    print(f"\nFinished at iter={max_iters}, last loss(avg)={loss_ls[-1]:.2e}, best loss={bestFitness:.2e}")

    return Result(best_w=bestParams, best_fit=bestFitness, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)
