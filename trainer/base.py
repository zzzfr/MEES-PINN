import jax.numpy as jnp
from src.utils import multi_SimManager
from . import xNES_Adam, AGA, CMAES, PSO, GA, xNES_NAG, NSGA2, MOEAD
from .utils import save_result


def trainer(pde, policy, train_task, config, iter_num):

    sim_mgr = multi_SimManager(n_repeats=1, test_n_repeats=1, pop_size=0, n_evaluations=1,
                               policy_net=policy, train_vec_task=train_task, valid_vec_task=train_task,
                               seed=config.seed)

    get_f = get_fitness
    result = None
    if config.algo == "xNES_Adam":
        result = xNES_Adam.train(get_f, policy, sim_mgr, max_iters=config.max_iters, lr=config.lr)
    elif config.algo == 'xNES_NAG':
        result = xNES_NAG.train(get_f, policy, sim_mgr, max_iters=config.max_iters, lr=config.lr)
    elif config.algo == 'CMAES':
        result = CMAES.train(get_f, policy, sim_mgr, pop_size=config.pop_size, init_stdev=config.init_stdev, max_iters=config.max_iters)
    elif config.algo == 'GA':
        result = GA.train(get_f, policy, sim_mgr, pop_size=config.pop_size, max_iters=config.max_iters)
    elif config.algo == 'PSO':
        result = PSO.train(get_f, policy, sim_mgr, pop_size=config.pop_size, max_iters=config.max_iters)
    elif config.algo == 'AGA':
        result = AGA.train(get_f, sim_mgr, popsize=config.pop_size, params_num=policy.num_params, max_iters=config.max_iters)
    elif config.algo == 'MOEAD':
        get_f = get_multi_fitness
        result = MOEAD.train(get_f, sim_mgr, pop_size=config.pop_size, params_num=policy.num_params, max_iters=config.max_iters)
    elif config.algo == 'NSGA2':
        get_f = get_multi_fitness
        result = NSGA2.train(get_f, sim_mgr, pop_size=config.pop_size, params_num=policy.num_params, max_iters=config.max_iters)
    if result is None:
        raise NotImplementedError("The current method has not bemmen selected")

    save_result(pde=pde, policy=policy, train_task=train_task, result=result, config=config, iter_num=iter_num)


def get_fitness(sim_mgr, samples):
    losses, _ = sim_mgr.eval_params(params=samples, test=False)
    scores = jnp.sum(losses, axis=1)
    return -losses, scores


def get_multi_fitness(sim_mgr, samples):
    scores, _ = sim_mgr.eval_params(params=samples, test=False)
    return scores

