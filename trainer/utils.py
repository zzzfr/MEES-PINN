import os
import jax.numpy as jnp
from pathlib import Path
from flax.core.frozen_dict import unfreeze, freeze
from flax import serialization
import pandas as pd
import numpy as np


def flat_to_params_tree(policy, flat_vector):
    flat1 = jnp.array([flat_vector])  # (1, P)
    this_dict = policy.format_params_fn(flat1)
    new_dict = unfreeze(this_dict)
    for m_ in new_dict:
        for p_ in new_dict[m_]:
            for k_ in new_dict[m_][p_]:
                new_dict[m_][p_][k_] = new_dict[m_][p_][k_][0]
    return freeze(new_dict)


def save_best_params(pde, method_name, net_arch, params_dir, params_tree):
    filename = f"{pde}_{method_name}_{net_arch}.msgpack"
    path = params_dir / filename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(params_tree))
    os.replace(tmp_path, path)
    return path


def get_pde_name(pde):
    if hasattr(pde, '__name__'):
        return pde.__name__.split('.')[-1]

    if hasattr(pde, '__class__') and hasattr(pde.__class__, '__name__'):
        return pde.__class__.__name__

    if isinstance(pde, str):
        return pde.split('.')[-1] if '.' in pde else pde

    return str(pde).split(' ')[0].replace('<', '').replace('>', '')


def save_result(pde, policy, train_task, result, config, iter_num):
    pde_class = get_pde_name(pde)
    method_name = config.algo
    net_arch = config.hidden_layers

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TARGET_BASE = PROJECT_ROOT / 'multi_train_log' /f"log_{iter_num}" / method_name
    LOSS_DIR = TARGET_BASE / "loss_iters"
    M_LOSS_DIR = TARGET_BASE / "various_loss"
    RESULT_DIR = TARGET_BASE / "result"
    LOSS_TIME = TARGET_BASE / "loss_time_csv"
    PARAMS_DIR = TARGET_BASE / "params"
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    M_LOSS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_TIME.mkdir(parents=True, exist_ok=True)
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    params_tree = flat_to_params_tree(policy, result.best_w)
    save_path = save_best_params(pde_class, method_name, net_arch, PARAMS_DIR, params_tree)

    print(f"{pde_class} {method_name} {net_arch} 's best loss has been saved：loss(avg)={result.best_fit:.4e} → {save_path}")

    
    iter_time_cumsum = np.cumsum(result.iter_time_ls)
    df_log = pd.DataFrame({
        "iter": np.arange(1, len(result.loss_ls) + 1),
        "cum_time": iter_time_cumsum,
        "loss": result.loss_ls
    })
    loss_time_csv_path = LOSS_TIME / f"{pde_class}{net_arch}_IterTime_Loss.csv"
    df_log.to_csv(loss_time_csv_path, index=False)


    # save various loss
    df_various_ls = pd.DataFrame(result.various_loss_ls, columns=['pde_loss', 'ic_loss', 'bc_loss', 'data_loss'])
    various_ls_csv_path = M_LOSS_DIR / f"{pde_class}{net_arch}_M_LOSS.csv"
    df_various_ls.to_csv(various_ls_csv_path, index=False)


    
    if train_task.X_data is not None:
        X_input = train_task.X_data
        Y_true = train_task.Y_data
        model = policy.net

        flat_best = jnp.array([result.best_w])
        this_dict = policy.format_params_fn(flat_best)
        new_dict = unfreeze(this_dict)
        for m_ in new_dict:
            for p_ in new_dict[m_]:
                for k_ in new_dict[m_][p_]:
                    new_dict[m_][p_][k_] = new_dict[m_][p_][k_][0]
        params_tree = freeze(new_dict)

        derivs = model.derivatives(params_tree, X_input)

        if Y_true.shape[1] == 1:
            u_pred = np.asarray(derivs['u'])
            df = pd.DataFrame({
                'x': X_input[:, 0],
                'y': X_input[:, 1] if X_input.shape[1] >= 2 else 0,
                't': X_input[:, 2] if X_input.shape[1] >= 3 else 0,
                'u_true': Y_true[:, 0],
                'u_pred': u_pred[:, 0],
            })
        elif Y_true.shape[1] == 2:
            u_pred = np.asarray(derivs['u'])
            v_pred = np.asarray(derivs['v'])
            df = pd.DataFrame({
                'x': X_input[:, 0],
                'y': X_input[:, 1],
                't': X_input[:, 2] if X_input.shape[1] == 3 else 0,
                'u_true': Y_true[:, 0],
                'v_true': Y_true[:, 1],
                'u_pred': u_pred[:, 0],
                'v_pred': v_pred[:, 0],
            })
        elif Y_true.shape[1] == 3:
            u_pred = np.asarray(derivs['u'])
            v_pred = np.asarray(derivs['v'])
            p_pred = np.asarray(derivs['p'])
            df = pd.DataFrame({
                'x': X_input[:, 0],
                'y': X_input[:, 1],
                't': X_input[:, 2],
                'u_true': Y_true[:, 0],
                'v_true': Y_true[:, 1],
                'p_true': Y_true[:, 2],
                'u_pred': u_pred[:, 0],
                'v_pred': v_pred[:, 0],
                'p_pred': p_pred[:, 0],
            })
        else:
            raise ValueError(f"Unsupported output dimension: {Y_true.shape[1]}")

        csv_path = RESULT_DIR / f"{pde_class}{net_arch}_Result.csv"
        df.to_csv(csv_path, index=False)






