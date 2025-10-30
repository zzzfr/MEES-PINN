import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from src.data import DataSampler_T,LowDiscrepancySampler 
from typing import Sequence

import numpy as np

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import DataLoader, SimManager, addbc, stack_outputs, ic_fitter
from src.nn import BaseNN
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
ref_dir = project_root / 'ref'

BatchSize_eq = 8192
BatchSize_data = 4096
@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]

class PINN(BaseNN):
    """
        PINN for 2D Heat2D_LongTimes:
        PDE:
            ∂u/∂t - 0.001 * (∂²u/∂x² + ∂²u/∂y²) - 5 * sin(K * u²) * (1 + 2 * sin(t * π / 4)) * sin(m1 * π * x1) * sin(m2 * π * x2) = 0
        
        IC:

            u(x1, x2, 0) = sin(INITIAL_COEF_1 * x1) * sin(INITIAL_COEF_2 * x2) = 0
     
        BC: 
            u(0, x2, t) = 0  u(1, x2, t) = 0
            u(x1, 0, t) = 0  u(x1, 1, t) = 0
    """
    def derivatives(self, params, X):
        def forward(z):
            out = self.apply(params, z[None, :])
            return out[0]

        def u_fn(z): return forward(z)[0]

        
        grads_u = jax.vmap(jax.grad(u_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)

    
        u = jax.vmap(u_fn)(X).reshape(-1, 1)

        
        return {
            'u': u,
            'u_xx': hess_u[:, 0, 0].reshape(-1, 1),  # ∂²u/∂x²
            'u_yy': hess_u[:, 1, 1].reshape(-1, 1),  # ∂²u/∂y²
            'u_t': grads_u[:, 2:3],  # ∂u/∂t
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, datapath=ref_dir/'heat_longtime.dat', bbox=[0, 1, 0, 1, 0, 100], k=1, m1=4, m2=2):
       
        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([1, ])

       
        self.K = k
        self.m1 = m1
        self.m2 = m2
        self.INITIAL_COEF_1 = 4 * np.pi
        self.INITIAL_COEF_2 = 3 * np.pi

       
        self.bbox = bbox
        self.geom = geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        time_domain = geometry.TimeDomain(bbox[4], bbox[5])
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)

        
        self.output_dim = 1
        self.input_dim = self.geom_time.dim if self.geom_time is not None else self.geom.dim
        if hidden_layers is not None:
            parts = hidden_layers.split('*')
            width, depth = parts
            self.net = PINN(width=int(width), depth=int(depth), input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        
        self.seed = 0  
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)  
        self.num_params = self.param_size
        self.layout = ['u', 'u_xx', 'u_yy', 'u_t']

       
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        
        def ic_func(x):
            return jnp.sin(self.INITIAL_COEF_1 * x[:, 0:1]) * jnp.sin(self.INITIAL_COEF_2 * x[:, 1:2])
        bc_config = [{
            'component': 0,
            'function': ic_func,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }]

        self.bcs = addbc(bc_config, self.geom_time)

        # --- pde points ---
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all
        self.X_pde = self.pde_data
        self.Y_pde = np.zeros(shape=(len(self.X_pde), self.output_dim))

        # --- data points ---
        def data_load(path):
            loader = DataLoader()
            loader.load(path, input_dim=self.input_dim, output_dim=self.output_dim, t_transpose=True)
            Data = loader.ref_data
            X_data = jnp.array(Data[:, :self.input_dim], jnp.float32)
            Y_data = jnp.array(Data[:, self.input_dim:], jnp.float32)

            return X_data, Y_data

        self.X_data, self.Y_data = data_load(datapath)

        # --- mini-batch ---
        self.BatchSize_eq = BatchSize_eq
        self.BatchSize_data = BatchSize_data
        domain_bounds = [
            [self.bbox[0], self.bbox[1]],
            [self.bbox[2], self.bbox[3]],
            [self.bbox[4], self.bbox[5]],
        ]
        self.pde_sampler = LowDiscrepancySampler(self.X_pde, self.Y_pde, domain_bounds)

        if len(self.X_data) > self.BatchSize_data:
            self.is_batch = True
            self.data_size = self.BatchSize_data
            self.data_sampler = LowDiscrepancySampler(self.X_data, self.Y_data, domain_bounds)
        else:
            self.is_batch = False
            self.data_size = len(self.X_data)

        # --- reset / step ---
        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq) 
            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data
            masks = [bc.filter(X_eq) for bc in self.bcs]
            X_batch = np.concatenate((X_eq, X_d), axis=0)
            Y_batch = np.concatenate((Y_eq, Y_d), axis=0)
            return State(obs=X_batch, labels=Y_batch, bcs_masks=masks)

        def step_fn(states, actions):
            def single_loss_fn(s, a):
                reward = -self.loss_fn(pred=a, X_batch=s.obs, Y_batch=s.labels, bcs_masks=s.bcs_masks)
                return reward

            rewards = jax.vmap(single_loss_fn)(states, actions)
            done = jnp.ones((actions.shape[0],), dtype=jnp.int32)
            return states, rewards, done

        self._reset_fn = reset_fn
        self._step_fn = step_fn

    def _init_params(self):
        key = random.PRNGKey(self.seed)
        dummy = jnp.zeros((1, self.input_dim))
        self.params_tree = self.net.init(key, dummy)
        self.param_size, self.fmt = get_params_format_fn(self.params_tree)

    def update_seed(self, seed):
        self.seed = seed
        self._init_params()

    def pde_fn(self, pred, X=None):
       
        u = pred[:, 0:1]
        u_t = pred[:, 1:2]
        u_xx = pred[:, 2:3]
        u_yy = pred[:, 3:4]

        x1 = X[:, 0:1]
        x2 = X[:, 1:2]
        t = X[:, 2:3]

        r_u = u_t - 0.001 * (u_xx + u_yy) - 5 * jnp.sin(self.K * u ** 2) * (1 + 2 * jnp.sin(t * np.pi / 4)) * jnp.sin(self.m1 * np.pi * x1) * jnp.sin(self.m2 * np.pi * x2)

        return r_u

    def data_fn(self, Y_ref, pred, mask):
        u_true = Y_ref[-self.data_size:, 0:1]
        u_pred = pred[-self.data_size:, 0:1]
        loss_tmp = (u_pred - u_true) ** 2

        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        pde_pred = pred[:pde_size, :]
        X_pde = X_batch[:pde_size, :]

        # ---- PDE loss ----
        r_all = self.pde_fn(pde_pred, X_pde)  # shape: (N_pde, 2) -> [res_u, res_v]
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)
        interior_mask = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior_mask[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior_mask) + 1e-8)

        # ---- IC / BC loss ----
        bc_loss_sum, ic_loss_sum = 0.0, 0.0
        bc_count, ic_count = 0, 0

        for bc, mask in zip(self.bcs, bcs_masks):
            mask_f = mask[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_pde)  # shape: (N_pde, 1)
            term = jnp.sum((err ** 2) * mask_f) / (jnp.sum(mask_f) + 1e-8)

            if isinstance(bc, IC):
                ic_loss_sum += term
                ic_count += 1
            else:
                bc_loss_sum += term
                bc_count += 1

        ic_loss = ic_loss_sum / (ic_count + 1e-8)
        bc_loss = bc_loss_sum / (bc_count + 1e-8)

        # ---- Data loss ----
        data_loss = self.data_fn(Y_batch, pred, interior_mask)

        loss = jnp.hstack([self.pde_lambda * pde_loss, self.ic_lambda * ic_loss, self.bc_lambda * bc_loss, self.data_lambda * data_loss])

        return loss

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)


class PINNsPolicy(PolicyNetwork):
    def __init__(self, net, num_params, format_params_fn, grad_keys):
        self.net = net
        self.num_params = num_params
        self.format_params_fn = format_params_fn
        self.grad_keys = grad_keys

    def get_actions(self,
                    t_states: TaskState,
                    flat_params: jnp.ndarray,
                    p_states: PolicyState):
        params_tree = self.format_params_fn(flat_params)

        obs = t_states.obs

        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)
            return stack_outputs(outs, self.grad_keys)

        actions = jax.vmap(f_single)(params_tree, obs)
        return actions, p_states
