import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from src.data import DataSampler_T, LowDiscrepancySampler
import numpy as np

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import DataLoader, SimManager, addbc, stack_outputs, ic_fitter
from src.nn import BaseNN
from pathlib import Path
from typing import Sequence

project_root = Path(__file__).resolve().parent.parent.parent.parent
ref_dir = project_root / 'ref'

BatchSize_eq = 4096
BatchSize_data = 4096


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]



class PINN(BaseNN):
    """
    PINN for 1D Kuramoto–Sivashinsky Equation:
    PDE:
        ∂u/∂t + α * u * ∂u/∂x + β * ∂²u/∂x² + γ * ∂⁴u/∂x⁴ = 0

    IC:
        u(x, 0) = cos(x) * (1 + sin(x))

    BC:
        u(0, t) = u(2π, t)
        ∂u/∂x (0, t) = ∂u/∂x (2π, t)
    """

    def derivatives(self, params, X):
        def forward(z):
            out = self.apply(params, z[None, :])
            return out[0]
        
        def u_fn(z):
            return forward(z)[0] 

        grads_u = jax.vmap(jax.grad(u_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)
        u_xx_fn = lambda z: jax.hessian(u_fn)(z)[0, 0] 
        u_xxxx = jax.vmap(lambda z: jax.hessian(u_xx_fn)(z)[0, 0])(X)

        # u 预测值
        u = jax.vmap(u_fn)(X).reshape(-1, 1)

        return {
            'u': u,
            'u_x': grads_u[:, 0:1],        # ∂u/∂x
            'u_t': grads_u[:, 1:2],        # ∂u/∂t
            'u_xx': hess_u[:, 0, 0:1],     # ∂²u/∂x²
            'u_xxxx': u_xxxx.reshape(-1, 1)  # ∂⁴u/∂x⁴
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, datapath=ref_dir/'Kuramoto_Sivashinsky.dat', bbox=[0, 2 * np.pi, 0, 1], alpha=100 / 16, beta=100 / (16 * 16), gamma=100 / (16**4)):
        
        self.max_steps = 1
        self.obs_shape = tuple([2, ])
        self.act_shape = tuple([1, ])

        
        self.alpha = alpha    
        self.beta = beta
        self.gamma = gamma

        
        self.bbox = bbox
        self.geom = geometry.Interval(self.bbox[0], self.bbox[1])
        time_domain = geometry.TimeDomain(self.bbox[2], self.bbox[3])
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
        self.layout = ['u', 'u_x', 'u_t', 'u_xx', 'u_xxxx']


        
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        

        bc_config = [{
            'component': 0,
            'function': (lambda x: jnp.cos(x[:, 0:1]) * (1 + jnp.sin(x[:, 0:1]))),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }]

        self.bcs = addbc(bc_config, self.geom_time)

        # --- pde points ---
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all
        self.X_pde = self.pde_data
        self.Y_pde = jnp.zeros_like(self.X_pde[:, :1])

        # --- data points ---
        def data_load(path):
            loader = DataLoader()
            loader.load(path, input_dim=self.input_dim, output_dim=self.output_dim, t_transpose=False)
            Data = loader.ref_data
            X_data = jnp.array(Data[:, :self.input_dim], jnp.float32)
            Y_data = jnp.array(Data[:, self.input_dim:], jnp.float32)

            return X_data, Y_data

        self.X_data, self.Y_data = data_load(datapath)

        # --- mini batch ---
        self.BatchSize_eq = BatchSize_eq
        self.BatchSize_data = BatchSize_data
        domain_bounds = [
            [self.bbox[0], self.bbox[1]],  # x
            [self.bbox[2], self.bbox[3]],  # t
        ]
        self.pde_sampler = LowDiscrepancySampler(self.X_pde, self.Y_pde, domain_bounds)
        if len(self.X_data) > self.BatchSize_data:
            self.is_batch = True
            self.data_size = self.BatchSize_data
            self.data_sampler = LowDiscrepancySampler(self.X_data, self.Y_data, domain_bounds)
        else:
            self.is_batch = False
            self.data_size = len(self.X_data)

        # ---  reset / step ---
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

    def pde_fn(self, pred):
        
        
        u, u_t, u_x, u_xx, u_xxxx = (
            pred[:, 0:1],  # u
            pred[:, 1:2],  # u_t (∂u/∂t)
            pred[:, 2:3],  # u_x (∂u/∂x)
            pred[:, 3:4],  # u_xx (∂²u/∂x²)
            pred[:, 4:5],  # u_xxxx (∂⁴u/∂x⁴)
        )

       
        r_u = u_t + self.alpha * u * u_x + self.beta * u_xx + self.gamma * u_xxxx

        return r_u  

    def data_fn(self, Y_ref, pred): 
        u_ref = Y_ref[:, 0:1]
        u_pred = pred[-Y_ref.shape[0]:, 0:1] 

        loss_tmp = (u_pred - u_ref) ** 2
        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        X_eq = X_batch[:pde_size]
        pde_pred = pred[:pde_size, :]
        data_pred = pred[pde_size:]
        data_true = Y_batch[pde_size:]

        comb_mask = jnp.any(jnp.stack([mask[:pde_size] for mask in bcs_masks], axis=1), axis=1)

        # PDE loss
        r_all = self.pde_fn(pde_pred)
        interior = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior) + 1e-8)

        # IC/BC loss
        ic_loss_sum = 0.0
        bc_loss_sum = 0.0
        ic_category = 0
        bc_category = 0
        for bc, mask in zip(self.bcs, bcs_masks):
            mask_eq = mask[:pde_size]
            mask_f = mask_eq[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_eq)
            term = jnp.sum((err ** 2) * mask_f) / (jnp.sum(mask_f) + 1e-8)
            if isinstance(bc, IC):
                ic_loss_sum += term
                ic_category += 1
            else:
                bc_loss_sum += term
                bc_category += 1
        ic_loss = ic_loss_sum / (ic_category + 1e-8)
        bc_loss = bc_loss_sum / (bc_category + 1e-8)

        data_loss = self.data_fn(data_true, data_pred)

        loss = jnp.hstack([self.pde_lambda * pde_loss, self.ic_lambda * ic_loss, self.bc_lambda * bc_loss,self.data_lambda * data_loss])

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
        # flat → pytree
        params_tree = self.format_params_fn(flat_params)  # batched pytree

        
        obs = t_states.obs 

        # forward + derivatives
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)  # dict
            return stack_outputs(outs, self.grad_keys)  # (N_pts, 4)

        actions = jax.vmap(f_single)(params_tree, obs)  # (B, N_pts, 4)
        return actions, p_states
