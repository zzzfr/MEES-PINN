import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from src.data import DataSampler, LowDiscrepancySampler
from typing import Sequence
import numpy as np

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import DataLoader, SimManager, addbc, stack_outputs,ic_fitter
from src.utils.process_variables import process_variables

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
    PINN for Navier-Stokes 2D with Backstep Geometry:
    PDE:
        ∂u/∂t = ν * (∂²u/∂x² + ∂²u/∂y²) - u * ∂u/∂x - v * ∂u/∂y - ∂p/∂x
        ∂v/∂t = ν * (∂²v/∂x² + ∂²v/∂y²) - u * ∂v/∂x - v * ∂v/∂y - ∂p/∂y
        ∂u/∂x + ∂v/∂y = 0  
        
    IC:
        u(x, y, 0) = 0.0  
        v(x, y, 0) = 0.0  
        p(x, y, 0) = 0.0  

    BC:
        u(x, y, t) = 0   v(x, y, t) = 0
        u(x, y, t) = 0   v(x, y, t) = 0  

        u(x, y, t) = u_func(x, y)  
        v(x, y, t) = 0  
    """
    def derivatives(self, params, X):
        def forward(z):
            return self.apply(params, z[None, :])[0]

        def u_fn(z): return forward(z)[0]
        def v_fn(z): return forward(z)[1]
        def p_fn(z): return forward(z)[2]

        grads_u = jax.vmap(jax.grad(u_fn))(X)
        grads_v = jax.vmap(jax.grad(v_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)
        hess_v = jax.vmap(jax.hessian(v_fn))(X)
        grads_p = jax.vmap(jax.grad(p_fn))(X)

        u = jax.vmap(u_fn)(X)[:, None]
        v = jax.vmap(v_fn)(X)[:, None]
        p = jax.vmap(p_fn)(X)[:, None]

        return {
            'u': u,
            'v': v,
            'p': p,
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2],
            'u_xx': hess_u[:, 0, 0:1], 'u_yy': hess_u[:, 1, 1:2],
            'v_x': grads_v[:, 0:1], 'v_y': grads_v[:, 1:2],
            'v_xx': hess_v[:, 0, 0:1], 'v_yy': hess_v[:, 1, 1:2],
            'p_x': grads_p[:, 0:1], 'p_y': grads_p[:, 1:2],
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, datapath=ref_dir/'ns_0_obstacle.dat', nu=1 / 100,
        bbox=[0, 4, 0, 2],
        # obstacle={
        #     'circ': [(1, 0.5, 0.2), (2, 0.5, 0.3), (2.7, 1.6, 0.2)],
        #     'rec': [(2.8, 3.6, 0.8, 1.1)]
        # },
            obstacle={},):
 
        self.max_steps = 1
        self.obs_shape = tuple([2, ])
        self.act_shape = tuple([3, ])

 
        self.nu = nu


        eps = 1e-5
        self.bbox = bbox

        self.geom = geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        rec = geometry.Rectangle(xmin=[bbox[0] - eps, bbox[3] / 2], xmax=[bbox[1] / 2, bbox[3] + eps])
        self.geom = geometry.csg.CSGDifference(self.geom, rec)
        self.geom_time = None

        self.output_dim = 3
        self.input_dim = self.geom_time.dim if self.geom_time is not None else self.geom.dim
        if hidden_layers is not None:
            parts = hidden_layers.split('*')
            width, depth = parts
            self.net = PINN(width=int(width), depth=int(depth), input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        for name, configs in obstacle.items():
            if name == 'circ':
                for c in configs:
                    diff_geom = geometry.Disk(c[:2], c[2])
                    self.geom = geometry.csg.CSGDifference(self.geom, diff_geom)
            elif name == 'rec':
                for c in configs:
                    diff_geom = geometry.Rectangle(xmin=[c[0], c[2]], xmax=[c[1], c[3]])
                    self.geom = geometry.csg.CSGDifference(self.geom, diff_geom)


        self.seed = 0 
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)  
        self.num_params = self.param_size
        self.layout = ['u', 'v', 'p','u_x', 'u_y', 'u_xx', 'u_yy','v_x', 'v_y', 'v_xx', 'v_yy','p_x', 'p_y']


        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0


        def boundary_in(x, on_boundary):
            return on_boundary & jnp.isclose(x[:, 0], bbox[0]) & jnp.isclose(x[:, 1], bbox[2])

        def boundary_out(x, on_boundary):
            return on_boundary & jnp.isclose(x[:, 0], bbox[1]) & jnp.isclose(x[:, 1], bbox[3])

        def boundary_other(x, on_boundary):
            return on_boundary & ~(boundary_in(x, on_boundary) | boundary_out(x, on_boundary))

        def u_func(x):
            return x[:, 1:2] * (1 - x[:, 1:2]) * 4

        bc_config = [
            {
                'component': 0,
                'function': u_func,
                'bc': boundary_in,
                'type': 'dirichlet'
            },
            {
                'component': 1,
                'function': (lambda _: 0),
                'bc': boundary_in,
                'type': 'dirichlet'
            },
            {
                'component': 2,
                'function': (lambda _: 0),
                'bc': boundary_out,
                'type': 'dirichlet'
            },
            {
                'component': 0,
                'function': (lambda _: 0),
                'bc': boundary_other,
                'type': 'dirichlet'
            },
            {
                'component': 1,
                'function': (lambda _: 0),
                'bc': boundary_other,
                'type': 'dirichlet'
            }
        ]

        self.bcs = addbc(bc_config, self.geom)

        # --- pde points ---
        self.pde_data = DataSampler(self.geom, self.bcs, mul=4).train_x_all 
        self.X_pde = self.pde_data
        self.Y_pde = np.zeros(shape=(len(self.X_pde), self.output_dim))

        # --- data points ---
        def data_load(path):
            loader = DataLoader()
            loader.load(path, input_dim=self.input_dim, output_dim=self.output_dim, t_transpose=False)
            Data = loader.ref_data
            X_data = jnp.array(Data[:, :self.input_dim], jnp.float32)
            Y_data = jnp.array(Data[:, self.input_dim:], jnp.float32)

            return X_data, Y_data

        self.X_data, self.Y_data = data_load(datapath)

        # --- mini-batch ---
        self.BatchSize_eq = BatchSize_eq
        self.BatchSize_data = BatchSize_data
        domain_bounds = [
            [self.bbox[0], self.bbox[1]],  # x
            [self.bbox[2], self.bbox[3]],  # y
        ]
        self.pde_sampler = LowDiscrepancySampler(self.X_pde, self.Y_pde, domain_bounds)

        if len(self.X_data) > self.BatchSize_data:
            self.is_batch = True
            self.data_size = self.BatchSize_data
            self.data_sampler = LowDiscrepancySampler(self.X_data, self.Y_data, domain_bounds)
        else:
            self.is_batch = False
            self.data_size = len(self.X_data)

        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq) 
            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data
            masks = [jnp.array(bc.filter(X_eq), dtype=bool).reshape(-1) for bc in self.bcs]
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
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        u_x, u_y, u_xx, u_yy = pred[:, 3:4], pred[:, 4:5], pred[:, 5:6], pred[:, 6:7]
        v_x, v_y, v_xx, v_yy = pred[:, 7:8], pred[:, 8:9], pred[:, 9:10], pred[:, 10:11]
        p_x, p_y = pred[:, 11:12], pred[:, 12:13]

        r_u = u * u_x + v * u_y - p_x - self.nu * (u_xx + u_yy)
        r_v = u * v_x + v * v_y - p_y - self.nu * (v_xx + v_yy)

        r_continuity = u_x + v_y

        return jnp.hstack([r_u, r_v, r_continuity])  # shape (N, 3)

    def data_fn(self, Y_ref, pred, mask):
        u_true = Y_ref[-self.data_size:, 0:1]
        v_true = Y_ref[-self.data_size:, 1:2]
        p_true = Y_ref[-self.data_size:, 2:3]

        u_pred = pred[-self.data_size:, 0:1]
        v_pred = pred[-self.data_size:, 1:2]
        p_pred = pred[-self.data_size:, 2:3]

        loss_u = (u_pred - u_true) ** 2
        loss_v = (v_pred - v_true) ** 2
        loss_p = (p_pred - p_true) ** 2

        total_loss = (jnp.sum(loss_u) + jnp.sum(loss_v) + jnp.sum(loss_p)) / pred.shape[0]
        return total_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        if pred.ndim == 3 and pred.shape[1] == 1:
            pred2d = pred[:, 0, :]         # (N, layout_dim)
        elif pred.ndim == 2:
            pred2d = pred                  
        else:
            raise ValueError(f"Unexpected pred shape: {pred.shape}")

        pde_size = self.BatchSize_eq
        pde_pred = pred2d[:pde_size, :]     # (N_pde, layout_dim)
        X_pde = X_batch[:pde_size, :]


        for i, m in enumerate(bcs_masks):
            assert m.shape == (X_pde.shape[0],), \
                f"BC mask {i} shape mismatch: {m.shape}, expected {(X_pde.shape[0],)}"

        # ---- PDE loss ----
        r_all = self.pde_fn(pde_pred)     
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)
        interior_mask = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior_mask[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior_mask) + 1e-8)

        # ---- IC / BC loss ----
        bc_loss_sum, ic_loss_sum = 0.0, 0.0
        bc_count, ic_count = 0, 0
        for bc, mask in zip(self.bcs, bcs_masks):
            mask_f = mask[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_pde)  
            term = jnp.sum((err ** 2) * mask_f) / (jnp.sum(mask_f) + 1e-8)
            if isinstance(bc, IC):
                ic_loss_sum += term; ic_count += 1
            else:
                bc_loss_sum += term; bc_count += 1
        ic_loss = ic_loss_sum / (ic_count + 1e-8)
        bc_loss = bc_loss_sum / (bc_count + 1e-8)

        # ---- Data loss ----
        data_loss = self.data_fn(Y_batch, pred2d, interior_mask) 

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
        if obs.ndim == 2:
            obs = obs[None, ...]
        obs_shared = obs[0]

        def f_single(params):
            def per_point(x):
                return stack_outputs(self.net.derivatives(params, x[None, :]), self.grad_keys)
            return jax.vmap(per_point)(obs_shared)

        actions = jax.vmap(f_single)(params_tree)
        return actions, p_states
