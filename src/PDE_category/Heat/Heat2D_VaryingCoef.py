import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from scipy import interpolate
from src.data import DataSampler_T, LowDiscrepancySampler
import numpy as np
from typing import Sequence

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
    PINN for 2D heat equation with varying diffusion coefficient:

    PDE:
        ∂u/∂t - a(x,y) * (∂²u/∂x² + ∂²u/∂y²) = f(x,y,t)

    IC:
        u(x,y,0) = 0

    BC:
        u(x_min,y,t) = u(x_max,y,t) = 0
        u(x,y_min,t) = u(x,y_max,t) = 0
    """
    def derivatives(self, params, X):
        def forward(z):
            out = self.apply(params, z[None, :])
            return out[0]
        def u_fn(z): return forward(z)[0]

        grads_u = jax.vmap(jax.grad(u_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)
        u = jax.vmap(u_fn)(X).reshape(-1, 1)

        u_t = grads_u[:, 2:3]             # ∂u/∂t
        u_xx = hess_u[:, 0, 0:1]          # ∂²u/∂x²
        u_yy = hess_u[:, 1, 1:2]          # ∂²u/∂y²

        return {
            'u': u,
            'u_t': u_t,
            'u_xx': u_xx,
            'u_yy': u_yy           
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, datapath=ref_dir/'heat_darcy.dat', bbox=[0, 1, 0, 1, 0, 5], A=200, m=(1, 5, 1)):
        
        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([1, ])

        
        self.A = A
        self.m = m

       
        self.bbox = bbox
        self.geom = geometry.Rectangle(xmin=(self.bbox[0], self.bbox[2]), xmax=(self.bbox[1], self.bbox[3]))
        time_domain = geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)
       
        self.heat_2d_coef = np.loadtxt(ref_dir / 'heat_2d_coef_256.dat')
        self._coef_ready = False
        self._current_X_pde = None

        
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
        self.layout = ['u','u_t', 'u_xx','u_yy']

        
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

       
        def boundary_t0(x, on_initial):
           
            xx = jnp.atleast_2d(x)
            oi = jnp.ravel(on_initial)  
            return jnp.logical_and(oi, jnp.isclose(xx[:, 2], bbox[4]))

        def boundary_xb(x, on_boundary):
            
            xx = jnp.atleast_2d(x)
            ob = jnp.ravel(on_boundary)
            mask_edge = (
                jnp.isclose(xx[:, 0], bbox[0]) |
                jnp.isclose(xx[:, 0], bbox[1]) |
                jnp.isclose(xx[:, 1], bbox[2]) |
                jnp.isclose(xx[:, 1], bbox[3])
            )
            return jnp.logical_and(ob, mask_edge)


        bc_config = [{
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_xb,
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

        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq)
            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data

            X_batch = np.concatenate((X_eq, X_d), axis=0)
            Y_batch = np.concatenate((Y_eq, Y_d), axis=0)

            masks = [jnp.squeeze(bc.filter(X_eq)) for bc in self.bcs]

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

    def _prepare_coef_grid(self):
        if self._coef_ready:
            return
        coef = self.heat_2d_coef  # shape (..., 3): [x, y, a]
        xg = np.unique(coef[:, 0])
        yg = np.unique(coef[:, 1])
        nx, ny = len(xg), len(yg)
        idx = np.lexsort((coef[:, 1], coef[:, 0]))  
        A_grid = coef[idx][:, 2].reshape(nx, ny)

        self.xg = jnp.array(xg, dtype=jnp.float32)
        self.yg = jnp.array(yg, dtype=jnp.float32)
        self.A_grid = jnp.array(A_grid, dtype=jnp.float32)
        self._coef_ready = True

    def _a_nearest(self, x, y):
       
        self._prepare_coef_grid()
        xs = x.squeeze(-1)
        ys = y.squeeze(-1)
        ix = jnp.clip(jnp.searchsorted(self.xg, xs, side='left'), 0, self.xg.shape[0]-1)
        iy = jnp.clip(jnp.searchsorted(self.yg, ys, side='left'), 0, self.yg.shape[0]-1)
        a = self.A_grid[ix, iy]
        return a[:, None]
    def pde_fn(self, pred, X=None):
        X = self._current_X_pde
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]

        u = pred[:, 0:1]
        u_t = pred[:, 1:2]
        u_xx = pred[:, 2:3]
        u_yy = pred[:, 3:4]

        a = self._a_nearest(x, y)  # (N,1)

        f = self.A * jnp.sin(self.m[0] * jnp.pi * x) * \
            jnp.sin(self.m[1] * jnp.pi * y) * \
            jnp.sin(self.m[2] * jnp.pi * t)

        r_u = u_t - a * (u_xx + u_yy) - f
        return r_u
    
    def data_fn(self, Y_ref, pred): 
        u_ref = Y_ref[:, 0:1]
        u_pred = pred[-Y_ref.shape[0]:, 0:1] 

        loss_tmp = (u_pred - u_ref) ** 2
        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        pde_pred = pred[:pde_size, :]
        X_pde = X_batch[:pde_size, :]

        # ---- PDE loss ----
        self._current_X_pde = X_pde
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
        data_loss = self.data_fn(Y_batch, pred)

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

        # forward + derivatives
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)
            return stack_outputs(outs, self.grad_keys)

        actions = jax.vmap(f_single)(params_tree, obs)
        return actions, p_states
