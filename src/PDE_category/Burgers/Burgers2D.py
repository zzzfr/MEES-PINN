
import jax 
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
import numpy as np
import scipy
from typing import Sequence

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import addbc, stack_outputs, DataLoader
from src.data import DataSampler_T, LowDiscrepancySampler
from src.nn import BaseNN
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
ref_dir = project_root / 'ref'

BatchSize_eq = 2048
BatchSize_data = 2048

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]


class PINN(BaseNN):
    """
    PINN for Burgers2D :
    PDE:
        ∂u₁/∂t + u₁·∂u₁/∂x + u₂·∂u₁/∂y − ν·(∂²u₁/∂x² + ∂²u₁/∂y²) = 0
        ∂u₂/∂t + u₁·∂u₂/∂x + u₂·∂u₂/∂y − ν·(∂²u₂/∂x² + ∂²u₂/∂y²) = 0

    IC:
        u₁(x, y, 0) = ic₁(x, y)
        u₂(x, y, 0) = ic₂(x, y)

    BC (Periodic in x and y):
        u(0,   y, t) = u(L,   y, t)
        u(L,   y, t) = u(0,   y, t)
        u(x,   0, t) = u(x,   L, t)
        u(x,   L, t) = u(x,   0, t)
    """

    def derivatives(self, params, X):
        def forward(z):
            return self.apply(params, z[None, :])[0]

        def u_fn(z): return forward(z)[0]
        def v_fn(z): return forward(z)[1]

        u = jax.vmap(u_fn)(X).reshape(-1, 1)
        v = jax.vmap(v_fn)(X).reshape(-1, 1)
        grads_u = jax.vmap(jax.grad(u_fn))(X)
        grads_v = jax.vmap(jax.grad(v_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)
        hess_v = jax.vmap(jax.hessian(v_fn))(X)

        return {
            'u': u,
            'v': v,
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2], 'u_t': grads_u[:, 2:3],
            'u_xx': hess_u[:, 0, 0:1], 'u_yy': hess_u[:, 1, 1:2],
            'v_x': grads_v[:, 0:1], 'v_y': grads_v[:, 1:2], 'v_t': grads_v[:, 2:3],
            'v_xx': hess_v[:, 0, 0:1], 'v_yy': hess_v[:, 1, 1:2]
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, datapath=ref_dir / 'burgers2d_0.dat',
                 icpath=(ref_dir / 'burgers2d_init_u_0.dat', ref_dir / 'burgers2d_init_v_0.dat'),
                 L=4, T=1, nu=0.001):
        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([2, ])
        self.nu = nu

       
        self.bbox = [0, L, 0, L, 0, T]
        self.geom = geometry.Rectangle(self.bbox[0:4:2], self.bbox[1:4:2])
        time_domain = geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)


        self.output_dim = 2
        self.input_dim = self.geom_time.dim if self.geom_time is not None else self.geom.dim
        if hidden_layers is not None:
            parts = hidden_layers.split('*')
            width, depth = parts
            self.net = PINN(width=int(width), depth=int(depth),
                            input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        self.seed = 0
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)
        self.num_params = self.param_size
        self.layout = ['u', 'v', 'u_x', 'u_y', 'u_t', 'u_xx', 'u_yy',
                       'v_x', 'v_y', 'v_t', 'v_xx', 'v_yy']

        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

       
        self.ics = (np.loadtxt(icpath[0]), np.loadtxt(icpath[1]))
        self.ic_interpolators = (
            scipy.interpolate.LinearNDInterpolator(self.ics[0][:, :2], self.ics[0][:, 2:]),
            scipy.interpolate.LinearNDInterpolator(self.ics[1][:, :2], self.ics[1][:, 2:])
        )

        def ic_func(x, component):
            return self.ic_interpolators[component](x[:, :2])

        def boundary_ic(x, on_initial):
            time_cond = jnp.isclose(x[2], 0.0)
            return jnp.logical_and(on_initial, time_cond)

        def boundary_xb(x, on_boundary):
            cond1 = jnp.isclose(x[:, 0], 0.0)
            cond2 = jnp.isclose(x[:, 0], float(L))
            x_boundary = jnp.logical_or(cond1, cond2)
            return jnp.logical_and(on_boundary, x_boundary)

        def boundary_yb(x, on_boundary):
            cond1 = jnp.isclose(x[:, 1], 0.0)
            cond2 = jnp.isclose(x[:, 1], float(L))
            y_boundary = jnp.logical_or(cond1, cond2)
            return jnp.logical_and(on_boundary, y_boundary)

        bc_config = [
            {'component': 0, 'function': lambda x: ic_func(x, 0), 'bc': boundary_ic, 'type': 'ic'},
            {'component': 1, 'function': lambda x: ic_func(x, 1), 'bc': boundary_ic, 'type': 'ic'},
            {'component': 0, 'type': 'periodic', 'component_x': 0, 'bc': boundary_xb},
            {'component': 1, 'type': 'periodic', 'component_x': 0, 'bc': boundary_xb},
            {'component': 0, 'type': 'periodic', 'component_x': 1, 'bc': boundary_yb},
            {'component': 1, 'type': 'periodic', 'component_x': 1, 'bc': boundary_yb},
        ]

        self.bcs = addbc(bc_config, self.geom_time)

        
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all
        X_pde = self.pde_data
        self.ic_masks = [bc.filter(X_pde) for bc in self.bcs if isinstance(bc, IC)]
        self.ic_points = [X_pde[mask] for mask in self.ic_masks]
        self.ic_u = ic_func(X_pde[self.ic_masks[0]], 0)
        self.ic_v = ic_func(X_pde[self.ic_masks[0]], 1)
        self.Y_ic = np.hstack([self.ic_u, self.ic_v])

        def data_load(path):
            loader = DataLoader()
            loader.load(path, input_dim=self.input_dim, output_dim=self.output_dim, t_transpose=True)
            Data = loader.ref_data
            X_data = jnp.array(Data[:, :self.input_dim], jnp.float32)
            Y_data = jnp.array(Data[:, self.input_dim:], jnp.float32)
            return X_data, Y_data

        self.X_data, self.Y_data = data_load(datapath)
        self.X_pde = X_pde
        self.Y_pde = np.zeros(shape=(len(X_pde), self.output_dim))
        self.Y_pde[self.ic_masks[0]] = self.Y_ic

        
        self.is_batch = False
        self.BatchSize_eq = len(self.X_pde)
        self.BatchSize_data = len(self.X_data)
        self.data_size = len(self.X_data)

        
        def reset_fn(key):
            X_eq_all = self.X_pde
            Y_eq_all = self.Y_pde
            masks = [bc.filter(X_eq_all) for bc in self.bcs]
            X_batch = np.concatenate((X_eq_all, self.X_data), axis=0)
            Y_batch = np.concatenate((Y_eq_all, self.Y_data), axis=0)
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
        u, v = pred[:, 0:1], pred[:, 1:2]
        u_x, u_y, u_t = pred[:, 2:3], pred[:, 3:4], pred[:, 4:5]
        u_xx, u_yy = pred[:, 5:6], pred[:, 6:7]
        v_x, v_y, v_t = pred[:, 7:8], pred[:, 8:9], pred[:, 9:10]
        v_xx, v_yy = pred[:, 10:11], pred[:, 11:12]
        r_u = u_t + u * u_x + v * u_y - self.nu * (u_xx + u_yy)
        r_v = v_t + u * v_x + v * v_y - self.nu * (v_xx + v_yy)
        return jnp.hstack([r_u, r_v])

    def data_fn(self, Y_ref, pred, mask=None):
        u_true = Y_ref[-self.data_size:, 0:1]
        u_pred = pred[-self.data_size:, 0:1]
        loss_tmp = (u_pred - u_true) ** 2
        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        pde_pred = pred[:pde_size, :]
        X_pde = X_batch[:pde_size, :]
        Y_pde = Y_batch[:pde_size, :]

        # bcs_masks: list of boolean arrays, each shape (pde_size,)
        # convert to float masks for arithmetic (still JAX arrays; fine)
        # (if they already are floats it's fine)
        masks_f = [m.astype(pde_pred.dtype) for m in bcs_masks]

        # PDE loss: only on interior points (not on any BC)
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)   # bool per point
        interior = (~comb_mask).astype(pde_pred.dtype)              # 1.0 for interior, 0.0 for bc
        r_all = self.pde_fn(pde_pred)                               # (N, 2)
        r_masked = r_all * interior[:, None]                        # zero out BC points
        denom_interior = jnp.sum(interior) + 1e-8
        pde_loss = jnp.sum(r_masked ** 2) / denom_interior

        # === IC loss (use masks, avoid boolean indexing) ===
        ic_loss_sum = 0.0
        ic_category = 0
        # We iterate over self.bcs but use masks_f to compute masked MSE
        for bc, mask_f in zip(self.bcs, masks_f):
            if isinstance(bc, IC):
                # pde_pred[:, ic_category] is shape (N, 1) or (N,)
                pred_comp = pde_pred[:, ic_category:ic_category+1]  # shape (N,1)
                true_comp = Y_pde[:, ic_category:ic_category+1]     # shape (N,1)
                # compute squared error, mask it, sum then normalize by number of masked points
                se = (pred_comp - true_comp) ** 2                  # (N,1)
                masked_se = se * mask_f[:, None]                   # zero out non-IC points
                denom = jnp.sum(mask_f) + 1e-8
                ic_loss_sum += jnp.sum(masked_se) / denom
                ic_category += 1
            else:
                # For non-IC BCs (e.g. periodic), we try a masked residual approach:
                # If bc.error can accept masked (zeroed) X and pde_pred, you can use:
                #   err = bc.error(pde_pred * mask_f[:, None], X_pde * mask_f[:, None])
                # But many bc.error implementations expect compact arrays pde_pred[mask], X_pde[mask].
                # So here we fallback to a generic masked L2 between pde_pred and some target if available.
                # If bc.error requires indexing, we should modify bc.error to accept mask arrays.
                # As a conservative default, compute a masked L2 of pde_pred (no X usage).
                err_full = pde_pred * mask_f[:, None]  # zeroed-out elsewhere
                bc_loss_sum = jnp.sum(err_full ** 2) / (jnp.sum(mask_f) + 1e-8)
                # accumulate (we'll average later)
                # Note: this is a placeholder; replace with proper bc.error -> see note below.
                # accumulate to bc_loss_sum_placeholder (we'll combine outside)
                if 'bc_loss_acc' not in locals():
                    bc_loss_acc = bc_loss_sum
                    bc_count = 1
                else:
                    bc_loss_acc = bc_loss_acc + bc_loss_sum
                    bc_count += 1

        # finalize BC loss
        if 'bc_loss_acc' in locals():
            bc_loss = bc_loss_acc / (bc_count + 1e-8)
        else:
            bc_loss = 0.0

        ic_loss = ic_loss_sum / (ic_category + 1e-8)

        # data loss (unchanged)
        data_loss = self.data_fn(Y_batch, pred)

        loss = jnp.hstack([
            self.pde_lambda * pde_loss,
            self.ic_lambda * ic_loss,
            self.bc_lambda * bc_loss,
            self.data_lambda * data_loss
        ])
        return loss


    def reset(self, key):
        new_state = self._reset_fn(key)
        self.BatchSize_eq = len(self.X_pde)
        self.bcs_masks = new_state.bcs_masks
        self.bcs_points = [self.X_pde[m] for m in self.bcs_masks]
        return new_state

    def step(self, state, action):
        return self._step_fn(state, action)


class PINNsPolicy(PolicyNetwork):
    def __init__(self, net, num_params, format_params_fn, grad_keys):
        self.net = net
        self.num_params = num_params
        self.format_params_fn = format_params_fn
        self.grad_keys = grad_keys

    def get_actions(self, t_states: TaskState, flat_params: jnp.ndarray, p_states: PolicyState):
        params_tree = self.format_params_fn(flat_params)
        obs = t_states.obs
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)
            return stack_outputs(outs, self.grad_keys)
        actions = jax.vmap(f_single)(params_tree, obs)
        return actions, p_states
