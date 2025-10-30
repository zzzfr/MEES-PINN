import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
import numpy as np
from typing import Sequence

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import SimManager, addbc, stack_outputs
from src.data import DataSampler_T, LowDiscrepancySampler
from src.nn import BaseNN

BatchSize = 8192


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]


class PINN(BaseNN):
    """
    PINN for Wave2D_LongTime :
    PDE:
        ∂²u/∂t² − (∂²u/∂x² + a²·∂²u/∂y²) = 0

    IC:
        u(x, y, 0) =
            INITIAL_COEF_1·sin(m1·π·x)·sinh(n1·π·y) + INITIAL_COEF_2·sinh(m2·π·x)·sin(n2·π·y)

    BC (Dirichlet BC):
        u(x, y, t) = INITIAL_COEF_1·sin(m1·π·x)·sinh(n1·π·y)·cos(p1·π·t) + INITIAL_COEF_2·sinh(m2·π·x)·sin(n2·π·y)·cos(p2·π·t)
        for all (x, y) on the spatial boundary ∂Ω
    """

    def derivatives(self, params, X):
        def forward(z):
            return self.apply(params, z[None, :])[0]

        def u_fn(z): return jnp.squeeze(forward(z))

        u = jax.vmap(u_fn)(X).reshape(-1, 1)
        grads_u = jax.vmap(jax.grad(u_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)

        return {
            'u': u,
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2], 'u_t': grads_u[:, 2:3],
            'u_xx': hess_u[:, 0, 0:1], 'u_yy': hess_u[:, 1, 1:2], 'u_tt': hess_u[:, 2, 2:3],
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, bbox=[0, 1, 0, 1, 0, 100], a=np.sqrt(2), m1=1, m2=3, n1=1, n2=2, p1=1, p2=1):

        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([1, ])

        self.a = a


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
        self.layout = ['u', 'u_x', 'u_y', 'u_t', 'u_xx', 'u_yy', 'u_tt']


        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 0.0


        INITIAL_COEF_1 = 1
        INITIAL_COEF_2 = 1

        def ref_sol(x):
            return (
                    INITIAL_COEF_1 * jnp.sin(m1 * jnp.pi * x[:, 0:1]) * jnp.sinh(n1 * jnp.pi * x[:, 1:2]) * jnp.cos(
                p1 * jnp.pi * x[:, 2:3])
                    + INITIAL_COEF_2 * jnp.sinh(m2 * jnp.pi * x[:, 0:1]) * jnp.sin(n2 * jnp.pi * x[:, 1:2]) * jnp.cos(
                p2 * jnp.pi * x[:, 2:3])
            )

        bc_config = [
            {
                'component': 0,
                'function': ref_sol,
                'bc': (lambda _, on_initial: on_initial),
                'type': 'ic'
            },
            {
                'component': 0,
                'function': ref_sol,
                'bc': (lambda _, on_boundary: on_boundary),
                'type': 'dirichlet'
            }
        ]
        self.bcs = addbc(bc_config, self.geom_time)

        # --- pde points ---
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all
        self.X_pde = self.pde_data
        self.Y_pde = np.zeros(shape=(len(self.X_pde), self.output_dim))

        # --- data points ---
        self.X_data = None
        self.Y_data = None

        self.X_all = self.X_pde
        self.Y_all = self.Y_pde

        # --- mini batch ---
        self.batch_size = BatchSize
        domain_bounds = [
            [self.bbox[0], self.bbox[1]],  # [x_min,x_max]
            [self.bbox[2], self.bbox[3]],  # [y_min,y_max]
            [self.bbox[4], self.bbox[5]],  # [t_min,t_max]
        ]
        self.sampler = LowDiscrepancySampler(self.X_all, self.Y_all, domain_bounds)


        def reset_fn(key):
            X_batch, Y_batch = self.sampler.get_batch(batch_size=BatchSize) 
            masks = [bc.filter(X_batch) for bc in self.bcs]
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
        """根据网络输出action + 坐标计算 pde 残差"""
        u = pred[:, 0:1]
        u_x, u_y, u_t = pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
        u_xx, u_yy, u_tt = pred[:, 4:5], pred[:, 5:6], pred[:, 6:7]

        r = u_tt - (u_xx + self.a**2 * u_yy)

        return r

    def data_fn(self, Y_ref, pred, mask):
        u_ref = Y_ref[:, 0:1]
        u_pred = pred[:, 0:1]
        loss_tmp = (u_pred - u_ref) ** 2

        # data_loss = jnp.sum(loss_tmp ** 2) / pred.shape[0]
        loss_masked = loss_tmp * mask[:, None]
        data_loss = jnp.sum(loss_masked) / (jnp.sum(mask))
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)

        # PDE loss
        r_all = self.pde_fn(pred)
        interior = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior) + 1e-8)

        # IC & BC loss
        ic_loss_sum = 0.0
        bc_loss_sum = 0.0
        ic_category = 0
        bc_category = 0
        for bc, mask in zip(self.bcs, bcs_masks):

            mask_f = mask[:, None].astype(pred.dtype)
            err = bc.error(pred, X_batch)
            term = jnp.sum((err ** 2) * mask_f) / (jnp.sum(mask_f) + 1e-8)
            if isinstance(bc, IC):
                ic_loss_sum += term
                ic_category += 1
            else:
                bc_loss_sum += term
                bc_category += 1
        ic_loss = ic_loss_sum / (ic_category + 1e-8)
        bc_loss = bc_loss_sum / (bc_category + 1e-8)

        # data loss
        data_loss = 0.0

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
