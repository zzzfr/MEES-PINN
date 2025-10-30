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
from src.data import DataSampler, LowDiscrepancySampler
from src.nn import BaseNN

BatchSize = 4096


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]


class PINN(BaseNN):
    """
    PINN for Wave1D :
    PDE:
        ∂²u/∂t² − C²·∂²u/∂x² = 0

    IC:
        u(x,0) = sin(π·x/scale) + 0.5·sin(a·π·x/scale)
        ∂u/∂t(x,0) = 0

    BC (Dirichlet spatial):
        u(0,   t)     = ref_sol([0,   t])
        u(scale, t)   = ref_sol([scale, t])
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
            'u_x': grads_u[:, 0:1], 'u_t': grads_u[:, 1:2],
            'u_xx': hess_u[:, 0, 0:1], 'u_tt': hess_u[:, 1, 1:2],
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, C=2, bbox=[0, 1, 0, 1], scale=1, a=4):
        self.max_steps = 1
        self.obs_shape = tuple([2, ])
        self.act_shape = tuple([1, ])

        self.C = C
        self.scale = scale
        self.a = a

        self.bbox = bbox
        self.geom = geometry.Rectangle(xmin=[self.bbox[0], self.bbox[2]], xmax=[self.bbox[1], self.bbox[3]])
        self.geom_time = None

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
        self.layout = ['u', 'u_x', 'u_t', 'u_xx', 'u_tt']

        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 0.0

        def ref_sol(x):
            x = x / scale
            return (jnp.sin(np.pi * x[:, 0:1]) * jnp.cos(2 * np.pi * x[:, 1:2]) + 0.5 * jnp.sin(
                a * jnp.pi * x[:, 0:1]) * jnp.cos(2 * a * jnp.pi * x[:, 1:2]))

        self.ref_sol = ref_sol

        def boundary_x0(x, on_boundary):
            return jnp.logical_and(on_boundary, (jnp.isclose(x[:, 0], self.bbox[0]) | jnp.isclose(x[:, 0], self.bbox[1])))

        def boundary_t0(x, on_boundary):
            return jnp.logical_and(on_boundary, jnp.isclose(x[:, 1], self.bbox[2]))

        bc_config = [
            {
                'component': 0,
                'function': (lambda _: 0),
                'bc': boundary_t0,
                'type': 'neumann'
            },
            {
                'component': 0,
                'function': ref_sol,
                'bc': boundary_t0,
                'type': 'dirichlet'
            },
            {
                'component': 0,
                'function': ref_sol,
                'bc': boundary_x0,
                'type': 'dirichlet'
            }
        ]

        self.bcs = addbc(bc_config, self.geom)

        # --- pde points ---
        self.pde_data = DataSampler(self.geom, self.bcs, mul=1).train_x_all
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
            [self.bbox[2], self.bbox[3]],  # [t_min,t_max]
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
        u = pred[:, 0:1]
        u_x, u_t = pred[:, 1:2], pred[:, 2:3],
        u_xx, u_tt = pred[:, 3:4], pred[:, 4:5]

        r = u_tt - self.C**2 * u_xx

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
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior))

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
        data_loss = self.data_fn(Y_batch, pred, interior)

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
