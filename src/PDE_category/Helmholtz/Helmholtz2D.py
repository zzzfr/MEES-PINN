import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from typing import Sequence
from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import addbc, stack_outputs
from src.data import DataSampler, LowDiscrepancySampler
from src.nn import BaseNN

BatchSize = 2048


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]



class PINN(BaseNN):
    """
    PINN for Helmholtz2D :
    PDE:
        ∂²u/∂x² + ∂²u/∂y² + k²·u = f(x,y),
        f(x,y) = sin(A₀·π·x/scale) · sin(A₁·π·y/scale) · (k² − π²·(A₀² + A₁²)/scale²)

    BC (Dirichlet BC):
        u(0,   y) = 0   u(scale,   y) = 0
        u(x,   0) = 0   u(x,   scale)   = 0
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
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2],
            'u_xx': hess_u[:, 0, 0].reshape(-1, 1), 'u_yy': hess_u[:, 1, 1].reshape(-1, 1),
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, scale=1, A=(4, 4), k=1):
        
        self.max_steps = 1
        self.obs_shape = tuple([2, ])
        self.act_shape = tuple([1, ])

        self.A = A
        self.scale = scale
        self.k = k

        
        self.bbox = [0, self.scale, 0, self.scale]
        self.geom = geometry.Rectangle(xmin=[0, 0], xmax=[self.scale, self.scale])
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
        self.layout = ['u', 'u_x', 'u_y', 'u_xx', 'u_yy']

        
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 0.0

        
        def ref_sol(x):
            return jnp.sin(A[0] * jnp.pi * x[:, 0:1] / scale) * jnp.sin(A[1] * jnp.pi * x[:, 1:2] / scale)

        self.ref_sol = ref_sol

        bc_config = [
            {
                'component': 0,
                'function': (lambda _: 0),
                'bc': (lambda _, on_boundary: on_boundary),
                'type': 'dirichlet',
            },
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
            [self.bbox[2], self.bbox[3]],  # [y_min,y_max]
        ]
        self.sampler = LowDiscrepancySampler(self.X_all, self.Y_all, domain_bounds)

        # --- reset / step ---
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
        u_x, u_y = pred[:, 1:2], pred[:, 2:3]
        u_xx, u_yy = pred[:, 3:4], pred[:, 4:5]

        def f(x):
            return jnp.sin(self.A[0] * jnp.pi * x[:, 0:1] / self.scale) * jnp.sin(self.A[1] * jnp.pi * x[:, 1:2] / self.scale) * \
                   (self.k ** 2 - jnp.pi ** 2 * (self.A[0] ** 2 + self.A[1] ** 2) / self.scale ** 2)

        r = u_xx + u_yy + self.k**2 * u - f(X)

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
        r_all = self.pde_fn(pred, X=X_batch)
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
        params_tree = self.format_params_fn(flat_params)  # batched pytree

        obs = t_states.obs

        # forward + derivatives
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)
            return stack_outputs(outs, self.grad_keys)

        actions = jax.vmap(f_single)(params_tree, obs)
        return actions, p_states
