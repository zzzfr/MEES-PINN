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
    PINN for PoissonInv :
    PDE:
        ∂/∂x [ a(x,y) ∂u/∂x ] + ∂/∂y [ a(x,y) ∂u/∂y ] + f_src(x,y) = 0

    Data Loss (pointset):
        u(x_i, y_i) = u_ref(x_i, y_i) + noise

    BC (Dirichlet BC) for a:
        a(x,y) = 1 / [1 + x² + y² + (x−1)² + (y−1)²],   (x,y) ∈ ∂Ω
    """

    def derivatives(self, params, X):
        def forward(z):
            return self.apply(params, z[None, :])[0]  # shape = (2,)

        def u_fn(z): return forward(z)[0]

        def a_fn(z): return forward(z)[1]

        def div_au(z):
            def au_x(x): return a_fn(x) * jax.grad(u_fn)(x)[0]

            def au_y(x): return a_fn(x) * jax.grad(u_fn)(x)[1]

            return jax.grad(au_x)(z)[0] + jax.grad(au_y)(z)[1]

        grads_u = jax.vmap(jax.grad(u_fn))(X)  # shape = (N, 2)
        u = jax.vmap(u_fn)(X).reshape(-1, 1)  # shape = (N, 1)
        a = jax.vmap(a_fn)(X).reshape(-1, 1)
        d_au = jax.vmap(div_au)(X).reshape(-1, 1)

        return {
            'u': u,
            'a': a,
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2],
            'd_au': d_au
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, bbox=[0, 1, 0, 1]):
        
        self.max_steps = 1
        self.obs_shape = tuple([2, ])
        self.act_shape = tuple([2, ])

       
        self.bbox = bbox
        self.geom = geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        self.geom_time = None

        
        self.output_dim = 2
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
        self.layout = ['u', 'a', 'u_x', 'u_y', 'd_au']

        
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

       
        bc_config = [
            {
            'component': 1,
            'function': self.a_ref,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet',
            'name': 'bc_a',
        }]

        self.bcs = addbc(bc_config, self.geom)

        # --- pde points ---
        self.pde_data = DataSampler(self.geom, self.bcs, mul=4).train_x_all
        self.X_pde = self.pde_data

        # --- data points ---
        self.X_data = self.X_pde
        self.Y_data = self.ref_data_fn(self.X_data)

        self.X_all = self.X_data
        self.Y_all = self.Y_data

        # --- mini batch ---
        self.batch_size = BatchSize
        domain_bounds = [
            [self.bbox[0], self.bbox[1]],   #[x_min,x_max]
            [self.bbox[2], self.bbox[3]],   #[y_min,y_max]
        ]
        self.sampler = LowDiscrepancySampler(self.X_all, self.Y_all, domain_bounds)

        # ---reset / step ---
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
        d_au = pred[:, 4:5]

        def f_src(Z):
            x, y = Z[:, 0:1], Z[:, 1:2]
            return (
                    2 * jnp.pi ** 2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * self.a_ref(Z)
                    + 2 * jnp.pi * ((2 * x + 1) * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
                            + (2 * y + 1) * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
                    ) * self.a_ref(Z) ** 2
                    )

        r = d_au + f_src(X)

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
        r_all = self.pde_fn(pred, X_batch)
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

    @staticmethod
    def a_ref(X):
        x, y = X[:, 0:1], X[:, 1:2]
        return 1 / (1 + x ** 2 + y ** 2 + (x - 1) ** 2 + (y - 1) ** 2)

    @staticmethod
    def u_ref(X):
        x, y = X[:, 0:1], X[:, 1:2]
        return jnp.sin(jnp.pi * x) * jnp.sin(np.pi * y)

    def ref_data_fn(self, X):
        return np.array(self.u_ref(X) + np.random.normal(loc=0, scale=0.1))     # add noise

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
