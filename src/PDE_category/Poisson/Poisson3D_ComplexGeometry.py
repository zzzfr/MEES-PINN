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
from src.utils import SimManager, addbc, stack_outputs, DataLoader
from src.data import DataSampler, LowDiscrepancySampler
from src.nn import BaseNN
from pathlib import Path

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
    PDE:
        -μ(x,y,z)·(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²) + k(x,y,z)²·u = f(x,y,z),
        where
            μ(x,y,z) = mu0   if z < interface_z,
                     = mu1   otherwise;
            k(x,y,z)² = k0²  if z < interface_z,
                      = k1²  otherwise;
            f(x,y,z) = A0·exp(sin(m0·π·x)+sin(m1·π·y)+sin(m2·π·z))·((x²+y²+z²−1)/(x²+y²+z²+1))
                     + A1·sin(m0·π·x)·sin(m1·π·y)·sin(m2·π·z)

    BC (Neumann BC):
        ∂u/∂n = 0 on all boundaries (outer hypercube faces and internal sphere surfaces)
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
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2], 'u_z': grads_u[:, 2:3],
            'u_xx': hess_u[:, 0, 0:1], 'u_yy': hess_u[:, 1, 1:2], 'u_zz': hess_u[:, 2, 2:3],
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, datapath=ref_dir/"poisson_3d.dat",
                bbox=[0, 1, 0, 1, 0, 1], interface_z=0.5,
                circ=[(0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2), (0.2, 0.8, 0.7, 0.1), (0.6, 0.2, 0.3, 0.1)],
                A=(20, 100), m=(1, 10, 5), k=(8, 10), mu=(1, 1)):


        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([1, ])


        self.interface_z = interface_z
        self.circ = circ
        self.A = A
        self.m = m
        self.k = k
        self.mu = mu


        self.bbox = bbox
        geom = geometry.Hypercube(xmin=self.bbox[0::2], xmax=self.bbox[1::2])
        for i in range(len(circ)):
            sphere = geometry.Sphere(circ[i][0:3], circ[i][3])
            geom = geometry.csg.CSGDifference(geom, sphere)
        self.geom = geom
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
        self.layout = ['u', 'u_x', 'u_y', 'u_z', 'u_xx', 'u_yy', 'u_zz']



        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0


        bc_config = [
            {
                'component': 0,
                'function': (lambda x: 0),
                'bc': (lambda _, on_boundary: on_boundary),
                'type': 'neumann'
            }
        ]

        self.bcs = addbc(bc_config, self.geom)

        # --- pde points ---
        self.pde_data = DataSampler(self.geom, self.bcs, mul=4).train_x_all
        X_pde = self.pde_data
        self.X_pde = X_pde
        self.Y_pde = np.zeros(shape=(len(X_pde), self.output_dim))

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
            [self.bbox[0], self.bbox[1]],  # [x_min,x_max]
            [self.bbox[2], self.bbox[3]],  # [y_min,y_max]
            [self.bbox[4], self.bbox[5]],  # [z_min,z_max]
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
        u_xx, u_yy, u_zz = pred[:, 4:5], pred[:, 5:6], pred[:, 6:7]

        def f_src(xyz):
            x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
            X2_sum = x ** 2 + y ** 2 + z ** 2
            part1 = jnp.exp(
                jnp.sin(self.m[0] * np.pi * x) + jnp.sin(self.m[1] * np.pi * y) + jnp.sin(self.m[2] * np.pi * z)) \
                * (X2_sum - 1) / (X2_sum + 1)

            part2 = jnp.sin(self.m[0] * np.pi * x) * jnp.sin(self.m[1] * np.pi * y) * jnp.sin(self.m[2] * np.pi * z)

            return self.A[0] * part1 + self.A[1] * part2

        mus = jnp.where(X[:, 2] < self.interface_z, self.mu[0], self.mu[1]).reshape(-1, 1)
        ks = jnp.where(X[:, 2] < self.interface_z, self.k[0] ** 2, self.k[1] ** 2).reshape(-1, 1)

        r = -mus * (u_xx + u_yy + u_zz) + ks * u - f_src(X)
        return r

    def data_fn(self, Y_ref, pred, mask=None):
        u_true = Y_ref[- self.data_size:, 0:1]
        u_pred = pred[- self.data_size:, 0:1]
        loss_tmp = (u_pred - u_true) ** 2

        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        pde_pred = pred[:pde_size, :]
        X_pde = X_batch[:pde_size, :]
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)

        # PDE loss

        r_all = self.pde_fn(pde_pred, X_pde)
        interior = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior))

        # IC & BC loss
        ic_loss_sum = 0.0
        bc_loss_sum = 0.0
        ic_category = 0
        bc_category = 0
        for bc, mask in zip(self.bcs, bcs_masks):

            mask_f = mask[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_pde)
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

        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)
            return stack_outputs(outs, self.grad_keys)

        actions = jax.vmap(f_single)(params_tree, obs)
        return actions, p_states
