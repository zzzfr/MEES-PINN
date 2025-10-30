import numpy as np
from EAPINN import config

DEFAULT_NUM_DOMAIN_POINTS = 8192
DEFAULT_NUM_BOUNDARY_POINTS = 2048
DEFAULT_NUM_TEST_POINTS = 8192
DEFAULT_NUM_INITIAL_POINTS = 2048


class DataSampler:
    def __init__(
        self,
        geometry,
        bcs,
        mul=1,
        train_distribution="Hammersley",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
    ):
        self.geom = geometry
        self.bcs = bcs
        self.num_domain = DEFAULT_NUM_DOMAIN_POINTS * mul
        self.num_boundary = DEFAULT_NUM_BOUNDARY_POINTS * mul

        self.train_distribution = train_distribution
        self.anchors = None if anchors is None else anchors.astype(config.real(np))
        self.exclusions = exclusions

        self.soln = solution
        self.auxiliary_var_fn = auxiliary_var_function
        self.num_test = num_test

        self.train_x_all = None
        self.train_x_bc = None
        self.num_bcs = None

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.train_aux_vars = None
        self.test_aux_vars = None

        self.pde_points()

    def pde_points(self):
        X = np.empty((0, self.geom.dim), dtype=config.real(np))
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                X = self.geom.uniform_points(self.num_domain, boundary=False)
            else:
                X = self.geom.random_points(self.num_domain, random=self.train_distribution)
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geom.random_boundary_points(
                    self.num_boundary, random=self.train_distribution
                )
            X = np.vstack((tmp, X))
        if self.anchors is not None:
            X = np.vstack((self.anchors, X))
        if self.exclusions is not None:
            X = np.array([x for x in X if not any(np.allclose(x, y) for y in self.exclusions)])
        self.train_x_all = X
        return X

    def bc_points(self):
        x_bcs = [bc.collocation_points(self.train_x_all) for bc in self.bcs]
        self.num_bcs = list(map(len, x_bcs))
        self.train_x_bc = (
            np.vstack(x_bcs)
            if x_bcs
            else np.empty((0, self.train_x_all.shape[-1]), dtype=config.real(np))
        )
        return self.train_x_bc

    def train_next_batch(self, batch_size=None):
        self.train_x_all = None
        self.train_x_bc = None
        X_pde = self.train_points()
        X_bc = self.bc_points()
        self.train_x = np.vstack((X_bc, X_pde))
        self.train_y = self.soln(self.train_x) if self.soln else None
        if self.auxiliary_var_fn:
            self.train_aux_vars = self.auxiliary_var_fn(self.train_x).astype(config.real(np))
        return self.train_x, self.train_y, self.train_aux_vars

    def test_points(self):
        X = (
            self.geom.uniform_points(self.num_test, boundary=False)
            if self.test_x is None and self.num_test
            else self.test_x
        )
        X = np.vstack((self.train_x_bc, X))
        return X

    def test(self):
        X = None
        if self.num_test is None:
            X = self.train_x
        else:
            X = self.test_points()
        self.test_x = X
        self.test_y = self.soln(self.test_x) if self.soln else None
        if self.auxiliary_var_fn:
            self.test_aux_vars = self.auxiliary_var_fn(self.test_x).astype(config.real(np))
        return self.test_x, self.test_y, self.test_aux_vars

    def resample_train_points(self, pde_points=True, bc_points=True):
        if pde_points:
            self.train_x_all = None
        if bc_points:
            self.train_x_bc = None
        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        return self.train_next_batch()

    def add_anchors(self, anchors):
        raise NotImplementedError

    def replace_with_anchors(self, anchors):
        raise NotImplementedError


class DataSampler_T(DataSampler):

    def __init__(
            self,
            geometryxtime,
            ic_bcs,
            mul=1,
            train_distribution="Hammersley",
            anchors=None,
            exclusions=None,
            solution=None,
            num_test=None,
            auxiliary_var_function=None,
    ):
        self.num_initial = DEFAULT_NUM_INITIAL_POINTS * mul
        super().__init__(
            geometryxtime,
            ic_bcs,
            mul,
            train_distribution=train_distribution,
            anchors=anchors,
            exclusions=exclusions,
            solution=solution,
            num_test=num_test,
            auxiliary_var_function=auxiliary_var_function,
        )

    def pde_points(self):
        X = super().pde_points()
        if self.num_initial > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geom.random_initial_points(
                    self.num_initial, random=self.train_distribution
                )
            if self.exclusions is not None:

                def is_not_excluded(x):
                    return not np.any([np.allclose(x, y) for y in self.exclusions])

                tmp = np.array(list(filter(is_not_excluded, tmp)))
            X = np.vstack((tmp, X))
        self.train_x_all = X
        return X
