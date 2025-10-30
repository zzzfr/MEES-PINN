import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree

BatchSize = 1024


class LowDiscrepancySampler:

    def __init__(self, X_all, Y_all, domain_bounds):
        self.X_all = X_all
        self.Y_all = Y_all
        self.dim = X_all.shape[1]
        self.domain_bounds = np.array(domain_bounds)
        self.sobol = qmc.Sobol(self.dim, scramble=False)
        self.tree = cKDTree(self.X_all)

        m = int(np.ceil(np.log2(len(X_all))))
        pts01 = self.sobol.random_base2(m)[:len(X_all)]
        self.full_sequence = qmc.scale(
            pts01,
            self.domain_bounds[:, 0],
            self.domain_bounds[:, 1]
        )
        self.index = 0

    def get_batch(self, batch_size=BatchSize):
        start = self.index
        end = start + batch_size

        if end <= len(self.full_sequence):
            seq = self.full_sequence[start:end]
        else:
            seq = np.vstack([
                self.full_sequence[start:],
                self.full_sequence[: end % len(self.full_sequence)]
            ])
        self.index = end % len(self.full_sequence)

        _, idx = self.tree.query(seq, k=1)
        return self.X_all[idx], self.Y_all[idx]


