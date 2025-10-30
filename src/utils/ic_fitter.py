
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from .data_trans import DataLoader


class ICDataFitter:
    def __init__(self, datapath, input_dim, output_dim, t0=0.0, t_transpose=True):

        loader = DataLoader()
        loader.load(datapath, input_dim=input_dim, output_dim=output_dim, t_transpose=t_transpose)
        data = loader.ref_data  # numpy.ndarray, shape (N, input_dim+output_dim)

        coords = data[:, :input_dim]  
        values = data[:, input_dim:] 


        eps = 1e-6
        mask = np.isclose(coords[:, -1], t0, atol=eps)
        pts = coords[mask, :-1]  
        vals = values[mask].squeeze()  

        self._interp = LinearNDInterpolator(pts, vals)

    def __call__(self, x):

        y = self._interp(x)
        return y

    def sample(self, mode="all", size=None, random_state=None):

        pts = self._interp.xi  
        vals = self._interp.values
        if mode == "all" or size is None:
            return pts, vals
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(pts), size, replace=False)
        return pts[idx], vals[idx]
