

import numpy as np


class DataLoader:

    def __init__(self):
        self.ref_data = None

    def load(self, datapath, input_dim, output_dim, t_transpose=False):
        data = np.loadtxt(datapath, comments="%").astype(np.float32)
        if t_transpose:
            data = self._trans_time_data_to_dataset(data, datapath, input_dim, output_dim)
        self.ref_data = data

    def get_static_dataset(self, input_dim, output_dim):
        if self.ref_data is None:
            raise ValueError("No data loaded; call load() first.")
        # assume each row is [x1, x2, ..., x_input_dim, u1, ..., u_output_dim]
        X = self.ref_data[:, :input_dim]
        y = self.ref_data[:, input_dim:input_dim + output_dim]
        return X, y

    def get_time_dataset(self, input_dim, output_dim):
        return self.get_static_dataset(input_dim, output_dim)

    def get_dataset(self, input_dim, output_dim):
        return self.get_static_dataset(input_dim, output_dim)

    def _trans_time_data_to_dataset(self, data, datapath, input_dim, output_dim):

        total_cols = data.shape[1]
        num_times = (total_cols - input_dim + 1) // output_dim

        times = None
        with open(datapath, "r") as f:
            for line in f:
                if line.startswith("%") and line.count("@") == num_times * output_dim:
                    parts = line.split("@")[1:]

                    def extract(s):
                        idx = s.find("t=")
                        return float(s[idx + 2:].split()[0]) if idx != -1 else None

                    times = [extract(p) for p in parts]
                    break
        if times is None or any(t is None for t in times):
            raise ValueError("Can not explan")

        times = np.array(times[::output_dim], dtype=np.float32)


        num_spatial = input_dim - 1

        N = data.shape[0]  
        flat_coords = []
        for i in range(num_spatial):
            coord = data[:, i]  # shape (N,)
            flat_coords.append(np.repeat(coord, num_times))  # shape (N * num_times)

        flat_coords.append(np.tile(times, N))  # shape (N * num_times,)


        flat_outputs = []
        for oc in range(output_dim):  
            pieces = []
            for j in range(num_times):
                col = num_spatial + j * output_dim + oc
                pieces.append(data[:, col])  # shape (N,)
            flat_outputs.append(np.concatenate(pieces))  # shape (N * num_times,)
        flat_u = np.stack(flat_outputs, axis=1)  # shape (N * num_times, output_dim)


        return np.concatenate([np.stack(flat_coords, axis=1), flat_u], axis=1)

