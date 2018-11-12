import os
import h5py
import numpy as np


def load_hdf5(hdf5_path):
    """Load hdf5 data.
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)

    return x, y


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
