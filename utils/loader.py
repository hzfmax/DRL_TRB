import os.path as osp

import numpy as np
import pandas as pd
from user_config import DATA_DIR


def get_victoria_data(line_name='Victoria', read=True):
    filepath = osp.join(DATA_DIR, f'{line_name}.npy')
    data = np.load(filepath, allow_pickle=True).item()
    return data
