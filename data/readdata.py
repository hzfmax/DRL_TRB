import os.path as osp

import numpy as np
import pandas as pd


# ROOT = osp.abspath(osp.dirname(__file__))
# DATA_DIR = osp.join(ROOT, 'data')
# DEFAULT_LOG_DIR = osp.join(ROOT, 'log')

# filepath = osp.join(DATA_DIR, f'{line_name}.npy')
# data = np.load("/home/hzfmax/DRL_TRB/data", allow_pickle=True).item()


# input_data = np.load(r"/home/hzfmax/DRL_TRB/data/Victoria.npy",allow_pickle=True)
# print(input_data.shape)
# data = input_data.reshape(1,-1)
# print(data.shape)
# print(data)
# np.savetxt(r"/home/hzfmax/DRL_TRB/data/Victoria.txt",data,delimiter=',')

context = np.load('/home/hzfmax/DRL_TRB/data/Victoria.npy',allow_pickle=True,encoding="latin1")
print(context)