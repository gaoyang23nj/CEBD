import numpy as np
import tensorflow as tf
import os

dir = "../Main/collect_data/scenario9/"
files = os.listdir(dir)
file_path = ""
for file in files:
    file_path = os.path.join(dir, file)
    print(file)
    print(file_path)矩阵合并
data = np.load(file_path)
print(data.files)
data = np.load('D:/13-ML/Simulation_ONE/Main/collect_data/scenario9/1.npz')
print(data['y'].shape)
print(data['x_d'].shape)
# data['x_ind']
