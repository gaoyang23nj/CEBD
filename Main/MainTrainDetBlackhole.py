import shutil
import multiprocessing
import time

import numpy as np
import tensorflow as tf
import os
import re

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 得到的 y(99行1列) x_d_new(99行3列) x_ind_new(99行100*3列);
# 99是条目个数 因为本节点不会评判自己; 1,3,300(可以改成297个 99*3)
def process_data_npz(file_path, data_srcnode):
    # print(file_path)
    data = np.load(file_path)
    # 取出数据； 并删除 本节点的评价
    y = data['y']
    x_d = data['x_d']
    x_ind = data['x_ind']
    y_new = np.delete(y, data_srcnode, axis=0)
    x_d_new = np.delete(x_d, data_srcnode, axis=0)
    x_ind_new = np.delete(x_ind, data_srcnode, axis = 0)
    return y_new, x_d_new, x_ind_new

# 从直接证据里 训练分类器
def train_from_DirectEvidence(y_final, x1_final, x2_final):
    # 只利用x1属性
    X_train, X_test, y_train, y_test = train_test_split(x1_final, y_final, test_size=0.25)
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='relu',input_shape=[3,]),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=100, epochs=10)
    model.evaluate(X_train, y_train, verbose=2)
    model.evaluate(X_test, y_test, verbose=2)
    savemodel_file_path = os.path.join(dir, 'ML\\deve_model.h5')
    model.save(savemodel_file_path)
    # m = tf.keras.models.load_model(savemodel_file_path)
    # m.evaluate(X_test, y_test, verbose=2)
    print('num_train: {}, {}; num_test, {}, {};'.format(y_train.shape, np.sum(y_train), y_test.shape, np.sum(y_test)))

# 从间接证据里 训练分类器
def train_from_inDirectEvidence(y_final, x1_final, x2_final):
    # 只利用x2属性
    X_train, X_test, y_train, y_test = train_test_split(x2_final, y_final, test_size = 0.3)
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(300, activation='relu',input_shape=[300,]),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=100, epochs=10)
    model.evaluate(X_train, y_train, verbose=2)
    model.evaluate(X_test, y_test, verbose=2)
    savemodel_file_path = os.path.join(dir, 'ML\\indeve_model.h5')
    model.save(savemodel_file_path)
    # m = tf.keras.models.load_model(savemodel_file_path)
    # m.evaluate(X_test, y_test, verbose=2)
    print('num_train: {}, {}; num_test, {}, {};'.format(y_train.shape, np.sum(y_train), y_test.shape, np.sum(y_test)))

# 从文件中收集数据
def collect_data_totrain(dir):
    # 把文件名 和 对应的数据源 洗出来
    npz_tunple_list = []
    npz_dirs = os.listdir(dir)
    for npz_dir in npz_dirs:
        npz_dir_path = os.path.join(dir, npz_dir)
        if os.path.isdir(npz_dir_path):
            files = os.listdir(npz_dir_path)
            for file in files:
                npz_file_path = os.path.join(npz_dir_path, file)
                m = re.match(r'(\d*).npz', file)
                if m is None:
                    continue
                data_srcnode = int(m.group(1))
                npz_tunple_list.append((npz_file_path, data_srcnode))
    # 总的数据条目数未知; 通过逐个文件读取 获得训练数据集合
    num_npzfile = len(npz_tunple_list)
    # y.shape 99*1;     # x1.shape 99*3;     # x2.shape 99*300
    y_final = np.zeros(shape=(99 * num_npzfile,))
    x1_final = np.zeros(shape=(99 * num_npzfile, 3))
    x2_final = np.zeros(shape=(99 * num_npzfile, 300))
    for i in range(len(npz_tunple_list)):
        (npz_file_path, data_srcnode) = npz_tunple_list[i]
        y, x1, x2 = process_data_npz(npz_file_path, data_srcnode)
        # 对于0.npz 从0行到99行
        y_final[i * 99: (i + 1) * 99] = y
        x1_final[i * 99: (i + 1) * 99] = x1
        x2_final[i * 99: (i + 1) * 99] = x2
    return y_final, x1_final, x2_final

if __name__ == "__main__":
    # dir = "D:\\13-ML\\Simulation_ONE\\Main\\collect_data"
    dir = "..\\Main\\collect_data"
    ml_dir = dir + "\\ML"
    if os.path.exists(ml_dir):
        shutil.rmtree(ml_dir)
        print('delete dir ' + ml_dir)
    os.makedirs(ml_dir)
    print('add dir ' + ml_dir)

    # 把文件名 和 对应的数据源 洗出来
    y_final, x1_final, x2_final = collect_data_totrain(dir)

    train_from_DirectEvidence(y_final, x1_final, x2_final)
    print('Direct Eve Training... Completed!')
    train_from_inDirectEvidence(y_final, x1_final, x2_final)
    print('InDirect Eve Training... Completed!')


