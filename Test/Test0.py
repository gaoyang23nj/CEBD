import shutil
import multiprocessing
import time
import datetime

import numpy as np
import tensorflow as tf
import os
import re

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def process_data_npz(file_path, data_srcnode):
    print(file_path)
    data = np.load(file_path)
    y = data['y']
    y_new = np.delete(y, data_srcnode, axis = 0)
    x_d = data['x_d']
    x_d_new = np.delete(x_d, data_srcnode, axis = 0)
    x_ind = data['x_ind']
    x_ind_new = np.delete(x_ind, data_srcnode, axis = 0)
    return y_new, x_d_new, x_ind_new

# 从直接证据里 训练分类器
def train_from_DirectEvidence():
    dir = "../Main/collect_data/scenario9/"
    files = os.listdir(dir)
    file_path = ""
    data_srcnode = 0
    # 合计有100个数据来源
    # y.shape 99*1
    y_final = np.zeros(shape=(99 * 100,))
    # x1.shape 99*3
    x1_final = np.zeros(shape=(99 * 100, 3))
    # x2.shape 99*300
    x2_final = np.zeros(shape=(99 * 100, 300))
    for file in files:
        file_path = os.path.join(dir, file)
        m = re.match(r'(\d*).npz', file)
        if m is None:
            continue
        data_srcnode = int(m.group(1))
        # 收集所有data 训练模型
        y, x1, x2 = process_data_npz(file_path, data_srcnode)
        y_final[data_srcnode * 99 : (data_srcnode+1) * 99] = y
        x1_final[data_srcnode * 99: (data_srcnode + 1) * 99] = x1
        x2_final[data_srcnode * 99: (data_srcnode + 1) * 99] = x2
    # 只利用x1属性
    X_train, X_test, y_train, y_test = train_test_split(x1_final, y_final, test_size=0.3)
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
    model.fit(X_train, y_train, epochs=5)
    model.evaluate(X_test, y_test, verbose=2)
    savemodel_file_path = os.path.join(dir, 'ML/deve_model.h5')
    model.save(savemodel_file_path)
    # m = tf.keras.models.load_model(savemodel_file_path)
    # m.evaluate(X_test, y_test, verbose=2)
    print('num_train: {}, {}; num_test, {}, {};'.format(y_train.shape, np.sum(y_train), y_test.shape, np.sum(y_test)))

# 从间接证据里 训练分类器
def train_from_inDirectEvidence():
    dir = "../Main/collect_data/scenario9/"
    files = os.listdir(dir)
    file_path = ""
    data_srcnode = 0
    # 合计有100个数据来源
    # y.shape 99*1
    y_final = np.zeros(shape=(99 * 100,))
    # x1.shape 99*3
    x1_final = np.zeros(shape=(99 * 100, 3))
    # x2.shape 99*300
    x2_final = np.zeros(shape=(99 * 100, 300))
    for file in files:
        file_path = os.path.join(dir, file)
        m = re.match(r'(\d*).npz', file)
        if m is None:
            continue
        data_srcnode = int(file.split('.')[0])
        # 收集所有data 训练模型
        y, x1, x2 = process_data_npz(file_path, data_srcnode)
        y_final[data_srcnode * 99 : (data_srcnode+1) * 99] = y
        x1_final[data_srcnode * 99: (data_srcnode + 1) * 99] = x1
        x2_final[data_srcnode * 99: (data_srcnode + 1) * 99] = x2
    # 只利用x1属性
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
    model.fit(X_train, y_train, epochs=5)
    model.evaluate(X_test, y_test, verbose=2)
    savemodel_file_path = os.path.join(dir, 'ML/indeve_model.h5')
    model.save(savemodel_file_path)
    # m = tf.keras.models.load_model(savemodel_file_path)
    # m.evaluate(X_test, y_test, verbose=2)
    print('num_train: {}, {}; num_test, {}, {};'.format(y_train.shape, np.sum(y_train), y_test.shape, np.sum(y_test)))



MAX = 10
MAX_String = 'Max Process Ability'
def process_q(save_d_model_file_path, save_ind_model_file_path, q_input, q_output):
    num_to_process = 0
    print('.........Process Running...pid[{}]'.format(os.getpid()))

    d_model = tf.keras.models.load_model(save_d_model_file_path)
    ind_model = tf.keras.models.load_model(save_ind_model_file_path)
    while True:
        em = q_input.get(True)
        if em is None:
            break
        d_predict = d_model.predict(em[1])
        ind_predict = ind_model.predict(em[2])
        num_to_process = num_to_process + 1
        print('.........Process Running...pid[{}],no.{}'.format(os.getpid(), num_to_process))
        if num_to_process >= MAX:
            res = (em[0], d_predict[0][1], ind_predict[0][1], MAX_String)
            q_output.put(res)
            break
        else:
            res = (em[0], d_predict[0][1], ind_predict[0][1])
            q_output.put(res)


if __name__ == "__main__":
    # dir = "../Main/collect_data/scenario9/ML"
    # if os.path.exists(dir):
    #     shutil.rmtree(dir)
    # # os.makedirs(dir)
    # os.mkdir(dir)
    # p = multiprocessing.Process(target=train_from_DirectEvidence, args=())
    # p.start()
    # p.join()
    # print('Direct Eve Training... Completed!')
    #
    # p = multiprocessing.Process(target=train_from_inDirectEvidence, args=())
    # p.start()
    # p.join()
    # print('InDirect Eve Training... Completed!')

    dir = "../Main/collect_data/scenario9/"
    save_d_model_file_path = os.path.join(dir, 'ML/deve_model.h5')
    save_ind_model_file_path = os.path.join(dir, 'ML/indeve_model.h5')



    # tGPU_start = time.time()
    # for i in range(predict_times):
    #     manager = multiprocessing.Manager()
    #     return_dict = manager.dict()
    #     j = multiprocessing.Process(target=tryonce_process_gpu, args=(save_d_model_file_path, save_ind_model_file_path, d_attrs, ind_attrs,  return_dict))
    #     j.start()
    #     j.join()
    #     res_d = return_dict["res_d"]
    #     res_ind = return_dict["res_ind"]
    #     print('return_dict: d:{} ind:{}'.format(res_d, res_ind))
    # print(return_dict["res_d"])
    # tGPU_end = time.time()
    # tGPU = tGPU_end - tGPU_start

    print('Main Running...............................................0')

    tCPU_start = time.time()
    q_input = multiprocessing.Queue()
    q_output = multiprocessing.Queue()
    # manager = multiprocessing.Manager()

    # return_dict = manager.dict()
    j = multiprocessing.Process(target=process_q, args=(save_d_model_file_path, save_ind_model_file_path, q_input, q_output))
    j.daemon = True
    j.start()
    print('Main Running...............................................1')

    # 预测执行的次数
    predict_times = 50
    key = 0
    for i in range(predict_times):
        d_attrs = np.random.rand(1, 3)
        ind_attrs = np.random.rand(1, 300)

        em = (key, d_attrs.copy(), ind_attrs.copy())
        key = key + 1
        q_input.put(em)

        res = q_output.get(True)
        if len(res) == 4 and res[3] == MAX_String:
            j = multiprocessing.Process(target=process_q,
                                        args=(save_d_model_file_path, save_ind_model_file_path, q_input, q_output))
            j.daemon = True
            j.start()
        print(res)
        print('....{}'.format(i))


    q_input.put(None)

    tCPU_end = time.time()
    tCPU = tCPU_end - tCPU_start
    print('GPU:{}'.format(tCPU))
    # print('GPU:{}  CPU:{}'.format(tGPU, tCPU))