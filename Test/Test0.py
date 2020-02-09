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
    dir = "../Main/collect_data/"
    save_d_model_file_path = os.path.join(dir, 'ML/deve_model.h5')
    save_ind_model_file_path = os.path.join(dir, 'ML/indeve_model.h5')

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