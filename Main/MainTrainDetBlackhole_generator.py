import shutil
import multiprocessing
import time
import math
import random

import numpy as np
import tensorflow as tf
import os
import re

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def direct_and_indirect(lines_train, lines_val, h5_filepath):
    num_train = len(lines_train)
    num_val = len(lines_val)
    print('num_train, num_val: {},{}'.format(num_train, num_val))
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(303, name="input"),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', name='output')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    print(model.summary)
    batch_size = 1
    model.fit_generator(mygenerator(lines_train, batch_size),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=mygenerator(lines_val, batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=10,
                        initial_epoch=0)

    model.save(h5_filepath)

def test(lines_test, h5_filepath):
    num_testlines = len(lines_test)
    # y.shape 99*1;     # x1.shape 99*3;     # x2.shape 99*300
    y_final = np.zeros(shape=(99 * num_testlines,))
    x_final = np.zeros(shape=(99 * num_testlines, 303))
    for i in range(num_testlines):
        y, x1, x2 = process_data_npz(lines_test[i])
        # 对于0.npz 从0行到99行
        y_final[i * 99: (i + 1) * 99] = y
        x_final[i * 99: (i + 1) * 99] = np.hstack((x1,x2))

    model = tf.keras.models.load_model(h5_filepath)
    # model.evaluate(X_train, Y_train, verbose=2)
    # model.evaluate(X_test, Y_test, verbose=2)
    y_pred_raw = model.predict(x_final)
    y_predict = (y_pred_raw[:, 1] > 0.5).astype(int)
    print(tf.math.confusion_matrix(y_final, y_predict, num_classes=2))

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # pd.DataFrame(history.history).plot(figsize = (8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.show()

# 得到的 y(99行1列) x_d_new(99行3列) x_ind_new(99行100*3列);
# 99是条目个数 因为本节点不会评判自己; 1,3,300(可以改成297个 99*3)
def process_data_npz(file_path):
    file_path = file_path.rstrip()
    # print(file_path)
    data = np.load(file_path)
    data_srcnode = data['datasrc']
    # 取出数据； 并删除 本节点的评价
    y = data['y']
    x_d = data['x_d']
    x_ind = data['x_ind']
    y_new = np.delete(y, data_srcnode, axis=0)
    x_d_new = np.delete(x_d, data_srcnode, axis=0)
    x_ind_new = np.delete(x_ind, data_srcnode, axis = 0)
    return y_new, x_d_new, x_ind_new

# 从文件中收集数据
def build_anno(eve_dir, annotation_path):
    # 把文件名 和 对应的数据源 洗出来
    npz_tunple_list = []
    npz_dirs = os.listdir(eve_dir)
    for npz_dir in npz_dirs:
        npz_dir_path = os.path.join(eve_dir, npz_dir)
        if os.path.isdir(npz_dir_path):
            files = os.listdir(npz_dir_path)
            for file in files:
                npz_file_path = os.path.join(npz_dir_path, file)
                # 只处理npz文件
                m = re.match(r'(\d*).npz', file)
                if m is None:
                    continue
                data_srcnode = int(m.group(1))
                npz_tunple_list.append((npz_file_path, data_srcnode))
    # 总的数据条目数未知; 但我们知道每个文件里有99条记录; 通过逐个文件读取 获得训练数据集合
    num_npzfile = len(npz_tunple_list)
    # y.shape 99*1;     # x1.shape 99*3;     # x2.shape 99*300
    f = open(annotation_path, 'w+')
    for i in range(len(npz_tunple_list)):
        (npz_file_path, data_srcnode) = npz_tunple_list[i]
        f.write(npz_file_path)
        f.write('\n')
    f.close()

def mygenerator(annotation_lines, batch_size):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print('begin_i_n:{},{}'.format(i,n))
    # i = (i + 1) % n; 遍历够一遍了 就重排shuffle;
    while True:
        x = []
        y = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            y_new, x_d_new, x_ind_new = process_data_npz(annotation_lines[i])
            tmp_x = np.hstack((x_d_new, x_ind_new))
            tmp_y = y_new
            x.append(tmp_x)
            y.append(tmp_y)
            i = (i + 1) % n
        input = np.array(x)
        output = np.array(y)
        input = np.reshape(input, (-1, 303))
        output = np.reshape(output, (-1, 1))
        yield ({'input': input}, {'output': output})

if __name__ == "__main__":
    # 获取所有GPU组成list
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    # 设置按需申请
    # 由于我这里仅有一块GPU,multi-GPU需要for一下
    tf.config.experimental.set_memory_growth(gpus[0], True)

    eve_dir = "..\\Main\\collect_data_blackhole"
    ml_dir = "..\\Main\\ML_blackhole"
    if os.path.exists(ml_dir):
        shutil.rmtree(ml_dir)
        print('delete dir ' + ml_dir)
    os.makedirs(ml_dir)
    print('add dir ' + ml_dir)
    h5_filepath = ml_dir + "\\model.h5"

    x = tf.random.uniform([1, 1])
    tmp = x.device.endswith('GPU:0')
    print('On GPU:{}'.format(tmp))

    # 把文件名 和 对应的数据源 洗出来
    # y_final, x_final = collect_data_totrain(dir)

    annotation_path = "..\\Main\\anno_blackhole"
    if os.path.exists(annotation_path):
        shutil.rmtree(annotation_path)
    os.makedirs(annotation_path)
    anno_file = os.path.join(annotation_path, "anno.txt")

    build_anno(eve_dir, anno_file)

    # 0.7:0.1:0.2 train val test
    val_test_split = 0.2
    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(anno_file) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_test = int(len(lines) * val_test_split)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val - num_test
    # lines[:num_train] lines[num_train:num_val] lines[num_train+num_val:]

    direct_and_indirect(lines[: num_train], lines[num_train : num_train + num_val], h5_filepath)

    test(lines[num_train + num_val:], h5_filepath)


    # train_from_DirectEvidence(y_final, x1_final, x2_final)
    # print('Direct Eve Training... Completed!')
    # train_from_inDirectEvidence(y_final, x1_final, x2_final)
    # print('InDirect Eve Training... Completed!')


