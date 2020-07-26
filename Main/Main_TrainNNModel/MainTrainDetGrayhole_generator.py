import shutil
import multiprocessing
import time
import math
import random

import numpy as np
import tensorflow as tf
import os
import re
import datetime

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

NUM_of_INPUTS = 301
MAX_RUNNING_TIMES = 864000
NUM_LINES_In_One_File = 1000

def direct_and_indirect(lines_train, lines_val, h5_filepath):
    num_train = len(lines_train)
    num_val = len(lines_val)
    print('num_train, num_val: {},{}'.format(num_train, num_val))
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(NUM_of_INPUTS, name="input"),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', name='output')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    print(model.summary)

    # 参数设置
    log_dir = 'logs_gray\\'
    # update_freq == 'batch'
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=False, save_best_only=False, save_freq='epoch')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    batch_size = 8
    # verbose = 1 默认; verbose=2 only epoch
    model.fit_generator(mygenerator(lines_train, batch_size),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=mygenerator(lines_val, batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        verbose=2,
                        epochs=10,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint, early_stopping])
    model.save(h5_filepath)

def mygenerator(annotation_lines, batch_size):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print('\nbegin_i_n:{},{}'.format(i,n))

    # i = (i + 1) % n; 遍历够一遍了 就重排shuffle;
    while True:
        input = np.zeros((NUM_LINES_In_One_File * batch_size, NUM_of_INPUTS))
        output = np.zeros((NUM_LINES_In_One_File * batch_size, 1))
        # x = []
        # y = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            y, x = process_data_npz(annotation_lines[i])
            tmp_input = np.array(x)
            tmp_output = np.array(y)
            input[b * NUM_LINES_In_One_File: (b + 1) * NUM_LINES_In_One_File] = x
            output[b * NUM_LINES_In_One_File: (b + 1) * NUM_LINES_In_One_File] = y
            i = (i + 1) % n
        yield ({'input': input}, {'output': output})

def test(lines_test, h5_filepath):
    print(datetime.datetime.now())
    matrix_conf = np.zeros(shape=(2,2), dtype='int')
    model = tf.keras.models.load_model(h5_filepath)
    num_testlines = len(lines_test)
    i = 0
    batchsize = 8
    y_final = np.zeros(shape=(NUM_LINES_In_One_File*batchsize, 1))
    x_final = np.zeros(shape=(NUM_LINES_In_One_File*batchsize, NUM_of_INPUTS))
    print('test:{} (100)\t actually:{}'.format(num_testlines, int(num_testlines / batchsize) * batchsize))
    while i < int(num_testlines/batchsize)*batchsize:
        for b in range(batchsize):
            y, x = process_data_npz(lines_test[i])
            # 对于0.npz 从0行到99行; 完成一个file
            y_final[b*NUM_LINES_In_One_File:(b+1)*NUM_LINES_In_One_File, :] = y.copy()
            x_final[b*NUM_LINES_In_One_File:(b+1)*NUM_LINES_In_One_File, :] = x.copy()
            i = i+1
        # 对一个batchsize的file进行预测处理
        y_pred_raw = model.predict(x_final)
        y_predict = (y_pred_raw[:, 1] > 0.5).astype(int)
        tmp = tf.math.confusion_matrix(y_final, y_predict, num_classes=2)
        matrix_conf = matrix_conf + tmp
        if i % 5000 == 0:
            print(datetime.datetime.now())
            print(tmp)
            print(matrix_conf)
    print(datetime.datetime.now())
    print(matrix_conf)

def process_data_npz(file_path):
    file_path = file_path.rstrip()
    # print(file_path)
    data = np.load(file_path)
    # 取出数据； 并删除 本节点的评价
    y = data['y']
    x = data['x']
    x = x.astype('float64')
    x[:, 0] = x[:, 0] / MAX_RUNNING_TIMES
    # x = np.delete(x, 0, axis=1)
    return y, x

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
    # 总的数据条目数未知; 但我们知道每个文件里有1000 即NUM_of_LINES_in_FILE条记录; 通过逐个文件读取 获得训练数据集合
    num_npzfile = len(npz_tunple_list)
    # y.shape 1000*1;     # x1.shape 1000*3;
    f = open(annotation_path, 'w+')
    for i in range(len(npz_tunple_list)):
        (npz_file_path, data_srcnode) = npz_tunple_list[i]
        f.write(npz_file_path)
        f.write('\n')
    f.close()

if __name__ == "__main__":
    # 获取所有GPU组成list
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    # 设置按需申请
    # 由于我这里仅有一块GPU,multi-GPU需要for一下
    tf.config.experimental.set_memory_growth(gpus[0], True)

    eve_dir = "E:\\collect_data_grayhole"
    ml_dir = "../ML_grayhole"
    if not os.path.exists(ml_dir):
        os.makedirs(ml_dir)
        print('add dir ' + ml_dir)
    else:
        print(ml_dir + ' exist')
    h5_filepath = os.path.join(ml_dir, "model.h5")
    if os.path.exists(h5_filepath):
        os.remove(h5_filepath)

    annotation_dir = "../anno_grayhole"
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
        print('add dir ' + annotation_dir)
    else:
        print(annotation_dir + ' exist')
    anno_filepath = os.path.join(annotation_dir, "anno.txt")
    if os.path.exists(anno_filepath):
        os.remove(anno_filepath)

    build_anno(eve_dir, anno_filepath)

    # 0.7:0.1:0.2 train val test
    val_test_split = 0.2
    # 0.1用于验证，0.9用于训练
    val_val_split = 0.2
    with open(anno_filepath) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_test = int(len(lines) * val_test_split)
    num_val = int(len(lines) * val_val_split)
    num_train = len(lines) - num_val - num_test
    # lines[:num_train] lines[num_train:num_val] lines[num_train+num_val:]

    direct_and_indirect(lines[: num_train], lines[num_train : num_train + num_val], h5_filepath)

    test(lines[num_train + num_val:], h5_filepath)


