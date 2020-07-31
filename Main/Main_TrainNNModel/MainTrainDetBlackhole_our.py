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

# 对于每条数据,每个view来说, 有10个值作为属性
NUM_of_DIMENSIONS = 10
NUM_of_DIRECT_INPUTS = 8
NUM_of_INDIRECT_INPUTS = 9
# NUM_of_COMBINE_INPUTS = 301
NUM_of_COMBINE_INPUTS = 298
MAX_RUNNING_TIMES = 864000
NUM_LINES_In_One_File = 1000

def extract_direct_data(x, ll):
    assert x.shape[0] == ll
    # 6个x数值
    runtime_data = x[:, 0]
    d_data = x[:, 0:NUM_of_DIMENSIONS]
    ind_data = x[:, NUM_of_DIMENSIONS:]

    input = np.zeros((ll, NUM_of_DIRECT_INPUTS), dtype='float64')
    # time                                      t_{c} / t_{w}
    # input[:, 6] = np.divide(d_data[:, 0], MAX_RUNNING_TIMES)
    input[:, 0] = np.divide(d_data[:, 0], MAX_RUNNING_TIMES)

    # d_data[:, 1] / d_data[:, 7]                N_{snd}^{snd}(i)/(N_{snd}^{all} + 1)
    input[:, 1] = np.divide(d_data[:, 1], d_data[:, 7] + 1)
    # d_data[:, 5] / d_data[:, 7]                N_{snd}^{src}(i)/(N_{snd}^{all} + 1)
    input[:, 2] = np.divide(d_data[:, 5], d_data[:, 7] + 1)
    # d_data[:, 6] / d_data[:, 7]                N_{snd}^{dst}(i)/(N_{snd}^{all} + 1)
    input[:, 3] = np.divide(d_data[:, 6], d_data[:, 7] + 1)

    # d_data[:, 2] / d_data[:, 8]               N_{rcv}^{rcv}(i)/(N_{rcv}^{all} + 1)
    input[:, 4] = np.divide(d_data[:, 2], d_data[:, 8] + 1)
    # d_data[:, 3] / d_data[:, 8]               N_{rcv}^{src}(i)/(N_{rcv}^{all} + 1)
    input[:, 5] = np.divide(d_data[:, 3], d_data[:, 8] + 1)
    # d_data[:, 4] / d_data[:, 8]               N_{rcv}^{dst}(i)/(N_{rcv}^{all} + 1)
    input[:, 6] = np.divide(d_data[:, 4], d_data[:, 8] + 1)

    # add get_receive_from_and_pktsrc()对应的值 N_{rcv}^{ss}(i)/(N_{snd}^{all} + 1)
    input[:, 7] = np.divide(d_data[:, 9], d_data[:, 8] + 1)

    return input

def mygenerator_direct_data(annotation_lines, batch_size):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print('\nbegin_i_n:{},{}'.format(i,n))

    # i = (i + 1) % n; 遍历够一遍了 就重排shuffle;
    while True:
        input = np.zeros((NUM_LINES_In_One_File * batch_size, NUM_of_DIRECT_INPUTS))
        output = np.zeros((NUM_LINES_In_One_File * batch_size, 1))
        #
        index = 0
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            y, x, ll = process_data_npz(annotation_lines[i])
            # 属性整合
            x = extract_direct_data(x, ll)

            input[index: index + ll, :] = x
            output[index: index + ll, :] = y
            #
            index = index + ll
            i = (i + 1) % n
        res_input = input[0:index, :]
        res_output = output[0:index, :]

        yield ({'input': res_input}, {'output': res_output})

def train_direct(lines_train, lines_val, h5_direct_filepath):
    num_train = len(lines_train)
    num_val = len(lines_val)
    print('num_train, num_val: {},{}'.format(num_train, num_val))
    direct_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(NUM_of_DIRECT_INPUTS, name="input"),
        tf.keras.layers.Dense(NUM_of_DIRECT_INPUTS, activation='relu'),
        tf.keras.layers.Dense(NUM_of_DIRECT_INPUTS, activation='relu'),
        tf.keras.layers.Dense(NUM_of_DIRECT_INPUTS, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', name='output')
    ])
    direct_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    direct_model.summary()

    # 参数设置
    log_dir = 'logs_black_time_direct\\'
    # update_freq == 'batch'
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=False, save_best_only=False, save_freq='epoch')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    batch_size = 1
    # verbose = 1 默认; verbose=2 only epoch
    direct_model.fit_generator(mygenerator_direct_data(lines_train, batch_size),
                                steps_per_epoch=max(1, num_train//batch_size),
                                validation_data=mygenerator_direct_data(lines_val, batch_size),
                                validation_steps=max(1, num_val // batch_size),
                                verbose=2,
                                epochs=10,
                                initial_epoch=0,
                                callbacks=[logging, checkpoint, early_stopping])
    direct_model.save(h5_direct_filepath)

def extract_indirect_data(x, y, ll):
    assert y.shape[0] == ll and y.shape[1] == 1
    assert 0 == x.shape[1] % NUM_of_DIMENSIONS
    # 6个x数值
    runtime_data = x[:, 0]
    ind_data = x[:, NUM_of_DIMENSIONS:]

    num_of_views = int(ind_data.shape[1] / NUM_of_DIMENSIONS)
    output = np.repeat(y, num_of_views, axis=1).reshape(-1,1)

    ll = ll * num_of_views
    input = np.zeros((ll, NUM_of_INDIRECT_INPUTS), dtype='float64')
    # delta time (1000, 98) / (1000,1)
    runtime_data = np.expand_dims(runtime_data, axis=1)
    tmp = np.repeat(runtime_data, num_of_views, axis=1)

    input[:,0] = np.true_divide(tmp - ind_data[:, 0: num_of_views], tmp+1).reshape(-1,1).squeeze()
    # (1000, 98) / 1
    input[:,1] = np.true_divide(ind_data[:, 0: num_of_views], MAX_RUNNING_TIMES).reshape(-1,1).squeeze()

    # (1000, 98) / (1000, 98)
    # ind_data[:, 1* num_of_views: 2* num_of_views]
    input[:,2] = np.true_divide(ind_data[:, 1* num_of_views: 2* num_of_views],
                                ind_data[:, 7* num_of_views: 8* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 5 * num_of_views: 6 * num_of_views]
    input[:,3] = np.true_divide(ind_data[:, 5 * num_of_views: 6 * num_of_views],
                                ind_data[:, 7* num_of_views: 8* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 6 * num_of_views: 7 * num_of_views]
    input[:,4] = np.true_divide(ind_data[:, 6 * num_of_views: 7 * num_of_views],
                                ind_data[:, 7 * num_of_views: 8 * num_of_views]+1).reshape(-1,1).squeeze()

    # ind_data[:, 2* num_of_views: 3* num_of_views]
    input[:,5] = np.true_divide(ind_data[:, 2* num_of_views: 3* num_of_views],
                                ind_data[:, 8* num_of_views: 9* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 3* num_of_views: 4* num_of_views]
    input[:,6] = np.true_divide(ind_data[:, 3* num_of_views: 4* num_of_views],
                                ind_data[:, 8* num_of_views: 9* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 4* num_of_views: 5* num_of_views]
    input[:,7] = np.true_divide(ind_data[:, 4* num_of_views: 5* num_of_views],
                                ind_data[:, 8* num_of_views: 9* num_of_views]+1).reshape(-1,1).squeeze()

    # N_{rcv}^{ss}(i)/(N_{rcv}^{all} + 1)
    input[:,8] = np.true_divide(ind_data[:, 9 * num_of_views: 10 * num_of_views],
                                 ind_data[:, 8 * num_of_views: 9 * num_of_views] + 1).reshape(-1, 1).squeeze()
    return input, output, ll

def mygenerator_indirect_data(annotation_lines, batch_size):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print('\nbegin_i_n:{},{}'.format(i,n))

    # i = (i + 1) % n; 遍历够一遍了 就重排shuffle;
    while True:
        input = np.zeros((NUM_LINES_In_One_File * batch_size*100, NUM_of_INDIRECT_INPUTS))
        output = np.zeros((NUM_LINES_In_One_File * batch_size*100, 1))
        #
        index = 0
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            y, x, ll = process_data_npz(annotation_lines[i])
            # 属性整合
            x, y, ll = extract_indirect_data(x, y, ll)

            input[index: index + ll, :] = x
            output[index: index + ll, :] = y
            #
            index = index + ll
            i = (i + 1) % n
        res_input = input[0:index, :]
        res_output = output[0:index, :]

        yield ({'input': res_input}, {'output': res_output})

def train_indirect(lines_train, lines_val, h5_indirect_filepath):
    num_train = len(lines_train)
    num_val = len(lines_val)
    print('num_train, num_val: {},{}'.format(num_train, num_val))
    indirect_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(NUM_of_INDIRECT_INPUTS, name="input"),
        tf.keras.layers.Dense(NUM_of_INDIRECT_INPUTS, activation='relu'),
        tf.keras.layers.Dense(NUM_of_INDIRECT_INPUTS, activation='relu'),
        tf.keras.layers.Dense(NUM_of_INDIRECT_INPUTS, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', name='output')
    ])
    indirect_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    indirect_model.summary()

    # 参数设置
    log_dir = 'logs_black_time_indirect\\'
    # update_freq == 'batch'
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=False, save_best_only=False, save_freq='epoch')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    batch_size = 1
    # verbose = 1 默认; verbose=2 only epoch
    indirect_model.fit_generator(mygenerator_indirect_data(lines_train, batch_size),
                                steps_per_epoch=max(1, num_train//batch_size),
                                validation_data=mygenerator_indirect_data(lines_val, batch_size),
                                validation_steps=max(1, num_val // batch_size),
                                verbose=2,
                                epochs=10,
                                initial_epoch=0,
                                callbacks=[logging, checkpoint, early_stopping])
    indirect_model.save(h5_indirect_filepath)


# 采用更好的方案 速度更快 约9分钟
def direct_and_indirect_test(lines_test,  h5_direct_filepath, h5_indirect_filepath):
    print("{} begin direct_and_indirect_test\n".format(datetime.datetime.now()))
    matrix_conf = np.zeros(shape=(2,2), dtype='int')
    d_model = tf.keras.models.load_model(h5_direct_filepath)
    ind_model = tf.keras.models.load_model(h5_indirect_filepath)
    num_testlines = len(lines_test)
    i = 0
    batch_size = 8

    # y_final = np.zeros(shape=(NUM_LINES_In_One_File * batch_size, 1))
    # x_final = np.zeros(shape=(NUM_LINES_In_One_File * batch_size, NUM_of_DIRECT_INPUTS))
    # # indirect
    # ind_input = np.zeros((NUM_LINES_In_One_File * batch_size, NUM_of_INDIRECT_INPUTS))
    # output = np.zeros((NUM_LINES_In_One_File * batch_size, 1))
    # # direct
    # d_input = np.zeros((NUM_LINES_In_One_File * batch_size, NUM_of_DIRECT_INPUTS))
    # output = np.zeros((NUM_LINES_In_One_File * batch_size, 1))

    print('test:{} (100)\t actually:{}'.format(num_testlines, int(num_testlines / batch_size) * batch_size))
    while i < int(num_testlines/batch_size)*batch_size:
        y, x, ll = process_data_npz(lines_test[i])

        ind_x, ind_y, ind_ll = extract_indirect_data(x, y, ll)
        d_x = extract_direct_data(x, ll)

        num_of_views = int(ind_y.shape[0] / ll)

        ind_predict_y = ind_model.predict(ind_x)
        tmp_ind = ind_predict_y[:,1].reshape(-1, num_of_views)
        d_predict_y = d_model.predict(d_x)
        tmp_d = d_predict_y[:, 1].reshape(-1, 1)
        tmp_res = np.hstack((tmp_d, tmp_ind))
        final_res = np.sum(tmp_res, axis=1) / tmp_res.shape[1]
        y_predict = (final_res > 0.5).astype(int)
        tmp_conf_matrix = tf.math.confusion_matrix(y, y_predict, num_classes=2)
        matrix_conf = matrix_conf + tmp_conf_matrix
        if i % 10000 == 0:
            print("{}\t {}".format(i*10000,datetime.datetime.now()))
            # print(tmp_conf_matrix)
            # print(matrix_conf)
        i = i+1
    print("{} complete direct_and_indirect_test...\n".format(datetime.datetime.now()))
    print(matrix_conf)
    # 正类是恶意blackhole节点
    accuracy = (matrix_conf[0,0]+matrix_conf[1,1])/np.sum(matrix_conf)
    # precison = 真阳/(真阳+假阳)
    precision = matrix_conf[1,1]/(matrix_conf[1,1]+matrix_conf[0,1])
    # reacall = 真阳/(真阳+假阴)
    recall = matrix_conf[1, 1] / (matrix_conf[1, 1] + matrix_conf[1, 0])
    print('\n[direct_and_indirect_test over!] accuracy={} precision={} recall={}\n'.format(accuracy, precision, recall))

def process_data_npz(file_path):
    file_path = file_path.rstrip()
    # print(file_path)
    data = np.load(file_path)
    # 取出数据； 并删除 本节点的评价
    y = data['y']
    x = data['x']
    ll = data['length']
    return y, x, ll

# 从文件中收集数据
def build_anno(eve_dir, annotation_path):
    # 把文件名 和 对应的数据源 洗出来
    npz_tunple_list = []
    print(eve_dir, annotation_path)
    scenario_dirs = os.listdir(eve_dir)
    print(eve_dir, annotation_path)
    for scenario_dir in scenario_dirs:
        scenario_path = os.path.join(eve_dir, scenario_dir)
        if not os.path.isdir(scenario_path):
            continue
        files = os.listdir(scenario_path)
        for file in files:
            npz_file_path = os.path.join(scenario_path, file)
            # 只处理npz文件
            m = re.match(r'(\d*).npz', file)
            if m is None:
                continue
            # 文件名
            npzfile_gen_time = int(m.group(1))
            npz_tunple_list.append((npz_file_path, npzfile_gen_time))
    # 总的数据条目数未知; 但我们知道每个文件里有100条记录; 通过逐个文件读取 获得训练数据集合
    num_npzfile = len(npz_tunple_list)
    f = open(annotation_path, 'w+')
    for i in range(len(npz_tunple_list)):
        (npz_file_path, npzfile_gen_time) = npz_tunple_list[i]
        f.write(npz_file_path)
        f.write('\n')
    f.close()

def train_combine(lines_train, lines_val, h5_combine_filepath):
    num_train = len(lines_train)
    num_val = len(lines_val)
    print('num_train, num_val: {},{}'.format(num_train, num_val))
    combine_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(NUM_of_COMBINE_INPUTS, name="input"),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', name='output')
    ])
    combine_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    combine_model.summary()

    # 参数设置
    log_dir = 'logs_black_combine\\'
    # update_freq == 'batch'
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=False, save_best_only=False, save_freq='epoch')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    batch_size = 8
    # verbose = 1 默认; verbose=2 only epoch
    combine_model.fit_generator(mygenerator_combine_data(lines_train, batch_size),
                                steps_per_epoch=max(1, num_train//batch_size),
                                validation_data=mygenerator_combine_data(lines_val, batch_size),
                                validation_steps=max(1, num_val // batch_size),
                                verbose=2,
                                epochs=10,
                                initial_epoch=0,
                                callbacks=[logging, checkpoint, early_stopping])
    combine_model.save(h5_combine_filepath)


def extract_combine_data(x, ll):
    assert x.shape[0] == ll

    # 6个x数值
    runtime_data = x[:, 0]
    d_data = x[:, 0:NUM_of_DIMENSIONS]
    ind_data = x[:, NUM_of_DIMENSIONS:]
    # is not 99, is 98
    num_of_views = int(ind_data.shape[1] / NUM_of_DIMENSIONS)

    # 1*1
    dect_time = np.zeros((ll, 1), dtype='float64')
    dect_time[:, 0] = np.divide(d_data[:, 0], MAX_RUNNING_TIMES)

    # 1*3
    d_input = np.zeros((ll, 3), dtype='float64')
    # d_data[:,1]
    d_input[:, 0] = d_data[:, 1]
    # d_data[:,2]
    d_input[:, 1] = d_data[:, 2]
    # d_data[:,9]
    d_input[:, 2] = d_data[:, 3]

    # 99*3
    ind_input = np.zeros((ll, 3*num_of_views), dtype='float64')

    # send
    ind_input[:, 0: num_of_views] = ind_data[:, num_of_views: 2*num_of_views]
    # receive
    ind_input[:, num_of_views: 2*num_of_views] = ind_data[:, 2*num_of_views: 3*num_of_views]
    # receive_src
    ind_input[:, 2*num_of_views: 3*num_of_views] = ind_data[:, 9*num_of_views: 10*num_of_views]

    input = np.hstack((dect_time, d_input, ind_input))

    return input

def mygenerator_combine_data(annotation_lines, batch_size):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print('mygenerator_combine_data begin_i_n:{},{}\n'.format(i,n))

    # i = (i + 1) % n; 遍历够一遍了 就重排shuffle;
    while True:
        input = np.zeros((NUM_LINES_In_One_File * batch_size, NUM_of_COMBINE_INPUTS))
        output = np.zeros((NUM_LINES_In_One_File * batch_size, 1))
        #
        index = 0
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            y, x, ll = process_data_npz(annotation_lines[i])
            # 属性整合
            x = extract_combine_data(x, ll)

            input[index: index + ll, :] = x
            output[index: index + ll, :] = y
            #
            index = index + ll
            i = (i + 1) % n
        res_input = input[0:index, :]
        res_output = output[0:index, :]

        yield ({'input': res_input}, {'output': res_output})

def combine_test(lines_test, h5_combine_filepath):
    print("{} begin combine_test\n".format(datetime.datetime.now()))

    matrix_conf = np.zeros(shape=(2,2), dtype='int')
    combine_model = tf.keras.models.load_model(h5_combine_filepath)
    num_testlines = len(lines_test)
    i = 0
    batch_size = 8

    print('test:{} (100)\t actually:{}'.format(num_testlines, int(num_testlines / batch_size) * batch_size))
    while i < int(num_testlines/batch_size)*batch_size:
        y, x, ll = process_data_npz(lines_test[i])

        x = extract_combine_data(x, ll)

        combine_predict_y = combine_model.predict(x)
        y_predict = (combine_predict_y[:,1] > 0.5).astype(int)
        tmp_conf_matrix = tf.math.confusion_matrix(y, y_predict, num_classes=2)
        matrix_conf = matrix_conf + tmp_conf_matrix
        if i % 10000 == 0:
            print("{}\t {}".format(i*10000,datetime.datetime.now()))
            # print(tmp_conf_matrix)
            # print(matrix_conf)
        i = i+1
    print("{} complete combine_test...\n".format(datetime.datetime.now()))
    print(matrix_conf)
    # 正类是恶意blackhole节点
    accuracy = (matrix_conf[0,0]+matrix_conf[1,1])/np.sum(matrix_conf)
    # precison = 真阳/(真阳+假阳)
    precision = matrix_conf[1,1]/(matrix_conf[1,1]+matrix_conf[0,1])
    # reacall = 真阳/(真阳+假阴)
    recall = matrix_conf[1, 1] / (matrix_conf[1, 1] + matrix_conf[1, 0])
    print('\n[combine_test over!] accuracy={} precision={} recall={}\n'.format(accuracy, precision, recall))

if __name__ == "__main__":
    # 获取所有GPU组成list
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    cpus = tf.config.experimental.list_physical_devices('CPU')
    print(cpus)

    if len(gpus) > 0:
        # 设置按需申请
        # 由于我这里仅有一块GPU,multi-GPU需要for一下
        tf.config.experimental.set_memory_growth(gpus[0], True)

    eve_dirs = ["E:\\collect_data_blackhole_time", "E:\\collect_data_grayhole_time"]

    annotation_dir = ".\\anno_blackhole_grayhole_time"
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
        print('add dir ' + annotation_dir)
    else:
        print(annotation_dir + ' exist')
    anno_bk_filepath = os.path.join(annotation_dir, "anno_blackhole.txt")
    if os.path.exists(anno_bk_filepath):
        os.remove(anno_bk_filepath)
    print(eve_dirs[0], anno_bk_filepath)
    build_anno(eve_dirs[0], anno_bk_filepath)

    anno_gk_filepath = os.path.join(annotation_dir, "anno_grayhole.txt")
    if os.path.exists(anno_gk_filepath):
        os.remove(anno_gk_filepath)
    print(eve_dirs[1], anno_gk_filepath)
    build_anno(eve_dirs[1], anno_gk_filepath)

    anno_filepath = anno_bk_filepath

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

    # print("与之前的计算结果 略有差别~ 少了自己关于自己的结论。")
    # print("\tMethod-1:\t(1)train combine_model.\t(2)use combine_model result as predict value.\n")
    # ml_dir = "..\\Main\\ML_blackhole_time"
    # if not os.path.exists(ml_dir):
    #     os.makedirs(ml_dir)
    #     print('add dir ' + ml_dir)
    # else:
    #     print(ml_dir + ' exist')
    # h5_combine_filepath = os.path.join(ml_dir, "combine_model.h5")
    #
    # #下面开始训练combine model
    # # if os.path.exists(h5_combine_filepath):
    # #     os.remove(h5_combine_filepath)
    # # train_combine(lines[: num_train], lines[num_train : num_train + num_val], h5_combine_filepath)
    #
    # combine_test(lines[num_train + num_val:], h5_combine_filepath)

    print("************************************")

    print("\tMethod-2:\t(1)train direct_model.h5 and indirect_model.h5 respectively;\t(2)use average as predict value.\n")
    ml_dir = "..\\ML_blackhole_time"
    if not os.path.exists(ml_dir):
        os.makedirs(ml_dir)
        print('add dir ' + ml_dir)
    else:
        print(ml_dir + ' exist')
    h5_direct_filepath = os.path.join(ml_dir, "our_direct_model.h5")
    h5_indirect_filepath = os.path.join(ml_dir, "our_indirect_model.h5")

    # 下面开始训练direct_model 和 indirect model
    if os.path.exists(h5_direct_filepath):
        os.remove(h5_direct_filepath)
    if os.path.exists(h5_indirect_filepath):
        os.remove(h5_indirect_filepath)
    train_indirect(lines[: num_train], lines[num_train : num_train + num_val], h5_indirect_filepath)
    train_direct(lines[: num_train], lines[num_train : num_train + num_val], h5_direct_filepath)

    direct_and_indirect_test(lines[num_train + num_val:], h5_direct_filepath, h5_indirect_filepath)




