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
from sklearn.svm import SVC
from sklearn.externals import joblib

# 对于每条数据,每个view来说, 有10个值作为属性
NUM_of_DIMENSIONS = 10
NUM_of_DIRECT_INPUTS = 8
NUM_of_INDIRECT_INPUTS = 9
# NUM_of_COMBINE_INPUTS = 301
NUM_of_COMBINE_INPUTS = 298
MAX_RUNNING_TIMES = 864000
NUM_LINES_In_One_File = 1000
batch_size = 100
INDIRECT_USE_SIZE = 5000

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

def get_direct_data_x_y(annotation_lines, batch_size):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print('\nbegin_i_n:{},{}'.format(i,n))
    
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

    total = np.hstack((input, output))
    np.random.shuffle(total)
    res_input = total[0:index, :-1]
    res_output = total[0:index, -1]
    return res_input, res_output

def train_direct(lines_train, lines_val, h5_direct_filepath):
    num_train = len(lines_train)
    num_val = len(lines_val)
    print('num_train, num_val: {},{}'.format(num_train, num_val))

    res_input, res_output = get_direct_data_x_y(lines_train, batch_size)
    clf = SVC(gamma='scale')
    # clf = SVC()
    print('Training*********')
    clf.fit(res_input, res_output)
    
    x_test, y_test = get_direct_data_x_y(lines_val, batch_size)
    print('Testing*********')
    y_predict = clf.predict(x_test)
    
    matrix_conf = tf.math.confusion_matrix(y_test, y_predict, num_classes=2)
    print("validate direct_model...\n".format(datetime.datetime.now()))
    print(matrix_conf)
    # 正类是恶意blackhole节点
    accuracy = (matrix_conf[0,0]+matrix_conf[1,1])/np.sum(matrix_conf)
    # precison = 真阳/(真阳+假阳)
    precision = matrix_conf[1,1]/(matrix_conf[1,1]+matrix_conf[0,1])
    # reacall = 真阳/(真阳+假阴)
    recall = matrix_conf[1, 1] / (matrix_conf[1, 1] + matrix_conf[1, 0])
    print('\n[direct_and_indirect_test over!] accuracy={} precision={} recall={}\n'.format(accuracy, precision, recall))
    joblib.dump(clf, h5_direct_filepath) 

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

def get_indiret_data_x_y(annotation_lines, batch_size):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print('\nbegin_i_n:{},{}'.format(i,n))

    input = np.zeros((NUM_LINES_In_One_File * batch_size * 100, NUM_of_INDIRECT_INPUTS))
    output = np.zeros((NUM_LINES_In_One_File * batch_size * 100, 1))
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
    total = np.hstack((input, output))
    np.random.shuffle(total)
    res = total[0:INDIRECT_USE_SIZE, :]
    res_input = res[0:index, :-1]
    res_output = res[0:index, -1]
    # res_output = np.squeeze(res_output)
    return res_input, res_output

def train_indirect(lines_train, lines_val, h5_indirect_filepath):
    num_train = len(lines_train)
    num_val = len(lines_val)
    print('num_train, num_val: {},{}'.format(num_train, num_val))

    res_input, res_output = get_indiret_data_x_y(lines_train, batch_size)
    clf = SVC(gamma='scale')
    # clf = SVC()
    print('Training*********')
    clf.fit(res_input, res_output)
    
    x_test, y_test = get_indiret_data_x_y(lines_val, batch_size)
    print('Testing*********')
    y_predict = clf.predict(x_test)
    
    matrix_conf = tf.math.confusion_matrix(y_test, y_predict, num_classes=2)
    print("validate indirect_model...\n".format(datetime.datetime.now()))
    print(matrix_conf)
    # 正类是恶意blackhole节点
    accuracy = (matrix_conf[0,0]+matrix_conf[1,1])/np.sum(matrix_conf)
    # precison = 真阳/(真阳+假阳)
    precision = matrix_conf[1,1]/(matrix_conf[1,1]+matrix_conf[0,1])
    # reacall = 真阳/(真阳+假阴)
    recall = matrix_conf[1, 1] / (matrix_conf[1, 1] + matrix_conf[1, 0])
    print('\n[validate indirect_model over!] accuracy={} precision={} recall={}\n'.format(accuracy, precision, recall))
    joblib.dump(clf, h5_indirect_filepath)

# 采用更好的方案 速度更快 约9分钟
def direct_and_indirect_test(lines_test, h5_direct_filepath, h5_indirect_filepath):
    print("{} begin direct_and_indirect_test\n".format(datetime.datetime.now()))
    matrix_conf = np.zeros(shape=(2,2), dtype='int')
    
    d_model = joblib.load(h5_direct_filepath)
    ind_model = joblib.load(h5_indirect_filepath)
    
    # num_testlines = len(lines_test)
    num_testlines = 100
    i = 0
    batch_size = 1

    # y_final = np.zeros(shape=(NUM_LINES_In_One_File * batch_size, 1))
    # x_final = np.zeros(shape=(NUM_LINES_In_One_File * batch_size, NUM_of_DIRECT_INPUTS))
    # # indirect
    # ind_input = np.zeros((NUM_LINES_In_One_File * batch_size, NUM_of_INDIRECT_INPUTS))
    # output = np.zeros((NUM_LINES_In_One_File * batch_size, 1))
    # # direct
    # d_input = np.zeros((NUM_LINES_In_One_File * batch_size, NUM_of_DIRECT_INPUTS))
    # output = np.zeros((NUM_LINES_In_One_File * batch_size, 1))

    print('test:{} (100)\t actually:{}'.format(num_testlines, int(num_testlines)))
    while i < int(num_testlines):
        y, x, ll = process_data_npz(lines_test[i])

        print("No. i:{} ll:{} \t {}".format(i, ll, datetime.datetime.now()))

        ind_x, ind_y, ind_ll = extract_indirect_data(x, y, ll)
        d_x = extract_direct_data(x, ll)

        num_of_views = int(ind_y.shape[0] / ll)

        ind_predict_y = ind_model.predict(ind_x)
        tmp_ind = ind_predict_y.reshape(-1, num_of_views)
        d_predict_y = d_model.predict(d_x)
        tmp_d = d_predict_y.reshape(-1, 1)
        tmp_res = np.hstack((tmp_d, tmp_ind))
        final_res = np.sum(tmp_res, axis=1) / tmp_res.shape[1]
        y_predict = (final_res > 0.35).astype(int)
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
def build_anno(eve_dirs, annotation_path):
    scenario_dirs = []
    # 把文件名 和 对应的数据源 洗出来
    npz_tunple_list = []
    print(eve_dirs, annotation_path)
    for eve_dir in eve_dirs:
        tmps = os.listdir(eve_dir)
        for tmp in tmps:
            tmp_scenario = os.path.join(eve_dir, tmp)
            scenario_dirs.append(tmp_scenario)
    print(scenario_dirs)
    for scenario_dir in scenario_dirs:
        # scenario_path = os.path.join(eve_dir, scenario_dir)
        scenario_path = scenario_dir
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

def train_for_RWP_dataset():
    #==========================    RWP trace blackhole node 检测模型的训练 ====================================#
    eve_dirs = ["E:\\collect_data_blackhole_time"]

    annotation_dir = ".\\anno_blackhole_grayhole_time"
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
        print('add dir ' + annotation_dir)
    else:
        print(annotation_dir + ' exist')
    anno_bk_filepath = os.path.join(annotation_dir, "anno_blackhole_1.txt")
    if os.path.exists(anno_bk_filepath):
        os.remove(anno_bk_filepath)
    # print(eve_dirs, anno_bk_filepath)
    build_anno(eve_dirs, anno_bk_filepath)

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

    print("************************************")

    print("\tMethod-2:\t(1)train direct_model.h5 and indirect_model.h5 respectively;\t(2)use average as predict value.\n")
    ml_dir = "..\\ML_blackhole_time"
    if not os.path.exists(ml_dir):
        os.makedirs(ml_dir)
        print('add dir ' + ml_dir)
    else:
        print(ml_dir + ' exist')

    h5_direct_filepath = os.path.join(ml_dir, "our_direct_model_1_SVM.pkl")
    h5_indirect_filepath = os.path.join(ml_dir, "our_indirect_model_1_SVM.pkl")

    # # 下面开始训练direct_model 和 indirect model
    if os.path.exists(h5_direct_filepath):
        os.remove(h5_direct_filepath)
    if os.path.exists(h5_indirect_filepath):
        os.remove(h5_indirect_filepath)
    train_indirect(lines[: num_train], lines[num_train : num_train + num_val], h5_indirect_filepath)
    train_direct(lines[: num_train], lines[num_train : num_train + num_val], h5_direct_filepath)

    direct_and_indirect_test(lines[num_train + num_val:], h5_direct_filepath, h5_indirect_filepath)

if __name__ == "__main__":
    print('SVM classify and average!..')
    
    #==========================    shanghai trace blackhole node 检测模型的训练 ====================================#
    # train_for_shanghai_dataset()

    #==========================    RWP trace blackhole node SVM 检测模型的训练 ====================================#
    train_for_RWP_dataset()




