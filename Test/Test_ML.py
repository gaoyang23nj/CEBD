import numpy as np
import tensorflow as tf
import os
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
        data_srcnode = int(file.split('.')[0])
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
    print('num_train: {}, {}; num_test, {}, {};'.format(y_train.shape, np.sum(y_train), y_test.shape, np.sum(y_test)))

if __name__ == "__main__":
    train_from_DirectEvidence()
    # train_from_inDirectEvidence()