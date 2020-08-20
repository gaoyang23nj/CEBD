import numpy as np
import os
import tensorflow as tf
import multiprocessing

NUM_of_DIMENSIONS = 10
NUM_of_DIRECT_INPUTS = 8
NUM_of_INDIRECT_INPUTS = 9
MAX_RUNNING_TIMES = 864000

ProcessCtl_dict_time = dict()
ProcessCtl_dict_time["running_label"] = False
ProcessCtl_dict_time["key"] = 0
Detect_and_ban_time_q_input = multiprocessing.Queue()
Detect_and_ban_time_q_output = multiprocessing.Queue()


class DetectProcessManager(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_the_instance'):
            cls._the_instance = object.__new__(cls, *args, **kwargs)
        return cls._the_instance

    def __init__(self):
        global ProcessCtl_dict_time
        global Detect_and_ban_time_q_input
        global Detect_and_ban_time_q_output
        tmp_dir = "..\\Main\\ML_blackhole_time"
        direct_model_file_path = os.path.join(tmp_dir, 'our_direct_model.h5')
        indirect_model_file_path = os.path.join(tmp_dir, 'our_indirect_model.h5')
        self.model_files_path = (direct_model_file_path, indirect_model_file_path)

        self.MAX_Ability = (10000, 'Max Process Ability', 'Continue')
        if not ProcessCtl_dict_time["running_label"]:
            ProcessCtl_dict_time["running_label"] = True
            j = multiprocessing.Process(target=self.process_predict_blackhole_d_ind_direct_coll, args=(
                self.model_files_path, self.MAX_Ability, Detect_and_ban_time_q_input, Detect_and_ban_time_q_output))
            j.daemon = True
            j.start()

    def request(self, x, y):
        global ProcessCtl_dict_time
        global Detect_and_ban_time_q_input
        global Detect_and_ban_time_q_output

        request_element = (ProcessCtl_dict_time["key"], x, y)
        ProcessCtl_dict_time["key"] = ProcessCtl_dict_time["key"] + 1
        Detect_and_ban_time_q_input.put(request_element)

        result_element = Detect_and_ban_time_q_output.get(True)
        if result_element[3] == self.MAX_Ability[1]:
            j = multiprocessing.Process(target=self.process_predict_blackhole_d_ind_direct_coll, args=(
                self.model_files_path, self.MAX_Ability, Detect_and_ban_time_q_input, Detect_and_ban_time_q_output))
            j.daemon = True
            j.start()
        return result_element

    @staticmethod
    def close_process():
        global ProcessCtl_dict_time
        global Detect_and_ban_time_q_input

        if ProcessCtl_dict_time["running_label"]:
            Detect_and_ban_time_q_input.put(None)
            ProcessCtl_dict_time["running_label"] = False

    @staticmethod
    def cal_conf_matrix(y_true, y_predict, num_classes):
        res = np.zeros((num_classes, num_classes), dtype='int')
        res[y_true][y_predict] = 1
        return res

    @staticmethod
    def extract_indirect_data(x, y, ll):
        assert y.shape[0] == ll
        assert 0 == x.shape[1] % NUM_of_DIMENSIONS
        # 6个x数值
        runtime_data = x[:, 0]
        ind_data = x[:, NUM_of_DIMENSIONS:]

        num_of_views = int(ind_data.shape[1] / NUM_of_DIMENSIONS)
        y = np.expand_dims(y, 0)
        output = np.repeat(y, num_of_views, axis=1).reshape(-1, 1)

        ll = ll * num_of_views
        ind_in = np.zeros((ll, NUM_of_INDIRECT_INPUTS), dtype='float64')
        # delta time (1000, 98) / (1000,1)
        runtime_data = np.expand_dims(runtime_data, axis=1)
        tmp = np.repeat(runtime_data, num_of_views, axis=1)

        ind_in[:, 0] = np.true_divide(tmp - ind_data[:, 0: num_of_views], tmp+1).reshape(-1, 1).squeeze()
        # (1000, 98) / 1
        ind_in[:, 1] = np.true_divide(ind_data[:, 0: num_of_views], MAX_RUNNING_TIMES).reshape(-1, 1).squeeze()

        # (1000, 98) / (1000, 98)
        # ind_data[:, 1* num_of_views: 2* num_of_views]
        ind_in[:, 2] = np.true_divide(ind_data[:, 1 * num_of_views: 2 * num_of_views],
                                      ind_data[:, 7 * num_of_views: 8 * num_of_views]+1).reshape(-1, 1).squeeze()
        # ind_data[:, 5 * num_of_views: 6 * num_of_views]
        ind_in[:, 3] = np.true_divide(ind_data[:, 5 * num_of_views: 6 * num_of_views],
                                      ind_data[:, 7 * num_of_views: 8 * num_of_views]+1).reshape(-1, 1).squeeze()
        # ind_data[:, 6 * num_of_views: 7 * num_of_views]
        ind_in[:, 4] = np.true_divide(ind_data[:, 6 * num_of_views: 7 * num_of_views],
                                      ind_data[:, 7 * num_of_views: 8 * num_of_views]+1).reshape(-1, 1).squeeze()

        # ind_data[:, 2* num_of_views: 3* num_of_views]
        ind_in[:, 5] = np.true_divide(ind_data[:, 2 * num_of_views: 3 * num_of_views],
                                      ind_data[:, 8 * num_of_views: 9 * num_of_views]+1).reshape(-1, 1).squeeze()
        # ind_data[:, 3* num_of_views: 4* num_of_views]
        ind_in[:, 6] = np.true_divide(ind_data[:, 3 * num_of_views: 4 * num_of_views],
                                      ind_data[:, 8 * num_of_views: 9 * num_of_views]+1).reshape(-1, 1).squeeze()
        # ind_data[:, 4* num_of_views: 5* num_of_views]
        ind_in[:, 7] = np.true_divide(ind_data[:, 4 * num_of_views: 5 * num_of_views],
                                      ind_data[:, 8 * num_of_views: 9 * num_of_views]+1).reshape(-1, 1).squeeze()

        # N_{rcv}^{ss}(i)/(N_{rcv}^{all} + 1)
        ind_in[:, 8] = np.true_divide(ind_data[:, 9 * num_of_views: 10 * num_of_views],
                                      ind_data[:, 8 * num_of_views: 9 * num_of_views] + 1).reshape(-1, 1).squeeze()

        # add get_receive_from_and_pktsrc()对应的值[]

        return ind_in, output, ll

    @staticmethod
    def extract_direct_data(x, ll):
        assert x.shape[0] == ll
        # 6个x数值
        # runtime_data = x[:, 0]
        d_data = x[:, 0:NUM_of_DIMENSIONS]
        # ind_data = x[:, NUM_of_DIMENSIONS:]

        d_input = np.zeros((ll, NUM_of_DIRECT_INPUTS), dtype='float64')

        # time                                      t_{c} / t_{w}
        # input[:, 6] = np.divide(d_data[:, 0], MAX_RUNNING_TIMES)
        d_input[:, 0] = np.divide(d_data[:, 0], MAX_RUNNING_TIMES)

        # d_data[:, 1] / d_data[:, 7]                N_{snd}^{snd}(i)/(N_{snd}^{all} + 1)
        d_input[:, 1] = np.divide(d_data[:, 1], d_data[:, 7] + 1)
        # d_data[:, 5] / d_data[:, 7]                N_{snd}^{src}(i)/(N_{snd}^{all} + 1)
        d_input[:, 2] = np.divide(d_data[:, 5], d_data[:, 7] + 1)
        # d_data[:, 6] / d_data[:, 7]                N_{snd}^{dst}(i)/(N_{snd}^{all} + 1)
        d_input[:, 3] = np.divide(d_data[:, 6], d_data[:, 7] + 1)

        # d_data[:, 2] / d_data[:, 8]               N_{rcv}^{rcv}(i)/(N_{rcv}^{all} + 1)
        d_input[:, 4] = np.divide(d_data[:, 2], d_data[:, 8] + 1)
        # d_data[:, 3] / d_data[:, 8]               N_{rcv}^{src}(i)/(N_{rcv}^{all} + 1)
        d_input[:, 5] = np.divide(d_data[:, 3], d_data[:, 8] + 1)
        # d_data[:, 4] / d_data[:, 8]               N_{rcv}^{dst}(i)/(N_{rcv}^{all} + 1)
        d_input[:, 6] = np.divide(d_data[:, 4], d_data[:, 8] + 1)

        # add get_receive_from_and_pktsrc()对应的值 N_{rcv}^{ss}(i)/(N_{snd}^{all} + 1)
        d_input[:, 7] = np.divide(d_data[:, 9], d_data[:, 8] + 1)

        return d_input

    def process_predict_blackhole_d_ind_direct_coll(self, files_path, max_ability, q_input, q_output):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        num_to_process = 0
        print('.........Process Running...pid[{}]'.format(os.getpid()))
        d_model = tf.keras.models.load_model(files_path[0])
        ind_model = tf.keras.models.load_model(files_path[1])
        while True:
            em = q_input.get(True)
            if em is None:
                break
            x = em[1]
            i_isSelfish = em[2]
            y_final = np.zeros(1, dtype='int')
            y_final[0] = i_isSelfish

            ind_x, ind_y, ind_ll = self.extract_indirect_data(x, y_final, 1)
            d_x = self.extract_direct_data(x, 1)

            num_of_views = int(ind_x.shape[0] / 1)

            ind_predict_y = ind_model.predict(ind_x)
            # 转化为行向量
            tmp_ind = ind_predict_y[:, 1].reshape(-1, num_of_views)
            d_predict_y = d_model.predict(d_x)
            tmp_d = d_predict_y[:, 1].reshape(-1, 1)
            # tmp_res = np.hstack((tmp_d, tmp_ind))
            # final_res = np.sum(tmp_res, axis=1) / tmp_res.shape[1]
            # isB_predict = final_res > 0.5

            # y_predict = np.zeros((1), dtype='int')
            # y_predict[0] = int(isB_predict)
            # conf_matrix = np.array(tf.math.confusion_matrix(y_final, y_predict, num_classes=2))
            num_to_process = num_to_process + 1
            # print('.........Process Running...pid[{}],no.{}'.format(os.getpid(), num_to_process))
            if num_to_process >= max_ability[0]:
                # 到达 最大可执行数目
                res = (em[0], tmp_d, tmp_ind, max_ability[1])
                q_output.put(res)
                break
            else:
                # 仍然可以执行
                res = (em[0], tmp_d, tmp_ind, max_ability[2])
                q_output.put(res)
