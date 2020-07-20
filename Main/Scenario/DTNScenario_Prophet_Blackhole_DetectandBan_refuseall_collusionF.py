import datetime

from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt
from Main.DTNNodeBuffer_Detect import DTNNodeBuffer_Detect
from Main.DTNNodeBuffer_Detect_coll import DTNNodeBuffer_Detect_coll

import copy
import numpy as np
import math
import os
import tensorflow as tf
import multiprocessing
import re
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

NUM_of_DIMENSIONS = 10
NUM_of_DIRECT_INPUTS = 7
NUM_of_INDIRECT_INPUTS = 8
MAX_RUNNING_TIMES = 864000

def cal_conf_matrix(y_true, y_predict, num_classes):
    res = np.zeros((num_classes,num_classes), dtype = 'int')
    res[y_true][y_predict] = 1
    return res

def extract_indirect_data(x, y, ll):
    assert y.shape[0] == ll
    assert 0 == x.shape[1] % NUM_of_DIMENSIONS
    # 6个x数值
    runtime_data = x[:, 0]
    ind_data = x[:, NUM_of_DIMENSIONS:]

    num_of_views = int(ind_data.shape[1] / NUM_of_DIMENSIONS)
    y = np.expand_dims(y,0)
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
    input[:,2] = np.true_divide(ind_data[:, 1* num_of_views: 2* num_of_views], ind_data[:, 7* num_of_views: 8* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 5 * num_of_views: 6 * num_of_views]
    input[:,3] = np.true_divide(ind_data[:, 5 * num_of_views: 6 * num_of_views], ind_data[:, 7* num_of_views: 8* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 6 * num_of_views: 7 * num_of_views]
    input[:,4] = np.true_divide(ind_data[:, 6 * num_of_views: 7 * num_of_views], ind_data[:, 7 * num_of_views: 8 * num_of_views]+1).reshape(-1,1).squeeze()

    # ind_data[:, 2* num_of_views: 3* num_of_views]
    input[:,5] = np.true_divide(ind_data[:, 2* num_of_views: 3* num_of_views], ind_data[:, 8* num_of_views: 9* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 3* num_of_views: 4* num_of_views]
    input[:,6] = np.true_divide(ind_data[:, 3* num_of_views: 4* num_of_views], ind_data[:, 8* num_of_views: 9* num_of_views]+1).reshape(-1,1).squeeze()
    # ind_data[:, 4* num_of_views: 5* num_of_views]
    input[:,7] = np.true_divide(ind_data[:, 4* num_of_views: 5* num_of_views], ind_data[:, 8* num_of_views: 9* num_of_views]+1).reshape(-1,1).squeeze()

    # add get_receive_from_and_pktsrc()对应的值[]

    return input, output, ll

def extract_direct_data(x, ll):
    assert x.shape[0] == ll
    # 6个x数值
    runtime_data = x[:, 0]
    d_data = x[:, 0:NUM_of_DIMENSIONS]
    ind_data = x[:, NUM_of_DIMENSIONS:]

    input = np.zeros((ll, NUM_of_DIRECT_INPUTS), dtype='float64')
    # d_data[:, 1] / d_data[:, 7]
    input[:, 0] = np.divide(d_data[:, 1], d_data[:, 7] + 1)
    # d_data[:, 5] / d_data[:, 7]
    input[:, 1] = np.divide(d_data[:, 5], d_data[:, 7] + 1)
    # d_data[:, 6] / d_data[:, 7]
    input[:, 2] = np.divide(d_data[:, 6], d_data[:, 7] + 1)

    # d_data[:, 2] / d_data[:, 8]
    input[:, 3] = np.divide(d_data[:, 2], d_data[:, 8] + 1)
    # d_data[:, 3] / d_data[:, 8]
    input[:, 4] = np.divide(d_data[:, 3], d_data[:, 8] + 1)
    # d_data[:, 4] / d_data[:, 8]
    input[:, 5] = np.divide(d_data[:, 4], d_data[:, 8] + 1)

    # add get_receive_from_and_pktsrc()对应的值

    # time
    input[:, 6] = np.divide(d_data[:, 0], MAX_RUNNING_TIMES)
    return input

def process_predict_blackhole_d_ind_direct_coll(files_path, max_ability, q_input, q_output):
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
        y_final = np.zeros((1), dtype='int')
        y_final[0] = i_isSelfish

        ll = 1
        ind_x, ind_y, ind_ll = extract_indirect_data(x, y_final, 1)
        d_x = extract_direct_data(x, 1)

        num_of_views = int(ind_x.shape[0] / 1)

        ind_predict_y = ind_model.predict(ind_x)
        # 转化为行向量
        tmp_ind = ind_predict_y[:,1].reshape(-1, num_of_views)
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


ProcessCtl_dict_time = dict()
ProcessCtl_dict_time["running_label"] = False
ProcessCtl_dict_time["key"] = 0
DectectandBan_time_q_input = multiprocessing.Queue()
DectectandBan_time_q_output = multiprocessing.Queue()

# 使用训练好的model 在消息投递时候 增加对对端节点的判定
# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_Prophet_Blackhole_DectectandBan_refuseall_collusionF(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, list_selfish, new_normal_indices, new_coll_indices, coll_pairs, num_of_nodes, buffer_size, total_runningtime):
        # tf的调用次数
        self._tmpCallCnt = 0
        self.scenarioname = scenarioname
        self.list_selfish = list_selfish
        self.list_normal = new_normal_indices

        # 所有colluded节点
        self.list_coll = new_coll_indices
        # collusion对
        self.list_coll_pairs = coll_pairs
        # colluded节点对应的bk节点
        self.list_coll_corres_bk = []
        for ele in self.list_coll_pairs:
            (coll_node_id, bk_node_id) = ele
            self.list_coll_corres_bk.append(bk_node_id)

        self.num_of_nodes = num_of_nodes
        # 为各个node建立虚拟空间 <buffer+router>
        self.listNodeBuffer = []
        self.listRouter = []
        # 为了打印 获得临时的分类结果 以便观察分类结果; 从1到9(从0.1到0.9) 最后一个time block
        self.index_time_block = 1
        self.MAX_RUNNING_TIMES = total_runningtime
        # 为各个node建立检测用的 证据存储空间 BufferDetect
        self.listNodeBufferDetect = []
        for node_id in range(num_of_nodes):
            if node_id in self.list_selfish:
                tmpRouter = RoutingBlackhole(node_id, num_of_nodes)
            else:
                # 其中包含collusion节点
                tmpRouter = RoutingProphet(node_id, num_of_nodes)
            self.listRouter.append(tmpRouter)
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size)
            self.listNodeBuffer.append(tmpBuffer)
            if node_id in self.list_coll:
                coll_node_id = -1
                bk_node_id = -1
                for ele in self.list_coll_pairs:
                    (coll_node_id, bk_node_id) = ele
                    if coll_node_id == node_id:
                        break
                tmpBuffer_Detect = DTNNodeBuffer_Detect_coll(node_id, num_of_nodes, bk_node_id)
            else:
                tmpBuffer_Detect = DTNNodeBuffer_Detect(node_id, num_of_nodes)
            self.listNodeBufferDetect.append(tmpBuffer_Detect)


        # 加载训练好的模型 load the trained model (d_eve and ind_eve as input)
        dir = "..\\Main\\ML_blackhole_time"
        direct_model_file_path = os.path.join(dir, 'direct_model.h5')
        indirect_model_file_path = os.path.join(dir, 'indirect_model.h5')
        self.model_files_path = (direct_model_file_path, indirect_model_file_path)

        self.MAX_Ability = (10000, 'Max Process Ability', 'Continue')
        global ProcessCtl_dict_time
        global DectectandBan_time_q_input
        global DectectandBan_time_q_output
        if not ProcessCtl_dict_time["running_label"]:
            ProcessCtl_dict_time["running_label"] = True
            j = multiprocessing.Process(target=process_predict_blackhole_d_ind_direct_coll, args=(
                self.model_files_path, self.MAX_Ability, DectectandBan_time_q_input, DectectandBan_time_q_output))
            j.daemon = True
            j.start()
        # 保存真正使用的结果: self.DetectResult[0,1] False_Positive ; self.DetectResult[1,0] False_Negative
        self.DetectResult = np.zeros((2,2),dtype='int')
        # tmp 临时结果
        self.tmp0_DetectResult = np.zeros((2, 2), dtype='int')
        self.tmp_DetectResult = np.zeros((2, 20), dtype='int')
        # 矩阵属性可以考虑更改
        self.num_of_att = 10

        # collusion 检查的阈值; 需要经验确定
        self.collusion_alpha = 0.025
        # 记录collusion检测的评价结果 并 用list记录下来/带上时间；
        # 合作的bk
        self.coll_corr_bk_sum_evalu = 0.0
        self.coll_corr_bk_num_evalu = 0
        self.coll_corr_bk_recd_list = []
        # 没合作的bk
        self.bk_sum_evalu = 0.0
        self.bk_num_evalu = 0
        self.bk_recd_list = []
        # colluded节点
        self.coll_sum_evalu = 0.0
        self.coll_num_evalu = 0
        self.coll_recd_list = []
        # normal节点
        self.normal_sum_evalu = 0.0
        self.normal_num_evalu = 0
        self.normal_recd_list = []

        # 结果保存到文件中

        # 记录检测的精确程度
        self.coll_DetectRes = np.zeros((2,2),dtype='int')

        return

    # tmp_ 保存时间线上状态; 事态的发展会保证，self.index_time_block 必然不会大于10
    def __update_tmp_conf_matrix(self, gentime, isEndoftime):
        assert(self.index_time_block <= 10)
        if (isEndoftime == True) or (gentime >= 0.1 * self.index_time_block * self.MAX_RUNNING_TIMES):
            index = self.index_time_block - 1
            tmp_ = self.DetectResult - self.tmp0_DetectResult
            self.tmp_DetectResult[:, index * 2 : index*2+2] = tmp_
            self.tmp0_DetectResult = self.DetectResult.copy()
            self.index_time_block = self.index_time_block + 1
        return

    def __print_tmp_conf_matrix(self):
        # end of time; 最后一次刷新
        self.__update_tmp_conf_matrix(-1, True)
        output_str = '{}_tmp_state\n'.format(self.scenarioname)
        # self.DetectResult self.DetectdEve self.DetectindEve
        output_str += 'self.tmp_DetectResult:\n{}\n'.format(self.tmp_DetectResult)
        return output_str

    def __print_collusion(self):
        coll_corr_bk_eva = self.coll_corr_bk_sum_evalu / self.coll_corr_bk_num_evalu
        bk_eva = self.bk_sum_evalu / self.bk_num_evalu
        coll_eva = self.coll_sum_evalu / self.coll_num_evalu
        normal_eva = self.normal_sum_evalu / self.normal_num_evalu

        output_str = '{}_collusion_state\n'.format(self.scenarioname)
        output_str += 'coll_corr_bk_eva:{}\nbk_eva:{}\ncoll_eva:{}\nnormal_eva:{}\n'.format(coll_corr_bk_eva, bk_eva, coll_eva, normal_eva)
        output_str += 'coll_DetectRes:\n{}\n'.format(self.coll_DetectRes)

        short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = '' + short_time + '.npz'
        tmp_num_pairs = len(self.list_coll_pairs)
        tmp_ratio_bk = len(self.list_selfish) / (len(self.list_normal) + len(self.list_coll) + len(self.list_selfish))
        # # 结果保存到文件中
        self.collfilter_recd_path = "..\\collfilter_"+short_time+"_pair"+str(tmp_num_pairs)+"_ratio_0_"+str(int(10*tmp_ratio_bk))+".npz"
        np.savez(self.collfilter_recd_path, coll_corr_bk_recd = self.coll_corr_bk_recd_list,
                 bk_recd = self.bk_recd_list, coll_recd = self.coll_recd_list, normal_recd = self.normal_recd_list,
                 num_paris = tmp_num_pairs, ratio_bk = tmp_ratio_bk)
        return output_str

    def print_res(self, listgenpkt):
        output_str_whole = self.__print_res_whole(listgenpkt)
        output_str_pure, succ_ratio, avg_delay = self.__print_res_pure(listgenpkt)
        # 打印混淆矩阵
        output_str_state = self.__print_conf_matrix()
        output_str_tmp_state = self.__print_tmp_conf_matrix()
        output_str_coll = self.__print_collusion()
        print(output_str_whole + output_str_pure + output_str_state + output_str_tmp_state + output_str_coll)
        # 不必进行标签值 和 属性值 的保存
        # self.print_eve_res()
        # 使得预测进程终止
        global ProcessCtl_dict_time
        global DectectandBan_time_q_input
        if ProcessCtl_dict_time["running_label"] == True:
            DectectandBan_time_q_input.put(None)
            ProcessCtl_dict_time["running_label"] = False
        outstr = output_str_whole + output_str_pure + output_str_state + output_str_tmp_state + output_str_coll
        percent_selfish = len(self.list_selfish) / self.num_of_nodes
        res = (succ_ratio, avg_delay, self.DetectResult, self.tmp_DetectResult)
        config = (percent_selfish, 1)
        return outstr, res, config

    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        self.__update_tmp_conf_matrix(gentime, False)
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.listNodeBuffer[src_id].gennewpkt(newpkt)
        return

    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        a_send = self.listNodeBufferDetect[a_id].get_send_values()
        a_receive = self.listNodeBufferDetect[a_id].get_receive_values()
        a_send_all = self.listNodeBufferDetect[a_id].get_send_all()
        a_receive_all = self.listNodeBufferDetect[a_id].get_receive_all()
        a_receive_src = self.listNodeBufferDetect[a_id].get_receive_src_values()
        a_receive_dst = self.listNodeBufferDetect[a_id].get_receive_dst_values()
        a_send_src = self.listNodeBufferDetect[a_id].get_send_src_values()
        a_send_dst = self.listNodeBufferDetect[a_id].get_send_dst_values()
        a_receive_from_and_src = self.listNodeBufferDetect[a_id].get_receive_from_and_pktsrc()

        b_send = self.listNodeBufferDetect[b_id].get_send_values()
        b_receive = self.listNodeBufferDetect[b_id].get_receive_values()
        b_send_all = self.listNodeBufferDetect[b_id].get_send_all()
        b_receive_all = self.listNodeBufferDetect[b_id].get_receive_all()
        b_receive_src = self.listNodeBufferDetect[b_id].get_receive_src_values()
        b_receive_dst = self.listNodeBufferDetect[b_id].get_receive_dst_values()
        b_send_src = self.listNodeBufferDetect[b_id].get_send_src_values()
        b_send_dst = self.listNodeBufferDetect[b_id].get_send_dst_values()
        b_receive_from_and_src = self.listNodeBufferDetect[b_id].get_receive_from_and_pktsrc()

        self.listNodeBufferDetect[b_id].renewindeve(runningtime, a_id, a_send, a_receive, a_send_all, a_receive_all,
                                                    a_receive_src, a_receive_dst, a_send_src, a_send_dst,
                                                    a_receive_from_and_src)
        self.listNodeBufferDetect[a_id].renewindeve(runningtime, b_id, b_send, b_receive, b_send_all, b_receive_all,
                                                    b_receive_src, b_receive_dst, b_send_src, b_send_dst,
                                                    b_receive_from_and_src)

        bool_BH_a_wch_b = self.__detect_blackhole(a_id, b_id, runningtime)
        bool_BH_b_wch_a = self.__detect_blackhole(b_id, a_id, runningtime)
        # 如果有一方不同意 则停止
        if bool_BH_a_wch_b or bool_BH_a_wch_b:
            return

        # ================== 控制信息 交换==========================
        # 对称操作!!!
        # 获取 b_node Router 向各节点的值(带有老化计算)
        P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        P_a_any = self.listRouter[a_id].get_values_before_up(runningtime)
        # 根据 b_node Router 保存的值, a_node更新向各其他node传递值 (带有a-b响应本次相遇的更新)
        self.listRouter[a_id].notifylinkup(runningtime, b_id, P_b_any)
        self.listRouter[b_id].notifylinkup(runningtime, a_id, P_a_any)
        if isinstance(self.listRouter[a_id], RoutingBlackhole) and isinstance(self.listRouter[b_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是blackhole b_id是blackhole==========================
            self.__sendpkt_toblackhole(runningtime, a_id, b_id)
            self.__sendpkt_toblackhole(runningtime, b_id, a_id)
        elif isinstance(self.listRouter[a_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是blackhole b_id是正常prophet==========================
            self.__sendpkt(runningtime, a_id, b_id)
            self.__sendpkt_toblackhole(runningtime, b_id, a_id)
        elif isinstance(self.listRouter[b_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是正常prophet b_id是blackhole==========================
            self.__sendpkt_toblackhole(runningtime, a_id, b_id)
            self.__sendpkt(runningtime, b_id, a_id)
        elif (not isinstance(self.listRouter[a_id], RoutingBlackhole)) and (not isinstance(self.listRouter[b_id], RoutingBlackhole)):
            # ================== 报文 交换==========================
            self.__sendpkt(runningtime, a_id, b_id)
            self.__sendpkt(runningtime, b_id, a_id)

    # 报文发送 a_id -> b_id
    def __sendpkt_toblackhole(self, runningtime, a_id, b_id):
        P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        P_a_any = self.listRouter[a_id].get_values_before_up(runningtime)
        # 准备从a到b传输的pkt 组成的list<这里保存的是deepcopy>
        totran_pktlist = []
        # b_listpkt_hist = self.listNodeBuffer[b_id].getlistpkt_hist()
        # a_listpkt_hist = self.listNodeBuffer[a_id].getlistpkt_hist()
        b_listpkt_hist = []
        a_listpkt_hist = []
        # 1) b_id 告诉 a_id: b_id有哪些pkt
        b_listpkt = self.listNodeBuffer[b_id].getlistpkt()
        a_listpkt = self.listNodeBuffer[a_id].getlistpkt()
        # hist列表 和 当前内存里都没有 来自a的pkt   a才有必要传输
        for a_pkt in a_listpkt:
            isDuplicateExist = False
            for bpktid_hist in b_listpkt_hist:
                if a_pkt.pkt_id == bpktid_hist:
                    isDuplicateExist = True
                    break
            if not isDuplicateExist:
                for bpkt in b_listpkt:
                    if a_pkt.pkt_id == bpkt.pkt_id:
                        isDuplicateExist = True
                        break
            if not isDuplicateExist:
                cppkt = copy.deepcopy(a_pkt)
                if a_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, cppkt)
                totran_pktlist.append(cppkt)
                break
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
            elif P_a_any[tmp_pkt.dst_id] < P_b_any[tmp_pkt.dst_id]:
                # # 利用model进行判定 b_id是否是blackhole
                # bool_BH = self.__detect_blackhole(a_id, b_id, runningtime)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)

                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
                # blackhole b_id立刻发动
                self.listNodeBuffer[b_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)

    # 报文发送 a_id -> b_id
    def __sendpkt(self, runningtime, a_id, b_id):
        P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        P_a_any = self.listRouter[a_id].get_values_before_up(runningtime)
        # 准备从a到b传输的pkt 组成的list<这里保存的是deepcopy>
        totran_pktlist = []
        # b_listpkt_hist = self.listNodeBuffer[b_id].getlistpkt_hist()
        # a_listpkt_hist = self.listNodeBuffer[a_id].getlistpkt_hist()
        b_listpkt_hist = []
        a_listpkt_hist = []
        # 1) b_id 告诉 a_id: b_id有哪些pkt
        b_listpkt = self.listNodeBuffer[b_id].getlistpkt()
        a_listpkt = self.listNodeBuffer[a_id].getlistpkt()
        # hist列表 和 当前内存里都没有 来自a的pkt   a才有必要传输
        for a_pkt in a_listpkt:
            isDuplicateExist = False
            for bpktid_hist in b_listpkt_hist:
                if a_pkt.pkt_id == bpktid_hist:
                    isDuplicateExist = True
                    break
            if not isDuplicateExist:
                for bpkt in b_listpkt:
                    if a_pkt.pkt_id == bpkt.pkt_id:
                        isDuplicateExist = True
                        break
            if not isDuplicateExist:
                cppkt = copy.deepcopy(a_pkt)
                if a_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, cppkt)
                totran_pktlist.append(cppkt)
                break
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
            elif P_a_any[tmp_pkt.dst_id] < P_b_any[tmp_pkt.dst_id]:
                # # 利用model进行判定 b_id是否是blackhole
                # bool_BH = self.__detect_blackhole(a_id, b_id, runningtime)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)

                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)

    def __detect_blackhole(self, a_id, b_id, runningtime):
        theBufferDetect = self.listNodeBufferDetect[a_id]
        # a和b相遇 来自a的是直接证据
        # i提供给a一些证据 作为间接证据

        d_attrs = np.zeros((1, 1 * self.num_of_att), dtype='int')
        d_attrs[0][0] = runningtime
        d_attrs[0][1] = ((theBufferDetect.get_send_values())[b_id]).copy()
        d_attrs[0][2] = ((theBufferDetect.get_receive_values())[b_id]).copy()
        d_attrs[0][3] = ((theBufferDetect.get_receive_src_values())[b_id]).copy()
        d_attrs[0][4] = ((theBufferDetect.get_receive_dst_values())[b_id]).copy()
        d_attrs[0][5] = ((theBufferDetect.get_send_src_values())[b_id]).copy()
        d_attrs[0][6] = ((theBufferDetect.get_send_dst_values())[b_id]).copy()
        d_attrs[0][7] = theBufferDetect.get_send_all().copy()
        d_attrs[0][8] = theBufferDetect.get_receive_all().copy()
        d_attrs[0][9] = ((theBufferDetect.get_receive_from_and_pktsrc())[b_id]).copy()

        mask = [True] * (self.num_of_nodes)
        mask[a_id] = False
        mask[b_id] = False
        n = self.num_of_nodes - 2
        ind_attrs = np.zeros((1, n * self.num_of_att), dtype='int')
        # 来自各个节点评价b_id; (1)作为间接证据 除去a_id的观察 (2)去除嫌疑 去掉b_id的观察
        tmp_send = (theBufferDetect.get_ind_send_values())[:, b_id].transpose().copy()
        tmp_receive = (theBufferDetect.get_ind_receive_values())[:, b_id].transpose().copy()
        tmp_receive_src = (theBufferDetect.get_ind_receive_src_values())[:, b_id].transpose().copy()
        tmp_receive_dst = (theBufferDetect.get_ind_receive_dst_values())[:, b_id].transpose().copy()
        tmp_send_src = (theBufferDetect.get_ind_send_src_values())[:, b_id].transpose().copy()
        tmp_send_dst = (theBufferDetect.get_ind_send_dst_values())[:, b_id].transpose().copy()

        tmp_time = (theBufferDetect.get_ind_time())[b_id].transpose().copy()
        tmp_send_all = (theBufferDetect.get_ind_send_all())[b_id].transpose().copy()
        tmp_receive_all = (theBufferDetect.get_ind_receive_all())[b_id].transpose().copy()
        tmp_receive_from_and_pktsrc = (theBufferDetect.get_ind_receive_from_and_pktsrc())[:, b_id].transpose().copy()

        ind_attrs[0][0: n] = tmp_time
        ind_attrs[0][n: 2 * n] = tmp_send[mask]
        ind_attrs[0][2 * n: 3 * n] = tmp_receive[mask]
        ind_attrs[0][3 * n: 4 * n] = tmp_receive_src[mask]
        ind_attrs[0][4 * n: 5 * n] = tmp_receive_dst[mask]
        ind_attrs[0][5 * n: 6 * n] = tmp_send_src[mask]
        ind_attrs[0][6 * n: 7 * n] = tmp_send_dst[mask]

        ind_attrs[0][7 * n: 8 * n] = tmp_send_all
        ind_attrs[0][8 * n: 9 * n] = tmp_receive_all
        ind_attrs[0][9 * n: 10 * n] = tmp_receive_from_and_pktsrc[mask]

        new_x = np.hstack((d_attrs, ind_attrs))
        # tf的调用次数 加1
        self._tmpCallCnt = self._tmpCallCnt + 1

        i_isSelfish = int(b_id in self.list_selfish)

        to_collusion_index = np.arange(self.num_of_nodes)
        to_collusion_index = to_collusion_index[mask]
        to_collusion_index = to_collusion_index.reshape((-1, self.num_of_nodes-2))
        # to_collusion_index.reshape((-1, len(to_collusion_index)))

        # 加载模型；进行预测
        global ProcessCtl_dict_time
        global DectectandBan_time_q_input
        global DectectandBan_time_q_output
        request_element = (ProcessCtl_dict_time["key"], new_x.copy(), i_isSelfish)
        ProcessCtl_dict_time["key"] = ProcessCtl_dict_time["key"] + 1
        DectectandBan_time_q_input.put(request_element)

        result_element = DectectandBan_time_q_output.get(True)
        # boolBlackhole = result_element[1]
        # conf_matrix = result_element[2]
        d_predict = result_element[1]
        ind_predict = result_element[2]
        if result_element[3] == self.MAX_Ability[1]:
            j = multiprocessing.Process(target=process_predict_blackhole_d_ind_direct_coll, args=(
                self.model_files_path, self.MAX_Ability, DectectandBan_time_q_input, DectectandBan_time_q_output))
            j.daemon = True
            j.start()

        # collusion filtering; 返回 corrupted node对应的id 和 filtering后的ind_predict
        res_coll_id, res_coll_filtering = self.__detect_collusion(ind_predict, to_collusion_index)
        tmp_res = np.hstack((d_predict, res_coll_filtering))
        final_res = np.sum(tmp_res, axis=1) / tmp_res.shape[1]
        boolBlackhole = final_res > 0.5

        # 只从正常节点的角度观察
        if a_id in self.list_normal:
            if b_id in self.list_coll_corres_bk:
                # b_id是合作的bk节点 记录下评价; coll以后评价有没有提高
                self.coll_corr_bk_sum_evalu = self.coll_corr_bk_sum_evalu + final_res
                self.coll_corr_bk_num_evalu =  self.coll_corr_bk_num_evalu + 1
                self.coll_corr_bk_recd_list.append((final_res, runningtime))
            elif b_id in self.list_selfish:
                # b_id是普通的bk节点 (没有colluded节点与b_id合作)
                self.bk_sum_evalu = self.bk_sum_evalu + final_res
                self.bk_num_evalu = self.bk_num_evalu + 1
                self.bk_recd_list.append((final_res, runningtime))
            elif b_id in self.list_coll:
                self.coll_sum_evalu = self.coll_sum_evalu + final_res
                self.coll_num_evalu = self.coll_num_evalu + 1
                self.coll_recd_list.append((final_res, runningtime))
            elif b_id in self.list_normal:
                self.normal_sum_evalu = self.normal_sum_evalu + final_res
                self.normal_num_evalu = self.normal_num_evalu + 1
                self.normal_recd_list.append((final_res, runningtime))
            else:
                print('Internal Err! CollusionF calculate res!')

        #  只有a_id是正常节点 才有观察的必要
        if a_id in self.list_normal:
            # 看看检测出来的准不准
            tmp = np.zeros((2,2), dtype='int')
            if (res_coll_id, b_id) in self.list_coll_pairs:
                tmp[0][0] = 1
            elif b_id in self.list_coll_corres_bk:
                # b_id是coll_corres_bk 存在的对应的colluded节点; 发生漏检
                # assert res_coll_id !=
                tmp[0][1] = 1
            elif res_coll_id != -1:
                # b_id也不是coll_bk; 也没有正确发现; 但还是以为有coll_id
                # 误报 真实为‘1’误以为‘0’
                tmp[1][0] = 1
            else:
                tmp[1][1] = 1
            self.coll_DetectRes = self.coll_DetectRes + tmp

        conf_matrix = cal_conf_matrix(i_isSelfish, int(boolBlackhole), num_classes=2)

        self.DetectResult = self.DetectResult + conf_matrix
        return boolBlackhole

    # collusion filtering; 返回 corrupted node对应的id 和 filtering后的ind_predict
    def __detect_collusion(self, ind_predict, to_collusion_index):
        assert ind_predict.shape[1] == to_collusion_index.shape[1]
        assert ind_predict.shape[0] == to_collusion_index.shape[0]
        assert ind_predict.shape[1] == self.num_of_nodes - 2
        assert ind_predict.shape[0] == 1
        dim = ind_predict.shape[1]
        # ind_predict = np.squeeze(ind_predict, axis=0)
        # to_collusion_index = np.squeeze(to_collusion_index, axis=0)
        one_collu_list = []
        for i in range(dim):
            # 0*(num_nodes - 3) 的
            mask = [True] * (dim)
            mask[i] = False
            mask = np.array(mask).reshape(-1,dim)
            tmp_leave_one_array = ind_predict[mask]
            tmp_leave_one_std = np.std(tmp_leave_one_array)
            # 记录leave one以后的std 和 对应的index
            one_collu_list.append((tmp_leave_one_std, to_collusion_index[0][i], i))
        # sort
        one_collu_list.sort(reverse=True)
        if (one_collu_list[0][0] - one_collu_list[1][0]) > self.collusion_alpha:
            coll_node_id = one_collu_list[0][1]

            mask = [True] * (dim)
            # 出问题的位置 为false
            mask[one_collu_list[0][2]] = False
            mask = np.array(mask).reshape(-1, dim)
            good_indirect_predict_res = ind_predict[mask]

            return coll_node_id, good_indirect_predict_res
        else:
            return -1, ind_predict

    # 改变检测buffer的值
    def __updatedectbuf_sendpkt(self, a_id, b_id, pkt_src_id, pkt_dst_id):
        self.listNodeBufferDetect[a_id].send_to_b(b_id)
        self.listNodeBufferDetect[a_id].send_to_pkt_src(pkt_src_id)
        self.listNodeBufferDetect[a_id].send_to_pkt_dst(pkt_dst_id)

        self.listNodeBufferDetect[b_id].receive_from_a(a_id)
        self.listNodeBufferDetect[b_id].receive_from_pkt_src(pkt_src_id)
        self.listNodeBufferDetect[b_id].receive_from_pkt_dst(pkt_dst_id)

        if a_id == pkt_src_id:
            self.listNodeBufferDetect[b_id].receive_from_and_pktsrc(a_id, pkt_src_id)

    def __print_conf_matrix(self):
        output_str = '{}_state\n'.format(self.scenarioname)
        output_str += 'self.list_selfish:\{}\n'.format(self.list_selfish)
        output_str += 'self.DetectResult:\n{}\n'.format(self.DetectResult)
        return output_str

    def __print_res_whole(self, listgenpkt):
        num_genpkt = len(listgenpkt)
        output_str = '{}_whole\n'.format(self.scenarioname)
        total_delay = 0
        total_succnum = 0
        total_pkt_hold = 0
        for i_id in range(len(self.listNodeBuffer)):
            list_succ = self.listNodeBuffer[i_id].getlistpkt_succ()
            tmp_succnum = 0
            for i_pkt in list_succ:
                tmp_delay = i_pkt.succ_time - i_pkt.gentime
                total_delay = total_delay + tmp_delay
                tmp_succnum = tmp_succnum + 1
            assert (tmp_succnum == len(list_succ))
            total_succnum = total_succnum + tmp_succnum

            list_pkt = self.listNodeBuffer[i_id].getlistpkt()
            total_pkt_hold = total_pkt_hold + len(list_pkt)
        succ_ratio = total_succnum/num_genpkt
        if total_succnum != 0:
            avg_delay = total_delay/total_succnum
            output_str += 'succ_ratio:{} avg_delay:{}\n'.format(succ_ratio, avg_delay)
        else:
            output_str += 'succ_ratio:{} avg_delay:null\n'.format(succ_ratio)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, num_genpkt, total_succnum)
        return output_str

    def __print_res_pure(self, listgenpkt):
        num_purepkt = 0
        for tunple in listgenpkt:
            (pkt_id, src_id, dst_id) = tunple
            if (not isinstance(self.listRouter[src_id], RoutingBlackhole)) and (not isinstance(self.listRouter[dst_id], RoutingBlackhole)):
                num_purepkt = num_purepkt + 1
        output_str = '{}_pure\n'.format(self.scenarioname)
        total_delay = 0
        total_succnum = 0
        total_pkt_hold = 0
        for i_id in range(len(self.listNodeBuffer)):
            if not isinstance(self.listRouter[i_id], RoutingBlackhole):
                list_succ = self.listNodeBuffer[i_id].getlistpkt_succ()
                tmp_succnum = 0
                for i_pkt in list_succ:
                    # 这样 src_id 和 dst_id 都是 正常prophet node
                    if not isinstance(self.listRouter[i_pkt.src_id], RoutingBlackhole):
                        tmp_delay = i_pkt.succ_time - i_pkt.gentime
                        total_delay = total_delay + tmp_delay
                        tmp_succnum = tmp_succnum + 1
                total_succnum = total_succnum + tmp_succnum

                list_pkt = self.listNodeBuffer[i_id].getlistpkt()
                total_pkt_hold = total_pkt_hold + len(list_pkt)
        succ_ratio = total_succnum/num_purepkt
        if total_succnum != 0:
            avg_delay = total_delay/total_succnum
            output_str += 'succ_ratio:{} avg_delay:{}\n'.format(succ_ratio, avg_delay)
        else:
            avg_delay = ()
            output_str += 'succ_ratio:{} avg_delay:null\n'.format(succ_ratio)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, num_purepkt, total_succnum)
        return output_str, succ_ratio, avg_delay

class RoutingProphet(object):
    def __init__(self, node_id, num_of_nodes, p_init=0.75, gamma=0.98, beta=0.25):
        self.node_id = node_id
        self.P_init = p_init
        self.Gamma = gamma
        self.Beta = beta
        self.num_of_nodes = num_of_nodes
        # aging的时间, 多少秒更新一次 30s, 现在是0.1s一个间隔
        self.secondsInTimeUnit = 30 * 10
        # 记录 a_id 与其他任何节点 之间的delivery prob, P_a_any
        self.delivery_prob = np.zeros(self.num_of_nodes, dtype='double')
        # 初始化 为 P_init
        for i in range(self.num_of_nodes):
            if i != self.node_id:
                self.delivery_prob[i] = self.P_init
        # 记录 两两之间的上次相遇时刻 以便计算相遇间隔
        self.lastAgeUpdate = 0

    # ===============================================  Prophet内部逻辑  ================================
    # 每隔一段时间执行 老化效应
    def __aging(self, running_time):
        duration = running_time - self.lastAgeUpdate
        k = math.floor(duration / self.secondsInTimeUnit)
        if k == 0:
            return
        # 更新了 大家都老化一下
        self.delivery_prob = self.delivery_prob * math.pow(self.Gamma, k)
        self.lastAgeUpdate = running_time

    # a 和 b 相遇 更新prob
    def __update(self, runningtime, a_id, b_id):
        # 取值之前要更新
        P_a_b = self.__getPredFor(runningtime, a_id, b_id)
        # 发生a-b相遇 更新
        self.delivery_prob[b_id] = P_a_b + (1 - P_a_b) * self.P_init

    # 传递效应, 遇见就更新
    def __transitive(self, runningtime, a_id, b_id, P_b_any):
        # 获取的时候 会进行老化操作
        P_a_b = self.__getPredFor(runningtime, a_id, b_id)
        # 获取b_id的delivery prob矩阵 的副本
        for c_id in range(self.num_of_nodes):
            if c_id == b_id or c_id == a_id:
                continue
            self.delivery_prob[c_id] = self.delivery_prob[c_id] + (1 - self.delivery_prob[c_id]) * \
                                       self.delivery_prob[b_id] * P_b_any[c_id] * self.Beta

    def __getPredFor(self, runningtime, a_id, b_id):
        assert(a_id == self.node_id)
        self.__aging(runningtime)
        return self.delivery_prob[b_id]

    # ========================= 提供给上层的功能 ======================================
    # 更新后, 提供 本node 的 delivery prob Matrix 给对端
    def get_values_before_up(self, runningtime):
        self.__aging(runningtime)
        return self.delivery_prob

    # 当a->b 相遇(linkup时候) 更新a->b相应的值
    def notifylinkup(self, runningtime, b_id, *args):
        # b到任何节点的值
        P_b_any = args[0]
        a_id = self.node_id
        # a-b相遇 产生增益
        self.__update(runningtime, a_id, b_id)
        # 借助b进行中转
        self.__transitive(runningtime, a_id, b_id, P_b_any)


class RoutingBlackhole(RoutingProphet):
    def __init__(self, node_id, num_of_nodes):
        super(RoutingBlackhole, self).__init__(node_id, num_of_nodes)
