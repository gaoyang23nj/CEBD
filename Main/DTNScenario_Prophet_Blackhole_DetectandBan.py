from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt
from Main.DTNNodeBuffer_Detect import DTNNodeBuffer_Detect

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


def tryonce_process_cpu(save_d_model_file_path, save_ind_model_file_path, d_attrs, ind_attrs, return_dict):
    with tf.device('CPU:0'):
        x = tf.random.uniform([1, 1])
        tmp = x.device.endswith('CPU:0')
        print('On CPU:{}'.format(tmp))
        assert x.device.endswith('CPU:0')
        d_model = tf.keras.models.load_model(save_d_model_file_path)
        ind_model = tf.keras.models.load_model(save_ind_model_file_path)
        # d_attrs = np.zeros((1, 3), dtype='int')
        # ind_attrs = np.zeros((1, 300), dtype='int')
        d_predict = d_model.predict(d_attrs)
        ind_predict = ind_model.predict(ind_attrs)
        print(d_predict)
        print(ind_predict)
        return_dict["res_d"] = d_predict[0][1]
        return_dict["res_ind"] = ind_predict[0][1]

def tryonce_process_gpu(save_d_model_file_path, save_ind_model_file_path, d_attrs, ind_attrs, return_dict):
    x = tf.random.uniform([1, 1])
    tmp = x.device.endswith('GPU:0')
    print('On GPU:{}'.format(tmp))
    assert x.device.endswith('GPU:0')
    d_model = tf.keras.models.load_model(save_d_model_file_path)
    ind_model = tf.keras.models.load_model(save_ind_model_file_path)
    # d_attrs = np.zeros((1, 3), dtype='int')
    # ind_attrs = np.zeros((1, 300), dtype='int')
    d_predict = d_model.predict(d_attrs)
    ind_predict = ind_model.predict(ind_attrs)
    print(d_predict)
    print(ind_predict)
    return_dict["res_d"] = d_predict[0][1]
    return_dict["res_ind"] = ind_predict[0][1]

# 使用训练好的model 在消息投递时候 增加对对端节点的判定
# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_Prophet_Blackhole_DectectandBan(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, list_selfish, num_of_nodes, buffer_size):
        # tf的调用次数
        self._tmpCallCnt = 0
        self.scenarioname = scenarioname
        self.list_selfish = list_selfish
        self.num_of_nodes = num_of_nodes
        # 为各个node建立虚拟空间 <buffer+router>
        self.listNodeBuffer = []
        self.listRouter = []
        # 为各个node建立检测用的 证据存储空间 BufferDetect
        self.listNodeBufferDetect = []
        for node_id in range(num_of_nodes):
            if node_id in list_selfish:
                tmpRouter = RoutingBlackhole(node_id, num_of_nodes)
            else:
                tmpRouter = RoutingProphet(node_id, num_of_nodes)
            self.listRouter.append(tmpRouter)
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size)
            self.listNodeBuffer.append(tmpBuffer)
            tmpBuffer_Detect = DTNNodeBuffer_Detect(node_id, num_of_nodes)
            self.listNodeBufferDetect.append(tmpBuffer_Detect)
        # 加载训练好的模型 load the trained model (d_eve and ind_eve as input)
        dir = "../Main/collect_data/scenario9/"
        self.save_d_model_file_path = os.path.join(dir, 'ML/deve_model.h5')
        self.save_ind_model_file_path = os.path.join(dir, 'ML/indeve_model.h5')
        return

    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.listNodeBuffer[src_id].gennewpkt(newpkt)
        return

    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # 交换直接评价信息，更新间接评价
        a_sendvalues = self.listNodeBufferDetect[a_id].get_send_values()
        a_receivevalues = self.listNodeBufferDetect[a_id].get_receive_values()
        a_receivesrcvalues = self.listNodeBufferDetect[a_id].get_receive_src_values()
        b_sendvalues = self.listNodeBufferDetect[b_id].get_send_values()
        b_receivevalues = self.listNodeBufferDetect[b_id].get_receive_values()
        b_receivesrcvalues = self.listNodeBufferDetect[a_id].get_receive_src_values()
        self.listNodeBufferDetect[b_id].renewindeve(runningtime, a_id, a_sendvalues, a_receivevalues, a_receivesrcvalues)
        self.listNodeBufferDetect[a_id].renewindeve(runningtime, b_id, b_sendvalues, b_receivevalues, b_receivesrcvalues)
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
            self.sendpkt_toblackhole(runningtime, a_id, b_id)
            self.sendpkt_toblackhole(runningtime, b_id, a_id)
        elif isinstance(self.listRouter[a_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是blackhole b_id是正常prophet==========================
            self.sendpkt(runningtime, a_id, b_id)
            self.sendpkt_toblackhole(runningtime, b_id, a_id)
        elif isinstance(self.listRouter[b_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是正常prophet b_id是blackhole==========================
            self.sendpkt_toblackhole(runningtime, a_id, b_id)
            self.sendpkt(runningtime, b_id, a_id)
        elif (not isinstance(self.listRouter[a_id], RoutingBlackhole)) and (not isinstance(self.listRouter[b_id], RoutingBlackhole)):
            # ================== 报文 交换==========================
            self.sendpkt(runningtime, a_id, b_id)
            self.sendpkt(runningtime, b_id, a_id)

    # 报文发送 a_id -> b_id
    def sendpkt_toblackhole(self, runningtime, a_id, b_id):
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
                self.updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
            elif P_a_any[tmp_pkt.dst_id] < P_b_any[tmp_pkt.dst_id]:
                # 利用model进行判定 b_id是否是blackhole
                bool_BH = self.detect_blackhole(a_id, b_id)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                if not bool_BH:
                    self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
                # blackhole b_id立刻发动
                self.listNodeBuffer[b_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)

    # 报文发送 a_id -> b_id
    def sendpkt(self, runningtime, a_id, b_id):
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
                self.updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
            elif P_a_any[tmp_pkt.dst_id] < P_b_any[tmp_pkt.dst_id]:
                # 利用model进行判定 b_id是否是blackhole
                bool_BH = self.detect_blackhole(a_id, b_id)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                if not bool_BH:
                    self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)

    def detect_blackhole(self, a_id, b_id):
        theBufferDetect = self.listNodeBufferDetect[a_id]
        viewofb = theBufferDetect.getviewofb(b_id)
        # 如果给出了 比较确定的结果
        if (viewofb == 1) or (viewofb == -1) or (viewofb == -2):
            # 如果确定是正常节点(-1)返回False 或者 异常节点(1)返回True； 则直接返回结果（是否为异常？）以便于进行路由
            return viewofb == 1
        # 否则 viewofb == 0
        assert viewofb == 0
        d_attrs = np.zeros((1,3), dtype='int')
        d_attrs[0][0] = ((theBufferDetect.get_send_values())[b_id]).copy()
        d_attrs[0][1] = ((theBufferDetect.get_receive_values())[b_id]).copy()
        d_attrs[0][2] = ((theBufferDetect.get_receive_src_values())[b_id]).copy()
        # 还需归一化转化
        ind_attrs = np.zeros((1,300), dtype='int')
        ind_attrs[0][0: self.num_of_nodes] = ((theBufferDetect.get_ind_send_values().transpose())[b_id,:]).copy()
        ind_attrs[0][self.num_of_nodes: 2 * self.num_of_nodes] = \
            ((theBufferDetect.get_ind_receive_values().transpose())[b_id,:]).copy()
        ind_attrs[0][2 * self.num_of_nodes: 3 * self.num_of_nodes] = \
            ((theBufferDetect.get_ind_receive_src_values().transpose())[b_id,:]).copy()
        # tf的调用次数 加1
        self._tmpCallCnt = self._tmpCallCnt + 1

        # 加载模型；进行预测
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        j = multiprocessing.Process(target=tryonce_process_cpu, args=(self.save_d_model_file_path, self.save_ind_model_file_path, d_attrs, ind_attrs, return_dict))
        j.start()
        j.join()
        res_d = return_dict["res_d"]
        res_ind = return_dict["res_ind"]
        print('return_dict: d:{} ind:{}'.format(res_d, res_ind))

        # 执行判定流程 & 结果更新过程
        # 如果 判定结果不具有足够的倾向性 (只要任何一个判定不具有倾向性 即可做出判断)
        abs_att = 0.15
        if ((math.fabs(res_d - 0.5) < abs_att) or (math.fabs(res_d - 0.5) < abs_att)):
            # pass
            boolBlackhole = False
        # 如果 判定结果具有足够的倾向性 （两个判定都足够倾向 d_eve ind_eve）
        else:
            # 如果 直接证据 和 间接证据 相悖, 本次路由放过去, 需要后续研究；
            # 可能存在虚假证据正在干扰；或者直接证据不足
            if (res_d-0.5)*(res_ind-0.5) < 0:
                boolBlackhole = False
                # 进行虚假证据鉴定
            else:
                if res_ind > 0.5 + abs_att:
                    assert res_d > 0.5 + abs_att
                    boolBlackhole = True
                    # 1表示恶意节点
                    theBufferDetect.setviewofb(b_id, 1)
                else:
                    assert res_ind < 0.5 - abs_att
                    assert res_d < 0.5 - abs_att
                    boolBlackhole = False
                    # -1表示正常节点
                    theBufferDetect.setviewofb(b_id, -1)
        # # 判定
        # boolBlackhole = ((res_d > 0.65) or (res_ind > 0.65))
        # 如果确认了结果；否则。。。
        isSelfish = (b_id in self.list_selfish)
        print('[{}] a({}) predict b({}): d_eve_predict({}), ind_eve_predict({}), {}, Truth:{}'.
              format(self._tmpCallCnt, a_id, b_id, res_d, res_ind, boolBlackhole, isSelfish))
        return boolBlackhole

    # 改变检测buffer的值
    def updatedectbuf_sendpkt(self, a_id, b_id, src_id, dst_id):
        self.listNodeBufferDetect[a_id].sendtoj(b_id)
        self.listNodeBufferDetect[b_id].receivefromi(a_id)
        if src_id == a_id:
            self.listNodeBufferDetect[b_id].receivefromsrc(a_id)

    def print_res(self, listgenpkt):
        output_str_whole = self.print_res_whole(listgenpkt)
        output_str_pure = self.print_res_pure(listgenpkt)
        # 不必进行标签值 和 属性值 的保存
        # self.print_eve_res()
        return output_str_whole + output_str_pure

    def print_res_whole(self, listgenpkt):
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
        print(output_str)
        return output_str

    def print_res_pure(self, listgenpkt):
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
            output_str += 'succ_ratio:{} avg_delay:null\n'.format(succ_ratio)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, num_purepkt, total_succnum)
        print(output_str)
        return output_str

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
