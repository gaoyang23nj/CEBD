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

NUM_of_DIMENSIONS = 10
NUM_of_DIRECT_INPUTS = 7
NUM_of_INDIRECT_INPUTS = 8
MAX_RUNNING_TIMES = 864000

# *** 未完成
# 应该加入 黑名单LBL 等到时候超时才放出
# 0值ER 可能会使TR虚高
class DTNNodeBuffer_Detect_MDS(object):
    def __init__(self, node_id, num_of_nodes):
        self.node_id = node_id
        self.w_max = 100
        self.tr_init = 0.5
        self.tr_reduced_init = 0.4
        self.TR_list = np.ones(num_of_nodes, dtype='float')*self.tr_init
        # 对于Prophet来说
        # TR更新用的变化值
        self.tr_gamma = 0.04
        self.tr_rho = 0.09
        self.tr_lambda = 0.06
        # failstep 判断阈值
        self.Nth_b = 0.8
        self.Nth_a = 4
        self.NRth = 0.7
        # TR阈值
        self.TRth_evil = 0.3
        self.TRth_friend = 0.8
        # 保存的信息, ER等
        self.ER_list = []
        # 临时ER 在传输报文过程前启用, 传输结束后加入到ER_list,并重置
        self.tmp_ER = {"partner_id":-1, "send_to_partner":0, "recv_from_partner":0, "gensend_to_partner":0, "running":0}
        # 超时时间1个小时
        self.expriation = 3600*10
        self.LBL_list = []
        self.LBL_resume_time_list = []

    def get_ER_list(self):
        return self.ER_list.copy()

    def detect_node_j(self, j_id, j_ER_list, runningtime):
        j_is_blackhole = False
        # 检查LBL
        if j_id in self.LBL_list:
            j_is_blackhole = True
            # 如果j在黑名单LBL上 则查看j的恢复时刻
            index_j_id = self.LBL_list.index(j_id)
            # 规定的恢复时刻已经到了
            if self.LBL_resume_time_list[index_j_id] <= runningtime:
                j_is_blackhole = False
                self.LBL_list.pop(index_j_id)
                self.LBL_resume_time_list.pop(index_j_id)
            return j_is_blackhole
        # 开始检测
        fail_step = 0
        N_send = 0
        N_recv = 0
        N_jsend = 0
        for one_ER in j_ER_list:
            # N_send = N_send + one_ER["send_to_partner"]
            # N_recv = N_recv + one_ER["recv_from_partner"]
            # N_jsend = N_jsend + one_ER["gensend_to_partner"]
            N_send = N_send + one_ER[1]
            N_recv = N_recv + one_ER[2]
            N_jsend = N_jsend + one_ER[3]
        # 第5步
        # 计算 theta
        if N_recv == 0:
            N_recv = 1
        theta = (N_send + 0.0) / N_recv
        Nth = self.__get_Nth(len(j_ER_list))
        if theta < Nth:
            fail_step = fail_step + 1
        # 第6步
        if N_send == 0:
            N_send = 1
        psi = (N_jsend + 0.0) / N_send
        if psi >= self.NRth:
            fail_step = fail_step + 1
        # 更新 TR值
        if fail_step == 0:
            self.TR_list[j_id] = self.TR_list[j_id] + self.tr_lambda
        elif fail_step == 1:
            self.TR_list[j_id] = self.TR_list[j_id] - self.tr_gamma
        elif fail_step == 2:
            self.TR_list[j_id] = self.TR_list[j_id] - self.tr_rho
        else:
            assert 0 == 0
        # 返回判断结果
        if self.TR_list[j_id] < self.TRth_evil:
            # 应该加入 黑名单LBL 等到时候超时才放出
            j_is_blackhole = True
            # 不可有重复
            assert not j_id in self.LBL_list
            self.LBL_list.append(j_id)
            self.LBL_resume_time_list.append(runningtime + self.expriation)
            self.TR_list[j_id] = self.tr_reduced_init
        elif self.TR_list[j_id] < self.TRth_evil:
            j_is_blackhole = False

        return j_is_blackhole

    def __get_Nth(self, len_ER_list):
        w = len_ER_list
        if w == 0:
            w = 1
        Nth = self.Nth_b - (self.Nth_a / w)
        return Nth

    # begin a new encounter with a node; input the encountered node's id
    def begin_new_encounter(self, partner_id):
        self.tmp_ER["partner_id"] = partner_id
        self.tmp_ER["running"] = 1

    # 结束目前的encounter
    def end_new_encounter(self, partner_id):
        assert self.tmp_ER["partner_id"] == partner_id
        new_ER = (self.tmp_ER["partner_id"], self.tmp_ER["send_to_partner"], self.tmp_ER["recv_from_partner"], self.tmp_ER["gensend_to_partner"])
        # 窗口大小限制
        if len(self.ER_list) <= self.w_max:
            self.ER_list.append(new_ER)
        else:
            self.ER_list.pop(0)
            self.ER_list.append(new_ER)
        self.tmp_ER["partner_id"] = -1
        self.tmp_ER["send_to_partner"] = 0
        self.tmp_ER["recv_from_partner"] = 0
        self.tmp_ER["gensend_to_partner"] = 0
        self.tmp_ER["running"] = 0

    def send_one_pkt_to_partner(self, partner_id, pkt_src):
        assert self.tmp_ER["partner_id"] == partner_id
        self.tmp_ER["send_to_partner"] = self.tmp_ER["send_to_partner"] + 1
        if pkt_src == self.node_id:
            self.tmp_ER["gensend_to_partner"] = self.tmp_ER["gensend_to_partner"] + 1

    def receive_one_pkt_from_partner(self, partner_id):
        assert self.tmp_ER["partner_id"] == partner_id
        self.tmp_ER["recv_from_partner"] = self.tmp_ER["recv_from_partner"] + 1


# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_Prophet_Blackhole_MDS(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, list_selfish, num_of_nodes, buffer_size, total_runningtime):
        # tf的调用次数
        self._tmpCallCnt = 0
        self.scenarioname = scenarioname
        self.list_selfish = list_selfish
        self.num_of_nodes = num_of_nodes
        # 为各个node建立虚拟空间 <buffer+router>
        self.listNodeBuffer = []
        self.listRouter = []
        # 为了打印 获得临时的分类结果 以便观察分类结果; 从1到9(从0.1到0.9) 最后一个time block
        self.index_time_block = 1
        self.MAX_RUNNING_TIMES = total_runningtime
        # 为各个node建立检测用的 证据存储空间 BufferDetect
        self.listNodeBufferDetect_MDS = []
        for node_id in range(num_of_nodes):
            if node_id in list_selfish:
                tmpRouter = RoutingBlackhole(node_id, num_of_nodes)
            else:
                tmpRouter = RoutingProphet(node_id, num_of_nodes)
            self.listRouter.append(tmpRouter)
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size)
            self.listNodeBuffer.append(tmpBuffer)
            # 基于ER的检测系统
            tmpBuffer_Detect  = DTNNodeBuffer_Detect_MDS(node_id, num_of_nodes)
            self.listNodeBufferDetect_MDS.append(tmpBuffer_Detect)

        # 保存真正使用的结果: self.DetectResult[0,1] False_Positive ; self.DetectResult[1,0] False_Negative
        self.DetectResult = np.zeros((2,2),dtype='int')
        # tmp 临时结果
        self.tmp0_DetectResult = np.zeros((2, 2), dtype='int')
        self.tmp_DetectResult = np.zeros((2, 20), dtype='int')
        # 矩阵属性可以考虑更改
        self.num_of_att = 10
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

    def print_res(self, listgenpkt):
        output_str_whole = self.__print_res_whole(listgenpkt)
        output_str_pure, succ_ratio, avg_delay = self.__print_res_pure(listgenpkt)
        # 打印混淆矩阵
        output_str_state = self.__print_conf_matrix()
        output_str_tmp_state = self.__print_tmp_conf_matrix()
        print(output_str_whole + output_str_pure + output_str_state + output_str_tmp_state)
        # 不必进行标签值 和 属性值 的保存
        # self.print_eve_res()
        outstr = output_str_whole + output_str_pure + output_str_state + output_str_tmp_state
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

    # routing接到指令aid和bid相遇，开始进行消息交换 a_id <-> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # 交换直接评价信息，更新间接评价
        bool_BH_a_to_b = self.__detect_blackhole(a_id, b_id, runningtime)
        bool_BH_b_to_a = self.__detect_blackhole(b_id, a_id, runningtime)


        # 任何一方认为对方是blackhole, 就会拒绝报文交换; 如果 ER 0值交换 会产生提高TR
        if bool_BH_a_to_b or bool_BH_b_to_a:
            # theBufferDetect_a = self.listNodeBufferDetect_MDS[a_id]
            # theBufferDetect_a.begin_new_encounter(b_id)
            # theBufferDetect_b = self.listNodeBufferDetect_MDS[b_id]
            # theBufferDetect_b.begin_new_encounter(a_id)
            # theBufferDetect_a.end_new_encounter(b_id)
            # theBufferDetect_b.end_new_encounter(a_id)
            return

        # 通知 DTNNodeBuffer_Detect_MDS, 建立新的ER; a和b各自单独执行
        theBufferDetect_a = self.listNodeBufferDetect_MDS[a_id]
        theBufferDetect_a.begin_new_encounter(b_id)
        theBufferDetect_b = self.listNodeBufferDetect_MDS[b_id]
        theBufferDetect_b.begin_new_encounter(a_id)

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

        # 通知 DTNNodeBuffer_Detect_MDS, 把这个新的ER归档
        theBufferDetect_a.end_new_encounter(b_id)
        theBufferDetect_b.end_new_encounter(a_id)

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
                # 利用model进行判定 b_id是否是blackhole
                # bool_BH = self.__detect_blackhole(a_id, b_id, runningtime)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                # if not bool_BH:
                #     self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
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
                # 利用model进行判定 b_id是否是blackhole
                # bool_BH = self.__detect_blackhole(a_id, b_id, runningtime)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                # if not bool_BH:
                #     self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)

    # a节点检测b节点
    def __detect_blackhole(self, a_id, b_id, runningtime):
        theBufferDetect_a = self.listNodeBufferDetect_MDS[a_id]

        theBufferDetect_b = self.listNodeBufferDetect_MDS[b_id]
        ER_list = theBufferDetect_b.get_ER_list()

        bool_BH = theBufferDetect_a.detect_node_j(b_id, ER_list, runningtime)

        y_predict = np.zeros((1), dtype='int')
        y_predict[0] = int(bool_BH)
        i_isSelfish = int(b_id in self.list_selfish)
        y_final = np.zeros((1), dtype='int')
        y_final[0] = i_isSelfish
        conf_matrix = np.array(tf.math.confusion_matrix(y_final, y_predict, num_classes=2))

        self.DetectResult = self.DetectResult + conf_matrix
        return bool_BH

    # 改变检测buffer的值
    def __updatedectbuf_sendpkt(self, a_id, b_id, pkt_src_id, pkt_dst_id):
        # 有一个pkt从a_id 发送给 b_id; a 和 b 的MDS中 临时ER要响应的增加
        theBufferDetect_a = self.listNodeBufferDetect_MDS[a_id]
        theBufferDetect_a.send_one_pkt_to_partner(b_id, pkt_src_id)
        theBufferDetect_b = self.listNodeBufferDetect_MDS[b_id]
        theBufferDetect_b.receive_one_pkt_from_partner(a_id)

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
