# 对每次需要判定时候 都记录进行记录 作为模型的训练数据
import datetime
import shutil
from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt
from Main.DTNNodeBuffer_Detect import DTNNodeBuffer_Detect

import copy
import numpy as np
import math
import os

# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_Prophet_Blackhole_toDetect_time(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, list_selfish, num_of_nodes, buffer_size, total_runningtime, isPrintDetectAtt):
        self.isPrint = isPrintDetectAtt
        self.scenarioname = scenarioname
        self.list_selfish = list_selfish
        self.num_of_nodes = num_of_nodes
        # 为了打印临时证据 以便训练更好的分类器; 从1到9(从0.1到0.9) 最后一个time block
        self.index_time_block = 1
        self.MAX_RUNNING_TIMES = total_runningtime
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
        # 实验所产生的证据 保存路径
        self.eve_dir = ".//collect_data_blackhole_time//" + self.scenarioname
        if os.path.exists(self.eve_dir):
            shutil.rmtree(self.eve_dir)
        os.makedirs(self.eve_dir)
        # 证据文件 1*(3+1)+99*(3+1); 该矩阵不断更新，直到写入成为文件为止;
        # 矩阵属性可以考虑更改
        self.matrix_x = np.zeros((0, 99*9))
        self.matrix_y = np.zeros((0, 1))
        # 目前矩阵里有效值 的 个数
        self.matrix_index = 0
        self.limit_length_matrix = 1000
        return

    # 打印结果
    def print_res(self, listgenpkt):
        # 执行流程终止
        # 记录全部归档
        self.__save_eve_res_final()

        output_str_whole = self.__print_res_whole(listgenpkt)
        output_str_pure, succ_ratio, avg_delay = self.__print_res_pure(listgenpkt)
        print(output_str_whole + output_str_pure)
        # # 进行标签值 和 属性值 的保存; 以便于offline训练model
        outstr = output_str_whole + output_str_pure
        res = (succ_ratio, avg_delay)
        config = ()
        return outstr, res, config

    # 生成新报文
    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        # tmp_ 保存时间线上状态; 事态的发展会保证，self.index_time_block 必然不会大于10
        assert (self.index_time_block <= 10)
        if gentime >= 0.1 * self.index_time_block * self.MAX_RUNNING_TIMES:
            # 过程中的结果
            add_str = '_0_' + str(self.index_time_block)
            # self.__print_eve_res(basedir= self.eve_dir + add_str, isEndoftime=False)
            self.index_time_block = self.index_time_block + 1
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.listNodeBuffer[src_id].gennewpkt(newpkt)
        return

    # 相遇事件发生 交换报文
    # routing接到指令aid和bid相遇，开始进行消息交换a_id <-> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # 交换直接评价信息，更新间接评价
        a_send = self.listNodeBufferDetect[a_id].get_send_values()
        a_receive = self.listNodeBufferDetect[a_id].get_receive_values()
        a_send_all = self.listNodeBufferDetect[a_id].get_send_all()
        a_receive_all = self.listNodeBufferDetect[a_id].get_receive_all()
        a_receive_src = self.listNodeBufferDetect[a_id].get_receive_src_values()
        a_receive_dst = self.listNodeBufferDetect[a_id].get_receive_dst_values()
        a_send_src = self.listNodeBufferDetect[a_id].get_send_src_values()
        a_send_dst = self.listNodeBufferDetect[a_id].get_send_dst_values()


        b_send = self.listNodeBufferDetect[b_id].get_send_values()
        b_receive = self.listNodeBufferDetect[b_id].get_receive_values()
        b_send_all = self.listNodeBufferDetect[b_id].get_send_all()
        b_receive_all = self.listNodeBufferDetect[b_id].get_receive_all()
        b_receive_src = self.listNodeBufferDetect[b_id].get_receive_src_values()
        b_receive_dst = self.listNodeBufferDetect[b_id].get_receive_dst_values()
        b_send_src = self.listNodeBufferDetect[b_id].get_send_src_values()
        b_send_dst = self.listNodeBufferDetect[b_id].get_send_dst_values()

        self.listNodeBufferDetect[b_id].renewindeve(runningtime, a_id, a_send, a_receive, a_send_all, a_receive_all, a_receive_src, a_receive_dst, a_send_src, a_send_dst)
        self.listNodeBufferDetect[a_id].renewindeve(runningtime, b_id, b_send, b_receive, b_send_all, b_receive_all, b_receive_src, b_receive_dst, b_send_src, b_send_dst)
        # 进行标签值 和 属性值 的保存; 以便于offline训练model
        self.__save_eve_res(self.eve_dir, a_id, b_id, runningtime)
        self.__save_eve_res(self.eve_dir, b_id, a_id, runningtime)
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
            if tmp_pkt.dst_id == b_id or P_a_any[tmp_pkt.dst_id] < P_b_any[tmp_pkt.dst_id]:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)

    # 改变检测buffer的值
    def __updatedectbuf_sendpkt(self, a_id, b_id, pkt_src_id, pkt_dst_id):
        self.listNodeBufferDetect[a_id].send_to_b(b_id)
        self.listNodeBufferDetect[a_id].send_to_pkt_src(pkt_src_id)
        self.listNodeBufferDetect[a_id].send_to_pkt_dst(pkt_dst_id)

        self.listNodeBufferDetect[b_id].receive_from_a(a_id)
        self.listNodeBufferDetect[b_id].receive_from_pkt_src(pkt_src_id)
        self.listNodeBufferDetect[b_id].receive_from_pkt_dst(pkt_dst_id)

    # 保存所有的证据 因为self.matrix_x里可能仍有剩余
    def __save_eve_res_final(self):
        assert self.matrix_index == self.matrix_x.shape[0]

        short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = '' + short_time + '.npz'
        filename = os.path.join(self.eve_dir, filename)
        np.savez(filename, y=self.matrix_y, x=self.matrix_x, length = self.matrix_index)
        self.matrix_x = np.zeros((0, 99 * 9))
        self.matrix_y = np.zeros((0, 1))
        self.matrix_index = 0

    # 从a观察b 收集证据
    def __save_eve_res(self, basedir, a_id, b_id, runningtime):
        assert self.matrix_index == self.matrix_x.shape[0]

        theBufferDetect = self.listNodeBufferDetect[a_id]
        # a和b相遇 来自a的是直接证据
        # i提供给a一些证据 作为间接证据

        d_attrs = np.zeros((1, 1*9), dtype='int')
        d_attrs[0][0] = runningtime
        d_attrs[0][1] = ((theBufferDetect.get_send_values())[b_id]).copy()
        d_attrs[0][2] = ((theBufferDetect.get_receive_values())[b_id]).copy()
        d_attrs[0][3] = ((theBufferDetect.get_receive_src_values())[b_id]).copy()
        d_attrs[0][4] = ((theBufferDetect.get_receive_dst_values())[b_id]).copy()
        d_attrs[0][5] = ((theBufferDetect.get_send_src_values())[b_id]).copy()
        d_attrs[0][6] = ((theBufferDetect.get_send_dst_values())[b_id]).copy()
        d_attrs[0][7] = theBufferDetect.get_send_all().copy()
        d_attrs[0][8] = theBufferDetect.get_receive_all().copy()

        mask = [True]* (self.num_of_nodes)
        mask[a_id] = False
        mask[b_id] = False
        n = self.num_of_nodes-2
        ind_attrs = np.zeros((1, n*9), dtype='int')
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

        ind_attrs[0][0: n] = tmp_time
        ind_attrs[0][n: 2*n] = tmp_send[mask]
        ind_attrs[0][2*n: 3*n] = tmp_receive[mask]
        ind_attrs[0][3*n: 4*n] = tmp_receive_src[mask]
        ind_attrs[0][4*n: 5*n] = tmp_receive_dst[mask]
        ind_attrs[0][5*n: 6*n] = tmp_send_src[mask]
        ind_attrs[0][6*n: 7*n] = tmp_send_dst[mask]

        ind_attrs[0][7*n: 8*n] = tmp_send_all
        ind_attrs[0][8*n: 9*n] = tmp_receive_all

        label = np.zeros((1, 1), dtype='int')
        if b_id in self.list_selfish:
            label[0][0] = 1

        # 1*9 + (98)*9
        new_x = np.hstack((d_attrs, ind_attrs))
        self.matrix_x = np.vstack((self.matrix_x, new_x))
        self.matrix_y = np.vstack((self.matrix_y, label))
        self.matrix_index = self.matrix_index + 1
        # 如果足够长 就生成文件；并重置 matrix
        if self.matrix_x.shape[0] >= self.limit_length_matrix:
            short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename = '' + short_time + '.npz'
            filename = os.path.join(self.eve_dir, filename)
            np.savez(filename, y=self.matrix_y, x=self.matrix_x, length=self.matrix_index)
            self.matrix_x = np.zeros((0, 99*9))
            self.matrix_y = np.zeros((0, 1))
            self.matrix_index = 0
        return

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
