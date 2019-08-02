# -*- coding: UTF-8 -*-
# 2016TMC
# Detecting Colluding Blackhole and Greyhole Attacks in Delay Tolerant Networks
# SDBG
import numpy as np
from RoutingBase import RoutingBase
import copy

class RoutingSDBG(RoutingBase):
    def __init__(self, theBufferNode, numofnodes):
        super(RoutingSDBG, self).__init__(theBufferNode)
        self.node_id = self.theBufferNode.node_id
        self.numofnodes = numofnodes
        self.ER_sn = 1
        self.sig = 'sig'+self.node_id
        # 保存ERW
        self.ER_List = []
        self.w = 100
        # MS: 100 * [j_node_id, latest_sn_j, latest_time_j, self.TR_init, 0(未设置)/1(设置)]
        arr1 = np.arange(self.numofnodes).reshape(self.numofnodes, 1)
        arr2 = np.zeros((self.numofnodes, 4), dtype='double')
        self.MS = np.hstack([arr1, arr2])
        # 建立临时的SL 和 RL 集合
        self.tmpRSLs_js = []
        self.tmpSLs = []
        self.tmpRLs = []
        # SDBG 的 阈值参数; 需要经验 选择
        self.Th_RR = 4
        self.Th_SFR = 0.08
        self.TR_init = 0.5
        self.TR_init = 0.5
        self.Dec_gamma = 0.2
        self.Dec_rho = 0.2
        self.Inc_lambda = 0.5
        self.Th_good = 2
        self.Th_mal = -2
        self.Th_FXS = 2400


    # =============================================SDBG独有的功能======================================================
    # 当与 node j 相遇完成, 即将离开的时候
    # node_j 的 node_id, 本次ER的序列号(j给出的), 时间, i发给j的报文, i从j收到的报文
    def __genEncounter(self, j_node_id, j_ER_sn, timestamp, SL_i, RL_i, j_sig):
        # 从SL RL构建
        newER = (self.node_id, j_node_id, self.ER_sn, j_ER_sn, timestamp, SL_i, RL_i, self.sig, j_sig)
        # ER List最长为w
        if len(self.ER_List) >= self.w:
            self.ER_List.pop(0)
        assert(len(self.ER_List) < self.w)
        self.ER_List.append(newER)
        self.ER_sn = self.ER_sn + 1


    # 更新对于node_j的观察
    def __updateMS(self, j_node_id, latest_sn_j, latest_time_j):
        assert((j_node_id >= 0) and (j_node_id < self.numofnodes) and (j_node_id != self.node_id))
        # 如果是第一次出现 TR
        if self.MS[j_node_id, 4] == 0:
            self.MS[j_node_id, 1] = latest_sn_j
            self.MS[j_node_id, 2] = latest_time_j
            self.MS[j_node_id, 3] = self.TR_init
            self.MS[j_node_id, 4] = 1
        else:
            if self.MS[j_node_id, 1] < latest_sn_j:
                self.MS[j_node_id, 1] = latest_sn_j
            if self.MS[j_node_id, 2] < latest_time_j:
                self.MS[j_node_id, 2] = latest_time_j


    # 处理对方的ER序列, 得到 RR SFR
    def getRRSFR_ERofJ(self, ERW):
        N_RS = 0
        N_RNS = 0
        N_selfsend = 0
        N_send = 0
        # 记录有多少pkt在ERW里面没有被 传送出去
        tmpPktidList = []
        tmpPktList = []
        for tmpER in ERW:
            (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j, j_sig, k_sig) = tmpER
            # 接收到的
            for tmpRece in RL_j:
                (pkt_id, src_id, dst_id) = tmpRece
                # 记录报文接收到了
                tmpPktList.append((pkt_id, src_id, dst_id, timestamp, k_node_id))
                tmpPktidList.append(pkt_id)
            # 发送出去的
            for tmpSend in SL_j:
                (pkt_id, src_id, dst_id) = tmpSend
                if src_id == j_node_id:
                    N_selfsend = N_selfsend + 1
                if pkt_id in tmpPktidList:
                    idx = tmpPktidList.index(pkt_id)
                    oldtimestamp = tmpPktList[idx][3]
                    assert(oldtimestamp < timestamp)
                    N_RS = N_RS + 1
                    tmpPktidList.pop(idx)
                    tmpPktList.pop(idx)
            N_send = N_send + len(SL_j)
        assert(len(tmpPktList) == len(tmpPktidList))
        N_RNS = len(tmpPktList)
        RR = N_RS / N_RNS
        SFR = N_selfsend / N_send
        # return N_RS, N_RNS, N_selfsend, N_send, RR, SFR
        return RR, SFR


    # 解决collude 同谋攻击问题, 处理ER序列, 计算FXS的结果
    def getFXS_ERofJ(self, ERW):
        # self.numofnodes * 3 矩阵 记录; id, freq, send, FXS
        arr1 = np.arange(self.numofnodes).reshape(self.numofnodes,1)
        arr2 = np.zeros((self.numofnodes, 3), dtype='int')
        tmp_kcnt_mat = np.hstack([arr1, arr2])
        for tmpER in ERW:
            (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j, j_sig, k_sig) = tmpER
            tmp_kcnt_mat[k_node_id, 1] = tmp_kcnt_mat[k_node_id, 1] + 1
            tmp_kcnt_mat[k_node_id, 2] = tmp_kcnt_mat[k_node_id, 2] + len(SL_j)
        # 对应元素想乘 np.multiply()或 *
        tmp_kcnt_mat[:, 3] = tmp_kcnt_mat[:, 1] * tmp_kcnt_mat[:, 2]
        return tmp_kcnt_mat[:,3]

    # SDBG路由算法的主程序
    def main_process(self, j_node_id, ERW):
        dropping = False
        collusion = False
        RR, SFR = self.getRRSFR_ERofJ(ERW)
        if RR < self.Th_RR:
            # 对应的TR值修改
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_gamma
            dropping = True
        if SFR > self.Th_SFR:
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_rho
            dropping = True
        # 得到一维矩阵 各个node的FXS值
        FXSs = self.getFXS_ERofJ(ERW)
        for k_id in range(np.size(FXSs)):
            if FXSs[k_id] > self.Th_FXS:
                # 排除 建立 ERW_
                ERW_ = []
                for tmpER in ERW:
                    (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j, j_sig, k_sig) = tmpER
                    if k_node_id != k_id:
                        ERW_.insert(tmpER)
                RR_, SFR_ = self.getRRSFR_ERofJ(ERW_)
                if RR_ < self.Th_RR:
                    # 对应的TR值修改
                    self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_gamma
                    self.MS[k_id, 3] = self.MS[k_id, 3] - self.Dec_gamma
                    collusion = True
                if SFR_ > self.Th_SFR:
                    self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_rho
                    self.MS[k_id, 3] = self.MS[k_id, 3] - self.Dec_rho
                    collusion = True
        if dropping == False and collusion == False:
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] + self.Inc_lambda
        return



    # =============================================router 收到 DTNNodeBuffer的通知======================================================
    # 返回 ER_sn 和 签名; 方便对面node_router 写入新的ER
    def getSnSigRouter(self):
        return self.ER_sn, self.sig


    # 返回ERW, 方便对面node_router 根据ERW进行TR评价
    def getERWforlinkdown(self):
        return copy.deepcopy(self.ER_List)


    def notifylinkup(self, b_id, runningtime, *args):
        # 准备新ER的增加
        self.tmpRSLs_js.append(b_id)
        self.tmpSLs.append([])
        self.tmpRLs.append([])
        # 取出ERW
        ERW = args[0]
        self.main_process(b_id, ERW)


    # linkdown事件 传输结束 生成ER
    def notifylinkdown(self, b_id, runningtime, *args):
        j_sn, j_sig = args[0:]
        idx = self.tmpRSLs_js.index(b_id)
        j_SL = copy.deepcopy(self.tmpSLs[idx])
        j_RL = copy.deepcopy(self.tmpRLs[idx])
        self.__genEncounter(b_id, j_sn, runningtime, j_SL, j_RL, j_sig)
        self.__updateMS(b_id, j_sn, runningtime)
        # 传输结束 准备结束
        self.tmpRSLs_js.pop(idx)
        self.tmpSLs.pop(idx)
        self.tmpRLs.pop(idx)


    # 收到报文 更新RL
    def decideAddafterRece(self, a_id, i_pkt):
        # 如果对方节点 reputation 小于 mal阈值; 不与它交互, 不接收它的报文
        if self.MS[a_id, 3] < self.Th_mal:
            return False
        idx = self.tmpRSLs_js.index(a_id)
        self.tmpRLs[idx].append((i_pkt.pkt_id, i_pkt.src_id, i_pkt.dst_id))
        return True


    # 发送报文 更新SL
    def decideDelafterSend(self, b_id, i_pkt):
        idx = self.tmpRSLs_js.index(b_id)
        self.tmpSLs[idx].append((i_pkt.pkt_id, i_pkt.src_id, i_pkt.dst_id))
        return False


    # =============================Router的核心函数 1)转发哪些pkt; 2)转发优先级调整=============================
    # 如果是恶意节点 不与它交互 不发不收
    def gettranpktlist(self, runningtime, b_id, listb, a_id, lista):
        if self.MS[b_id, 3] < self.Th_mal:
            return []
        totran_pktlist = super().gettranpktlist()
        return totran_pktlist