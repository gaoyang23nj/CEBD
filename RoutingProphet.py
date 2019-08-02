import numpy as np
import math

from RoutingBase import RoutingBase
# Prophet方法 需要和ONE对照一下
# def __init__(self, theBufferNode, numofnodes, p_init=0.75, gamma=0.98, beta=0.25):
# def __init__(self, theBufferNode, numofnodes):

class RoutingProphet(RoutingBase):
    def __init__(self, theBufferNode, numofnodes, p_init=0.75, gamma=0.98, beta=0.25):
        p_init = 0.75
        gamma = 0.98
        beta = 0.25
        super(RoutingProphet, self).__init__(theBufferNode)
        self.P_init = p_init
        self.Gamma = gamma
        self.Beta = beta
        self.numofnodes = numofnodes
        # aging的时间, 多少秒更新一次
        self.secondsInTimeUnit = 30
        # 记录 a_id 与其他任何节点 之间的delivery prob, P_a_any
        self.delivery_prob = np.zeros(self.numofnodes, dtype='double')
        # 初始化 为 P_init
        for i in range(self.numofnodes):
            if i != self.theBufferNode.node_id:
                self.delivery_prob[i] = self.P_init
        # 记录 两两之间的上次相遇时刻 以便计算相遇间隔
        self.lastAgeUpdate = 0
        self.waiting_time = np.zeros((self.numofnodes, self.numofnodes), dtype='double')


    # ===============================================  Prophet内部逻辑  ================================
    # a 和 b 相遇 更新prob
    def __update(self, runningtime, a_id, b_id):
        assert(a_id == self.theBufferNode.node_id)
        # 取值之前要更新
        P_a_b = self.getPredFor(runningtime, b_id)
        self.delivery_prob[b_id] = P_a_b + (1 - P_a_b) * self.P_init


    # 每隔一段时间执行 老化效应
    def __aging(self, running_time):
        duration = running_time - self.lastAgeUpdate
        k = math.floor(duration / self.secondsInTimeUnit)
        if k == 0:
            return
        self.delivery_prob = self.delivery_prob * math.pow(self.Gamma, k)
        self.lastAgeUpdate = running_time


    # 传递效应, 遇见就更新
    def __transitive(self, runningtime, a_id, b_id):
        # 获取的时候 会进行老化操作
        P_a_b = self.getPredFor(runningtime, b_id)
        # 获取b_id的delivery prob矩阵 的副本
        P_b_any = self.getdeliverprobM(b_id)
        for c_id in range(self.numofnodes):
            if c_id == b_id or c_id == a_id:
                continue
            self.delivery_prob[c_id] = self.delivery_prob[c_id] + (1 -  self.delivery_prob[c_id])*self.delivery_prob[b_id]*P_b_any[c_id]*self.Beta


    def getPredFor(self, runningtime, a_id, b_id):
        assert(a_id == self.theBufferNode.node_id)
        self.__aging(runningtime)
        return self.delivery_prob[b_id]


    def getCntPredFor(self, runningtime, a_id, b_id):
        assert (a_id != self.theBufferNode.node_id)
        return self.theBufferNode.getCntPredFor(runningtime, a_id, b_id)

    # 获取b_id的 delivery prob 向上取 和 向下执行
    def getdeliverprobM(self, b_id):
        if b_id == self.theBufferNode.node_id:
            return self.delivery_prob
        else:
            return self.theBufferNode.getdeliverprobM(b_id)


    # 当a->b 相遇(linkup时候) 更新a->b相应的值
    def linkup(self, runningtime, a_id, b_id):
        self.__update(a_id, b_id)
        self.__transitive()

   # 顺序地得到准备传输的list(b_id里没有的pkt), dst_id是b_id的pkt应该最先传
    def gettranpktlist(self, runningtime, b_id, listb, a_id, lista):
        assert(a_id == self.theBufferNode.node_id)
        totran_pktlist_high = []
        totran_pktlist_low = []
        for i_pkt in lista:
            isiexist = False
            # 如果pkt_id相等, 是同一个pkt 不必再传输
            for j_pkt in listb:
                if i_pkt.pkt_id == j_pkt.pkt_id:
                    isiexist = True
                    break
            if isiexist == False:
                # 如果pkt的dst_id就是b, 找到目的 应该优先传输
                if i_pkt.dst_id == b_id:
                    totran_pktlist_high.append(i_pkt)
                # 作为relay 只有devliery prob更大 的时候 才转发
                else:
                    tmp = self.getCntPredFor(runningtime, b_id, i_pkt.dst_id)
                    if tmp > self.getPredFor(runningtime, a_id, i_pkt.dst_id):
                        totran_pktlist_low.append((tmp, i_pkt))
        totran_pktlist = totran_pktlist_high
        if len(totran_pktlist_low) > 0:
            # 按照优先级决定 传输顺序
            totran_pktlist_low = sorted(totran_pktlist_low, key = lambda x: x[0])
            for tunple in totran_pktlist_low:
                totran_pktlist.append(tunple[1])
        return totran_pktlist


    # 增加 不需要检查
    def decideAddafterRece(self, a_id, target_pkt):
        isAdd = True
        return isAdd


    # Prophet 单副本转发
    def decideDelafterSend(self, b_id, i_pkt):
        isDel = True
        return isDel



