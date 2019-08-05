import numpy as np
import math

from DTNPkt import DTNPkt
from RoutingBase import RoutingBase
# Maxprop方法 需要和ONE对照一下
# MaxProp: Routing for Vehicle-Based Disruption-Tolerant Networks
# 2006 infocom
# 为难！！ 需要两个list 阈值还要重新调整


class DTNTrackPkt(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        super(DTNTrackPkt, self).__init__(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.track = [src_id]



class RoutingMaxProp(RoutingBase):
    def __init__(self, theBufferNode, numofnodes):
        super(RoutingMaxProp, self).__init__(theBufferNode)
        self.numofnodes = numofnodes
        # 保存直连概率 初始化为 1/(|s|-1)
        self.probs = np.zeros(self.numofnodes)
        for i in range(self.numofnodes):
            if i != self.theBufferNode.node_id:
                self.probs[i] = 1/(self.numofnodes - 1)
        # 对全局的认识, 依赖于其他node copy
        self.all_probs = np.zeros((self.numofnodes, self.numofnodes), dtype='double')
        for i in range(self.numofnodes):
            if i == self.theBufferNode.node_id:
                self.all_probs[i, :] = self.probs[i]
                continue
            for j in range(self.numofnodes):
                self.all_probs[i, j] = 0
        # 记录传输量
        self.connet_record.append()

    def get_values_before_up(self):
        self.probs[b_id] = self.probs[b_id] + 1
        self.probs = self.probs * (1 / np.sum(self.probs))
        return self.probs

    def get_values_before_down(self):
        pass

    # 响应 linkup 事件, 调整两个node之间的delivery prob
    # args 带的是 对方的prob
    def notify_link_up(self, running_time, b_id, *args):
        tmp_prob_b = args[0]
        self.all_probs[b_id, :] = tmp_prob_b

    # error! 计算接触时间间隔 得到X 调整阈值
    def notify_link_down(self, running_time, b_id, *args):
        pass

    # 通过Djk算法 求出cost 从b_id到dst_id
    def __cal_likelihood(self, b_id, dst_id):
        res_cost = float("inf")
        list_process = [(b_id, 0)]
        while len(list_process) > 0:
            if list_process[0][0] == dst_id:
                if res_cost > list_process[0][1]:
                    res_cost = list_process[0][1]
            # 把邻居放进去 标记起来
            for i in range(self.numofnodes):
                cost_i = 1 - self.all_probs[list_process[0][0]][i]
                new_cost = list_process[0][1] + cost_i
                # 查找cost 准备更新
                for j in range(len(list_process)):
                    (id, cost) = list_process[j]
                    if id == i:
                        if cost > new_cost:
                            list_process[j] = (id, new_cost)
                        continue
            list_process.pop(0)
            # 验证是否都大于 res_cost 是则可以退出
            is_return = True
            for j in range(len(list_process)):
                if list_process[j][1] < res_cost:
                    is_return = False
            if is_return:
                return res_cost
        return res_cost

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
                else:
                    cost = self.__cal_likelihood(b_id, i_pkt.dst_id)
                    totran_pktlist_low.append((cost, len(i_pkt.track), i_pkt))
        totran_pktlist = totran_pktlist_high
        if len(totran_pktlist_low) > 0:
            # 按照优先级决定 传输顺序
            totran_pktlist_low = sorted(totran_pktlist_low, key=lambda x: x[0])
            for tunple in totran_pktlist_low:
                totran_pktlist.append(tunple[2])
        return totran_pktlist

    # 作为relay, 接收a_id发来的i_pkt吗？
    def decideAddafterRece(self, a_id, i_pkt):
        isAdd = True
        return isAdd

    # 发送i_pkt给b_id 以后，决定要不要 从内存中删除
    def decideDelafterSend(self, b_id, i_pkt):
        isDel = False
        return isDel




