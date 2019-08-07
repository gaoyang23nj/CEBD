import numpy as np
import math
import copy

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


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


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
        # 为了计算传输量 需要临时保存 传输的报文数 1）保存对端id号 2）保存本次连接传输的报文
        self.tmpRSLid = []
        self.tmpRSL = []
        # 记录平均传输量;
        # self.cnt_volume[0,0]是发生次数 self.cnt_volume[0,1]是传输量累加值 self.cnt_volume[0,1]是平均值 即x
        self.cnt_volume = np.zeros((1, 3))
        # buffer阈值 p初始化为最大值; threshold (跳数的阈值) 初始化为0
        # self.p = self.theBufferNode.maxsize
        self.threshold = 0

    # 基于 更新阈值(cnt) 用于排列报文的比较器
    def __compare_msg(self, pkt1, pkt2):
        if pkt1.pkt_id == pkt2.pkt_id:
            return 0
        # 跳数少的pkt 优先级更高
        if (len(pkt1.track)-1 < self.threshold) and (len(pkt2.track)-1 >= self.threshold):
            return -1
        elif (len(pkt1.track)-1 >= self.threshold) and (len(pkt2.track)-1 < self.threshold):
            return 1
        # 都是跳数小于阈值的  hop更小的优先级更高
        if (len(pkt1.track)-1 < self.threshold) and (len(pkt2.track)-1 < self.threshold):
            return len(pkt1.track) - len(pkt2.track)
        cost1 = self.__cal_likelihood(self.theBufferNode.node_id, pkt1.pkt_id)
        cost2 = self.__cal_likelihood(self.theBufferNode.node_id, pkt2.pkt_id)
        return cost1 - cost2

    # 根据 p值 更新阈值(cnt)
    def __update_threshold(self, p):
        if len(self.theBufferNode.list_pkt) == 0:
            self.threshold = 0
            return
        # 按照跳数 对 node buffer里的报文进行排序:
        copy_list_pkthop = []
        for tmp_pkt in self.theBufferNode.list_pkt:
            tmp_hopcount = len(tmp_pkt.track) - 1
            # tmp_likelihood = self.__cal_likelihood(self.theBufferNode.node_id, tmp_pkt.dst_id)
            copy_list_pkthop.append((tmp_hopcount, copy.deepcopy(tmp_pkt)))
        copy_list_pkthop = sorted(copy_list_pkthop, key=lambda x: x[0])
        self.threshold = 0
        for (hop, tmp_pkt) in copy_list_pkthop:
            if p > 0:
                p -= tmp_pkt.pkt_size
            else:
                # 阈值 应该是 跳数+1
                self.threshold = len(tmp_pkt.track)
                break
        return

    # 通过Djk算法 求出cost 从b_id到dst_id
    def __cal_likelihood(self, b_id, dst_id):
        res_cost = float("inf")
        list_process = [(b_id, 0)]
        while len(list_process) > 0:
            if list_process[0][0] == dst_id:
                if res_cost > list_process[0][1]:
                    res_cost = list_process[0][1]
                list_process.pop(0)
                continue
            # 把邻居放进去 标记起来
            for i in range(self.numofnodes):
                tmp_src_id = list_process[0][0]
                # 指向自己的邻居指针 没意义 不要加
                if tmp_src_id == i:
                    continue
                cost_i = 1 - self.all_probs[tmp_src_id][i]
                # 准备加入新项 （i, new_cost）
                new_cost = list_process[0][1] + cost_i
                # 查找cost 是不是变得更小？ 1）如果变得更小 需要更新 2）如果不存在 需要新增
                is_update = False
                for j in range(len(list_process)):
                    (id, cost) = list_process[j]
                    if id == i:
                        if cost > new_cost:
                            list_process[j] = (i, new_cost)
                        is_update = True
                        break
                if not is_update:
                    list_process.append((i, cost_i))
            list_process.pop(0)
            # 验证是否都大于 res_cost 是则可以退出
            is_return = True
            for j in range(len(list_process)):
                if list_process[j][1] < res_cost:
                    is_return = False
                    break
            if is_return:
                return res_cost
        return res_cost

    # ===================== 主要接口
    # 新的相遇事件 加1 然后归一化
    def get_values_before_up(self, runningtime):
        return self.probs

    # 响应 linkup 事件, 调整两个node之间的delivery prob
    # args 带的是 对方的prob
    def notify_link_up(self, running_time, b_id, *args):
        tmp_prob_b = args[0]
        # 对端还没来及执行: 1)加1; 2）归一化
        tmp_prob_b[self.theBufferNode.node_id] = self.probs[self.theBufferNode.node_id] + 1
        tmp_prob_b = tmp_prob_b * (1 / np.sum(tmp_prob_b))
        # 本端 执行更新
        self.probs[b_id] = self.probs[b_id] + 1
        self.probs = self.probs * (1 / np.sum(self.probs))
        # 对端的信息放入 node的全局视图里
        self.all_probs[b_id, :] = tmp_prob_b
        # 保存 临时传输报文数目
        self.tmpRSLid.append(b_id)
        self.tmpRSL.append([])

    # error! 计算接触时间间隔 得到X 调整阈值
    def notify_link_down(self, running_time, b_id, *args):
        idx = self.tmpRSLid.index(b_id)
        tmp_size = 0
        for tmp_pkt in self.tmpRSL[idx]:
            tmp_size = tmp_size + tmp_pkt.pkt_size
        self.cnt_volume[0, 0] += 1
        self.cnt_volume[0, 1] += tmp_size
        self.cnt_volume[0, 2] = self.cnt_volume[0, 1] / self.cnt_volume[0, 0]
        if self.cnt_volume[0, 2] < self.theBufferNode.maxsize / 2:
            p = self.cnt_volume[0, 2]
        elif (self.cnt_volume[0, 2] >= self.theBufferNode.maxsize/2) and (self.cnt_volume[0, 2] < self.theBufferNode.maxsize):
            if self.cnt_volume[0, 2] < self.theBufferNode.maxsize-self.cnt_volume[0, 2]:
                p = self.cnt_volume[0, 2]
            else:
                p = self.theBufferNode.maxsize-self.cnt_volume[0, 2]
        else:
            p = 0
        # 更新 threshold
        self.__update_threshold(p)
        # 传输结束 准备结束
        self.tmpRSLid.pop(idx)
        self.tmpRSL.pop(idx)

    # 顺序地得到准备传输的list(b_id里没有的pkt), dst_id是b_id的pkt应该最先传
    def gettranpktlist(self, runningtime, b_id, listb, a_id, lista, *args):
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
                    totran_pktlist_low.append(i_pkt)
        copy_list = sorted(totran_pktlist_low, key=cmp_to_key(self.__compare_msg))
        totran_pktlist = totran_pktlist_high
        if len(totran_pktlist_low) > 0:
            # 按照优先级决定 传输顺序
            for tunple in copy_list:
                totran_pktlist.append(tunple)
        return totran_pktlist

    # 作为relay, 接收a_id发来的i_pkt吗？
    # 记得 增加跳数
    def decideAddafterRece(self, a_id, i_pkt):
        isAdd = True
        # 获取 内存状态 给出准备清除的pktlist
        if self.theBufferNode.occupied_size + i_pkt.pkt_size > self.theBufferNode.maxsize:
            copy_list = sorted(self.theBufferNode.list_pkt, key=cmp_to_key(self.__compare_msg))
            while self.theBufferNode.occupied_size + i_pkt.pkt_size > self.theBufferNode.maxsize:
                self.theBufferNode.occupied_size -= copy_list[-1].pkt_size
                self.theBufferNode.deletepktbypktid(copy_list[-1].pkt_id)
        # 记录这个pkt
        idx = self.tmpRSLid.index(a_id)
        self.tmpRSL[idx].append((i_pkt.pkt_id, i_pkt.src_id, i_pkt.dst_id, i_pkt.pkt_size))
        return isAdd

    # 发送i_pkt给b_id 以后，决定要不要 从内存中删除
    def decideDelafterSend(self, b_id, i_pkt):
        isDel = False
        # 记录这个pkt
        idx = self.tmpRSLid.index(b_id)
        self.tmpRSL[idx].append((i_pkt.pkt_id, i_pkt.src_id, i_pkt.dst_id, i_pkt.pkt_size))
        return isDel




