import copy
import math
import numpy as np

from Routing.RoutingBase import RoutingBase

# 2018TONSM
# A Dynamic Trust Framework for Opportunistic Mobile Social Networks
# 2019-07-27
# 采用 DTNTrackPkt  # 3个维度 connectivity, fitness, satisfaction # two-hop ack 和 Final ack


class RoutingEric(RoutingBase):

    def __init__(self, theBufferNode, numofnodes):
        super(RoutingEric, self).__init__(theBufferNode)
        self.numofnodes = numofnodes
        # 在i看来, j作为中继的次数
        self.n_fwd = np.zeros(self.numofnodes, dtype=int)
        # i和j 相遇次数
        self.n_meet = np.zeros(self.numofnodes, dtype=int)
        # j接收i的报文个数
        self.n_rec = np.zeros(self.numofnodes, dtype=int)
        # 在i看来 j作为src发送报文的个数
        self.n_src = np.zeros(self.numofnodes, dtype=int)
        # 在i看来 j拒绝报文的个数
        self.n_deny = np.zeros(self.numofnodes, dtype=int)
        # j准备发送的消息 <2-hop ACK 在linkup时候触发>; 设置较短的声明周期
        self.ack_info = []
        # 在i看来 i收到有关j的ack报文的个数
        self.n_ack = np.zeros(self.numofnodes, dtype=int)
        # 2-hop ack的老化时间 上限
        self.time_diff = 3600
        # final ack
        self.final_ackinfo = []
        # 2-hop ack的老化时间 上限
        self.final_time_diff = 36000
        # final ack中出现的次数
        self.n_final_ack = np.zeros(self.numofnodes, dtype=int)
        # 被 本端放入balcklist的 node id 组成list
        self.blacklist = []

        # 直接 trust 值 [connectivity, fitness, satisfaction]
        self.T_ij_d = np.zeros((self.numofnodes, 3), dtype=float)
        # 间接 trust 值 [connectivity, fitness, satisfaction]
        self.T_ij_ind = np.zeros((self.numofnodes, 3), dtype=float)
        # aggregate trust 值 [connectivity, fitness, satisfaction]
        self.T_ij_agg = np.zeros((self.numofnodes, 3), dtype=float)
        # 记录上次 i和j 直接交互时间; 以利于完成 公式(6)
        self.time_lastupdate_d = np.zeros(self.numofnodes, dtype=int)
        # 记录上次 i和j 上次j被推荐的时间; 以利于完成 公式(7)
        self.time_lastupdate_ind = np.zeros(self.numofnodes, dtype=int)
        self.tao = 0.5
        self.gamma = 0.7
        self.delta_time = 18000
        self.dec_lambda = 0.999
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.5
        self.weight = np.array((0.1, 0.45, 0.45))
        # 发送阈值(fitness值) 接收阈值(经过wight之后的 T_agg值)
        self.receiving_threshold = 0.5
        self.sending_threshold = 0.5
        # blacklist的阈值
        self.blacklist_threshold = 0.2
        # 标志位, 本次connection j是否 出现 在path在里面
        self.tmp_Conn = []
        self.tmp_Conn_Record = []

    def __cal_sim(self, i_blacklist, j_blacklist):
        # 交集 并集
        intersection_list = []
        union_list = []
        for tmp in i_blacklist:
            if tmp in j_blacklist:
                intersection_list.append(tmp)
        union_list.extend(i_blacklist)
        for tmp in j_blacklist:
            if tmp not in union_list:
                union_list.append(tmp)
        return len(intersection_list) / len(union_list)

    def __update_blacklist(self):
        self.blacklist.clear()
        for j_id in range(self.numofnodes):
            if (self.T_ij_agg[j_id, 1] < self.blacklist_threshold) or \
                    (self.T_ij_agg[j_id, 1] < self.blacklist_threshold):
                self.blacklist.append(j_id)

    def __update_trust_d(self, running_time, j_id, T_ij_d_new):
        for x in range(3):
            # direct trust update
            tmp_factor = math.exp(-1 * self.dec_lambda * (running_time - self.time_lastupdate_d[j_id]))
            if self.delta_time < running_time - self.time_lastupdate_d[j_id]:
                self.T_ij_d[j_id][x] = tmp_factor * self.T_ij_d[j_id][x]
            else:
                self.T_ij_d[j_id][x] = self.alpha * tmp_factor * self.T_ij_d[j_id][x] + (1 - self.alpha) * T_ij_d_new

    def __update_trust_ind(self, running_time, j_id, T_ij_ind_new):
        for x in range(3):
            # indirect trust update
            tmp_factor_ind = math.exp(-1 * self.dec_lambda * (running_time - self.time_lastupdate_ind[j_id]))
            if self.delta_time < running_time - self.time_lastupdate_ind[j_id]:
                self.T_ij_ind[j_id][x] = tmp_factor_ind * self.T_ij_ind[j_id][x]
            else:
                self.T_ij_ind[j_id][x] = self.beta * tmp_factor_ind * self.T_ij_ind[j_id][x] + \
                                         (1 - self.beta) * T_ij_ind_new

    def __update_trust_agg(self, j_id):
        for x in range(3):
            # aggregation trust
            self.T_ij_agg[j_id][x] = self.gamma * self.T_ij_d[j_id][x] + (1 - self.gamma) * T_ij_ind[j_id][x]

    # !!!!!!!!! 这个函数 何时 触发？
    # 间接属性计算
    def __cal_ind_x(self, m_id, T_jm_d, j_blacklist):
        T_im_ind = np.zeros(3, dtype=float)
        for x in range(3):
            tmp_list = []
            # 遍历 合适的 node_j
            for j_id in range(self.numofnodes):
                if j_id in self.blacklist:
                    continue
                elif self.T_ij_d[j_id][x] > self.tao:
                    # 公式(5)
                    param = (0, j_id, self.T_ij_d[j_id][x], T_jm_d[m_id][x], self.T_ij_d[j_id][x] * T_jm_d[m_id][x])
                    tmp_list.append(param)
                else:
                    # 公式(5)
                    sim = self.__cal_sim(j_blacklist)
                    if sim > self.gamma:
                        param = (1, j_id, sim, T_jm_d[m_id][x], sim * T_jm_d[m_id][x])
                        tmp_list.append(param)
                # 公式(5) 在这个地方说得有点很奇怪 求和公式无法解释
                # !!!!!!!!!!!!! 按照我的理解 这两种方式过滤得到的 推荐, 应该一视同仁
            # 计算
            v1 = 0.0
            v2 = 0.0
            for tmp_tuple in tmp_list:
                v1 = v1 + tmp_tuple[2]
                v2 = v2 + tmp_tuple[4]
            T_im_ind[x] = v2 / v1
        return T_im_ind

    # 直接属性计算
    def __cal_connectivity(self, b_id):
        tmp_value = 2 * self.n_fwd[b_id] + self.n_meet[b_id]
        T_c = tmp_value / (tmp_value + self.numofnodes)
        return T_c

    def __cal_fitness(self, b_id):
        v1 = self.n_fwd[b_id] + self.n_rec[b_id]
        v2 = self.n_src[b_id] + self.n_deny[b_id]
        T_h = (v1 + 1) / (v1 + v2 + 2)
        return T_h

    def __cal_satisfaction(self, b_id, is_in_path):
        if is_in_path:
            T_s = (self.final_ackinfo[b_id] + self.ack_info[b_id]) / (self.final_ackinfo[b_id] + self.n_rec[b_id] + 1)
        else:
            T_s = (self.ack_info[b_id]) / (self.n_rec[b_id] + 1)
        return T_s

    # 提取出来的函数
    def __extract_final_ack_info(self, b_id, to_final_ackinfo):
        for tmp_final_info in to_final_ackinfo:
            if tmp_final_info in self.final_ackinfo:
                continue
            (to_id, info, pkt_id, runningtime) = tmp_final_info
            idx = self.tmp_Conn.index(b_id)
            if to_id == self.theBufferNode.node_id:
                list_id = info.split('->')
                for tmp_id in list_id:
                    self.n_final_ack[tmp_id] += 1
                    if tmp_id in self.tmp_Conn_Record[idx]:
                        self.tmp_Conn_Record[idx].append(tmp_id)
            else:
                self.final_ackinfo.append(copy.deepcopy(tmp_final_info))

    def __extract_ack_info(self, b_id, to_ackinfo):
        for tmp_info in to_ackinfo:
            # 如果本端 已有 这个info, 无需处理
            if tmp_info in self.ack_info:
                continue
            (to_id, info, pkt_id, runningtime) = tmp_info
            # 如果本端 没有这个info
            if to_id == self.theBufferNode.node_id:
                node_j = int(info.split('->')[0])
                # 目的是自己; node_j的ack 加1
                self.n_ack[node_j] += 1
            else:
                # 目的不是自己; 更新自己的to_sendinginfo; 加入记录 准备传给他人
                self.ack_info.append(copy.deepcopy(tmp_info))

    # ====================== 主要接口 =======================
    # 返回给对端 我所保存的ack_info final_ackinfo;
    def get_values_before_up(self, runningtime):
        # 注意, 太老的ACK 应该先进行老化删除
        # 按照触发时间(ack生成时间) 排列 ack_info
        self.ack_info = sorted(self.ack_info, key = lambda x: x[3])
        # 删掉太老的 2-hop ack info; 防止序号混乱 采用copy方案
        cp_ackinfo = copy.deepcopy(self.ack_info)
        for idx in range(cp_ackinfo):
            if cp_ackinfo[idx][3] + self.time_diff > runningtime:
                self.ack_info.pop(idx)
        self.final_ackinfo = sorted(self.final_ackinfo, key=lambda x: x[3])
        # 删掉太老的 final ack info; 防止序号混乱 采用copy方案
        cp_finalackinfo = copy.deepcopy(self.final_ackinfo)
        for idx in range(cp_finalackinfo):
            if cp_finalackinfo[idx][3] + self.final_time_diff > runningtime:
                self.final_ackinfo.pop(idx)
        return self.ack_info, self.final_ackinfo

    def notify_link_up(self, running_time, b_id, *args):
        to_ackinfo = args[0]
        to_final_ackinfo = args[1]
        # 处理收到的 2-hop ack
        self.__extract_ack_info(b_id, to_ackinfo)
        # 处理收到的 final ack
        self.__extract_final_ack_info(b_id, to_final_ackinfo)
        # 累计
        self.n_meet[b_id] += 1
        # 开始新的link 用于计算公式(3) path(j)
        self.tmp_Conn.append(b_id)
        self.tmp_Conn_Record.append([])

    def decideAddafterRece(self, runningtime, a_id, i_pkt):
        # 1.如果j作为中继node 记录j的贡献 生成2-hop ack
        if len(i_pkt.track) >= 2:
            for idx in range(1, len(i_pkt.track)):
                # 计算n_fwd j作为中继 次数
                tmp_id = i_pkt.track[idx]
                self.n_fwd[tmp_id] += 1
            # 启动 2-hop ACK
            to_id = i_pkt.track[-2]
            info = '{}->{}'.format(i_pkt.track[-1], self.theBufferNode.node_id)
            tmp_info = (to_id, info, i_pkt.pkt_id, runningtime)
            # 加入新的 2-hop ack
            self.ack_info.append(tmp_info)
        # 计数 j作为src的次数
        src_id = i_pkt.track[0]
        self.n_src[src_id] += 1
        # 若 fitness 小于 阈值, 则 refuse 转发 pkt
        if self.T_ij_agg[1] < self.receiving_threshold:
            return False, RoutingBase.Rece_Code_DenyPkt
        else:
            # pkt 的 track 增加
            i_pkt.track.append(self.theBufferNode.node_id)
            return True, RoutingBase.Rece_Code_AcceptPkt

    def notify_receive_succ(self, runningtime, a_id, i_pkt):
        # final ack; 成功投递制造final ack
        to_id = i_pkt.src_id
        # info 形如 0->1->2->3->4, 0是src 4是dst
        info = ''
        for tmp_id in i_pkt.track:
            info = info + '{}->'.format(tmp_id)
            # 计算n_fwd j作为中继 次数
            if tmp_id != i_pkt.src_id:
                self.n_fwd[tmp_id] += 1
        info = info + str(self.theBufferNode.node_id)
        tmp_info = (to_id, info, i_pkt.pkt_id, runningtime)
        # 加入新的 final ack
        self.final_ackinfo.append(tmp_info)
        # 计数 j作为src的次数
        src_id = i_pkt.track[0]
        self.n_src[src_id] += 1

    def decideDelafterSend(self, b_id, i_pkt):
        # j接收i的报文 的个数
        self.n_rec[b_id] += 1
        return False

    def notify_deny(self, b_id, i_pkt):
        self.n_deny[b_id] += 1

    def gettranpktlist(self, runningtime, b_id, listb, a_id, lista, *args):
        # 公式(9)
        T_ij = self.weight * self.T_ij_agg[b_id, :]
        if T_ij > self.sending_threshold:
            to_tran_list = super().gettranpktlist(runningtime, b_id, listb, a_id, lista, args)
            return to_tran_list
        else:
            return []

    def get_values_before_down(self, runningtime):
        return self.blacklist, self.T_ij_d

    def notify_link_down(self, running_time, b_id, *args):
        j_blacklist = args[0]
        T_jm_d = args[1]
        # 1.直接评价值更新
        T_d_c = self.__cal_connectivity(b_id)
        T_d_h = self.__cal_fitness(b_id)
        idx = self.tmp_Conn_Record.index(b_id)
        if b_id in self.tmp_Conn_Record[idx]:
            T_d_s = self.__cal_satisfaction(b_id, True)
        else:
            T_d_s = self.__cal_satisfaction(b_id, False)
        T_ij_d_new = np.matrix([T_d_c, T_d_h, T_d_s])
        self.__update_trust_d(running_time, b_id, T_ij_d_new)
        self.time_lastupdate_d[b_id] = running_time
        # 2.间接评价值更新 (根据 j所推荐的)
        for m_id in range(self.numofnodes):
            T_ij_ind_new = self.__cal_ind_x(m_id, T_jm_d, j_blacklist)
            self.__update_trust_ind(running_time, m_id, T_ij_ind_new)
        # 3.合成值 更新;  权重下 直接值+间接值
        self.__update_trust_agg(b_id)
        # 4.刷新blacklist
        self.__update_blacklist()

# 1.get_values_before_up 获取本端 2-hop ack 和 final ack
# 2.notify_link_up 处理对端 ack info;=> 改变n_ack 和 n_final_ack; => 改变 T_d_s 直接 statisfication; 记录 n_meet
# 3.gettranpktlist 使用 T_ij 过滤不符合条件的 对端节点
# 4.notify_succ 生成 Final ack; 记录n_src; 记录n_fwd
# 5.decideAdd 生成 2-hop ack; 记录n_src; 记录n_fwd
# 6.decideDel 记录n_rec
# 7.notify_link_down

