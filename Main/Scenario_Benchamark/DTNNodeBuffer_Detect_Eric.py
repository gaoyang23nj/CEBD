import numpy as np
import math
import copy

# *** 未完成
# 应该加入 黑名单LBL 等到时候超时才放出
# 0值ER 可能会使TR虚高
class DTNNodeBuffer_Detect_Eric(object):
    def __init__(self, node_id, num_of_nodes):
        self.node_id = node_id
        self.num_of_nodes = num_of_nodes

        self.two_hop_ack_list = []
        self.final_ack_list = []

        # direct
        # 和各个节点(e.g. 节点j) meet的次数, n_meet
        self.n_meet = np.zeros(self.num_of_nodes)
        # 从(e.g. 节点j) 转发给 ‘我’的次数, n_fwd 且j不是源节点
        self.n_fwd = np.zeros(self.num_of_nodes)

        self.n_rec = np.zeros(self.num_of_nodes)
        self.n_src = np.zeros(self.num_of_nodes)
        self.n_deny = np.zeros(self.num_of_nodes)

        self.n_ack = np.zeros(self.num_of_nodes)
        self.n_Fack = np.zeros(self.num_of_nodes)

        # ackid生成器 保证不会重复计算ack个数//每个节点都有唯一的ackid
        self.ack_id = 0
        self.record_two_hop_ack_list = []
        self.record_final_ack_list = []

        self.init = 0.5
        # Connectivity
        self.dir_T_d_c = np.ones(self.num_of_nodes)*self.init
        # Fitness
        self.dir_T_d_h = np.ones(self.num_of_nodes)*self.init
        # Satisfaction
        self.dir_T_d_s = np.ones(self.num_of_nodes)*self.init
        # 更新时间
        self.update_time = np.zeros(self.num_of_nodes)
        self.last_d_time = np.zeros(self.num_of_nodes)
        # 直接证据老化更新参数
        self.decay_alpha = 0.3
        self.decay_lambda = 0.0000001

        # database 关于其他节点的直接评价
        self.database_c = np.zeros((self.num_of_nodes, self.num_of_nodes))
        self.database_h = np.zeros((self.num_of_nodes, self.num_of_nodes))
        self.database_s = np.zeros((self.num_of_nodes, self.num_of_nodes))
        self.last_ind_update_time_c = np.zeros(self.num_of_nodes)
        self.last_ind_update_time_h = np.zeros(self.num_of_nodes)
        self.last_ind_update_time_s = np.zeros(self.num_of_nodes)
        # self.last_recommend_time = np.zeros(self.num_of_nodes)
        self.any_recommend = 0
        # Connectivity
        self.indir_T_d_c = np.ones(self.num_of_nodes)*self.init
        # Fitness
        self.indir_T_d_h = np.ones(self.num_of_nodes)*self.init
        # Satisfaction
        self.indir_T_d_s = np.ones(self.num_of_nodes)*self.init
        self.indir_tao = 0.3
        self.decay_beta = 0.3
        self.decay_delta = 0.0000001

        # 综合直接和间接评价
        self.agg_T_c = np.zeros(self.num_of_nodes)
        self.agg_T_h = np.zeros(self.num_of_nodes)
        self.agg_T_s = np.zeros(self.num_of_nodes)
        # 直接证据 和 间接证据 的 平衡
        self.agg_gamma = 0.3

        # 最终评价
        self.w1 = 0.3
        self.w2 = 0.3
        self.w3 = 0.4
        self.T_value = np.zeros(self.num_of_nodes)
        self.final_threshhold = 0

        self.rec_ack_id_list = []

    # begin a new encounter with a node; input the encountered node's id
    def begin_new_encounter(self, partner_id, runningtime):
        self.n_meet[partner_id] = self.n_meet[partner_id] + 1
        self.last_d_time[partner_id] = runningtime
        two_hop_ack = copy.deepcopy(self.two_hop_ack_list)
        final_ack = copy.deepcopy(self.final_ack_list)
        return two_hop_ack, final_ack

    # 结束目前的encounter
    def end_new_encounter(self, partner_id):
        pass

    def send_one_pkt_to_partner(self, partner_id, pkt):
        # 记录对端接收的报文数
        self.n_rec[partner_id] = self.n_rec[partner_id] + 1

    def record_ack(self, tmp_two_hop_ack, tmp_final_ack):
        for ele in tmp_two_hop_ack:
            # 过滤 是否 是发送给我的
            if ele[0] != self.node_id:
                continue
                # 通过EP方式把ack传递出去
                # self.two_hop_ack_list.append(copy.deepcopy(ele))
            # else:
            # 过滤 是否 已经记录过id
            if ele[2] in self.rec_ack_id_list:
                continue
            self.rec_ack_id_list.append(ele[2])
            tmp_partner_id = ele[1]
            self.n_ack[tmp_partner_id] = self.n_ack[tmp_partner_id] + 1

        for ele in tmp_final_ack:
            # 过滤 是否 是发送给我的
            if ele[0] != self.node_id:
                continue
                # 通过EP方式把ack传递出去
                # self.final_ack_list.append(copy.deepcopy(ele))
            # else:
            # 过滤 是否 已经记录过 id
            if ele[2] in self.rec_ack_id_list:
                continue
            self.rec_ack_id_list.append(ele[2])
            for tmp_partner_id in ele[1]:
                self.n_Fack[tmp_partner_id] = self.n_Fack[tmp_partner_id] + 1



    def receive_one_pkt_from_partner(self, partner_id, pkt, runningtime):
        # ack的格式 发送给谁\关于谁\ack的唯一标识id
        # 记录对端发送的报文数
        if pkt.src_id != partner_id:
            self.n_fwd[partner_id] = self.n_fwd[partner_id] + 1
            # 准备ack
            if pkt.dst_id == self.node_id:
                assert pkt.src_id == pkt.track[0]
                # final ack
                assert pkt.track[-1] == partner_id
                # 唯一标识
                final_ack_id = 'final_ackid_{}_nodeid_{}'.format(self.ack_id, self.node_id)
                self.ack_id = self.ack_id + 1
                self.final_ack_list.append((pkt.src_id, tuple(pkt.track), final_ack_id, pkt.pkt_id, self.node_id))
                print('New Ack {}'.format(final_ack_id))
            else:
                # two-hop ack
                assert pkt.track[-1] == partner_id
                # ack谁/ 是哪个报文/ 我是谁
                two_hop_ack_id = 'twohop_ackid_{}_nodeid_{}'.format(self.ack_id, self.node_id)
                self.ack_id = self.ack_id + 1
                self.two_hop_ack_list.append((pkt.track[-2], partner_id, two_hop_ack_id, pkt.pkt_id, self.node_id))
                print('New Ack {}'.format(two_hop_ack_id))
        else:
            self.n_src[partner_id] = self.n_src[partner_id] + 1

    def __cal_local_T_value(self, partner_id, runningtime):
        tmp_fwd = self.n_fwd[partner_id]
        tmp_meet = self.n_meet[partner_id]
        tmp_rec = self.n_rec[partner_id]
        tmp_src = self.n_src[partner_id]
        # Connectivity
        T_d_c = (2* tmp_fwd + tmp_meet)/(2*tmp_fwd + tmp_meet + self.num_of_nodes)
        # Fitness
        T_d_h = (tmp_fwd + tmp_rec + 1)/(tmp_src + tmp_fwd + tmp_rec + 0 + 2)

        # Satisfaction
        tmp_Fack = self.n_Fack[partner_id]
        tmp_ack = self.n_ack[partner_id]
        if tmp_Fack == 0:
            T_d_s = tmp_ack / (tmp_rec + 1)
        else:
            T_d_s = (tmp_Fack + tmp_ack) / (tmp_Fack + tmp_rec + 1)

        oldtime = self.update_time[partner_id]
        newtime = runningtime
        delta_t = runningtime - self.update_time[partner_id]
        assert delta_t >= 0
        # 时间老化公式
        # 在这之间交互过  (2)0 更新 交互 // (1)0 交互 更新
        if self.last_d_time[partner_id] > self.update_time[partner_id]:
            self.dir_T_d_c[partner_id] = self.decay_alpha * math.exp(-1 * delta_t * self.decay_lambda)*\
                                         self.dir_T_d_c[partner_id] \
                                         + (1 - self.decay_alpha)*T_d_c
            self.dir_T_d_h[partner_id] = self.decay_alpha * math.exp(-1 * delta_t * self.decay_lambda)*\
                                         self.dir_T_d_h[partner_id] \
                                         + (1 - self.decay_alpha)*T_d_h
            self.dir_T_d_s[partner_id] = self.decay_alpha * math.exp(-1 * delta_t * self.decay_lambda)*\
                                         self.dir_T_d_s[partner_id] \
                                         + (1 - self.decay_alpha)*T_d_s
        else:
            self.dir_T_d_c[partner_id] = math.exp(-1 * delta_t * self.decay_lambda) * self.dir_T_d_c[partner_id]
            self.dir_T_d_h[partner_id] = math.exp(-1 * delta_t * self.decay_lambda) * self.dir_T_d_h[partner_id]
            self.dir_T_d_s[partner_id] = math.exp(-1 * delta_t * self.decay_lambda) * self.dir_T_d_s[partner_id]
        self.update_time[partner_id] = runningtime
        return T_d_c, T_d_h, T_d_s

    def __cal_ind_T_value(self, partner_id, Tc, Th, Ts, agg_T, runningtime):
        self.database_c[partner_id,:] = Tc
        self.database_h[partner_id, :] = Th
        self.database_s[partner_id, :] = Ts
        # 时间老化公式
        # 在这之间交互过  (1)0 更新对j 没有对j推荐(有j以外的推荐) 现在更新对j // (2)0 更新对j 没有任何推荐 现在更新对j
        # (1) any推荐 > 对j的推荐> j的更新 (2) j的更新>any推荐
        # (如果自从上次更新以来) 1)没有接收到对j的推荐； => 上次更新时间 >上次推荐时间
        # 2)没有接收到任何推荐
        # 我认为文中所说的推荐 是 统合后的推荐
        for m in range(self.num_of_nodes):
            if m == self.num_of_nodes or m == partner_id:
                continue
            self.any_recommend = runningtime
            # if agg_T[m] > self.indir_tao:
            #     # 对m的推荐
            #     # self.last_recommend_time[m] = runningtime
            #     # 任何推荐
            #     self.any_recommend = runningtime

        if self.agg_T_c[partner_id] > self.indir_tao:
            # 准备更新这些地方的间接
            for m in range(self.num_of_nodes):
                if m == self.node_id or m == partner_id:
                    continue
                # 开始进行间接更新
                # 求出Ri集合
                fenmu = 0.
                fenzi = 0.
                tmp_recom = 0
                # 中间节点的集合 Ri
                for j in range(self.num_of_nodes):
                    if j == self.node_id or j == m:
                        continue
                    # 谁更新了？
                    if self.agg_T_c[j] > self.indir_tao:
                        fenmu = fenmu + self.agg_T_c[j]
                        fenzi = fenzi + self.dir_T_d_c[j] * self.database_c[j, m]
                        tmp_recom = tmp_recom + 1
                if tmp_recom >= 1:
                    # 对m的更新
                    value_curr = fenzi/fenmu
                    situa_int = int(self.last_ind_update_time_c[m] > self.any_recommend)
                    delta_time = runningtime - self.last_ind_update_time_c[m]
                    self.indir_T_d_c[m] = self.__decay_function(situa_int, self.indir_T_d_c[m], value_curr,
                                                                delta_time, self.decay_delta, self.decay_beta)
                    self.last_ind_update_time_c[m] = runningtime

        if self.agg_T_h[partner_id] > self.indir_tao:
            # 准备更新这些地方的间接
            for m in range(self.num_of_nodes):
                if m == self.node_id or m == partner_id:
                    continue
                # 开始进行间接更新
                # 求出Ri集合
                fenmu = 0.
                fenzi = 0.
                tmp_recom = 0
                # 中间节点的集合 Ri
                for j in range(self.num_of_nodes):
                    if j == self.node_id or j == m:
                        continue
                    # 谁更新了？
                    if self.agg_T_h[j] > self.indir_tao:
                        fenmu = fenmu + self.agg_T_h[j]
                        fenzi = fenzi + self.dir_T_d_h[j] * self.database_h[j, m]
                        tmp_recom = tmp_recom + 1
                if tmp_recom >= 1:
                    # 对m的更新
                    value_curr = fenzi/fenmu
                    situa_int = int(self.last_ind_update_time_h[m] > self.any_recommend)
                    delta_time = runningtime - self.last_ind_update_time_h[m]
                    self.indir_T_d_h[m] = self.__decay_function(situa_int, self.indir_T_d_h[m], value_curr,
                                                                delta_time, self.decay_delta, self.decay_beta)
                    self.last_ind_update_time_h[m] = runningtime

        if self.agg_T_s[partner_id] > self.indir_tao:
            # 准备更新这些地方的间接
            for m in range(self.num_of_nodes):
                if m == self.node_id or m == partner_id:
                    continue
                # 开始进行间接更新
                # 求出Ri集合
                fenmu = 0.
                fenzi = 0.
                tmp_recom = 0
                # 中间节点的集合 Ri
                for j in range(self.num_of_nodes):
                    if j == self.node_id or j == m:
                        continue
                    # 谁更新了？
                    if self.agg_T_s[j] > self.indir_tao:
                        fenmu = fenmu + self.agg_T_s[j]
                        fenzi = fenzi + self.dir_T_d_s[j] * self.database_s[j, m]
                        tmp_recom = tmp_recom + 1
                if tmp_recom > 1:
                    # 对m的更新
                    value_curr = fenzi/fenmu
                    situa_int = int(self.last_ind_update_time_s[m] > self.any_recommend)
                    delta_time = runningtime - self.last_ind_update_time_s[m]
                    self.indir_T_d_s[m] = self.__decay_function(situa_int, self.indir_T_d_s[m], value_curr,
                                                                delta_time, self.decay_delta, self.decay_beta)
                    self.last_ind_update_time_s[m] = runningtime
        return

    def __cal_agg_T(self, partner_id):
        self.agg_T_c[partner_id] = self.agg_gamma * self.dir_T_d_c[partner_id] + \
                                   (1 - self.agg_gamma) * self.indir_T_d_c[partner_id]
        self.agg_T_h[partner_id] = self.agg_gamma * self.dir_T_d_h[partner_id] + \
                                   (1 - self.agg_gamma) * self.indir_T_d_h[partner_id]
        self.agg_T_s[partner_id] = self.agg_gamma * self.dir_T_d_s[partner_id] + \
                                   (1 - self.agg_gamma) * self.indir_T_d_s[partner_id]

        self.T_value[partner_id] = self.w1*self.agg_T_c[partner_id] + self.w2 * self.agg_T_h[partner_id] \
                                   + self.w3 * self.agg_T_s[partner_id]
        return self.T_value[partner_id]

    def __decay_function(self, situa_int, value_pre, value_curr, delta_time, para_decay, para_weight):
        res = 0.
        if situa_int == 0:
            res = math.exp(-1 * delta_time * para_decay) * value_pre
        elif situa_int == 1:
            res = para_weight * math.exp(-1 * delta_time * para_decay) * value_pre \
                  + (1 - para_weight) * value_curr
        return res

    def detect_node_j(self, j_id, Tc, Th, Ts, agg_T, runningtime):
        self.__cal_local_T_value(j_id, runningtime)
        self.__cal_ind_T_value(j_id, Tc, Th, Ts, agg_T, runningtime)
        self.__cal_agg_T(j_id)
        boolgood = self.T_value[j_id] > self.final_threshhold
        isbk = not boolgood

        # 打印一下看看
        partner_id = j_id
        tmp_fwd = self.n_fwd[partner_id]
        tmp_meet = self.n_meet[partner_id]
        tmp_rec = self.n_rec[partner_id]
        tmp_src = self.n_src[partner_id]
        # Connectivity
        T_d_c = (2* tmp_fwd + tmp_meet)/(2*tmp_fwd + tmp_meet + self.num_of_nodes)
        # Fitness
        T_d_h = (tmp_fwd + tmp_rec + 1)/(tmp_src + tmp_fwd + tmp_rec + 0 + 2)

        # Satisfaction
        tmp_Fack = self.n_Fack[partner_id]
        tmp_ack = self.n_ack[partner_id]

        T_d_s1 = tmp_ack / (tmp_rec + 1)
        T_d_s2 = (tmp_Fack + tmp_ack) / (tmp_Fack + tmp_rec + 1)
        # if tmp_Fack == 0:
        #     T_d_s1 = tmp_ack / (tmp_rec + 1)
        # else:
        #     T_d_s2 = (tmp_Fack + tmp_ack) / (tmp_Fack + tmp_rec + 1)

        # return isbk, self.T_value[j_id], self.agg_T_c[j_id], self.agg_T_h[j_id], self.agg_T_s[j_id]
        return isbk, T_d_c, T_d_h, T_d_s1, T_d_s2

    def get_c_h_s_agg(self):
        Tc = self.dir_T_d_c.copy()
        Th = self.dir_T_d_h.copy()
        Ts = self.dir_T_d_s.copy()
        Tagg = self.T_value.copy()

        return Tc, Th, Ts, Tagg