import numpy as np

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
        # 超时时间4个小时
        self.expriation = 3600*10*0.5
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

