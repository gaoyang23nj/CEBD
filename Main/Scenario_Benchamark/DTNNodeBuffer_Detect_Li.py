import numpy as np


class DTNNodeBuffer_Detect_Li(object):
    def __init__(self, node_id, numofnodes, isbk):
        # ID_i
        self.node_id = node_id
        self.numofnodes = numofnodes
        # 该节点是不是bk
        self.isbk = isbk
        # sn_i
        self.ER_sn = 0

        self.gamma_init = 0.6
        # ttl:1 hour; 临到ttl blackhole节点才报告自己刚刚丢弃了报文; 遇到任何节点都称呼自己是最好的
        self.ttl = 1*360
        self.Gamma_list = [ self.gamma_init ] * self.numofnodes

        # 快速下降
        self.rho = 0.1
        # 缓慢上升
        self.delta = 0.1

        self.del_label = 3
        self.cnt_label = 4
        # 元素格式 （snd/rcv/del/cnt）发送/接收/删除/相遇
        self.ER_list = []

        self.r = 10

        if self.isbk == True:
            # 假装到ttl（接收后一定时间后 由self.ttl指定）而被删掉的报文；
            self.pret_del = []

        # 一次相遇
        self.tmp_ER = {"label":self.cnt_label, "self_id": -1, "partner_id": -1, "self_sn": -1, "partner_sn": -1, "running_time": -1,
                       "SL": [], "RL": []}

    # 刚刚和某个节点node j相遇 向node j发送i本地记录的ERW
    def get_local_ER_list(self):
        return self.ER_list.copy()

    def get_new_seq_for_ER(self):
        self.ER_sn = self.ER_sn + 1
        return self.ER_sn

    # 对方的id号, 对方的sn, runnningtime
    def begin_new_ER(self, node_j_id, node_j_sn, runnningtime):
        # 这里是保证同时 ER已经被清空
        assert (self.tmp_ER["self_id"] == -1)
        self.tmp_ER["label"] = self.cnt_label
        self.tmp_ER["self_id"] = self.node_id
        self.tmp_ER["partner_id"] = node_j_id
        self.tmp_ER["self_sn"] = self.ER_sn
        self.tmp_ER["partner_sn"] = node_j_sn
        self.tmp_ER["running_time"] = runnningtime
        # 看看是不是时候 把 del记录公布出来 从前到后
        tmp_del_label = 0
        if self.isbk:
            for tmp_del in self.pret_del:
                next_del_time = tmp_del[1]
                # 预定的删除时间 到了
                if next_del_time < runnningtime:
                    self.ER_list.append(tmp_del)
                    tmp_del_label = tmp_del_label + 1
            for i in range(tmp_del_label):
                self.pret_del.pop(0)

    # 我(本节点) 发送给其他节点的报文list
    def add_SL_to_new_ER(self, node_j, runningtime, SL):
        # print('{} pkts {}->{}'.format(len(SL), self.node_id, node_j))
        if len(SL) == 0:
            return
        assert self.tmp_ER["partner_id"] == node_j
        for (pkt_id, src_id, dst_id) in SL:
            self.tmp_ER["SL"].append((pkt_id, src_id, dst_id))

    # 我(本节点) 从其他节点接收的报文list
    def add_RL_to_new_ER(self, node_j, runningtime, RL):
        # print('{} pkts {}->{}'.format(len(RL), node_j, self.node_id))
        if len(RL) == 0:
            return
        assert self.tmp_ER["partner_id"] == node_j
        for (pkt_id, src_id, dst_id) in RL:
            self.tmp_ER["RL"].append((pkt_id, src_id, dst_id))
        if self.isbk:
            for (pkt_id, src_id, dst_id) in RL:
                # 现在接收报文 但是blackhole节点准备后 后面伪造的del记录
                self.pret_del.append((self.del_label, self.ttl + runningtime,
                                     pkt_id, src_id, dst_id))

    def end_new_ER(self):
        new_ER = (self.tmp_ER["label"], self.tmp_ER["self_id"], self.tmp_ER["partner_id"], self.tmp_ER["self_sn"],
                  self.tmp_ER["partner_sn"], self.tmp_ER["running_time"],
                  self.tmp_ER["SL"].copy(), self.tmp_ER["RL"].copy())
        # 整合ER 保存到ERw里
        self.ER_list.append(new_ER)
        while True:
            # 计算ER里面有多少 contact
            num_contact = 0
            for tmp in self.ER_list:
                if tmp[0] == self.cnt_label:
                    num_contact = num_contact + 1
            if num_contact > self.r:
                self.ER_list.pop(0)
            else:
                break
        self.tmp_ER["label"] = self.cnt_label
        self.tmp_ER["self_id"] = -1
        self.tmp_ER["partner_id"] = -1
        self.tmp_ER["self_sn"] = -1
        self.tmp_ER["partner_sn"] = -1
        self.tmp_ER["running_time"] = -1
        self.tmp_ER["SL"].clear()
        self.tmp_ER["RL"].clear()
        return

    def detect_contacted_node(self, node_j, ER_list):
        drop = False
        rcv = False
        snd = False
        for tmp_record in ER_list:
            if tmp_record[0] == self.del_label:
                drop = True
            else:
                # SL
                if (len(tmp_record[6]) > 0):
                    snd = True
                # RL
                if (len(tmp_record[7]) > 0):
                    rcv = True

        if drop:
            self.Gamma_list[node_j] = self.Gamma_list[node_j] * self.rho
        elif (rcv or snd):
            tmp_value = self.Gamma_list[node_j] + self.delta
            if tmp_value < 1:
                self.Gamma_list[node_j] = tmp_value
            else:
                self.Gamma_list[node_j] = 1
        else:
            tmp_value = self.Gamma_list[node_j] - self.delta
            if tmp_value > 0:
                self.Gamma_list[node_j] = tmp_value
            else:
                self.Gamma_list[node_j] = 0

        return self.Gamma_list[node_j]

    # 丢弃通知
    def notify_del_pkt(self, runningtime):
        self.ER_list.append((self.del_label, runningtime))
