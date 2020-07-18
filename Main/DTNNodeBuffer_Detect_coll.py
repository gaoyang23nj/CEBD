import numpy as np

class DTNNodeBuffer_Detect_coll(object):
    def __init__(self, node_id, num_of_nodes, bk_node_id):
        self.node_id = node_id
        self.bk_node_id = bk_node_id
        # 直接证据============================ a和b相遇时候能得到的
        # a发送给b的pkt个数为 self.send[j_node]
        self.send_values = np.zeros(num_of_nodes, dtype='int')
        # a从b接收的pkt个数为self.receive[i_node]
        self.receive_values = np.zeros(num_of_nodes, dtype='int')

        # 实际上也表示 a为中继b有关的pkt做出多少贡献
        # a接收到的所有pkt中 pkt的src是b 的 pkt个数;  self.receive_src_values[i]表示a接收的报文中src为i的pkt个数
        self.receive_src_values = np.zeros(num_of_nodes, dtype='int')
        # a接收到的所有pkt中 pkt的dst是b 的 pkt个数;  self.receive_dst_values[i]表示a接收的报文中dst为i的pkt个数
        self.receive_dst_values = np.zeros(num_of_nodes, dtype='int')
        # a发送的所有pkt中 pkt的src是b 的 pkt个数;  self.send_src_values[i]表示a发送的报文中src为i的pkt个数
        self.send_src_values = np.zeros(num_of_nodes, dtype='int')
        # a发送的所有pkt中 pkt的dst是b 的 pkt个数;  self.send_dst_values[i]表示a发送的报文中dst为i的pkt个数
        self.send_dst_values = np.zeros(num_of_nodes, dtype='int')

        #a发送给b的pkt中 pkt的src也是a的pkt个数;  self.receive_from_and_pktsrc[i]表示a发送的报文中src也是a的pkt个数
        self.receive_from_and_src = np.zeros(num_of_nodes, dtype='int')

        # a发送给所有节点 的pkt总数
        self.send_all = np.zeros(1, dtype='int')
        # a从所有节点接收 的pkt总数
        self.receive_all = np.zeros(1, dtype='int')

        # 间接证据============================ a和b相遇, i(除了a b以外的其他节点)向a提供的 关于b的证据
        # 每行 来自交换所得;
        # 行号i 代表 证据来自的节点i;   列号j 代表 所评价的节点j;  self.in_send_values[i,j] 表示  i发给对应节点j的报文数
        self.ind_send_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        # 行号i 代表 证据来自的节点i;   列号j 代表 所评价的节点j;  self.in_receive_values[i,j] 表示  i从节点j收到的报文数
        self.ind_receive_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')

        # 行号i 代表 证据来自的节点i;   列号j 代表 所评价的节点j;  self.in_receive_src_values[i,j] 表示  i收到的所有pkt中 pkt的src是b 的pkt个数
        self.ind_receive_src_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        # 行号i 代表 证据来自的节点i;   列号j 代表 所评价的节点j;  self.in_receive_src_values[i,j] 表示  i收到的所有pkt中 pkt的dst是b 的pkt个数
        self.ind_receive_dst_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        # 行号i 代表 证据来自的节点i;   列号j 代表 所评价的节点j;  self.in_receive_src_values[i,j] 表示  i发送的所有pkt中 pkt的src是b 的pkt个数
        self.ind_send_src_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        # 行号i 代表 证据来自的节点i;   列号j 代表 所评价的节点j;  self.in_receive_src_values[i,j] 表示  i发送的所有pkt中 pkt的dst是b 的pkt个数
        self.ind_send_dst_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')

        #行号i 代表 证据来自的节点i;   列号j 代表 所评价的节点j;  self.receive_from_and_pktsrc[i,j]表示i从j收到的报文中, pkt src也是a的pkt个数
        self.ind_receive_from_and_src = np.zeros((num_of_nodes, num_of_nodes), dtype='int')

        # 行号i 代表 证据来自的节点i; i所给出的证据 更新的时间
        self.ind_eve_updatetime = np.zeros(num_of_nodes, dtype='int')

        # 行号i 代表 证据来自的节点i; i发送给所有节点 的pkt总数
        self.ind_send_all = np.zeros(num_of_nodes, dtype='int')
        # 行号i 代表 证据来自的节点i; i从所有节点接收 的pkt总数
        self.ind_receive_all = np.zeros(num_of_nodes, dtype='int')

        # # 为了加快判定速度，对比较确定的结果，不再进行重复计算; 0 表示不确定节点，1 表示确定的恶意节点，-1表示确定的正常节点, -2表示自己
        # self.viewothers = np.zeros(num_of_nodes, dtype='int')
        # self.viewothers[node_id] = -2
        # 更新间隔控制
        # self.updatectl

    # def getviewofb(self, b_id):
    #     viewofb = self.viewothers[b_id].copy()
    #     return viewofb
    #
    # def setviewofb(self, b_id, viewofb):
    #     self.viewothers[b_id] = viewofb

    def send_to_b(self, b_id):
        self.send_values[b_id] = self.send_values[b_id] + 1
        self.send_all = self.send_all + 1
        # 针对 collusion的bk节点
        if b_id == self.bk_node_id:
            if self.send_values[b_id] > self.receive_values[b_id]:
                # 帮助 bk_id 伪造证据
                diff = self.send_values[b_id] - self.receive_values[b_id]
                self.receive_values[b_id] = self.receive_values[b_id] + diff
                self.receive_all = self.receive_all + diff

    def send_to_pkt_src(self, pkt_src_id):
        self.send_src_values[pkt_src_id] = self.send_src_values[pkt_src_id] + 1

    def send_to_pkt_dst(self, pkt_dst_id):
        self.send_dst_values[pkt_dst_id] = self.send_dst_values[pkt_dst_id] + 1

    def receive_from_a(self, a_id):
        self.receive_values[a_id] = self.receive_values[a_id] + 1
        self.receive_all = self.receive_all + 1
        if a_id == self.bk_node_id:
            if self.send_values[a_id] > self.receive_values[a_id]:
                # 帮助 bk_id 伪造证据
                diff = self.send_values[a_id] - self.receive_values[a_id]
                self.receive_values[a_id] = self.receive_values[a_id] + diff
                self.receive_all = self.receive_all + diff

    def receive_from_pkt_src(self, src_id):
        self.receive_src_values[src_id] = self.receive_src_values[src_id] + 1

    def receive_from_pkt_dst(self, dst_id):
        self.receive_dst_values[dst_id] = self.receive_dst_values[dst_id] + 1

    def receive_from_and_pktsrc(self, a_id, pkt_src_id):
        assert a_id == pkt_src_id
        self.receive_from_and_src[a_id] =  self.receive_from_and_src[a_id] + 1

    def renewindeve(self, runningtime, b_id, b_send, b_receive, b_send_all, b_receive_all, b_receive_src, b_receive_dst, b_send_src, b_send_dst, b_receive_from_and_src):
        # 现在相遇时刻 必然晚于 之前相遇时刻 assert一下
        assert (runningtime >= self.ind_eve_updatetime[b_id])

        self.ind_send_values[b_id, :] = b_send
        self.ind_receive_values[b_id, :] = b_receive

        self.ind_receive_src_values[b_id, :] = b_receive_src
        self.ind_receive_dst_values[b_id, :] = b_receive_dst

        self.ind_send_src_values[b_id, :] = b_send_src
        self.ind_send_dst_values[b_id, :] = b_send_dst

        self.ind_eve_updatetime[b_id] = runningtime
        self.ind_send_all[b_id] = b_send_all
        self.ind_receive_all[b_id] = b_receive_all

        self.ind_receive_from_and_src[b_id] = b_receive_from_and_src


    # =============================================================================================================
    def get_send_values(self):
        return self.send_values.copy()

    def get_receive_values(self):
        return self.receive_values.copy()

    def get_receive_src_values(self):
        return self.receive_src_values.copy()

    def get_receive_dst_values(self):
        return self.receive_dst_values.copy()

    def get_send_src_values(self):
        return self.send_src_values.copy()

    def get_send_dst_values(self):
        return self.send_dst_values.copy()

    def get_send_all(self):
        return self.send_all.copy()

    def get_receive_all(self):
        return self.receive_all.copy()

    def get_receive_from_and_pktsrc(self):
        return self.receive_from_and_src.copy()

    # =============================================================================================================
    def get_ind_send_values(self):
        return self.ind_send_values.copy()

    def get_ind_receive_values(self):
        return self.ind_receive_values.copy()

    def get_ind_receive_src_values(self):
        return self.ind_receive_src_values.copy()

    def get_ind_receive_dst_values(self):
        return self.ind_receive_dst_values.copy()

    def get_ind_send_src_values(self):
        return self.ind_send_src_values.copy()

    def get_ind_send_dst_values(self):
        return self.ind_send_dst_values.copy()

    def get_ind_time(self):
        return self.ind_eve_updatetime.copy()

    def get_ind_send_all(self):
        return self.ind_send_all.copy()

    def get_ind_receive_all(self):
        return self.ind_receive_all.copy()

    def get_ind_receive_from_and_pktsrc(self):
        return self.ind_receive_from_and_src.copy()