import numpy as np

class DTNNodeBuffer_Detect(object):
    def __init__(self, node_id, num_of_nodes):
        self.node_id = node_id
        # 直接证据======
        # 发送给j_node的pkt个数为self.send[j_node]
        self.send_values = np.zeros(num_of_nodes, dtype='int')
        # 从i_node接收的pkt个数为self.receive[i_node]
        self.receive_values = np.zeros(num_of_nodes, dtype='int')
        # 从i_node接收 且 i_node就是pkt的src
        self.receive_src_values = np.zeros(num_of_nodes, dtype='int')
        # 间接证据======每行 来自交换所得
        self.in_send_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        self.in_receive_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        self.in_receive_src_values = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        self.updatetime = np.zeros(num_of_nodes, dtype='int')

    def sendtoj(self, j_id):
        self.send_values[j_id] = self.send_values[j_id] + 1

    def receivefromi(self, i_id):
        self.receive_values[i_id] = self.receive_values[i_id] + 1

    def receivefromsrc(self, i_id):
        self.receive_src_values[i_id] = self.receive_src_values[i_id] + 1

    def renewindeve(self, runningtime, j_id, row_values_send, row_values_receive, row_values_receive_src):
        if runningtime >= self.updatetime[j_id]:
            self.in_send_values[j_id, :] = row_values_send
            self.in_receive_values[j_id, :] = row_values_receive
            self.in_receive_src_values[j_id, :] = row_values_receive_src
        self.updatetime[j_id] = runningtime

    def get_send_values(self):
        return self.send_values.copy()

    def get_receive_values(self):
        return self.receive_values.copy()

    def get_receive_src_values(self):
        return self.receive_src_values.copy()

    def get_ind_send_values(self):
        return self.in_send_values.copy()

    def get_ind_receive_values(self):
        return self.in_receive_values.copy()

    def get_ind_receive_src_values(self):
        return self.in_receive_src_values.copy()