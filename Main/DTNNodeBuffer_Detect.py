import numpy as np

class DTNNodeBuffer_Detect(object):
    def __init__(self, node_id, num_of_nodes):
        self.node_id = node_id
        # 直接证据======
        # 发送给j_node的pkt个数为self.send[j_node]
        self.send = np.zeros(num_of_nodes, dtype='int')
        # 从i_node接收的pkt个数为self.receive[i_node]
        self.receive = np.zeros(num_of_nodes, dtype='int')
        # 从i_node接收 且 i_node就是pkt的src
        self.receivefromsrc = np.zeros(num_of_nodes, dtype='int')
        # 间接证据======每行 来自交换所得
        self.in_send = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        self.in_receive = np.zeros((num_of_nodes, num_of_nodes), dtype='int')
        self.updatetime = np.zeros(num_of_nodes, dtype='int')

    def sendtoj(self, j_id):
        self.send[j_id] = self.send[j_id] + 1

    def receivefromi(self, i_id):
        self.receive[i_id] = self.receive[i_id] + 1

    def receivefromsrc(self, i_id):
        self.receivefromsrc[i_id] = self.receivefromsrc[i_id] + 1

    def renewindeve(self, runningtime, j_id, row_values_send, row_values_receive):
        if runningtime >= self.updatetime[j_id]:
            self.in_send[j_id, :] = row_values_send
            self.in_receive[j_id, :] = row_values_receive
        self.updatetime[j_id] = runningtime

    def getsendvalues(self):
        return self.send.copy()

    def getreceivevalues(self):
        return self.receive.copy()

    def getindevevalues(self):
        return self.in_send.copy(), self.in_receive.copy()