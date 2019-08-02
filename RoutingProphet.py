import numpy as np
import math

import RoutingBase

class RoutingProphet(RoutingBase):
    def __init__(self, buffer_node, numofnodes, p_init=0.75, gamma=0.98, beta=0.25):
        super(RoutingProphet, self).__init__(buffer_node)
        self.P_init = p_init
        self.Gamma = gamma
        self.Beta = beta
        self.numofnodes = numofnodes
        # 记录 两两之间的delivery prob
        self.delivery_prob = np.zeros((self.numofnodes, self.numofnodes), dtype='double')
        # 初始化 为 P_init
        for i in range(self.numofnodes):
            for j in range(self.numofnodes):
                if i != j:
                    self.delivery_prob[i, j] = self.P_init
        # 记录 两两之间的上次相遇时刻 以便计算相遇间隔
        self.waiting_time = np.zeros((self.numofnodes, self.numofnodes), dtype='double')


    # a 和 b 相遇 更新prob
    def __update(self, a_id, b_id):
        self.delivery_prob[a_id, b_id] = self.delivery_prob[a_id, b_id] + (1 - self.delivery_prob[a_id, b_id]) * self.P_init

    # 每隔一段时间执行 老化效应
    def __aging(self, a_id, b_id, duration):
        self.delivery_prob[a_id, b_id] = self.delivery_prob[a_id, b_id] * math.pow(self.Gamma, duration)

    # 传递效应, 1点动 则多点动
    def __tranproperty(self):


    # =======================================================================================
    # 当a->b 相遇 更新a->b相应的值
    def encounter(self, runningtime, a_id, b_id):
        duration = runningtime - self.waiting_time[a_id, b_id]
        self.__update(a_id, b_id)
        # 更新时间表 以便于下次计算
        self.waiting_time[a_id, b_id] = runningtime

    def


