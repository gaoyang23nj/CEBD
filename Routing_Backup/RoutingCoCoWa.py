# -*- coding: UTF-8 -*-
# 2015TMC
# CoCoWa: A Collaborative Contact-Based Watchdog for Detecting Selfish Nodes
# begin: 2019-08-09

# 没有明确的event事件定义 <不能实现>
# 相遇间隔的 lambda假设

from RoutingBase import RoutingBase


class RoutingCoCoWa(RoutingBase):
    # local watchdog => 3 events
    PosEvt = 1
    NegEvt = 2
    NoDetEvt = 3

    def __init__(self, theBufferNode, numofnodes):
        super(RoutingCoCoWa, self).__init__(theBufferNode)
        self.numofnodes = numofnodes
        # 对其他节点的reputation评价 由event探测事件进行修改
        self.all_rho = np.zeros((self.numofnodes,  1))
        # 记录 pair contact 事件分布
        self.all_lambda = np.zeros((self.numofnodes, self.numofnodes))
        # 参数
        # 单次增量
        self.delta = 2
        # state阈值
        self.theta = 3
        # neg 传播因子
        self.gamma = 0.9
        # p_d, p_fp, p_fn, p_c, p_m

    # Pos or Neg; Local or Indirect
    def update_rho(self, node_id, isPos, islocal):
        tmp = 1
        if islocal:
            tmp = self.delta
        if isPos:
            self.all_rho[node_id, 1] = self.all_rho[node_id, 1] + tmp
        else:
            self.all_rho[node_id, 1] = self.all_rho[node_id, 1] - tmp
        return self.all_rho[node_id, 1]

    # aging rho; 超时以后 相反事件生成 弥补之前的rho变化
    def aging_rho(self):
        # event stamp
        pass

    # switch state; 状态切换 每次改变reputation都有可能需要切换
    def switch_state(self):
        # 状态切换
        if self.all_rho[node_id, 1] >= self.theta:
            # 状态为Pos
        elif self.all_rho[node_id, 1] <= self.theta:
            # 状态为Neg
        else:
            # 状态为NoInfo

