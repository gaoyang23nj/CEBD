
# PROVEST: Provenance-Based Trust Model for Delay Tolerant Networks
# 2018 TDSC

from RoutingBase import RoutingBase
import numpy as np

# 采用TA分发key, Diffie-Hellman的办法不可行
# 对称密钥 K_st, outside attacker
# trust T_min
class RoutingProvest(RoutingBase):
    # local watchdog => 3 events
    PosEvt = 1
    NegEvt = 2
    NoDetEvt = 3

    def __init__(self, theBufferNode, numofnodes):
        super(RoutingProvest, self).__init__(theBufferNode)
        self.numofnodes = numofnodes
        # 3个维度的属性 availability integrity competence Tx
        self.all_trust = np.zeros((self.numofnodes, 3))


    # PI, provenance information; node_i k是之前MC的id, O_ik是直接观点(pos eve和neg eve的个数)
    # P_i_k = (i, k, O_ik(t))
    # mission message (MM)
    # message = [MM, P_0_o P_1_0 P_2_1 ... P_m_m-1] encrypted by key K_s_t
    # (k_0 k_n) ; k_n-1 = F(k_n); DN能解出前面的key并验证 层层加密

    # 更新4种T值
    def update_T(self):
        pass

    # 根据direct eve, 更新trust
    def update_trust_direct_eve(self):
        # 1) availability trust, sum=1, 是否connectivity
        (r_, s_, u_) = (1, 0, 0)
        (r_, s_, u_) = (0, 1, 0)
        # 2) integrity trust, sum=3,  是否完整 如果连通不上; 1. identity attack 2. fake recommendation 3.message modification
        (r_, s_, u_) = (0, 0, 3)
        # 3) competence trust, sum=1 <= sum=2, 1.energy 不考虑 2. cooperative behavior


    # 根据indirect eve, 更新trust
    def update_trust_indirect_eve(self):
        # 1) availability trust   1.j_id在j的PI里面  2.j_id 在本次PI匹配得上next PI 3.前一个MC和j 都>T_min
        (r_, s_, u_) = (1, 0, 0)
        # 无PI可用
        (r_, s_, u_) = (0, 0, 1)
        # 2) integrity trust



