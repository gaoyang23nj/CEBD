
# PROVEST: Provenance-Based Trust Model for Delay Tolerant Networks
# 2018 TDSC
# begin: 2019-08-09
# 只做blackhole

from RoutingBase import RoutingBase
import numpy as np

# 论文关注的点 attack model太多了 不知道针对blackhole/grayhole的具体办法？

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
        r = 1
        s = 1
        # 保存累计的[r, s, u]
        self.all_rsu = np.zeros((self.numofnodes, 3), dtype=int)
        # 为了计算T 需要临时保存本次的[r' s' u'],
        self.all_rsu_ = np.zeros((self.numofnodes, 3), dtype=int)

    # PI, provenance information; node_i k是之前MC的id, O_ik是直接观点(pos eve和neg eve的个数)
    # P_i_k = (i, k, O_ik(t))  O_ik(t)pos和neg个数
    # mission message (MM)
    # message = [MM, P_0_o P_1_0 P_2_1 ... P_m_m-1] encrypted by key K_s_t
    # (k_0 k_n) ; k_n-1 = F(k_n); DN能解出前面的key并验证 层层加密

    # 更新4种T值
    def update_T(self):
        pass

    # 根据direct eve, 更新trust; [r_, s_, u_]
    def __update_trust_direct_eve(self, b_id, is_trans):
        direct_eve = np.zeros((1, 3), dtype=int)
        # 1) availability trust, sum=1, 是否connectivity;
        # 无论是 normal node 还是 black(gray) hole 都会同意
        direct_eve = direct_eve + np.array([1, 0, 0])
        # 2) integrity trust, sum=3,  是否完整 如果连通不上; 1. identity attack 2. fake recommendation 3.message modification
        # 无论是 normal node 还是 black(gray) hole 都不会犯这种错误
        if is_trans:
            direct_eve = direct_eve + np.array([3, 0, 0])
        else:
            direct_eve = direct_eve + np.array([0, 0, 3])
        # 3) competence trust, sum=1 <= sum=2, 1.energy 不考虑 2.cooperative behavior 因为接触不到
        # 无论是 normal node 还是 black(gray) hole 都不会得知报文以后的发展
        direct_eve = direct_eve + np.array([0, 0, 1])
        self.all_rsu_[b_id, :] = self.all_rsu_[b_id, :] + direct_eve

    # indirect eve只有DN能做 通过带的track计算出来; 根据indirect eve, 更新trust [r_, s_, u_]
    def __update_trust_indirect_eve(self, j_id):
        indirect_eve = np.zeros((1, 3), dtype=int)
        # 1) availability trust   1.j_id在j的PI里面  2.j_id在本次PI 也在 next PI里的j_id 3.j的前一个MC和j 都>T_min
        # 如果 j的PI在里面 且正确;
        indirect_eve = indirect_eve + np.array([1, 0, 0])
        # 如果 没有 eve 可用
        indirect_eve = indirect_eve + np.array([0, 0, 1])
        # 2) integrity trust
        # 如果j的PI在MC里面 不会犯错 fake identity, fake recommendation(下一个MC > T_min), message modification
        indirect_eve = indirect_eve + np.array([3, 0, 0])
        # 如果j的PI不在里面已经
        indirect_eve = indirect_eve + np.array([0, 0, 3])
        # 3) energy, 不考虑; cooperativeness behavior,
        # 如果j的PI不在里面
        indirect_eve = indirect_eve + np.array([0, 0, 1])
        # 如果j的PI在里面 check: j的下一跳>T_min
        indirect_eve = indirect_eve + np.array([1, 0, 0])
        indirect_eve = indirect_eve + np.array([0, 1, 0])
        return indirect_eve

    # ====================== 主要接口 =======================
    def notify_link_up(self, running_time, b_id, *args):
        # 保存 临时传输报文数目
        self.tmpRSLid.append(b_id)
        self.tmpRSL.append([])

    #
    def notify_link_down(self, running_time, b_id, *args):


    def decideAddafterRece(self, a_id, i_pkt):
        # pkt 的 track 增加
        i_pkt.track.append(self.theBufferNode.node_id)
        # 收到报文
        self.__update_trust_direct_eve(a_id)




