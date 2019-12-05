
# PROVEST: Provenance-Based Trust Model for Delay Tolerant Networks
# 2018 TDSC
# begin: 2019-08-09
# 只做blackhole
# 这个论文做得很奇怪 我不能确定 实现的是否完全正确！！！！

from Routing.RoutingBase import RoutingBase
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
        # 保存累计的[r, s, u]
        self.all_rsu = np.zeros((self.numofnodes, 3), dtype=int)
        # 为了计算T 需要临时保存本次的[r' s' u'],
        self.all_rsu_ = np.zeros((self.numofnodes, 3), dtype=int)
        # 4种 T值
        self.T = np.zeros((self.numofnodes, 1), dtype=float)
        # 初始化评价值 否则routing将无法启动
        for tmp in range(self.numofnodes):
            self.all_rsu[tmp, 0] = 1
            self.all_rsu[tmp, 1] = 1
            self.T[tmp, 0] = 0.5
        # 转发阈值
        self.T_min = 0.3

    # PI, provenance information; node_i k是之前MC的id, O_ik是直接观点(pos eve和neg eve的个数)
    # P_i_k = (i, k, O_ik(t))  O_ik(t)pos和neg个数
    # mission message (MM)
    # message = [MM, P_0_o P_1_0 P_2_1 ... P_m_m-1] encrypted by key K_s_t
    # (k_0 k_n) ; k_n-1 = F(k_n); DN能解出前面的key并验证 层层加密

    # 更新4种T值
    def __update_T(self, b_id):
        total = self.all_rsu[b_id, 0] + self.all_rsu[b_id, 1] + self.all_rsu[b_id, 2] + self.all_rsu_[b_id, 0] + \
                self.all_rsu_[b_id, 1] + self.all_rsu_[b_id, 2]
        # pessi 悲观
        T_pessi = (self.all_rsu[b_id, 0] + self.all_rsu_[b_id, 0])/total
        # opti 乐观
        T_opti = (self.all_rsu[b_id, 0] + self.all_rsu[b_id, 2] + self.all_rsu_[b_id, 0] + self.all_rsu_[b_id, 2])/total
        # real 真实
        T_real = (self.all_rsu[b_id, 0] + self.all_rsu_[b_id, 0])/(self.all_rsu[b_id, 0] + self.all_rsu[b_id, 1] +
                                                                   self.all_rsu_[b_id, 0] + self.all_rsu_[b_id, 1])
        # hybrid 融合
        if self.all_rsu[b_id,0] < self.all_rsu[b_id,1]:
            T_hyb = T_pessi
        elif self.all_rsu[b_id,0] > self.all_rsu[b_id,1]:
            T_hyb = T_opti
        else:
            # self.all_rsu[b_id, 0] == self.all_rsu[b_id, 1]:
            T_hyb = T_real
        self.T[b_id] = T_hyb

    # 根据direct eve, 更新trust; [r_, s_, u_]
    def __update_trust_direct_eve(self, b_id):
        direct_eve = np.zeros((1, 3), dtype=int)
        # 1) availability trust, sum=1, 是否connectivity;
        # 无论是 normal node 还是 black(gray) hole 都会同意
        direct_eve = direct_eve + np.array([1, 0, 0])
        # 2) integrity trust, sum=3,  是否完整 如果连通不上; 1. identity attack 2. fake recommendation 3.message modification
        # 无论是 normal node 还是 black(gray) hole 都不会犯这种错误
        direct_eve = direct_eve + np.array([3, 0, 0])
        # 3) competence trust, sum=1 <= sum=2, 1.energy 不考虑 2.cooperative behavior 因为接触不到
        # 无论是 normal node 还是 black(gray) hole 都不会得知报文以后的发展
        direct_eve = direct_eve + np.array([0, 0, 1])
        return direct_eve

    # indirect eve只有DN能做 通过带的track计算出来; 根据indirect eve, 更新trust [r_, s_, u_]
    def __update_trust_indirect_eve(self, j_id, track):
        indirect_eve = np.zeros((1, 3), dtype=int)
        if not (j_id in track):
            indirect_eve = indirect_eve + np.array([0, 0, 5])
        else:
            idx = track.index(j_id)
            # 1) availability trust   1.j_id在j的PI里面  2.j_id在本次PI 也在 next PI里的j_id 3.j的前一个MC和j 都>T_min
            if (self.T[j_id] > self.T_min) and (self.T[track[idx + 1]] > self.T_min):
                indirect_eve = indirect_eve + np.array([1, 0, 0])
            else:
                indirect_eve = indirect_eve + np.array([0, 1, 0])
            # 2) integrity; 如果j的PI在MC里面 不会犯错 fake identity, fake recommendation(下一个MC > T_min), message modification
            if self.T[track[idx+1]] > self.T_min:
                indirect_eve = indirect_eve + np.array([3, 0, 0])
            else:
                indirect_eve = indirect_eve + np.array([0, 3, 0])
            # 3) energy, 不考虑; cooperativeness behavior;  如果j的PI在里面 check: j的下一跳>T_min
            if self.T[track[idx + 1]] > self.T_min:
                indirect_eve = indirect_eve + np.array([1, 0, 0])
            else:
                indirect_eve = indirect_eve + np.array([0, 1, 0])
        return indirect_eve

    # ====================== 主要接口 =======================
    # linkup的时候 准备记录本次传输的pkt
    def notify_link_up(self, running_time, b_id, *args):
        # i和j发生接触 更新direct eve.
        direct_eve = self.__update_trust_direct_eve(b_id)
        self.all_rsu_[b_id, :] = self.all_rsu_[b_id, :] + direct_eve

    # 成功投递
    def notify_receive_succ(self, a_id, i_pkt):
        # 执行indirect操作
        # !!!!!!!!!!!!!!! 实在不确定 这个地方 应该怎么实现？
        # !!!!!!!!!!!!!!! 这个paper 里 此处描述的很模糊？
        for tmp_id in range(self.numofnodes):
            # 排除 自己 和 对端, 评价其他所有的node_id
            if (tmp_id == self.theBufferNode.node_id) or (tmp_id == a_id):
                continue
            # 对于远端node_id 计算 indirect_eve
            indirect_eve = self.__update_trust_indirect_eve(a_id, i_pkt.track)
            self.all_rsu_[tmp_id, :] = self.all_rsu_[tmp_id, :] + indirect_eve

    # 从 a_id 获取一个报文 i_pkt
    def decideAddafterRece(self, runningtime, a_id, i_pkt):
        # pkt 的 track 增加
        i_pkt.track.append(self.theBufferNode.node_id)
        return True, RoutingBase.Rece_Code_AcceptPkt

    # linkdown时候 准备结束本次记录
    def notify_link_down(self, running_time, b_id, *args):
        self.__update_T(b_id)
        # 刷新 准备下次记录
        self.all_rsu[b_id, :] = self.all_rsu[b_id, :] + self.all_rsu_[b_id, :]
        self.all_rsu_[b_id, :] = np.zeros((1, 3), dtype=int)

    # 只对信任的node才传输
    def gettranpktlist(self, runningtime, b_id, listb, a_id, lista, *args):
        if self.T[b_id] > self.T_min:
            return super().gettranpktlist(runningtime, b_id, listb, a_id, lista, args)
        else:
            return []



