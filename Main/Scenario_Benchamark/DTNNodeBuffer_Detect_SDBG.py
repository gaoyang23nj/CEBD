import numpy as np

class DTNNodeBuffer_Detect_SDBG(object):
    def __init__(self, node_id, numofnodes):
        # ID_i
        self.node_id = self.node_id
        self.numofnodes = numofnodes
        # sn_i
        self.ER_sn = 0
        # sig_i 模拟一下好了
        self.sig = 'sig_' + self.node_id

        # 只在本地保存w个ER； presents them to the neighbors
        self.ER_List = []
        self.w = 100

        # 对其他剩余节点 都创建MS: (N-1)个MS
        # 100 * [j_node_id, latest_sn_j, latest_time_j, self.TR_init, 0(未设置)/1(设置)]
        arr1 = np.arange(self.numofnodes).reshape(self.numofnodes, 1)
        arr2 = np.zeros((self.numofnodes, 4), dtype='double')
        self.MS = np.hstack([arr1, arr2])

        # SDBG 的 阈值参数; 需要经验 选择;//从paper的图里 选择出来
        self.Th_RR = 4
        self.Th_SFR = 0.08

        self.Dec_gamma = 0.2
        self.Dec_rho = 0.2
        self.Inc_lambda = 0.5

        self.Th_FXS = 2400

        # TR的阈值 good mal; >good的 高优先级; <good且>mal的 低优先级; < mal 不会交换报文 只交流控制信息
        self.Th_good = 2
        self.Th_mal = -2

        # ********************************************
        # 为了在一次contact中构建ER记录 需要把交换的报文都记录下来
        self.tmp_ER = []

    # 刚刚和某个节点node j相遇 向node j发送i本地记录的ERW
    def get_local_ER_list(self):
        return self.ER_List.copy()

    def get_new_seq_for_ER(self):
        return self.ER_sn, self.sig

    # 对方的id号, 对方的sn, runnningtime
    def begin_new_ER(self, node_j_id, node_j_sn, runnningtime):
        # 这里是保证同时 只有一个ER存在
        assert len(self.tmp_ER) == 0
        self.tmp_ER.extend([self.node_id, node_j_id, self.ER_sn, node_j_sn, runnningtime])
        # 存在SL self.tmp_ER[5]
        self.tmp_ER.append([])
        # 存放RL self.tmp_ER[6]
        self.tmp_ER.append([])

    # 我(本节点) 发送给其他节点的报文list
    def add_SL_to_new_ER(self, node_j, runningtime, SL):
        print('{} pkts {}->{}'.format(len(SL), self.node_id, node_j, ))
        assert self.tmp_ER[1] == node_j and self.tmp_ER[4] == runningtime
        self.tmp_ER[5].extend(SL)


    # 我(本节点) 从其他节点接收的报文list
    def add_RL_to_new_ER(self, node_j, runningtime, RL):
        print('{} pkts {}->{}'.format(len(RL), node_j, self.node_id))
        assert self.tmp_ER[1] == node_j and self.tmp_ER[4] == runningtime
        self.tmp_ER[6].extend(RL)
        pass

    def end_new_ER(self, sig_j):
        self.tmp_ER.extend([self.sig, sig_j])
        self.ER_sn = self.ER_sn + 1
        # 整合ER 保存到ERw里
        self.ER_List.append(self.tmp_ER.copy())
        if len(self.ER_List) > self.w:
            self.ER_List.pop(0)
        self.tmp_ER.clear()
        assert len(self.tmp_ER) == 0
        assert len(self.ER_List) <= self.w
        node_j = self.ER_List[-1][1]
        num_snd = len(self.ER_List[5])
        num_rcv = len(self.ER_List[6])
        print('add new ER, {} pkts:{}->{} and {} pkts:{}->{}'.format(num_snd, self.node_id, node_j,
                                                                     num_rcv, node_j, self.node_id))
        return

    def detect_node_j(self, j_node_id, ERW):
        is_blackhole = False
        dropping = False
        collusion = False
        RR, SFR = self.__get_RR_SFR_from_ERW(ERW)
        if RR < self.Th_RR:
            # 对应的TR值修改
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_gamma
            dropping = True
        if SFR > self.Th_SFR:
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_rho
            dropping = True
        # 下面判断collusion
        # 得到一维矩阵 各个node的FXS值
        tmp_collusion_id_list = []
        FXSs = self.__get_FXS_ERW(ERW)
        for k_id in range(np.size(FXSs)):
            # 如果是encountered节点 或者 本节点 则无需判断
            if k_id == j_node_id or k_id == self.node_id:
                continue
            if FXSs[k_id] > self.Th_FXS:
                # 从ERW中排除 node k, 建立 ERW_
                ERW_ = []
                for tmpER in ERW:
                    (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j, j_sig, k_sig) = tmpER
                    if k_node_id != k_id:
                        ERW_.insert(tmpER)
                RR_, SFR_ = self.__get_RR_SFR_from_ERW(ERW_)
                if RR_ < self.Th_RR:
                    # 对应的TR值修改, node j 和 node k 都要惩罚
                    self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_gamma
                    # 暂不处理 collusion
                    # self.MS[k_id, 3] = self.MS[k_id, 3] - self.Dec_gamma
                    collusion = True
                    tmp_collusion_id_list.append(k_id)
                if SFR_ > self.Th_SFR:
                    self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_rho
                    # 暂不处理 collusion
                    # self.MS[k_id, 3] = self.MS[k_id, 3] - self.Dec_rho
                    collusion = True
                    tmp_collusion_id_list.append(k_id)
        if dropping == False:
        # if dropping == False and collusion == False:
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] + self.Inc_lambda
            is_blackhole = True
        return is_blackhole, tmp_collusion_id_list

    def __check_order(self):
        # 检查来自相遇节点j传递而来的ERW_j是否具有一致性; 遵守sequence and timestamp order
        # 省略
        # 通过ERW_j交叉检查 另一个节点k的一致性
        # 省略
        pass

    # 处理对方的ER序列, 得到 RR SFR
    def __get_RR_SFR_from_ERW(self, ERW):
        N_RS = 0
        N_RNS = 0
        N_selfsend = 0
        N_send = 0
        # 记录有多少pkt在ERW里面没有被 传送出去
        tmpPktidList = []
        tmpPktList = []
        for tmpER in ERW:
            (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j, j_sig, k_sig) = tmpER
            # 接收到的
            for tmpRece in RL_j:
                (pkt_id, src_id, dst_id) = tmpRece
                # 记录报文接收到了
                tmpPktList.append((pkt_id, src_id, dst_id, timestamp, k_node_id))
                tmpPktidList.append(pkt_id)
            # 发送出去的
            for tmpSend in SL_j:
                (pkt_id, src_id, dst_id) = tmpSend
                if src_id == j_node_id:
                    N_selfsend = N_selfsend + 1
                if pkt_id in tmpPktidList:
                    idx = tmpPktidList.index(pkt_id)
                    oldtimestamp = tmpPktList[idx][3]
                    assert (oldtimestamp < timestamp)
                    N_RS = N_RS + 1
                    tmpPktidList.pop(idx)
                    tmpPktList.pop(idx)
            N_send = N_send + len(SL_j)
        assert (len(tmpPktList) == len(tmpPktidList))
        N_RNS = len(tmpPktList)
        RR = N_RS / N_RNS
        SFR = N_selfsend / N_send
        # return N_RS, N_RNS, N_selfsend, N_send, RR, SFR
        return RR, SFR

    # 解决collude 同谋攻击问题, 处理ER序列, 计算FXS的结果
    def __get_FXS_ERW(self, ERW):
        # self.numofnodes * 3 矩阵 记录; id, freq, send, FXS
        arr1 = np.arange(self.numofnodes).reshape(self.numofnodes,1)
        arr2 = np.zeros((self.numofnodes, 3), dtype='int')
        tmp_kcnt_mat = np.hstack([arr1, arr2])
        for tmpER in ERW:
            (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j, j_sig, k_sig) = tmpER
            tmp_kcnt_mat[k_node_id, 1] = tmp_kcnt_mat[k_node_id, 1] + 1
            tmp_kcnt_mat[k_node_id, 2] = tmp_kcnt_mat[k_node_id, 2] + len(SL_j)
        # 对应元素想乘 np.multiply()或 *
        tmp_kcnt_mat[:, 3] = tmp_kcnt_mat[:, 1] * tmp_kcnt_mat[:, 2]
        return tmp_kcnt_mat[:,3]
