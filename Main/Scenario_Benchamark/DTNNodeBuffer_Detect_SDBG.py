import numpy as np

class DTNNodeBuffer_Detect_SDBG(object):
    def __init__(self, node_id, numofnodes):
        # ID_i
        self.node_id = node_id
        self.numofnodes = numofnodes
        # sn_i
        self.ER_sn = 0
        # sig_i 模拟一下好了
        self.sig = 'sig_' + str(self.node_id)

        # 只在本地保存w个ER； presents them to the neighbors
        self.ER_List = []
        self.w = 100

        # 对其他剩余节点 都创建MS: (N-1)个MS
        # 100 * [j_node_id, latest_sn_j, latest_time_j, self.TR_init, 0(未设置)/1(设置)]
        arr1 = np.arange(self.numofnodes).reshape(self.numofnodes, 1)
        arr2 = np.zeros((self.numofnodes, 4), dtype='double')
        self.MS = np.hstack([arr1, arr2])

        # SDBG 的 阈值参数; 需要经验 选择;//从paper的图里 选择出来
        self.Th_RR = 3
        self.Th_SFR = 0.08

        # self.Dec_gamma = 0.2
        # self.Dec_rho = 0.2
        # self.Inc_lambda = 0.5

        self.Dec_gamma = 0.04
        self.Dec_rho = 0.09
        self.Inc_lambda = 0.06

        self.Th_FXS = 2400

        # TR的阈值 good mal; >good的 高优先级; <good且>mal的 低优先级; < mal 不会交换报文 只交流控制信息
        # self.Th_good = 2
        # self.Th_mal = -2
        self.Th_good = 0.8
        self.Th_mal = -1.0

        # ********************************************
        # 为了在一次contact中构建ER记录 需要把交换的报文都记录下来
        self.tmp_ER = {"self_id": -1, "partner_id": -1, "self_sn": -1, "partner_sn": -1, "running_time": -1,
                       "SL": [], "RL": []}

    # 刚刚和某个节点node j相遇 向node j发送i本地记录的ERW
    def get_local_ER_list(self):
        return self.ER_List.copy()

    def get_new_seq_for_ER(self):
        self.ER_sn = self.ER_sn + 1
        return self.ER_sn, self.sig

    # 对方的id号, 对方的sn, runnningtime
    def begin_new_ER(self, node_j_id, node_j_sn, runnningtime):
        # 这里是保证同时 ER已经被清空
        assert (self.tmp_ER["self_id"] == -1)
        self.tmp_ER["self_id"] = self.node_id
        self.tmp_ER["partner_id"] = node_j_id
        self.tmp_ER["self_sn"] = self.ER_sn
        self.tmp_ER["partner_sn"] = node_j_sn
        self.tmp_ER["running_time"] = runnningtime

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
        if len(RL) == 0:
            return
        assert self.tmp_ER["partner_id"] == node_j
        for (pkt_id, src_id, dst_id) in RL:
            self.tmp_ER["RL"].append((pkt_id, src_id, dst_id))

    def end_new_ER(self, sig_j):
        # if len(self.tmp_ER["SL"]) + len(self.tmp_ER["RL"]) > 0:
        #     print('{}-> <-{} pkts id_{}<->id_{}'.format(len(self.tmp_ER["SL"]), len(self.tmp_ER["RL"]),
        #                                                 self.node_id, self.tmp_ER["partner_id"]))
        new_ER = (self.tmp_ER["self_id"], self.tmp_ER["partner_id"], self.tmp_ER["self_sn"],
                  self.tmp_ER["partner_sn"], self.tmp_ER["running_time"],
                  self.tmp_ER["SL"].copy(), self.tmp_ER["RL"].copy())
        # 整合ER 保存到ERw里
        self.ER_List.append(new_ER)
        if len(self.ER_List) > self.w:
            self.ER_List.pop(0)

        self.tmp_ER["self_id"] = -1
        self.tmp_ER["partner_id"] = -1
        self.tmp_ER["self_sn"] = -1
        self.tmp_ER["partner_sn"] = -1
        self.tmp_ER["running_time"] = -1
        self.tmp_ER["SL"].clear()
        self.tmp_ER["RL"].clear()
        # if len(self.ER_List[-1][5]) + len(self.ER_List[-1][6]) > 0:
        #     print('{}-> <-{} pkts id_{}<->id_{}'.format(len(self.ER_List[-1][5]), len(self.ER_List[-1][6]),
        #                                                 self.ER_List[-1][0], self.ER_List[-1][1]))
        return

    def detect_node_j(self, j_node_id, ERW):
        is_blackhole = False
        dropping = False
        collusion = False
        N_RS, N_RNS, N_selfsend, N_send = self.__get_RR_SFR_from_ERW(ERW)
        # 为了解决除以0的问题
        RR = N_RS / (N_RNS + 1)
        SFR = N_selfsend / (N_send + 1)

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

        if self.MS[j_node_id, 3] < self.Th_mal:
            is_blackhole = True
        return is_blackhole, tmp_collusion_id_list

    # 简化方式
    def detect_node_jv2(self, j_node_id, ERW, i_isSelfish):
        do_not_record = False
        is_blackhole = False
        dropping = False
        collusion = False
        # print('node_{} {}'.format(j_node_id, ERW))

        # N_RS, N_RNS, N_selfsend, N_send = self.__get_RR_SFR_from_ERW(ERW)
        N_RS, N_RNS = self.__get_NRS_NRNS(ERW)
        N_selfsend, N_send = self.__get_Nssend_Nsend(ERW)
        if N_RS==0 or N_RNS==0 or N_selfsend==0 or N_send==0:
            do_not_record = True
        # 为了解决除以0的问题
        RR = N_RS / (N_RNS + 0.00001)
        SFR = N_selfsend / (N_send + 0.00001)

        if RR < self.Th_RR:
            # 对应的TR值修改
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_gamma
            dropping = True
        if SFR > self.Th_SFR:
            dropping = True
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] - self.Dec_rho

        if dropping == False:
            self.MS[j_node_id, 3] = self.MS[j_node_id, 3] + self.Inc_lambda

        if self.MS[j_node_id, 3] < self.Th_mal:
            is_blackhole = True

        # 有没有降低阈值的必要
        # if i_isSelfish != is_blackhole:
        #     if i_isSelfish == 0:
        #         print('\033[31m RR:{} SFR:{} value:{} isSelfish:{} is_Bk:{} ERW_len:{} \033[0m'.format(RR, SFR, self.MS[j_node_id, 3],
        #                                                                           i_isSelfish, is_blackhole, len(ERW)))
        #     elif i_isSelfish == 1:
        #         print('\033[34m RR:{} SFR:{} value:{} isSelfish:{} is_Bk:{} ERW_len:{} \033[0m'.format(RR, SFR, self.MS[j_node_id, 3],
        #                                                                           i_isSelfish, is_blackhole, len(ERW)))

        return is_blackhole, [], i_isSelfish, (RR, SFR), do_not_record

    def __check_order(self):
        # 检查来自相遇节点j传递而来的ERW_j是否具有一致性; 遵守sequence and timestamp order
        # 省略
        # 通过ERW_j交叉检查 另一个节点k的一致性
        # 省略
        pass

        # 处理对方的ER序列, 得到 RR SFR

    def __get_NRS_NRNS(self, ERW):
        N_RS = 0
        N_RNS = 0
        # 记录有多少pkt在ERW里面没有被 传送出去
        tmpPktRList = []
        tmpPktRidList = []
        tmpPktSList = []
        tmpPktSidList = []
        for tmpER in ERW:
            (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j) = tmpER
            # 接收到的
            for tmpRece in RL_j:
                (pkt_id, src_id, dst_id) = tmpRece
                # 记录报文接收到了
                tmpPktRList.append((pkt_id, src_id, dst_id, timestamp, k_node_id))
                # tmpPktRidList.append(pkt_id)
            for tmpSend in SL_j:
                (pkt_id, src_id, dst_id) = tmpSend
                # 记录报文发送出去
                tmpPktSList.append((pkt_id, src_id, dst_id, timestamp, k_node_id))
                # tmpPktSidList.append(pkt_id)
        N_RS = len(tmpPktRList)
        tmp = 0
        if len(tmpPktRList)!=0 and len(tmpPktSList)!=0:
            pass
        # 从所有接收的报文中 去除 已经发送出去的
        for pkt_R in tmpPktRList:
            (pkt_id, src_id, dst_id, timestamp, k_node_id) = pkt_R
            for pkt_S in tmpPktSList:
                (pkt_ids, src_ids, dst_ids, timestamps, k_node_ids) = pkt_S
                if ((pkt_id == pkt_ids) and  (timestamps>=timestamp)):
                    tmp = tmp + 1
                    break
        N_RNS = N_RS- tmp
        return N_RS, N_RNS

    def __get_Nssend_Nsend(self, ERW):
        N_selfsend = 0
        N_send = 0
        tmpPktSList = []
        for tmpER in ERW:
            (j_node_id, k_node_id, j_ER_sn, k_ER_sn, timestamp, SL_j, RL_j) = tmpER
            N_send = N_send + len(SL_j)
            for tmpSend in SL_j:
                (pkt_id, src_id, dst_id) = tmpSend
                if src_id == j_node_id:
                    N_selfsend = N_selfsend + 1
        return N_selfsend, N_send

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
                    if oldtimestamp < timestamp:
                        # 如果之前接收 现在发送; 那么就属于已经转出的情况
                        N_RS = N_RS + 1
                        tmpPktidList.pop(idx)
                        tmpPktList.pop(idx)
                    else:
                        # 否则 可能是 重复把报文放进来了
                        print('send again!!! old_recv:{} now_snd:{}'.format(tmpPktList[idx], tmpSend))
            N_send = N_send + len(SL_j)
        assert (len(tmpPktList) == len(tmpPktidList))
        N_RNS = len(tmpPktList)
        return N_RS, N_RNS, N_selfsend, N_send

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
