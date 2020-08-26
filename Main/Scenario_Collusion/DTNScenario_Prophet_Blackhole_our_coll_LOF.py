import datetime
import copy
import numpy as np
import math

from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt
from Main.DTNNodeBuffer_Detect import DTNNodeBuffer_Detect
from Main.Scenario_Collusion.DetectProcessManager import DetectProcessManager

# NUM_of_DIMENSIONS = 10
# NUM_of_DIRECT_INPUTS = 8
# NUM_of_INDIRECT_INPUTS = 9
MAX_RUNNING_TIMES = 864000


# 使用训练好的model 在消息投递时候 增加对对端节点的判定
# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_Prophet_Blackhole_our_coll_LOF(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, list_selfish, new_normal_indices, new_coll_indices, coll_pairs, num_of_nodes, buffer_size, total_runningtime):
        # tf的调用次数
        self._tmpCallCnt = 0
        self.scenarioname = scenarioname
        self.list_selfish = list_selfish
        self.list_normal = new_normal_indices

        # 所有colluded节点
        self.list_coll = new_coll_indices
        # collusion对
        self.list_coll_pairs = coll_pairs
        # colluded节点对应的bk节点
        self.list_coll_corres_bk = []
        for ele in self.list_coll_pairs:
            (coll_node_id, bk_node_id) = ele
            self.list_coll_corres_bk.append(bk_node_id)

        self.num_of_nodes = num_of_nodes
        # 为各个node建立虚拟空间 <buffer+router>
        self.listNodeBuffer = []
        self.listRouter = []
        # 为了打印 获得临时的分类结果 以便观察分类结果; 从1到9(从0.1到0.9) 最后一个time block
        self.index_time_block = 1
        self.MAX_RUNNING_TIMES = total_runningtime
        # 为各个node建立检测用的 证据存储空间 BufferDetect
        self.listNodeBufferDetect = []
        print(self.scenarioname, self.list_coll_pairs, self.list_coll, self.list_coll_corres_bk)
        print(len(self.list_normal), len(self.list_selfish))
        for node_id in range(num_of_nodes):
            if node_id in self.list_selfish:
                tmpRouter = RoutingBlackhole(node_id, num_of_nodes)
            else:
                # 其中包含collusion节点
                tmpRouter = RoutingProphet(node_id, num_of_nodes)
            self.listRouter.append(tmpRouter)
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size)
            self.listNodeBuffer.append(tmpBuffer)
            tmpBuffer_Detect = DTNNodeBuffer_Detect(node_id, num_of_nodes)
            self.listNodeBufferDetect.append(tmpBuffer_Detect)

        # 加载训练好的模型 load the trained model (d_eve and ind_eve as input)
        self.pm = DetectProcessManager()

        # 保存真正使用的结果: self.DetectResult[0,1] False_Positive ; self.DetectResult[1,0] False_Negative
        self.DetectResult = np.zeros((2,2),dtype='int')
        # tmp 临时结果
        self.tmp0_DetectResult = np.zeros((2, 2), dtype='int')
        self.tmp_DetectResult = np.zeros((2, 20), dtype='int')
        # 矩阵属性可以考虑更改
        self.num_of_att = 10

        self.num_comm = 0
        
        # 记录不平衡的个数
        self.count_unb = 0
        self.forge_value = 0.1
        # 记录collusion检测的评价结果 并 用list记录下来/带上时间；
        # 合作的bk
        self.coll_corr_bk_sum_evalu = 0.0
        self.coll_corr_bk_num_evalu = 0
        self.coll_corr_bk_recd_list = []
        # 没合作的bk
        self.bk_sum_evalu = 0.0
        self.bk_num_evalu = 0
        self.bk_recd_list = []
        # colluded节点
        self.coll_sum_evalu = 0.0
        self.coll_num_evalu = 0
        self.coll_recd_list = []
        # normal节点
        self.normal_sum_evalu = 0.0
        self.normal_num_evalu = 0
        self.normal_recd_list = []

        # 结果保存到文件中

        # 记录检测的精确程度
        self.coll_DetectRes = np.zeros((2, 2), dtype='int')
        # tmp 临时结果
        self.tmp0_coll_DetectResult = np.zeros((2, 2), dtype='int')
        self.tmp_coll_DetectResult = np.zeros((2, 20), dtype='int')

        #
        self.bk_no_detect = 0
        self.sel_no_detect = 0
        return

    # tmp_ 保存时间线上状态; 事态的发展会保证，self.index_time_block 必然不会大于10
    def __update_tmp_conf_matrix(self, gentime, isEndoftime):
        assert(self.index_time_block <= 10)
        if (isEndoftime == True) or (gentime >= 0.1 * self.index_time_block * self.MAX_RUNNING_TIMES):
            index = self.index_time_block - 1
            # 对bk的检测
            tmp_ = self.DetectResult - self.tmp0_DetectResult
            self.tmp_DetectResult[:, index * 2 : index*2+2] = tmp_
            self.tmp0_DetectResult = self.DetectResult.copy()

            # 对coll的检测
            tmp_ = self.coll_DetectRes - self.tmp0_coll_DetectResult
            self.tmp_coll_DetectResult[:, index * 2: index * 2 + 2] = tmp_
            self.tmp0_coll_DetectResult = self.coll_DetectRes.copy()

            self.index_time_block = self.index_time_block + 1
        return

    def __print_tmp_conf_matrix(self):

        output_str = '{}_tmp_state\n'.format(self.scenarioname)
        # self.DetectResult self.DetectdEve self.DetectindEve
        output_str += 'self.tmp_DetectResult:\n{}\n'.format(self.tmp_DetectResult)
        return output_str

    def __print_collusion(self):
        short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = '' + short_time + '.npz'
        tmp_num_pairs = len(self.list_coll_pairs)
        tmp_ratio_bk = len(self.list_selfish) / (len(self.list_normal) + len(self.list_coll) + len(self.list_selfish))
        # # 结果保存到文件中
        self.collfilter_recd_path = "..\\collfilter_"+short_time+"_pair"+str(tmp_num_pairs)+"_ratio_0_"+str(int(10*tmp_ratio_bk))+".npz"
        np.savez(self.collfilter_recd_path, coll_corr_bk_recd=self.coll_corr_bk_recd_list,
                 bk_recd=self.bk_recd_list, coll_recd=self.coll_recd_list, normal_recd=self.normal_recd_list,
                 num_paris=tmp_num_pairs, ratio_bk=tmp_ratio_bk)

        coll_corr_bk_eva = self.coll_corr_bk_sum_evalu / self.coll_corr_bk_num_evalu
        bk_eva = self.bk_sum_evalu / self.bk_num_evalu

        coll_eva = self.coll_sum_evalu / self.coll_num_evalu

        normal_eva = self.normal_sum_evalu / self.normal_num_evalu

        output_str = '{}_collusion_state\n'.format(self.scenarioname)
        output_str += 'coll_corr_bk_eva:{}\nbk_eva:{}\ncoll_eva:{}\nnormal_eva:{}\n'.format(coll_corr_bk_eva, bk_eva, coll_eva, normal_eva)
        output_str += 'coll_DetectRes:\n{}\n'.format(self.coll_DetectRes)
        output_str += 'no detect count: coll_bk:{} sel:{}\n'.format(self.bk_no_detect, self.sel_no_detect)
        output_str += '{}_tmp_coll_state\n'.format(self.scenarioname)
        output_str += 'self.tmp_coll_DetectResult:\n{}\n'.format(self.tmp_coll_DetectResult)
        return output_str

    def print_res(self, listgenpkt):
        # end of time; 最后一次刷新
        self.__update_tmp_conf_matrix(-1, True)

        output_str_whole = self.__print_res_whole(listgenpkt)
        output_str_pure, succ_ratio, avg_delay, num_comm = self.__print_res_pure(listgenpkt)
        # 打印混淆矩阵
        output_str_state = self.__print_conf_matrix()
        output_str_tmp_state = self.__print_tmp_conf_matrix()
        output_str_coll = self.__print_collusion()
        print(output_str_whole + output_str_pure + output_str_state + output_str_tmp_state + output_str_coll)
        # 不必进行标签值 和 属性值 的保存
        # self.print_eve_res()

        # 使得预测进程终止
        self.pm.close_process()

        outstr = output_str_whole + output_str_pure + output_str_state + output_str_tmp_state + output_str_coll
        res = {'succ_ratio': succ_ratio, 'avg_delay': avg_delay, 'num_comm': num_comm,
               'DetectResult':self.DetectResult, 'tmp_DetectResult':self.tmp_DetectResult}
        ratio_bk_nodes = len(self.list_selfish) / self.num_of_nodes
        config = {'ratio_bk_nodes': ratio_bk_nodes, 'drop_prob': 1}
        return outstr, res, config

    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        self.__update_tmp_conf_matrix(gentime, False)
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.listNodeBuffer[src_id].gennewpkt(newpkt)
        return

    # routing接到指令aid和bid相遇，开始进行消息交换 a_id <-> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # 交换直接评价信息，更新间接评价
        a_send = self.listNodeBufferDetect[a_id].get_send_values()
        a_receive = self.listNodeBufferDetect[a_id].get_receive_values()
        a_send_all = self.listNodeBufferDetect[a_id].get_send_all()
        a_receive_all = self.listNodeBufferDetect[a_id].get_receive_all()
        a_receive_src = self.listNodeBufferDetect[a_id].get_receive_src_values()
        a_receive_dst = self.listNodeBufferDetect[a_id].get_receive_dst_values()
        a_send_src = self.listNodeBufferDetect[a_id].get_send_src_values()
        a_send_dst = self.listNodeBufferDetect[a_id].get_send_dst_values()
        a_receive_from_and_src = self.listNodeBufferDetect[a_id].get_receive_from_and_pktsrc()

        b_send = self.listNodeBufferDetect[b_id].get_send_values()
        b_receive = self.listNodeBufferDetect[b_id].get_receive_values()
        b_send_all = self.listNodeBufferDetect[b_id].get_send_all()
        b_receive_all = self.listNodeBufferDetect[b_id].get_receive_all()
        b_receive_src = self.listNodeBufferDetect[b_id].get_receive_src_values()
        b_receive_dst = self.listNodeBufferDetect[b_id].get_receive_dst_values()
        b_send_src = self.listNodeBufferDetect[b_id].get_send_src_values()
        b_send_dst = self.listNodeBufferDetect[b_id].get_send_dst_values()
        b_receive_from_and_src = self.listNodeBufferDetect[b_id].get_receive_from_and_pktsrc()

        self.listNodeBufferDetect[b_id].renewindeve(runningtime, a_id, a_send, a_receive, a_send_all, a_receive_all,
                                                    a_receive_src, a_receive_dst, a_send_src, a_send_dst,
                                                    a_receive_from_and_src)
        self.listNodeBufferDetect[a_id].renewindeve(runningtime, b_id, b_send, b_receive, b_send_all, b_receive_all,
                                                    b_receive_src, b_receive_dst, b_send_src, b_send_dst,
                                                    b_receive_from_and_src)
        bool_BH_a_wch_b = self.__detect_blackhole(a_id, b_id, runningtime)
        bool_BH_b_wch_a = self.__detect_blackhole(b_id, a_id, runningtime)
        # 如果有一方不同意 则停止
        if bool_BH_a_wch_b or bool_BH_b_wch_a:
            return

        # ================== 控制信息 交换==========================
        # 对称操作!!!
        # 获取 b_node Router 向各节点的值(带有老化计算)
        P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        P_a_any = self.listRouter[a_id].get_values_before_up(runningtime)
        # 根据 b_node Router 保存的值, a_node更新向各其他node传递值 (带有a-b响应本次相遇的更新)
        self.listRouter[a_id].notifylinkup(runningtime, b_id, P_b_any)
        self.listRouter[b_id].notifylinkup(runningtime, a_id, P_a_any)
        if isinstance(self.listRouter[a_id], RoutingBlackhole) and isinstance(self.listRouter[b_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是blackhole b_id是blackhole==========================
            self.__sendpkt_toblackhole(runningtime, a_id, b_id)
            self.__sendpkt_toblackhole(runningtime, b_id, a_id)
        elif isinstance(self.listRouter[a_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是blackhole b_id是正常prophet==========================
            self.__sendpkt(runningtime, a_id, b_id)
            self.__sendpkt_toblackhole(runningtime, b_id, a_id)
        elif isinstance(self.listRouter[b_id], RoutingBlackhole):
            # ================== 报文 交换; a_id是正常prophet b_id是blackhole==========================
            self.__sendpkt_toblackhole(runningtime, a_id, b_id)
            self.__sendpkt(runningtime, b_id, a_id)
        elif (not isinstance(self.listRouter[a_id], RoutingBlackhole)) and (not isinstance(self.listRouter[b_id], RoutingBlackhole)):
            # ================== 报文 交换==========================
            self.__sendpkt(runningtime, a_id, b_id)
            self.__sendpkt(runningtime, b_id, a_id)

    # 报文发送 a_id -> b_id
    def __sendpkt_toblackhole(self, runningtime, a_id, b_id):
        P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        P_a_any = self.listRouter[a_id].get_values_before_up(runningtime)
        # 准备从a到b传输的pkt 组成的list<这里保存的是deepcopy>
        totran_pktlist = []
        # b_listpkt_hist = self.listNodeBuffer[b_id].getlistpkt_hist()
        # a_listpkt_hist = self.listNodeBuffer[a_id].getlistpkt_hist()
        b_listpkt_hist = []
        a_listpkt_hist = []
        # 1) b_id 告诉 a_id: b_id有哪些pkt
        b_listpkt = self.listNodeBuffer[b_id].getlistpkt()
        a_listpkt = self.listNodeBuffer[a_id].getlistpkt()
        # hist列表 和 当前内存里都没有 来自a的pkt   a才有必要传输
        for a_pkt in a_listpkt:
            isDuplicateExist = False
            for bpktid_hist in b_listpkt_hist:
                if a_pkt.pkt_id == bpktid_hist:
                    isDuplicateExist = True
                    break
            if not isDuplicateExist:
                for bpkt in b_listpkt:
                    if a_pkt.pkt_id == bpkt.pkt_id:
                        isDuplicateExist = True
                        break
            if not isDuplicateExist:
                cppkt = copy.deepcopy(a_pkt)
                if a_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, cppkt)
                totran_pktlist.append(cppkt)
                break
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
                self.num_comm = self.num_comm + 1
            elif P_a_any[tmp_pkt.dst_id] < P_b_any[tmp_pkt.dst_id]:
                # # 利用model进行判定 b_id是否是blackhole
                # bool_BH = self.__detect_blackhole(a_id, b_id, runningtime)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)

                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
                # blackhole b_id立刻发动
                self.listNodeBuffer[b_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.num_comm = self.num_comm + 1

    # 报文发送 a_id -> b_id
    def __sendpkt(self, runningtime, a_id, b_id):
        P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        P_a_any = self.listRouter[a_id].get_values_before_up(runningtime)
        # 准备从a到b传输的pkt 组成的list<这里保存的是deepcopy>
        totran_pktlist = []
        # b_listpkt_hist = self.listNodeBuffer[b_id].getlistpkt_hist()
        # a_listpkt_hist = self.listNodeBuffer[a_id].getlistpkt_hist()
        b_listpkt_hist = []
        a_listpkt_hist = []
        # 1) b_id 告诉 a_id: b_id有哪些pkt
        b_listpkt = self.listNodeBuffer[b_id].getlistpkt()
        a_listpkt = self.listNodeBuffer[a_id].getlistpkt()
        # hist列表 和 当前内存里都没有 来自a的pkt   a才有必要传输
        for a_pkt in a_listpkt:
            isDuplicateExist = False
            for bpktid_hist in b_listpkt_hist:
                if a_pkt.pkt_id == bpktid_hist:
                    isDuplicateExist = True
                    break
            if not isDuplicateExist:
                for bpkt in b_listpkt:
                    if a_pkt.pkt_id == bpkt.pkt_id:
                        isDuplicateExist = True
                        break
            if not isDuplicateExist:
                cppkt = copy.deepcopy(a_pkt)
                if a_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, cppkt)
                totran_pktlist.append(cppkt)
                break
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
                self.num_comm = self.num_comm + 1
            elif P_a_any[tmp_pkt.dst_id] < P_b_any[tmp_pkt.dst_id]:
                # # 利用model进行判定 b_id是否是blackhole
                # bool_BH = self.__detect_blackhole(a_id, b_id, runningtime)
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)

                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.__updatedectbuf_sendpkt(a_id, b_id, tmp_pkt.src_id, tmp_pkt.dst_id)
                self.num_comm = self.num_comm + 1

    def __detect_blackhole(self, a_id, b_id, runningtime):
        theBufferDetect = self.listNodeBufferDetect[a_id]
        # a和b相遇 来自a的是直接证据
        # i提供给a一些证据 作为间接证据

        d_attrs = np.zeros((1, 1 * self.num_of_att), dtype='int')
        d_attrs[0][0] = runningtime
        d_attrs[0][1] = ((theBufferDetect.get_send_values())[b_id]).copy()
        d_attrs[0][2] = ((theBufferDetect.get_receive_values())[b_id]).copy()
        d_attrs[0][3] = ((theBufferDetect.get_receive_src_values())[b_id]).copy()
        d_attrs[0][4] = ((theBufferDetect.get_receive_dst_values())[b_id]).copy()
        d_attrs[0][5] = ((theBufferDetect.get_send_src_values())[b_id]).copy()
        d_attrs[0][6] = ((theBufferDetect.get_send_dst_values())[b_id]).copy()
        d_attrs[0][7] = theBufferDetect.get_send_all().copy()
        d_attrs[0][8] = theBufferDetect.get_receive_all().copy()
        d_attrs[0][9] = ((theBufferDetect.get_receive_from_and_pktsrc())[b_id]).copy()

        mask = [True] * self.num_of_nodes
        mask[a_id] = False
        mask[b_id] = False
        n = self.num_of_nodes - 2
        ind_attrs = np.zeros((1, n * self.num_of_att), dtype='int')
        # 来自各个节点评价b_id; (1)作为间接证据 除去a_id的观察 (2)去除嫌疑 去掉b_id的观察
        tmp_send = (theBufferDetect.get_ind_send_values())[:, b_id].transpose().copy()
        tmp_receive = (theBufferDetect.get_ind_receive_values())[:, b_id].transpose().copy()
        tmp_receive_src = (theBufferDetect.get_ind_receive_src_values())[:, b_id].transpose().copy()
        tmp_receive_dst = (theBufferDetect.get_ind_receive_dst_values())[:, b_id].transpose().copy()
        tmp_send_src = (theBufferDetect.get_ind_send_src_values())[:, b_id].transpose().copy()
        tmp_send_dst = (theBufferDetect.get_ind_send_dst_values())[:, b_id].transpose().copy()

        tmp_time = (theBufferDetect.get_ind_time())[b_id].transpose().copy()
        tmp_send_all = (theBufferDetect.get_ind_send_all())[b_id].transpose().copy()
        tmp_receive_all = (theBufferDetect.get_ind_receive_all())[b_id].transpose().copy()
        tmp_receive_from_and_pktsrc = (theBufferDetect.get_ind_receive_from_and_pktsrc())[:, b_id].transpose().copy()

        ind_attrs[0][0: n] = tmp_time
        ind_attrs[0][n: 2 * n] = tmp_send[mask]
        ind_attrs[0][2 * n: 3 * n] = tmp_receive[mask]
        ind_attrs[0][3 * n: 4 * n] = tmp_receive_src[mask]
        ind_attrs[0][4 * n: 5 * n] = tmp_receive_dst[mask]
        ind_attrs[0][5 * n: 6 * n] = tmp_send_src[mask]
        ind_attrs[0][6 * n: 7 * n] = tmp_send_dst[mask]

        ind_attrs[0][7 * n: 8 * n] = tmp_send_all
        ind_attrs[0][8 * n: 9 * n] = tmp_receive_all
        ind_attrs[0][9 * n: 10 * n] = tmp_receive_from_and_pktsrc[mask]

        new_x = np.hstack((d_attrs, ind_attrs))
        # tf的调用次数 加1
        self._tmpCallCnt = self._tmpCallCnt + 1

        i_isSelfish = int(b_id in self.list_selfish)

        to_collusion_index = np.arange(self.num_of_nodes)
        to_collusion_index = to_collusion_index[mask]
        to_collusion_index = to_collusion_index.reshape((-1, self.num_of_nodes-2))
        # to_collusion_index.reshape((-1, len(to_collusion_index)))

        # 加载模型；进行预测
        result_element = self.pm.request(new_x.copy(), i_isSelfish)

        # boolBlackhole = result_element[1]
        # conf_matrix = result_element[2]
        d_predict = result_element[1]
        ind_predict = result_element[2]

        # 如果一切正常 应该取得的值
        tmp_res = np.hstack((d_predict, ind_predict))
        before_res = np.sum(tmp_res, axis=1) / tmp_res.shape[1]
        threshold = 0.5

        # 进行collusion
        true_collude_id, ind_predict = self.__collusion(a_id, b_id, to_collusion_index, ind_predict)

        # 默认不足以撼动结果
        tmp_res = np.hstack((d_predict, ind_predict))
        coll_res = np.sum(tmp_res, axis=1) / tmp_res.shape[1]
        bool_black_hole = coll_res > threshold

        # if bool_black_hole:
        # LOF方法
        possible_coll_id, new_ind_predict, LOF_list, target = self.__detect_collusion_LOF(
            ind_predict, to_collusion_index, threshold)
        tmp_res = np.hstack((d_predict, new_ind_predict))
        final_res = np.sum(tmp_res, axis=1) / tmp_res.shape[1]
        # print(tmp_res, final_res, tmp_res.shape[0], tmp_res.shape[1])
        bool_black_hole = final_res > threshold

        # 显示/统计 collusion detection的效果
        self.evaluate_coll_detection(a_id, b_id, bool_black_hole, true_collude_id, possible_coll_id,
                                     before_res, coll_res, final_res, runningtime,
                                     LOF_list)

        conf_matrix = self.pm.cal_conf_matrix(i_isSelfish, int(bool_black_hole), num_classes=2)
        if i_isSelfish != int(bool_black_hole):
            # print("\033[42m", "a_id(id_{}) to b_id(id_{}) real:{} predict:{} value:{}".format(
            # a_id, b_id, i_isSelfish, boolBlackhole, final_res), "\033[0m")
            pass
        self.DetectResult = self.DetectResult + conf_matrix
        return bool_black_hole

    def evaluate_coll_detection(self, a_id, b_id, bool_black_hole, true_collude_id, possible_coll_id,
                                before_res, coll_res, final_res, runningtime, LOF_list):
        # 只从正常节点的角度观察;  a_id是正常节点 且 b_id被检测为blackhole
        if a_id in self.list_normal:
            if b_id in self.list_coll_corres_bk:
                # b_id是合作的bk节点 记录下评价; coll以后评价有没有提高
                self.coll_corr_bk_sum_evalu = self.coll_corr_bk_sum_evalu + coll_res
                self.coll_corr_bk_num_evalu = self.coll_corr_bk_num_evalu + 1
                self.coll_corr_bk_recd_list.append((coll_res, runningtime))
            elif b_id in self.list_selfish:
                # b_id是普通的bk节点 (没有colluded节点与b_id合作)
                self.bk_sum_evalu = self.bk_sum_evalu + coll_res
                self.bk_num_evalu = self.bk_num_evalu + 1
                self.bk_recd_list.append((coll_res, runningtime))
            elif b_id in self.list_coll:
                self.coll_sum_evalu = self.coll_sum_evalu + coll_res
                self.coll_num_evalu = self.coll_num_evalu + 1
                self.coll_recd_list.append((coll_res, runningtime))
            elif b_id in self.list_normal:
                self.normal_sum_evalu = self.normal_sum_evalu + coll_res
                self.normal_num_evalu = self.normal_num_evalu + 1
                self.normal_recd_list.append((coll_res, runningtime))
            else:
                print('Internal Err! CollusionF calculate res!')

        if a_id in self.list_normal:
            if b_id in self.list_coll_corres_bk:
                print('{}to{} before_res:{} coll_res:{} final_res:{}'.format(a_id, b_id, before_res, coll_res, final_res))

        # b: 1)selfish(coll_bk) 2)normal 3)coll
        # 只从正常节点的角度观察;  a_id是正常节点 且 b_id被检测为blackhole; 没检查出来 就先不打印了
        if a_id in self.list_normal:
            if (b_id in self.list_selfish) and (not bool_black_hole):
                if b_id in self.list_coll_corres_bk:
                    print('LOF hindden successfully')
                    self.bk_no_detect += 1
                else:
                    print('LOF no detect successfully')
                    self.sel_no_detect += 1
            elif (b_id in self.list_selfish) and bool_black_hole:
                # 看看检测出来的准不准
                tmp = np.zeros((2, 2), dtype='int')
                if (possible_coll_id, b_id) in self.list_coll_pairs:
                    assert true_collude_id == possible_coll_id
                    tmp[0][0] = 1
                    print("\033[42m",
                          "LOF value:{} b_id(id_{})找到了collude(id_{}) LOF_0:{}".format(final_res, b_id, true_collude_id,
                                                                                 LOF_list[0][0]),
                          "\033[0m", runningtime)
                elif b_id in self.list_coll_corres_bk:
                    # b_id是coll_corres_bk 存在的对应的colluded节点; 发生漏检
                    # assert res_coll_id !=
                    tmp[0][1] = 1
                    print("\033[41m",
                          "LOF value:{} a_id(id_{}) to b_id(id_{})有collude(id_{})但没被找到, 以为是id_{} LOF_0:{}".format(
                              final_res, a_id, b_id, true_collude_id, possible_coll_id, LOF_list[0][0]), "\033[0m time:", runningtime, LOF_list)
                elif possible_coll_id != -1:
                    # b_id也不是coll_bk; 也没有正确发现; 但还是以为有coll_id
                    # 误报 真实为‘1’误以为‘0’
                    assert b_id not in self.list_coll_corres_bk
                    tmp[1][0] = 1
                    print("\033[44m", "LOF value:{} a_id(id_{}) to b_id(id_{})没有collude但给出了，以为是id_{} LOF_0:{}".format(
                        final_res, a_id, b_id, possible_coll_id, LOF_list[0][0]), "\033[0m time:", runningtime, LOF_list)
                else:
                    tmp[1][1] = 1
                self.coll_DetectRes = self.coll_DetectRes + tmp
                print(self.coll_DetectRes)

    def __collusion(self, a_id, b_id, to_collusion_index, ind_predict):
        # 假设 collude节点 经过 多次尝试 使得神经网络能够给出想要的结果
        true_collude_id = -1
        # 只蒙蔽 good node (a_id)
        if (a_id in self.list_normal) and (b_id in self.list_coll_corres_bk):
            for ele in self.list_coll_pairs:
                if ele[1] == b_id:
                    true_collude_id = ele[0]
                    break
            # 已经是bk 不能 就还找不到对应的coll
            assert true_collude_id != -1
            # 各个评判节点 对应的node_id
            tmp_w = to_collusion_index.squeeze(axis=0)
            tmp_w = tmp_w.tolist()
            if true_collude_id in tmp_w:
                assert (true_collude_id, b_id) in self.list_coll_pairs
                target_loc = tmp_w.index(true_collude_id)
                # collude节点 伪造 好证据
                target_value = np.random.random() * self.forge_value
                ind_predict[0, target_loc] = target_value
                print('{}to{} change{} loc[{}] {}'.format(a_id, b_id, true_collude_id, target_loc, target_value))
            elif true_collude_id == a_id:
                print('Internal Err! --001')
                exit(1)
            elif true_collude_id == b_id:
                print('Internal Err! --003')
                exit(1)
            else:
                # b_id是coll_res_bk 但是pair里给出的true_coll 却没有提供信息
                print('Internal Err! --002')
                exit(1)
        return true_collude_id, ind_predict

    # collusion filtering; 返回 corrupted node对应的id 和 filtering后的ind_predict
    def __detect_collusion_LOF(self, ind_predict, to_collusion_index, threshold):
        # 预定义参数 最小的本地邻居个数 k=ng-1  k=(n-1)/2
        k = 20
        # o_i
        ind_predict_all = np.squeeze(ind_predict, axis=0)
        # 对应的id
        to_index_all = np.squeeze(to_collusion_index, axis=0)

        # for 每一个节点oi
        # 1.寻找最近的k个邻居 k邻距离d_k(oi) 邻居集合N_k(oi)
        # 2.Rd_k(oi <- ol) <- max(dk(ol),d(oi,ol),epison)
        # 3.LRD 本地可达密度 lrd_k(oi) <- |N_k(oi)| / sum( Rd_k(oj <- oi) )  //i邻域内的各个j
        # 4.LOF LOF_k(oi) <- sum( lrd_k(oj)/lrd_k(oi) ) / |N_k(oi)|        //i邻域内的各个j

        # rd(*) // p1,o // p2,o
        # rd(p) = |N(p)| / sum(rd(p,o)) //o是p邻域内各点
        # LOF(p) = sum(rd(o)/rd(p)) / |N(p)|
        assert ind_predict_all.shape == to_index_all.shape
        num_data = ind_predict_all.shape[0]

        # 所有的 index nodeid valueP Nk dk; Nk是邻域 里面包含(邻域距离,index,nodeid,value)
        list_oiNkdk = []
        # 1.寻找邻域d_k 和 N_k
        for oi in range(num_data):
            # oi对应的Nk和dk
            oi_ng_list_Nk = []
            oi_ng_tmp_dk = 0.
            tmp_list = []
            for tmp_oj in range(num_data):
                if tmp_oj == oi:
                    continue
                # d, index, 真正的id, 值
                ele = (math.fabs(ind_predict_all[oi] - ind_predict_all[tmp_oj]), tmp_oj,
                       to_index_all[tmp_oj], ind_predict_all[tmp_oj])
                tmp_list.append(ele)
            tmp_list.sort()
            oi_ng_tmp_dk = tmp_list[k-1][0]
            for ele in tmp_list:
                (dist, index, node_id, nodeValueP) = ele
                if dist <= oi_ng_tmp_dk:
                    oi_ng_list_Nk.append((dist, index, node_id, nodeValueP))
                else:
                    break
            oi_ele = (oi, to_index_all[oi], ind_predict_all[oi], oi_ng_list_Nk, oi_ng_tmp_dk)
            list_oiNkdk.append(oi_ele)
        # 2.rd矩阵 ol到oi的可达距离 Rd[oi, ol]
        Rd = np.ones((num_data, num_data), dtype='float')*-1
        for ol in range(num_data):
            for oi in range(num_data):
                assert list_oiNkdk[ol][0] == ol
                tmp_max_value = list_oiNkdk[ol][4]
                tmp_dist = math.fabs(ind_predict_all[ol] - ind_predict_all[oi])
                if tmp_dist > tmp_max_value:
                    tmp_max_value = tmp_dist
                if 0.000001 > tmp_max_value:
                    tmp_max_value = 0.000001
                # ol的dk   ol与oi距离  中间比较大的
                Rd[oi, ol] = tmp_max_value
        # 3.lrd // index lrd node_id
        lrd_list = []
        for p in range(num_data):
            assert list_oiNkdk[p][0] == p
            numerator = len(list_oiNkdk[p][3])
            denominator = 0.
            # 枚举p的邻域
            for o_ele in list_oiNkdk[p][3]:
                (dist_po, index_o, node_id_o, value_o) = o_ele
                denominator = denominator + Rd[p, index_o]
            lrd_list.append((p, numerator/denominator, to_index_all[p]))
        # 4.LOF
        LOF_list = []
        for p in range(num_data):
            assert list_oiNkdk[p][0] == p
            numerator = 0.
            denominator = len(list_oiNkdk[p][3])
            for o_ele in list_oiNkdk[p][3]:
                (dist_po, index_o, node_id_o, value_o) = o_ele
                assert lrd_list[index_o][0] == index_o
                assert lrd_list[p][0] == p
                numerator = numerator + lrd_list[index_o][1] / lrd_list[p][1]
            LOF_list.append((numerator/denominator, p, to_index_all[p], ind_predict_all[p]))
        # LOF值 描述了 节点多大可能是outlier
        LOF_list.sort(reverse=True)
        # print('LOF_list:', LOF_list[0], LOF_list[1], LOF_list[2], LOF_list[3])

        # possible_coll_id = LOF_list[0][2]
        # possible_coll_index = LOF_list[0][1]
        # if ind_predict_all[possible_coll_index] < threshold:
        #     mask = [True] * num_data
        #     # coll_node_id的位置 为false
        #     mask[possible_coll_index] = False
        #     new_ind_predict = ind_predict_all[mask].reshape(1, -1)
        # else:
        #     possible_coll_id = -1
        #     new_ind_predict = to_collusion_index

        # 在前 kk LOF中寻找  pred_value要足够小
        kk = 10
        diff_value = 0.15
        isFound = False
        LOF_th = 300000
        for i in range(kk):
            pred_value = LOF_list[i][3]
            LOF_value = LOF_list[i][0]
            if pred_value < threshold - diff_value and LOF_value > LOF_th:
                isFound = True
                possible_coll_index = LOF_list[i][1]
                possible_coll_id = LOF_list[i][2]
                mask = [True] * num_data
                # coll_node_id的位置 为false
                mask[possible_coll_index] = False
                new_ind_predict = ind_predict_all[mask].reshape(1, -1)
                # print(LOF_list[i])
                target = LOF_list[i]
                break
        if not isFound:
            possible_coll_id = -1
            new_ind_predict = ind_predict
            target = ()
        return possible_coll_id, new_ind_predict, LOF_list, target


    # 改变检测buffer的值
    def __updatedectbuf_sendpkt(self, a_id, b_id, pkt_src_id, pkt_dst_id):
        self.listNodeBufferDetect[a_id].send_to_b(b_id)
        self.listNodeBufferDetect[a_id].send_to_pkt_src(pkt_src_id)
        self.listNodeBufferDetect[a_id].send_to_pkt_dst(pkt_dst_id)

        self.listNodeBufferDetect[b_id].receive_from_a(a_id)
        self.listNodeBufferDetect[b_id].receive_from_pkt_src(pkt_src_id)
        self.listNodeBufferDetect[b_id].receive_from_pkt_dst(pkt_dst_id)

        if a_id == pkt_src_id:
            self.listNodeBufferDetect[b_id].receive_from_and_pktsrc(a_id, pkt_src_id)

    def __print_conf_matrix(self):
        output_str = '{}_state\n'.format(self.scenarioname)
        output_str += 'self.list_selfish:\{}\n'.format(self.list_selfish)
        output_str += 'self.DetectResult:\n{}\n'.format(self.DetectResult)
        return output_str

    def __print_res_whole(self, listgenpkt):
        num_genpkt = len(listgenpkt)
        output_str = '{}_whole\n'.format(self.scenarioname)
        total_delay = 0
        total_succnum = 0
        total_pkt_hold = 0
        for i_id in range(len(self.listNodeBuffer)):
            list_succ = self.listNodeBuffer[i_id].getlistpkt_succ()
            tmp_succnum = 0
            for i_pkt in list_succ:
                tmp_delay = i_pkt.succ_time - i_pkt.gentime
                total_delay = total_delay + tmp_delay
                tmp_succnum = tmp_succnum + 1
            assert (tmp_succnum == len(list_succ))
            total_succnum = total_succnum + tmp_succnum

            list_pkt = self.listNodeBuffer[i_id].getlistpkt()
            total_pkt_hold = total_pkt_hold + len(list_pkt)
        succ_ratio = total_succnum/num_genpkt
        if total_succnum != 0:
            avg_delay = total_delay/total_succnum
            output_str += 'succ_ratio:{} avg_delay:{}\n'.format(succ_ratio, avg_delay)
        else:
            output_str += 'succ_ratio:{} avg_delay:null\n'.format(succ_ratio)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, num_genpkt, total_succnum)
        return output_str

    def __print_res_pure(self, listgenpkt):
        num_purepkt = 0
        for tunple in listgenpkt:
            (pkt_id, src_id, dst_id) = tunple
            if (not isinstance(self.listRouter[src_id], RoutingBlackhole)) and (not isinstance(self.listRouter[dst_id], RoutingBlackhole)):
                num_purepkt = num_purepkt + 1
        output_str = '{}_pure\n'.format(self.scenarioname)
        total_delay = 0
        total_succnum = 0
        total_pkt_hold = 0
        for i_id in range(len(self.listNodeBuffer)):
            if not isinstance(self.listRouter[i_id], RoutingBlackhole):
                list_succ = self.listNodeBuffer[i_id].getlistpkt_succ()
                tmp_succnum = 0
                for i_pkt in list_succ:
                    # 这样 src_id 和 dst_id 都是 正常prophet node
                    if not isinstance(self.listRouter[i_pkt.src_id], RoutingBlackhole):
                        tmp_delay = i_pkt.succ_time - i_pkt.gentime
                        total_delay = total_delay + tmp_delay
                        tmp_succnum = tmp_succnum + 1
                total_succnum = total_succnum + tmp_succnum

                list_pkt = self.listNodeBuffer[i_id].getlistpkt()
                total_pkt_hold = total_pkt_hold + len(list_pkt)
        succ_ratio = total_succnum/num_purepkt
        if total_succnum != 0:
            avg_delay = total_delay/total_succnum
            output_str += 'succ_ratio:{} avg_delay:{} '.format(succ_ratio, avg_delay)
        else:
            avg_delay = ()
            output_str += 'succ_ratio:{} avg_delay:null '.format(succ_ratio)
        output_str += 'num_comm:{}\n'.format(self.num_comm)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, num_purepkt, total_succnum)
        return output_str, succ_ratio, avg_delay, self.num_comm

class RoutingProphet(object):
    def __init__(self, node_id, num_of_nodes, p_init=0.75, gamma=0.98, beta=0.25):
        self.node_id = node_id
        self.P_init = p_init
        self.Gamma = gamma
        self.Beta = beta
        self.num_of_nodes = num_of_nodes
        # aging的时间, 多少秒更新一次 30s, 现在是0.1s一个间隔
        self.secondsInTimeUnit = 30 * 10
        # 记录 a_id 与其他任何节点 之间的delivery prob, P_a_any
        self.delivery_prob = np.zeros(self.num_of_nodes, dtype='double')
        # 初始化 为 P_init
        for i in range(self.num_of_nodes):
            if i != self.node_id:
                self.delivery_prob[i] = self.P_init
        # 记录 两两之间的上次相遇时刻 以便计算相遇间隔
        self.lastAgeUpdate = 0

    # ===============================================  Prophet内部逻辑  ================================
    # 每隔一段时间执行 老化效应
    def __aging(self, running_time):
        duration = running_time - self.lastAgeUpdate
        k = math.floor(duration / self.secondsInTimeUnit)
        if k == 0:
            return
        # 更新了 大家都老化一下
        self.delivery_prob = self.delivery_prob * math.pow(self.Gamma, k)
        self.lastAgeUpdate = running_time

    # a 和 b 相遇 更新prob
    def __update(self, runningtime, a_id, b_id):
        # 取值之前要更新
        P_a_b = self.__getPredFor(runningtime, a_id, b_id)
        # 发生a-b相遇 更新
        self.delivery_prob[b_id] = P_a_b + (1 - P_a_b) * self.P_init

    # 传递效应, 遇见就更新
    def __transitive(self, runningtime, a_id, b_id, P_b_any):
        # 获取的时候 会进行老化操作
        P_a_b = self.__getPredFor(runningtime, a_id, b_id)
        # 获取b_id的delivery prob矩阵 的副本
        for c_id in range(self.num_of_nodes):
            if c_id == b_id or c_id == a_id:
                continue
            self.delivery_prob[c_id] = self.delivery_prob[c_id] + (1 - self.delivery_prob[c_id]) * \
                                       self.delivery_prob[b_id] * P_b_any[c_id] * self.Beta

    def __getPredFor(self, runningtime, a_id, b_id):
        assert(a_id == self.node_id)
        self.__aging(runningtime)
        return self.delivery_prob[b_id]

    # ========================= 提供给上层的功能 ======================================
    # 更新后, 提供 本node 的 delivery prob Matrix 给对端
    def get_values_before_up(self, runningtime):
        self.__aging(runningtime)
        return self.delivery_prob

    # 当a->b 相遇(linkup时候) 更新a->b相应的值
    def notifylinkup(self, runningtime, b_id, *args):
        # b到任何节点的值
        P_b_any = args[0]
        a_id = self.node_id
        # a-b相遇 产生增益
        self.__update(runningtime, a_id, b_id)
        # 借助b进行中转
        self.__transitive(runningtime, a_id, b_id, P_b_any)


class RoutingBlackhole(RoutingProphet):
    def __init__(self, node_id, num_of_nodes):
        super(RoutingBlackhole, self).__init__(node_id, num_of_nodes)
