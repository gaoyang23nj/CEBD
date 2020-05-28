# 对给出的node_id list, 分别进行 DTNBuffer缓存初始化;
# DTNBuffer下挂载  Routing规则
# DTNBuffer (视为 内存方向的node)的 receive 通过DTNScenario挂载
from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt

import copy

# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_EP(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, num_of_nodes, buffer_size):
        self.scenarioname = scenarioname
        # 为各个node建立虚拟空间 <内存+router>
        self.listNodeBuffer = []
        for node_id in range(num_of_nodes):
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size)
            self.listNodeBuffer.append(tmpBuffer)
        return

    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.listNodeBuffer[src_id].gennewpkt(newpkt)
        return

    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        self.sendpkt(runningtime, a_id, b_id)
        self.sendpkt(runningtime, b_id, a_id)

    # 报文发送 a_id -> b_id
    def sendpkt(self, runningtime, a_id, b_id):
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
        # Epidemic的路由方法 都转发
        for tmp_pkt in totran_pktlist:
            # 若抵达的是目的节点(即b_id == tmp_pkt.dst_id) 则a_id删掉该pkt的副本
            isReach = self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
            if isReach:
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)

    def print_res(self, listgenpkt):
        output_str = '{}\n'.format(self.scenarioname)
        total_delay = 0
        total_succnum = 0
        total_pkt_hold = 0
        for i_id in range(len(self.listNodeBuffer)):
            list_succ = self.listNodeBuffer[i_id].getlistpkt_succ()
            for i_pkt in list_succ:
                tmp_delay = i_pkt.succ_time - i_pkt.gentime
                total_delay = total_delay + tmp_delay
            tmp_succnum = len(list_succ)
            total_succnum = total_succnum + tmp_succnum

            list_pkt = self.listNodeBuffer[i_id].getlistpkt()
            total_pkt_hold = total_pkt_hold + len(list_pkt)
        succ_ratio = total_succnum / len(listgenpkt)
        if total_succnum != 0:
            avg_delay = total_delay / total_succnum
            output_str += 'succ_ratio:{} avg_delay:{}\n'.format(succ_ratio, avg_delay)
        else:
            avg_delay = ()
            output_str += 'succ_ratio:{} avg_delay:null\n'.format(succ_ratio)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, len(listgenpkt), total_succnum)
        print(output_str)
        res = (succ_ratio, avg_delay)
        config = ()
        return output_str, res, config