import numpy as np
import copy
import math
from DTNNodeBuffer import DTNNodeBuffer
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase

# 需要在pkt里加一个token属性
class DTNSWPkt(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size, token):
        super(DTNSWPkt, self).__init__(self, pkt_id, src_id, dst_id, gentime, pkt_size)
        self.token = token


    def __init__(self, dtnpkt, token):
        super(DTNSWPkt, self).__init__(self, dtnpkt.pkt_id, dtnpkt.src_id, dtnpkt.dst_id, dtnpkt.gentime, dtnpkt.pkt_size)
        self.token = token


class RoutingSparyandWait(RoutingBase):
    def __init__(self, inittoken = 8):
        self.inittoken = inittoken


    # 顺序地得到准备传输的list(b_id里没有的pkt), dst_id是b_id的pkt应该最先传
    def __gettranpktlist(self, a_id, b_id):
        totran_pktlist = []
        src_nodebuffer = self.listofnodebuffer[a_id]
        dst_nodebuffer = self.listofnodebuffer[b_id]
        for i_pkt in src_nodebuffer.listofpkt:
            isiexist = False
            # # 如果pkt_id
            for j_pkt in dst_nodebuffer.listofpkt:
                if i_pkt.pkt_id == j_pkt.pkt_id:
                    isiexist = True
                    break
            if isiexist == False:
                # 如果pkt的dst_id就是b, 找到目的 应该优先传输
                if i_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, i_pkt)
                else:
                    totran_pktlist.append(i_pkt)
        return totran_pktlist
