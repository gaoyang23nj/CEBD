import numpy as np
import copy
import math
from DTNNodeBuffer import DTNNodeBuffer
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase

# 需要在pkt里加一个token属性
class DTNSWPkt(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size, token):
        self.token = token
        super(DTNSWPkt,self).__init__(self, pkt_id, src_id, dst_id, gentime, pkt_size)


class RoutingSparyandWait(RoutingBase):
    def __init__(self, numofnodes):
        self.numofnodes = numofnodes
        # 传输速度 500kb/s
        self.transmitspeed = 500
        # 时间间隔 0.1s
        self.timestep = 0.1
        self.init_token = 8
        # <虚拟>生成node的内存空间
        self.listofnodebuffer = []
        for i in range(numofnodes):
            nodebuffer = DTNNodeBuffer(i, 100*1000)
            self.listofnodebuffer.append(nodebuffer)

        # 为了记录和对照方便 为每个节点记录成功投递的pkt list
        self.listofsuccpkt = []
        for i in range(numofnodes):
            tmppktlist = []
            self.listofsuccpkt.append(tmppktlist)

        # 建立正在传输的pkt_id 和 传输进度 的矩阵
        self.link_transmitpktid = np.zeros((self.numofnodes, self.numofnodes), dtype = 'int')
        self.link_transmitprocess = np.zeros((self.numofnodes,  self.numofnodes), dtype = 'int')


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
