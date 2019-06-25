import numpy as np
from DTNNodeBuffer import DTNNodeBuffer
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase


class RoutingEpidemic(RoutingBase):
    def __init__(self, numofnodes):
        self.numofnodes = numofnodes
        # 传输速度 500kb/s
        self.transmitspeed = 500
        # 时间间隔 0.1s
        self.timestep = 0.1
        self.listofnodebuffer = []
        # <虚拟>生成node的内存空间
        for i in range(numofnodes):
            nodebuffer = DTNNodeBuffer(i, 100*1000)
            self.listofnodebuffer.append(nodebuffer)
        # 建立正在传输的表格 和 link状态的矩阵
        self.link_state = np.zeros(self.numofnodes,  self.numofnodes)
        self.link_transmitprocess = np.zeros(self.numofnodes,  self.numofnodes)
        self.link_transmitpktid = np.zeros(self.numofnodes, self.numofnodes)

    # routing接到指令 在srcid生成一个pkt(srcid->dstid),并记录生成时间
    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        # tmpnodebuffer = self.listofnodebuffer[src_id]
        # 如果需要删除 按照drop old原则
        if self.listofnodebuffer[src_id].occupied_size + pkt_size >  self.listofnodebuffer[src_id].maxsize:
            self.__deletepkt(src_id, pkt_size)

        self.listofnodebuffer[src_id].addpkt(newpkt)


    # routing接到指令aid和bid相遇，开始进行消息交换
    # 如果a和b正在传输某个pkt, 则此时间间隔内应该帮助接着传
    # 否则 选择新的pkt开始传输
    def swappkt(self, a_id, b_id):
        # 可以传输的状态
        self.link_state[a_id][b_id] = 1
        if self.link_transmitpktid[a_id][b_id] != 0:
            # 继续传输之前的pkt
            return

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
                self.transimitpkttob(a_id, b_id, i_pkt)

    def transimitpkttob(self, a_id, b_id, i_pkt):
        if self.transmitspeed * self.timestep >= i_pkt.pkt_size:
            pass
        else:
            self.link_transmitpktid[a_id][b_id] = i_pkt.pkt_id
            self.link_transmitprocess[a_id][b_id] = self.transmitspeed * self.timestep + self.link_transmitprocess[a_id][b_id]



    # 建立一个hashmap
    def __isTransferring(self, a_id, b_id):
        return False


    def __deletepkt(self, src_id, pkt_size):
        thenodebuffer = self.listofnodebuffer[src_id]
        while thenodebuffer.occupied_size + pkt_size > thenodebuffer.maxsize:
            thenodebuffer.occupied_size = thenodebuffer.occupied_size - thenodebuffer.listofpkt[0].pkt_size
            thenodebuffer.listofpkt.pop(0)






