from DTNNodeBuffer import DTNNodeBuffer
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase

class RoutingEpidemic(RoutingBase):
    def __init__(self, numofnodes):
        self.numofnodes = numofnodes
        self.listofnodebuffer = []
        # <虚拟>生成node的内存空间
        for i in range(numofnodes):
            nodebuffer = DTNNodeBuffer(i, 100*1000)
            self.listofnodebuffer.append(nodebuffer)

    # routing接到指令 在srcid生成一个pkt(srcid->dstid),并记录生成时间
    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        # tmpnodebuffer = self.listofnodebuffer[src_id]
        # 如果需要删除 按照drop old原则
        if self.listofnodebuffer[src_id].occupied_size + pkt_size >  self.listofnodebuffer[src_id].maxsize:
            self.__deletepkt(src_id, pkt_size)

        self.listofnodebuffer[src_id].addpkt(newpkt)


    # routing接到指令aid和bid相遇，开始进行消息交换
    def swappkt(self, a_id, b_id):

        pass


    def __deletepkt(self, src_id, pkt_size):
        thenodebuffer = self.listofnodebuffer[src_id]
        while thenodebuffer.occupied_size + pkt_size > thenodebuffer.maxsize:
            thenodebuffer.occupied_size = thenodebuffer.occupied_size - thenodebuffer.listofpkt[0].pkt_size
            thenodebuffer.listofpkt.pop(0)






