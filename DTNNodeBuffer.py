import copy
from DTNPkt import DTNPkt

class DTNNodeBuffer(object):
    # buffersize = 100*1000 k, 即100M
    def __init__(self, id, maxsize=100*1000):
        self.id = id
        self.maxsize = maxsize
        self.occupied_size = 0
        self.listofpkt = []

    def addpkt(self, newpkt):
        cppkt = copy.deepcopy(newpkt)
        self.occupied_size = self.occupied_size + cppkt.pkt_size
        self.listofpkt.append(cppkt)

    def getlistpkt(self):
        relist = []
        for pkt in self.listofpkt:
            tunple = pkt.pkt_id
            relist.append(tunple)
        return  relist

    # 按照pkt_id删掉pkt
    def deletepktbypktid(self, pkt_id):
        isOK = False
        for pkt in self.listofpkt:
            if pkt_id == pkt.pkt_id:
                self.occupied_size = self.occupied_size - pkt.pkt_size
                self.listofpkt.remove(pkt)
                isOK = True
        return isOK

    # 老化机制 从头删除报文 提供至少pkt_size的空间
    def deletepktbysize(self, pkt_size):
        while self.occupied_size + pkt_size > self.maxsize:
            self.occupied_size = self.occupied_size - self.listofpkt[0].pkt_size
            self.listofpkt.pop(0)
        return