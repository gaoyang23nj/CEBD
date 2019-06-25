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



