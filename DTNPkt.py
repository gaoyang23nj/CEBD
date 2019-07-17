import copy
class DTNPkt(object):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        self.pkt_id = pkt_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.gentime = gentime
        self.pkt_size = pkt_size
        self.TTL = 0
        self.hops = 0


