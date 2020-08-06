
class DTNPkt(object):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        self.pkt_id = pkt_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.gentime = gentime
        self.pkt_size = pkt_size
        self.TTL = 0
        self.hops = 0
        self.succ_time = -1

class DTNPktSandW(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size, token):
        super(DTNPktSandW, self).__init__(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.token = token

class DTNPktTrack(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        super(DTNPktTrack, self).__init__(pkt_id, src_id, dst_id, gentime, pkt_size)
        # self.track = [src_id]
        self.track = []

class DTNPktPri(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size, pri):
        super(DTNPktPri, self).__init__(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.pri = pri
