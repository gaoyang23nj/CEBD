from Routing.RoutingEpidemic import RoutingEpidemic
from Routing.RoutingSparyandWait import *
from Routing.RoutingProphet import RoutingProphet
from Routing.RoutingMaxProp import *

from Routing.RoutingBlackhole import RoutingBlackhole

from Routing.RoutingProvest import RoutingProvest
from Routing.RoutingEric import RoutingEric
from Routing.RoutingSDBG import RoutingSDBG


class DTNNodeBuffer(object):
    # buffersize = 10*1000 k, 即10M; 每个报文100k
    def __init__(self, thescenario, node_id, maxsize):
        # 关联自己的场景
        self.theScenario = thescenario
        self.node_id = node_id
        self.maxsize = maxsize
        self.occupied_size = 0
        # <内存> 实时存储的pkt list, 从前往后（从0开始）pkt越来越老
        self.listofpkt = []
        # 历史上已经接收过的pkt id
        self.listofpktid_hist = []
        # 为了记录和对照方便 为每个节点记录成功投递的pkt list
        self.listofsuccpkt = []

    # =========================== 核心接口 提供传输pkt的名录; 生成报文; 接收报文
    def gennewpkt(self, newpkt):
        self.listofpktid_hist.append(newpkt.pkt_id)
        self.__mkroomaddpkt(newpkt, isgen=True)

    def receivepkt(self, runningtime, receivedpkt):
        self.listofpktid_hist.append(receivedpkt.pkt_id)
        cppkt = copy.deepcopy(receivedpkt)
        if isinstance(cppkt, DTNTrackPkt):
            cppkt.track.append(self.node_id)
        # 抵达目的节点
        if cppkt.dst_id == self.node_id:
            # 确定之前没有接收过这个pkt
            isReceivedBefore = False
            for succed_pkt in self.listofsuccpkt:
                if succed_pkt.pkt_id == cppkt.pkt_id:
                    isReceivedBefore = True
                    break
            if not isReceivedBefore:
                cppkt.succ_time = runningtime
                self.listofsuccpkt.append(cppkt)
        else:
            self.__mkroomaddpkt(cppkt, False)

    # 获取内存中的pkt list
    def getlistpkt(self):
        return self.listofpkt.copy()

    # 获取接触过的pkt_id [包括自己生成过的 和 自己接收过的]
    def getlistpkt_hist(self):
        return self.listofpktid_hist.copy()

    def getlistpkt_succ(self):
        return self.listofsuccpkt.copy()

    # ==============================================================================================================
    # 保证内存空间足够 并把pkt放在内存里; isgen 是否是生成新pkt
    def __mkroomaddpkt(self, newpkt, isgen):
        # 如果需要删除pkt以提供内存空间 按照drop old原则
        if self.occupied_size + newpkt.pkt_size > self.maxsize:
            print('delete pkt! in node_{}'.format(self.node_id))
            self.__deletepktbysize(newpkt.pkt_size)
        self.__addpkt(newpkt)

    # 老化机制 从头删除报文 提供至少pkt_size的空间
    def __deletepktbysize(self, pkt_size):
        while self.occupied_size + pkt_size > self.maxsize:
            self.occupied_size = self.occupied_size - self.listofpkt[0].pkt_size
            self.listofpkt.pop(0)
        return

    # 内存中增加pkt newpkt
    def __addpkt(self, newpkt):
        cppkt = copy.deepcopy(newpkt)
        self.occupied_size = self.occupied_size + cppkt.pkt_size
        # 如果需要记录 track
        if isinstance(cppkt, DTNTrackPkt):
            cppkt.track.append(self.node_id)
        self.listofpkt.append(cppkt)
        return
