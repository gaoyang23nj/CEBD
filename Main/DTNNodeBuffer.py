import copy

from Main.DTNPkt import *

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
        isDelPkt_for_room = self.__mkroomaddpkt(newpkt, isgen=True)
        return isDelPkt_for_room

    def receivepkt(self, runningtime, receivedpkt):
        # isReach 为 True 表示pkt已经投递到dest, 源节点不必再保留
        isDelPkt_for_room = False
        isReach = False
        cppkt = copy.deepcopy(receivedpkt)
        # if isinstance(cppkt, DTNPktTrack):
        #     cppkt.track.append(self.node_id)
        # 抵达目的节点
        if cppkt.dst_id == self.node_id:
            # 确定之前没有接收过这个pkt
            isReceivedBefore = False
            for succed_pkt in self.listofsuccpkt:
                if succed_pkt.pkt_id == cppkt.pkt_id:
                    isReceivedBefore = True
                    break
            if not isReceivedBefore:
                self.listofpktid_hist.append(receivedpkt.pkt_id)
                cppkt.succ_time = runningtime
                self.listofsuccpkt.append(cppkt)
            isReach = True
        else:
            self.listofpktid_hist.append(cppkt.pkt_id)
            isDelPkt_for_room = self.__mkroomaddpkt(cppkt, False)
        return isReach, isDelPkt_for_room

    # 获取内存中的pkt list
    def getlistpkt(self):
        return self.listofpkt.copy()

    # 获取接触过的pkt_id [包括自己生成过的 和 自己接收过的]
    def getlistpkt_hist(self):
        return self.listofpktid_hist.copy()

    def getlistpkt_succ(self):
        return self.listofsuccpkt.copy()

    # 按照pkt_id删掉pkt
    def deletepktbyid(self, runningtime, pkt_id):
        isFound = False
        for pkt in self.listofpkt:
            if pkt_id == pkt.pkt_id:
                self.occupied_size = self.occupied_size - pkt.pkt_size
                self.listofpkt.remove(pkt)
                isFound = True
        return isFound

    # ==============================================================================================================
    # 对指定的pkt_id 减少 changed_token个token
    def decrease_token(self, runningtime, pkt_id, changed_token):
        for tmp_pkt in self.listofpkt:
            if tmp_pkt.pkt_id == pkt_id:
                tmp_pkt.token = tmp_pkt.token - changed_token
                break

    # ==============================================================================================================
    # 保证内存空间足够 并把pkt放在内存里; isgen 是否是生成新pkt
    def __mkroomaddpkt(self, newpkt, isgen):
        isDel = False
        self.__addpkt(newpkt)
        # 如果需要删除pkt以提供内存空间 按照drop old原则
        while self.occupied_size > self.maxsize:
            # print('delete pkt! in node_{}'.format(self.node_id))
            self.occupied_size = self.occupied_size - self.listofpkt[0].pkt_size
            self.listofpkt.pop(0)
            isDel = True
        return isDel

    # 内存中增加pkt newpkt
    def __addpkt(self, newpkt):
        cppkt = copy.deepcopy(newpkt)
        self.occupied_size = self.occupied_size + cppkt.pkt_size
        # 如果需要记录 track
        if isinstance(cppkt, DTNPktTrack):
            cppkt.track.append(self.node_id)
        self.listofpkt.append(cppkt)
        return

# 带有优先级的buffer
class DTNNodeBufferPri(DTNNodeBuffer):
    def gennewpkt(self, newpkt):
        assert(isinstance(newpkt, DTNPktPri))
        super(DTNNodeBufferPri, self).gennewpkt(newpkt)

    # 内存中增加pkt newpkt
    def __addpkt(self, newpkt):
        cppkt = copy.deepcopy(newpkt)
        self.occupied_size = self.occupied_size + cppkt.pkt_size
        # 如果需要记录 track
        if isinstance(cppkt, DTNPktTrack):
            cppkt.track.append(self.node_id)
        # 找位置; 如果 缓存中pkt优先级 < 目标pkt优先级, 继续向后找位置; 如果=, 找到最后一个(同样优先级; 最后删除); 如果> 应该放到前面去
        target_idx = 0
        while target_idx < len(self.listofpkt):
            if self.listofpkt[target_idx].pri > newpkt.pri:
                break
            target_idx = target_idx + 1
        if target_idx > 0:
            self.listofpkt.insert(target_idx - 1, cppkt)
        else:
            # target_idx == 0
            self.listofpkt.insert(0, cppkt)
        return

    def receivepkt(self, runningtime, receivedpkt):
        assert(isinstance(receivedpkt, DTNPktPri))
        isReach = super(DTNNodeBufferPri, self).receivepkt(runningtime, receivedpkt)
        return isReach