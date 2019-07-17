import copy
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase
from RoutingEpidemic import RoutingEpidemic
from RoutingSparyandWait import *

# from RoutingBlackhole import RoutingBlackhole

class DTNNodeBuffer(object):
    # buffersize = 100*1000 k, 即100M
    def __init__(self, scenario, node_id, routingname, maxsize=100*1000):
        self.dtnscenario = scenario
        self.node_id = node_id
        self.maxsize = maxsize
        self.occupied_size = 0
        # <内存> 实时存储的pkt list
        self.listofpkt = []
        # 为了记录和对照方便 为每个节点记录成功投递的pkt list
        self.listofsuccpkt = []
        self.__attachRouter(routingname)


    def __attachRouter(self, routingname):
        if routingname == 'RoutingEpidemic':
            self.router = RoutingEpidemic()
        elif routingname == 'RoutingSparyandWait':
            self.router = RoutingSparyandWait(inittoken=2)


    # 内存中增加pkt newpkt
    def __addpkt(self, newpkt):
        cppkt = copy.deepcopy(newpkt)
        self.occupied_size = self.occupied_size + cppkt.pkt_size
        self.listofpkt.append(cppkt)
        return

    # 按照pkt_id删掉pkt
    def __deletepktbypktid(self, pkt_id):
        isOK = False
        for pkt in self.listofpkt:
            if pkt_id == pkt.pkt_id:
                self.occupied_size = self.occupied_size - pkt.pkt_size
                self.listofpkt.remove(pkt)
                isOK = True
        return isOK


    # 老化机制 从头删除报文 提供至少pkt_size的空间
    def __deletepktbysize(self, pkt_size):
        while self.occupied_size + pkt_size > self.maxsize:
            self.occupied_size = self.occupied_size - self.listofpkt[0].pkt_size
            self.listofpkt.pop(0)
        return


    # ==========================提供给Scenario的功能===========================================================================
    # 显示结果
    def showres(self):
        str = ''
        for pkt in self.listofsuccpkt:
            strtmp = 'pkt_{}:src_id(node_{})->dst_id(node_{}); '.format(pkt.pkt_id, pkt.src_id, pkt.dst_id)
            str = str + strtmp
        if len(self.listofsuccpkt) > 0:
            str = str + '\n'
        return len(self.listofsuccpkt), str

    # 按照id 找到pkt
    def findpktbyid(self,id):
        isFound = False
        pkt = 0
        for pkt in self.listofpkt:
            if id == pkt.pkt_id:
               isFound = True
               break
        return isFound, pkt


    # 询问router 准备传输的pkt 组成的list;  参照对方pktlist 现状, 计算准备传输的pktlist
    def gettranpktlist(self, b_id, listpkt):
        return self.router.gettranpktlist(b_id, listpkt, self.node_id, self.listofpkt)


    # 获取内存中的pkt_id list
    def getlistpkt(self):
        return self.listofpkt


    # 保证内存空间足够 并把pkt放在内存里; isgen 是否是生成新pkt
    def mkroomaddpkt(self, newpkt, isgen):
        # 按照需要 改装pkt
        if isgen and isinstance(self.router, RoutingSparyandWait):
            newpkt = DTNSWPkt(newpkt, self.router.inittoken)

        # 如果需要删除pkt以提供内存空间 按照drop old原则
        if self.occupied_size + newpkt.pkt_size > self.maxsize:
            self.__deletepktbysize(newpkt.pkt_size)
        self.__addpkt(newpkt)


    # 收到Scenario上的通知 i_pkt已经传输; 在runningtime时候 发送i_pkt给b_id
    def notifysentpkt(self, runningtime, b_id, i_pkt):
        isDelete = self.router.decideDelafterSend(b_id, i_pkt)
        if isDelete == True:
            self.__deletepktbypktid(i_pkt.pkt_id)
        return


    # 收到Scenario上的通知 i_pkt已经传输; 在runningtime时候 发送i_pkt给b_id
    def notifyreceivedpkt(self, runningtime, a_id, i_pkt):
        if i_pkt.dst_id == self.node_id:
            # 成功接收 加入成功接收的list
            isduplicate = False
            for j_pkt in self.listofsuccpkt:
                if i_pkt.pkt_id == j_pkt.pkt_id:
                    isduplicate = True
                    break
            # 只有之前没有接收到i_pkt 才会加入succlist
            if isduplicate == False:
                # append之前 需要 deepcopy
                target_pkt = copy.deepcopy(i_pkt)
                self.listofsuccpkt.append(target_pkt)
            return
        # 拷贝出一份 以防要修改一些值 hop; track; token等
        target_pkt = copy.deepcopy(i_pkt)
        isReceive = self.router.decideAddafterRece(a_id, target_pkt)
        if isReceive == True:
            self.mkroomaddpkt(target_pkt, isgen=False)
        return
