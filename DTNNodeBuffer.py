import copy
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase
from RoutingEpidemic import RoutingEpidemic
from RoutingSparyandWait import *
from RoutingProphet import RoutingProphet
from RoutingMaxProp import *

from RoutingBlackhole import RoutingBlackhole
from RoutingSDBG import RoutingSDBG

class DTNNodeBuffer(object):
    # b_id 将要复制pkt之前 给a_id的返回码 (a_id据此作出操作)
    # 报文投递至dst_id == b_id, 通知a_id可以删除了
    Rece_Code_ToDst = 1
    # 报文被b_id拒绝了 (显式拒绝)
    Rece_Code_DenyPkt = 2
    # 报文被b_id接收了 (接收 或者 隐式拒绝)
    Rece_Code_AcceptPkt = 3

    # buffersize = 10*1000 k, 即10M; 每个报文100k
    def __init__(self, scenario, node_id, routingname, numofnodes, maxsize=20*1000):
        self.theScenario = scenario
        self.node_id = node_id
        self.numofnodes = numofnodes
        self.maxsize = maxsize
        self.occupied_size = 0
        # <内存> 实时存储的pkt list
        self.listofpkt = []
        # 为了记录和对照方便 为每个节点记录成功投递的pkt list
        self.listofsuccpkt = []
        self.routingname = routingname
        self.__attach_router(routingname)

    def __attach_router(self, routingname):
        if routingname == 'RoutingEpidemic':
            self.theRouter = RoutingEpidemic(self)
        elif routingname == 'RoutingSparyandWait':
            self.theRouter = RoutingSparyandWait(self)
        elif routingname == 'RoutingProphet':
            self.theRouter = RoutingProphet(self, self.numofnodes)
        elif routingname == 'RoutingBlackhole':
            self.theRouter = RoutingBlackhole(self)
        elif routingname == 'RoutingSDBG':
            self.theRouter = RoutingSDBG(self, self.numofnodes)
        else:
            print('ERROR! 未知的router!')

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
    def getlocalusage(self):
        # str = ''
        succinlocalnode = []
        for pkt in self.listofsuccpkt:
            # strtmp = 'pkt_{}:src_id(node_{})->dst_id(node_{}); '.format(pkt.pkt_id, pkt.src_id, pkt.dst_id)
            # str = str + strtmp
            pktinfo = (pkt.pkt_id, pkt.src_id, pkt.dst_id)
            succinlocalnode.append(pktinfo)
        # if len(self.listofsuccpkt) > 0:
        #     str = str + '\n'
        return len(self.listofsuccpkt), succinlocalnode

    # 按照id 找到pkt
    def findpktbyid(self, id):
        isFound = False
        pkt = 0
        for pkt in self.listofpkt:
            if id == pkt.pkt_id:
               isFound = True
               break
        return isFound, pkt

    # 获取内存中的pkt_id list
    def getlistpkt(self):
        return self.listofpkt

    # 保证内存空间足够 并把pkt放在内存里; isgen 是否是生成新pkt
    def mkroomaddpkt(self, newpkt, isgen):
        # 按照需要 改装pkt
        # 如果需要删除pkt以提供内存空间 按照drop old原则
        if self.occupied_size + newpkt.pkt_size > self.maxsize:
            self.__deletepktbysize(newpkt.pkt_size)
        self.__addpkt(newpkt)

    # =========================== 核心接口 提供传输pkt的名录; 生成报文; 接收报文
    def notifygennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        # 按照需要 改装pkt
        if isinstance(self.theRouter, RoutingSparyandWait):
            init_token = 2
            newpkt = DTNSWPkt(pkt_id, src_id, dst_id, gentime, pkt_size, init_token)
        elif isinstance(self.theRouter, RoutingMaxProp):
            newpkt = DTNTrackPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        else:
            newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.mkroomaddpkt(newpkt, isgen=True)

    # 重要功能 事关报文复制！！！！
    # 收到Scenario上的通知 在runningtime时候 可以准备从a_id 拷贝i_pkt
    def notifyreceivedpkt(self, runningtime, a_id, i_pkt):
        #================================ code_1 成功抵达
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
            return DTNNodeBuffer.Rece_Code_ToDst
        #================================ code_2 显式拒绝
        # 竟然找到了这个pkt, 在这个pkt转发的过程中, 本pkt的其他副本已经成功
        isFound, pkt = self.findpktbyid(i_pkt.pkt_id)
        if isFound == True:
            return DTNNodeBuffer.Rece_Code_DenyPkt
        # ================================ code_3 (显、隐)接收 隐拒绝
        # 拷贝出一份 以防要修改一些值 hop; track; token等
        target_pkt = copy.deepcopy(i_pkt)
        # router决定是否真的要接收
        isReceive = self.theRouter.decideAddafterRece(a_id, target_pkt)
        if isReceive == True:
            self.mkroomaddpkt(target_pkt, isgen=False)
        return DTNNodeBuffer.Rece_Code_AcceptPkt

    # 收到Scenario上的通知 i_pkt已经传输; 在runningtime时候 发送i_pkt给b_id
    def notifysentpkt(self, runningtime, codeRece, b_id, i_pkt):
        # 若报文已经抵达目的, a_id同时做个验证保证真实
        if codeRece == DTNNodeBuffer.Rece_Code_ToDst and b_id == i_pkt.dst_id:
            isDelete = True
            self.__deletepktbypktid(i_pkt.pkt_id)
        # 若报文不被接收
        elif codeRece == DTNNodeBuffer.Rece_Code_DenyPkt:
            # do nothing
            return
        elif codeRece == DTNNodeBuffer.Rece_Code_AcceptPkt:
            isDelete = self.theRouter.decideDelafterSend(b_id, i_pkt)
            if isDelete == True:
                self.__deletepktbypktid(i_pkt.pkt_id)
            return
        else:
            print('ERROR! DTNBuffer 未知的接受码')
            pass

    # ====================== 核心接口 linkup linkdown时刻 a和 b 交换<控制信息>
    # 某些routing算法下 在linkup之前 需要给对方node的router传值
    def get_values_router_before_up(self, runningtime):
        return self.theRouter.get_values_before_up(runningtime)

    # 通知a_id： 与b_id 的 linkup事件
    def notify_link_up(self, b_id, running_time, *args):
        self.theRouter.notify_link_up(b_id, running_time, *args)

    # 某些routing算法下 在linkdown之前 需要给对方node的router传值
    def get_values_router_before_down(self, runningtime):
        return self.theRouter.get_values_before_down(runningtime)

    # 通知a_id： 与b_id 的 linkdown事件
    def notify_link_down(self, b_id, running_time, *args):
        self.theRouter.notify_link_down(b_id, running_time, *args)

    def get_values_router_before_tran(self, runningtime):
        return self.theRouter.get_values_before_tran(runningtime)

    # 询问router 准备传输的pkt 组成的list;  参照对方pktlist 现状, 计算准备传输的pktlist
    def gettranpktlist(self, runningtime, b_id, listpkt, *args):
        return self.theRouter.gettranpktlist(runningtime, b_id, listpkt, self.node_id, self.listofpkt, *args)

    # =================================== 提供给 MaxProp的接口 ================================

