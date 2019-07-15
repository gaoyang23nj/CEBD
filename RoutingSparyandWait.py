import numpy as np
import copy
import math
from DTNNodeBuffer import DTNNodeBuffer
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase

# 需要在pkt里加一个token属性
class DTNSWPkt(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size, token):
        self.token = token
        super(DTNSWPkt,self).__init__(self, pkt_id, src_id, dst_id, gentime, pkt_size)


class RoutingSparyandWait(RoutingBase):
    def __init__(self, numofnodes):
        self.numofnodes = numofnodes
        # 传输速度 500kb/s
        self.transmitspeed = 500
        # 时间间隔 0.1s
        self.timestep = 0.1
        self.init_token = 8
        # <虚拟>生成node的内存空间
        self.listofnodebuffer = []
        for i in range(numofnodes):
            nodebuffer = DTNNodeBuffer(i, 100*1000)
            self.listofnodebuffer.append(nodebuffer)

        # 为了记录和对照方便 为每个节点记录成功投递的pkt list
        self.listofsuccpkt = []
        for i in range(numofnodes):
            tmppktlist = []
            self.listofsuccpkt.append(tmppktlist)

        # 建立正在传输的pkt_id 和 传输进度 的矩阵
        self.link_transmitpktid = np.zeros((self.numofnodes, self.numofnodes), dtype = 'int')
        self.link_transmitprocess = np.zeros((self.numofnodes,  self.numofnodes), dtype = 'int')


    # routing接到指令 在srcid生成一个pkt(srcid->dstid),并记录生成时间
    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        newpkt = DTNSWPkt(pkt_id, src_id, dst_id, gentime, pkt_size, self.init_token)
        # tmpnodebuffer = self.listofnodebuffer[src_id]
        # 如果需要删除 按照drop old原则
        if self.listofnodebuffer[src_id].occupied_size + pkt_size > self.listofnodebuffer[src_id].maxsize:
            self.__deletepkt(src_id, pkt_size)

        self.listofnodebuffer[src_id].addpkt(newpkt)
        return


    # routing接到指令aid和bid相遇，开始进行消息交换
    def swappkt(self, a_id, b_id):
        # 可传输数据量
        transmitvolume = self.transmitspeed * self.timestep
        # 如果存在正在传输的pkt
        if self.link_transmitpktid[a_id][b_id] != 0:
            transmitvolume = self.continue_transmitting(a_id, b_id, transmitvolume)
        self.transmitting(a_id, b_id, transmitvolume)


    # 如果a和b正在传输某个pkt, 则此时间间隔内 继续传输 已经传输一部分的pkt
    # 返回 剩余的可传输量
    def continue_transmitting(self,a_id, b_id, transmitvolume):
        # 继续传输之前的pkt
        tmp_pktid = self.link_transmitpktid[a_id][b_id]
        (isfound, target_pkt) = self.listofnodebuffer[a_id].findpktbyid(tmp_pktid)
        # 既然正在传输的标志位(link_transmitpktid[a_id][b_id])存在
        # 就一定有传输量(link_transmitprocess[a_id][b_id])存在
        assert (isfound == True)
        # 既然正在传输的标志位(link_transmitpktid[a_id][b_id])存在, 就一定有传输量(link_transmitprocess[a_id][b_id])存在
        resumevolume = target_pkt.pkt_size - self.link_transmitprocess[a_id][b_id]
        if transmitvolume > resumevolume:
            remiantransmitvolume = transmitvolume - resumevolume
            # 本次传输结束 记得置空标志位
            self.link_transmitpktid[a_id][b_id] = 0
            self.link_transmitprocess[a_id][b_id] = 0
            self.copypkttobid(a_id, b_id, target_pkt)
            # 还剩一些可传输量
            return remiantransmitvolume
        elif transmitvolume < resumevolume:
            self.link_transmitprocess[a_id][b_id] = self.link_transmitprocess[a_id][b_id] + transmitvolume
            return 0
        else:
            self.link_transmitpktid[a_id][b_id] = 0
            self.link_transmitprocess[a_id][b_id] = 0
            self.copypkttobid(a_id, b_id, target_pkt)
            return 0


    # 否则 选择新的pkt开始传输
    def transmitting(self, a_id, b_id, transmitvolume):
        # 如果没有正在传输的pkt
        # 从a的buffer里 顺序查找 b的buffer里没有的pkt
        # 建立准备传输的pkt列表(这应该是一个优先级的list)
        totran_pktlist = self.__gettranpktlist(a_id, b_id)
        for i_pkt in totran_pktlist:
            # 开始传输i_pkt 可传输量消耗
            if i_pkt.pkt_size <= transmitvolume:
                transmitvolume = transmitvolume - i_pkt.pkt_size
                # 把报文复制给b_id
                self.copypkttobid(a_id, b_id, i_pkt)
            # 如果传不完了...
            else:
                self.link_transmitpktid[a_id][b_id] = i_pkt.pkt_id
                self.link_transmitprocess[a_id][b_id] = i_pkt.pkt_size - transmitvolume
                break
        return


    # 顺序地得到准备传输的list(b_id里没有的pkt), dst_id是b_id的pkt应该最先传
    def __gettranpktlist(self, a_id, b_id):
        totran_pktlist = []
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
                # 如果pkt的dst_id就是b, 找到目的 应该优先传输
                if i_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, i_pkt)
                else:
                    totran_pktlist.append(i_pkt)
        return totran_pktlist


    # EpidemicRouter复制报文从a_id给b_id
    def copypkttobid(self, a_id, b_id, i_pkt):
        # 成功投递给目的node,
        if i_pkt.dst_id == b_id:
            # 为了加入 succ ist, 需要去重复
            isduplicate = False
            for j_pkt in self.listofsuccpkt[b_id]:
                if i_pkt.pkt_id == j_pkt.pkt_id:
                    isduplicate = True
                    break
            # 只有之前没有接收到i_pkt 才会加入succlist
            if isduplicate == False:
                target_pkt = copy.deepcopy(i_pkt)
                self.listofsuccpkt[b_id].append(target_pkt)
            # 已经找到目的node, a_id就不必保留原来的副本了
            self.__deletepkt_id(a_id, i_pkt.pkt_id)
        else:
            if i_pkt.token > 1:
                target_pkt = copy.deepcopy(i_pkt)
                target_pkt.token = math.floor(i_pkt.token / 2)
                # 原先的token修改
                i_pkt.token = i_pkt.token - target_pkt.token
                self.listofnodebuffer[b_id].addpkt(target_pkt)
        return


    def linkdown(self, a_id, b_id):
        self.link_transmitpktid[a_id][b_id] = 0
        self.link_transmitprocess[a_id][b_id] = 0
        return


    def showres(self):
        # 获取成功投递的个数
        succnum = 0
        for inode_pktlist in self.listofsuccpkt:
            succnum = succnum + len(inode_pktlist)
        return succnum


    # 从src_id的list里 删掉指定pkt_id的报文
    def __deletepkt_id(self, src_id, pkt_id):
        thenodebuffer = self.listofnodebuffer[src_id]
        isOK = thenodebuffer.deletepktbypktid(pkt_id)
        assert(isOK)
        return


    # 为了在src_id 新增pkt, 需要从list开头开始提供pkt_size大小的空间
    def __deletepkt_size(self, src_id, pkt_size):
        thenodebuffer = self.listofnodebuffer[src_id]
        thenodebuffer.deletepktbysize(pkt_size)
        return


    # 提供给界面show的接口
    def getnodelist(self, node_id):
        node_pktlist = []
        for nodebuffer in self.listofnodebuffer:
            if node_id == nodebuffer.id:
                node_pktlist = nodebuffer.getlistpkt()
                break
        return node_pktlist