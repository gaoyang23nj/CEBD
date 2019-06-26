import numpy as np
import copy
from DTNNodeBuffer import DTNNodeBuffer
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase

class RoutingEpidemic(RoutingBase):
    def __init__(self, numofnodes):
        self.numofnodes = numofnodes
        # 传输速度 500kb/s
        self.transmitspeed = 500
        # 时间间隔 0.1s
        self.timestep = 0.1

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
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        # tmpnodebuffer = self.listofnodebuffer[src_id]
        # 如果需要删除 按照drop old原则
        if self.listofnodebuffer[src_id].occupied_size + pkt_size >  self.listofnodebuffer[src_id].maxsize:
            self.__deletepkt(src_id, pkt_size)

        self.listofnodebuffer[src_id].addpkt(newpkt)
        return

    # routing接到指令aid和bid相遇，开始进行消息交换
    # 如果a和b正在传输某个pkt, 则此时间间隔内应该帮助接着传
    # 否则 选择新的pkt开始传输
    def swappkt(self, a_id, b_id):
        # 可传输数据量
        transmitvolume = self.transmitspeed * self.timestep
        # 可以传输的状态
        if self.link_transmitpktid[a_id][b_id] != 0:
            # 继续传输之前的pkt
            tmp_pktid = self.link_transmitpktid[a_id][b_id]
            isfound = False
            for target_pkt in self.listofnodebuffer[a_id].listofpkt:
                if target_pkt.pkt_id == tmp_pktid:
                    isfound = True
                    break
            # 既然正在传输的标志位(link_transmitpktid[a_id][b_id])存在
            # 就一定有传输量(link_transmitprocess[a_id][b_id])存在
            assert(isfound == True)
            resumevolume = target_pkt.pkt_size-self.link_transmitprocess[a_id][b_id]
            if transmitvolume > resumevolume:
                transmitvolume = transmitvolume - resumevolume
                self.link_transmitpktid[a_id][b_id] = 0
                self.link_transmitprocess[a_id][b_id] = 0
                self.copypkttobid(b_id, target_pkt)
            elif transmitvolume < resumevolume:
                self.link_transmitprocess[a_id][b_id] = self.link_transmitprocess[a_id][b_id] + transmitvolume
                return
            else:
                self.link_transmitpktid[a_id][b_id] = 0
                self.link_transmitprocess[a_id][b_id] = 0
                self.copypkttobid(b_id, target_pkt)
                return

        # 从a的buffer里 顺序查找 b的buffer里没有的pkt
        # 建立准备传输的pkt列表
        totran_pktlist = self.__gettranpktlist(a_id, b_id)
        for i_pkt in totran_pktlist:
            # 开始传输i_pkt 可传输量消耗
            if i_pkt.pkt_size <= transmitvolume:
                transmitvolume = transmitvolume - i_pkt.pkt_size
                # 把报文复制给b_id
                self.copypkttobid(b_id, i_pkt)
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
                totran_pkt = copy.deepcopy(i_pkt)
                if totran_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, totran_pkt)
                else:
                    totran_pktlist.append(totran_pkt)
        return totran_pktlist

    # EpidemicRouter复制报文给b_id
    def copypkttobid(self, b_id, i_pkt):
        if i_pkt.dst_id == b_id:
            # 去重复
            isduplicate = False
            for j_pkt in self.listofsuccpkt[b_id]:
                if i_pkt.pkt_id == j_pkt.pkt_id:
                    isduplicate = True
                    break
            self.listofsuccpkt[b_id].append(i_pkt)
        else:
            self.listofnodebuffer[b_id].addpkt(i_pkt)


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

    # 建立一个hashmap
    def __isTransferring(self, a_id, b_id):
        return False


    def __deletepkt(self, src_id, pkt_size):
        thenodebuffer = self.listofnodebuffer[src_id]
        while thenodebuffer.occupied_size + pkt_size > thenodebuffer.maxsize:
            thenodebuffer.occupied_size = thenodebuffer.occupied_size - thenodebuffer.listofpkt[0].pkt_size
            thenodebuffer.listofpkt.pop(0)
        return






