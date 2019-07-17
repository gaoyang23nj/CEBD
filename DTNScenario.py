# 对给出的node_id list, 分别进行 DTNBuffer缓存初始化;
# DTNBuffer下挂载  Routing规则
# DTNBuffer (视为 内存方向的node)的 receive 通过DTNScenario挂载
import numpy as np
from DTNPkt import DTNPkt
from DTNNodeBuffer import DTNNodeBuffer
from DTNLogFiles import DTNLogFiles
class DTNScenario(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, list_idrouting):
        self.scenarioname = scenarioname
        # 传输速度 500kb/s
        self.transmitspeed = 500
        # 时间间隔 0.1s
        self.timestep = 0.1
        # 为各个node建立虚拟空间 <内存+router>
        self.listNodeBuffer = []
        for idrouting_tunple in list_idrouting:
            (node_id, routingname) = idrouting_tunple
            tmpBuffer = DTNNodeBuffer(self, node_id, routingname, maxsize=100*1000)
            self.listNodeBuffer.append(tmpBuffer)
        # 为了记录和对照方便 为每个节点记录成功投递的pkt list
        self.numofnodes = len(list_idrouting)
        # 建立正在传输的pkt_id 和 传输进度 的矩阵
        self.link_transmitpktid = np.zeros((self.numofnodes, self.numofnodes), dtype='int')
        self.link_transmitprocess = np.zeros((self.numofnodes, self.numofnodes), dtype='int')
        # 启动log
        self.filelog = DTNLogFiles()
        self.filelog.initlog(self.scenarioname)
        return

    # ==========================挂载在Scenario 上的 received、sent、gen接口=========================
    # 通知a_id: pkt已经发送
    def __notifypktsent(self, runningtime, a_id, b_id, i_pkt):
        self.listNodeBuffer[a_id].notifysentpkt(runningtime, b_id, i_pkt)
        return


    # 通知b_id: pkt需要接收<内存空间可能需要开辟>
    def __notifypktreceived(self, runningtime, b_id, a_id, i_pkt):
        self.listNodeBuffer[b_id].notifyreceivedpkt(runningtime, a_id, i_pkt)
        return


    # 通知newpkt.src_id: 新pkt生成<内存空间可能需要开辟>
    def __notifygennewpkt(self, runningtime, newpkt):
        self.listNodeBuffer[newpkt.src_id].mkroomaddpkt(newpkt)


    # =======================提供给DTNController的功能============================================
    # 提供给界面show的接口
    def getnodelist(self, node_id):
        return self.listNodeBuffer[node_id].getlistpkt()


    # scenario收到DTNcontroller指令, 打印routing结果
    def showres(self):
        # 获取成功投递的个数
        succnum = 0
        stroutput = self.scenarioname + 'succ_list: '
        for tmpnodebuffer in self.listNodeBuffer:
            tmpsuccnum, tmpstroutput = tmpnodebuffer.showres()
            stroutput = stroutput + tmpstroutput
            succnum = succnum + tmpsuccnum
        return succnum, stroutput


    # scenario收到DTNcontroller指令, 在srcid生成一个pkt(srcid->dstid)
    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.__notifygennewpkt(gentime, newpkt)
        return


    # scenario收到DTNcontroller指令, a_id <-> b_id 的 linkdown事件
    def linkdown(self, runningtime, a_id, b_id):
        self.link_transmitpktid[a_id][b_id] = 0
        self.link_transmitprocess[a_id][b_id] = 0
        self.filelog.insertlog(self.scenarioname, '[time_{}] [linkdown] a(node_{})<->b(node_{})\n'.format(
            runningtime, a_id, b_id))
        return


    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # 可传输数据量
        transmitvolume = self.transmitspeed * self.timestep
        # 如果存在正在传输的pkt
        if self.link_transmitpktid[a_id][b_id] != 0:
            transmitvolume = self.__c_transmitting(runningtime, a_id, b_id, transmitvolume)
        # 否则 选择新的pkt开始传输
        self.__transmitting(runningtime, a_id, b_id, transmitvolume)
        return


    # 如果a和b正在传输某个pkt, 则此时间间隔内 应当继续传输 之前的pkt
    # 返回 剩余的可传输量
    def __c_transmitting(self, runningtime, a_id, b_id, transmitvolume):
        # 继续传输之前的pkt
        tmp_pktid = self.link_transmitpktid[a_id][b_id]
        self.filelog.insertlog(self.scenarioname, '[time_{}] [c_tran] a(node_{})->b(node_{}):pkt(pkt_{}),progress({})\n'.format(
                                   runningtime, a_id, b_id, tmp_pktid, self.link_transmitprocess[a_id][b_id]))
        (isfound, target_pkt) = self.listNodeBuffer[a_id].findpktbyid(tmp_pktid)
        # 如果发生这种情况就是 刚好a_id把pkt传走了； 准备转发下一个pkt吧
        if isfound == False:
            self.filelog.insertlog(self.scenarioname,'[time_{}] [c_tran_intrupt] a(node_{})->b(node_{}):pkt(pkt_{})\n'.format(
                                       runningtime, a_id, b_id, tmp_pktid))
            self.link_transmitpktid[a_id][b_id] = 0
            self.link_transmitprocess[a_id][b_id] = 0
            return transmitvolume
        # 既然正在传输的标志位(link_transmitpktid[a_id][b_id])存在
        # 就一定有传输量(link_transmitprocess[a_id][b_id])存在
        assert (isfound == True)
        resumevolume = target_pkt.pkt_size - self.link_transmitprocess[a_id][b_id]
        if transmitvolume > resumevolume:
            remiantransmitvolume = transmitvolume - resumevolume
            # 本次传输结束 记得置空标志位
            self.link_transmitpktid[a_id][b_id] = 0
            self.link_transmitprocess[a_id][b_id] = 0
            # 通知 对应节点a_id pkt已发送; 通知 对应节点b_id pkt需接收;
            self.__notifypktsent(runningtime, a_id, b_id, target_pkt)
            self.__notifypktreceived(runningtime, b_id, a_id, target_pkt)
            # 还剩一些可传输量
            return remiantransmitvolume
        elif transmitvolume < resumevolume:
            self.link_transmitprocess[a_id][b_id] = self.link_transmitprocess[a_id][b_id] + transmitvolume
            return 0
        else:
            self.link_transmitpktid[a_id][b_id] = 0
            self.link_transmitprocess[a_id][b_id] = 0
            # 通知 对应节点a_id pkt已发送; 通知 对应节点b_id pkt需接收;
            self.__notifypktsent(runningtime, a_id, b_id, target_pkt)
            self.__notifypktreceived(runningtime, b_id, a_id, target_pkt)
            return 0


    # a_id -> b_id
    def __transmitting(self, runningtime, a_id, b_id, transmitvolume):
        # 如果没有正在传输的pkt, 从a的buffer里 顺序查找 b的buffer里没有的pkt
        # 建立准备传输的pkt列表(这应该是一个优先级的list)
        listpkt = self.listNodeBuffer[b_id].getlistpkt()
        totran_pktlist = self.listNodeBuffer[a_id].gettranpktlist(b_id, listpkt)
        for i_pkt in totran_pktlist:
            self.filelog.insertlog(self.scenarioname, '[time_{}] [tran] a(node_{})->b(node_{}):pkt(pkt_{})\n'.format(
                runningtime, a_id, b_id, i_pkt.pkt_id))
            # 开始传输i_pkt 可传输量消耗
            if i_pkt.pkt_size <= transmitvolume:
                transmitvolume = transmitvolume - i_pkt.pkt_size
                # 通知 对应节点a_id pkt已发送; 通知 对应节点b_id pkt需接收;
                self.__notifypktsent(runningtime, a_id, b_id, i_pkt)
                self.__notifypktreceived(runningtime, b_id, a_id, i_pkt)
            # 如果传不完了, 记录传输进度 transmitvolume
            else:
                self.link_transmitpktid[a_id][b_id] = i_pkt.pkt_id
                self.link_transmitprocess[a_id][b_id] = transmitvolume
                break
        return

