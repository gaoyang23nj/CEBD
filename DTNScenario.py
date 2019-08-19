# 对给出的node_id list, 分别进行 DTNBuffer缓存初始化;
# DTNBuffer下挂载  Routing规则
# DTNBuffer (视为 内存方向的node)的 receive 通过DTNScenario挂载
import numpy as np
from DTNPkt import DTNPkt
from DTNNodeBuffer import DTNNodeBuffer
from DTNLogFiles import DTNLogFiles
from RoutingSparyandWait import *

class DTNScenario(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, list_idrouting):
        self.scenarioname = scenarioname
        # 传输速度 500kb/s
        self.transmitspeed = 500
        # 时间间隔 0.1s
        self.timestep = 0.1
        # 为了记录和对照方便 为每个节点记录成功投递的pkt list
        self.numofnodes = len(list_idrouting)
        # 为各个node建立虚拟空间 <内存+router>
        self.listNodeBuffer = []
        # 记录坏节点的个数
        self.listselfishid = []
        for idrouting_tunple in list_idrouting:
            (node_id, routingname) = idrouting_tunple
            tmpBuffer = DTNNodeBuffer(self, node_id, routingname, self.numofnodes)
            if routingname == 'RoutingBlackhole':
                self.listselfishid.append(node_id)
            self.listNodeBuffer.append(tmpBuffer)
        # 建立正在传输的pkt_id 和 传输进度 的矩阵
        self.link_transmitpktid = np.zeros((self.numofnodes, self.numofnodes), dtype='int')
        self.link_transmitprocess = np.zeros((self.numofnodes, self.numofnodes), dtype='int')
        # 启动log
        self.filelog = DTNLogFiles()
        self.filelog.initlog(self.scenarioname)
        return

    # =======================提供给DTNController的功能============================================
    # 提供给界面show的接口
    def getnodelist(self, node_id):
        return self.listNodeBuffer[node_id].getlistpkt()

    def getselfishlist(self):
        return self.listselfishid

    # scenario收到DTNcontroller指令, 打印routing结果
    def showres(self):
        # 获取成功投递的个数
        total_succnum = 0
        # selfish->normal normal->selfish selfish->selfish
        selfish_succnum = 0
        # normal->normal
        normal_succnum = 0
        total_succ_delay = 0
        selfish_succ_delay = 0
        normal_succ_delay = 0
        for tmpnodebuffer in self.listNodeBuffer:
            tmpsucclist = tmpnodebuffer.getlocalusage()
            for pktinfo in tmpsucclist:
                (pkt_id, src_id, dst_id, gentime, succtime) = pktinfo
                if (src_id in self.listselfishid) or (dst_id in self.listselfishid):
                    selfish_succnum += 1
                    selfish_succ_delay += succtime - gentime
                total_succ_delay += (succtime - gentime)
            total_succnum = total_succnum + len(tmpsucclist)
        normal_succnum = total_succnum-selfish_succnum
        normal_succ_delay = total_succ_delay - selfish_succ_delay
        return total_succnum, normal_succnum, selfish_succnum, \
               total_succ_delay, normal_succ_delay, selfish_succ_delay

    # ==================== 核心接口 生成报文 响应linkdown linkup 交换报文 等事件 =====================================
    # scenario收到DTNcontroller指令, 在srcid生成一个pkt(srcid->dstid)
    # 通知newpkt.src_id: 新pkt生成<内存空间可能需要开辟>
    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        self.listNodeBuffer[src_id].notifygennewpkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        return

    # scenario收到DTNcontroller指令, a_id <-> b_id 的 linkdown事件
    def linkdown(self, running_time, a_id, b_id):
        # 1)从b_id获取对应参数*args 2) 通知a_id： 与b_id 的 linkup事件
        values_a = self.listNodeBuffer[a_id].get_values_router_before_down(running_time)
        values_b = self.listNodeBuffer[b_id].get_values_router_before_down(running_time)
        # 告诉 a linkup事件, 并传给a 有关b的控制信息
        self.listNodeBuffer[a_id].notify_link_down(running_time, b_id, values_b)
        # 告诉 b linkup事件, 并传给b 有关a的控制信息
        self.listNodeBuffer[b_id].notify_link_down(running_time, a_id, values_a)
        # 设置link正在传输值参数
        self.link_transmitpktid[a_id][b_id] = 0
        self.link_transmitprocess[a_id][b_id] = 0
        self.filelog.insertlog(self.scenarioname, '[time_{}] [linkdown] a(node_{})<->b(node_{})\n'.format(
            running_time, a_id, b_id))
        return

    # scenario收到DTNcontroller指令, a_id <-> b_id 的 linkup事件
    # 更新 a b的 内部状态, 交换信息
    def linkup(self, running_time, a_id, b_id):
        # 1)获取对面给出的参数以便评价 2)通知a_id： 与b_id 的 linkdown事件; [同上]
        values_a = self.listNodeBuffer[a_id].get_values_router_before_up(running_time)
        values_b = self.listNodeBuffer[b_id].get_values_router_before_up(running_time)
        self.listNodeBuffer[a_id].notify_link_up(running_time, b_id, values_b)
        self.listNodeBuffer[b_id].notify_link_up(running_time, a_id, values_a)
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

    # 如果a和b正在传输某个pkt, 则此时间间隔内 应当继续传输 之前的pkt, 返回 剩余的可传输量
    def __c_transmitting(self, runningtime, a_id, b_id, transmitvolume):
        # 继续传输之前的pkt
        tmp_pktid = self.link_transmitpktid[a_id][b_id]
        self.filelog.insertlog(self.scenarioname, '[time_{}] [c_tran] a(node_{})->b(node_{}):pkt(pkt_{}),progress({})\n'.format(
                                   runningtime, a_id, b_id, tmp_pktid, self.link_transmitprocess[a_id][b_id]))
        (isfound, target_pkt) = self.listNodeBuffer[a_id].findpktbyid(tmp_pktid)
        # 如果发生这种情况就是 刚好a_id把pkt传走了； 准备转发下一个pkt吧
        if isfound == False:
            self.filelog.insertlog(self.scenarioname, '[time_{}] [c_tran_intrupt] a(node_{})->b(node_{}):pkt(pkt_{})\n'.format(
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
            # 接收时 使用的是副本 可以先处理
            self.__sendandreceivepkt(a_id, b_id, runningtime, target_pkt)
            # 还剩一些可传输量
            return remiantransmitvolume
        elif transmitvolume < resumevolume:
            self.link_transmitprocess[a_id][b_id] = self.link_transmitprocess[a_id][b_id] + transmitvolume
            return 0
        else:
            self.link_transmitpktid[a_id][b_id] = 0
            self.link_transmitprocess[a_id][b_id] = 0
            # 通知 对应节点a_id pkt已发送; 通知 对应节点b_id pkt需接收;
            # 接收时 使用的是副本 可以先处理
            self.__sendandreceivepkt(a_id, b_id, runningtime, target_pkt)
            return 0

    # a_id -> b_id
    def __transmitting(self, runningtime, a_id, b_id, transmitvolume):
        # 如果没有正在传输的pkt, 从a的buffer里 顺序查找 b的buffer里没有的pkt
        # 1）建立准备传输的pkt列表(这应该是一个优先级的list)
        listpkt = self.listNodeBuffer[b_id].getlistpkt()
        # 2）传输前 对端node 提供一些 控制信息 协助判断是否要 转发该报文// i.e. Prophet
        values_b = self.listNodeBuffer[b_id].get_values_router_before_tran(runningtime)
        # 3) 由router判断哪些pkt需要传输
        totran_pktlist = self.listNodeBuffer[a_id].gettranpktlist(runningtime, b_id, listpkt, values_b)
        for target_pkt in totran_pktlist:
            self.filelog.insertlog(self.scenarioname, '[time_{}] [tran] a(node_{})->b(node_{}):pkt(pkt_{})\n'.format(
                runningtime, a_id, b_id, target_pkt.pkt_id))
            # 开始传输i_pkt 可传输量消耗
            if target_pkt.pkt_size <= transmitvolume:
                transmitvolume = transmitvolume - target_pkt.pkt_size
                # 通知 对应节点a_id pkt已发送; 通知 对应节点b_id pkt需接收;
                # 接收时 使用的是副本 可以先处理
                self.__sendandreceivepkt(a_id, b_id, runningtime, target_pkt)
            # 如果传不完了, 记录传输进度 transmitvolume
            else:
                self.link_transmitpktid[a_id][b_id] = target_pkt.pkt_id
                self.link_transmitprocess[a_id][b_id] = transmitvolume
                break
        return

    # a_id -> b_id 传输pkt: 传输量已经足够 正在拷贝
    def __sendandreceivepkt(self, a_id, b_id, runningtime, target_pkt):
        # b_id被通知: 将要收到来自a_id的pkt.
        # b_id做出对a_id的回应, b_id是否复制<开辟内存空间,修改字段>/是否欺骗
        codeRece = self.listNodeBuffer[b_id].notifyreceivedpkt(runningtime, a_id, target_pkt)
        # a_id被b_id通知: b_id已经收到了target_pkt
        # a_id将可能执行：修改原pkt字段(token), 删除pkt
        self.listNodeBuffer[a_id].notifysentpkt(runningtime, codeRece, b_id, target_pkt)
