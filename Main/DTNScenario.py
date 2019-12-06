# 对给出的node_id list, 分别进行 DTNBuffer缓存初始化;
# DTNBuffer下挂载  Routing规则
# DTNBuffer (视为 内存方向的node)的 receive 通过DTNScenario挂载
from Main.DTNNodeBuffer import DTNNodeBuffer


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
        # 记录坏节点 是哪些
        self.listselfishid = []
        for idrouting_tunple in list_idrouting:
            (node_id, routingname) = idrouting_tunple
            tmpBuffer = DTNNodeBuffer(self, node_id, routingname, self.numofnodes)
            if routingname == 'RoutingBlackhole':
                self.listselfishid.append(node_id)
            self.listNodeBuffer.append(tmpBuffer)
        return


    # ==================== 核心接口 生成报文 响应交换报文  =====================================
    # scenario收到DTNcontroller指令, 在srcid生成一个pkt(srcid->dstid)
    # 通知newpkt.src_id: 新pkt生成<内存空间可能需要开辟>
    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        self.listNodeBuffer[src_id].notifygennewpkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        return

    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        listpkt = self.listNodeBuffer[b_id].getlistpkt()
        # 2）传输前 对端node 提供一些 控制信息 协助判断是否要 转发该报文// i.e. Prophet
        values_b = self.listNodeBuffer[b_id].get_values_router_before_tran(runningtime)
        # 3) 由router判断哪些pkt需要传输
        totran_pktlist = self.listNodeBuffer[a_id].gettranpktlist(runningtime, b_id, listpkt, values_b)
        for target_pkt in totran_pktlist:
            codeRece = self.listNodeBuffer[b_id].notifyreceivedpkt(runningtime, a_id, target_pkt)
            # a_id被b_id通知: b_id已经收到了target_pkt
            # a_id将可能执行：修改原pkt字段(token), 删除pkt
            self.listNodeBuffer[a_id].notifysentpkt(runningtime, codeRece, b_id, target_pkt)
        return

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

