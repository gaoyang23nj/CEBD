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
            # 通知 对应节点a_id pkt已发送; 通知 对应节点b_id pkt需接收;
            # 接收时 使用的是副本 可以先处理
            self.__sendandreceivepkt(a_id, b_id, runningtime, target_pkt)

        return

