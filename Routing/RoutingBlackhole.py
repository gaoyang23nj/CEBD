from Routing.RoutingBase import RoutingBase


class RoutingBlackhole(RoutingBase):
    def __init__(self, theBufferNode):
        super(RoutingBlackhole, self).__init__(theBufferNode)

    def decideDelafterSend(self, b_id, i_pkt):
        is_del = False
        return is_del

    # 接收从a_id来的i_pkt以后, 决定要不要接收到内存里;
    # blackhole 永远不添加到自己的内存里
    def decideAddafterRece(self, runningtime, a_id, i_pkt):
        is_add = False
        return is_add, RoutingBase.Rece_Code_AcceptPkt

