from Routing.RoutingBase import RoutingBase


class RoutingEpidemic(RoutingBase):
    def __init__(self, theBufferNode):
        super(RoutingEpidemic, self).__init__(theBufferNode)

    # 作为relay, 接收a_id发来的i_pkt吗？
    def decideAddafterRece(self, runningtime, a_id, i_pkt):
        is_add = True
        return is_add, RoutingBase.Rece_Code_AcceptPkt

    # 发送i_pkt给b_id 以后，决定要不要 从内存中删除
    def decideDelafterSend(self, b_id, i_pkt):
        is_del = False
        return is_del

    def get_values_before_tran(self, runningtime):
        pass