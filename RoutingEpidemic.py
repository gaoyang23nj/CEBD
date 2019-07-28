import numpy as np
import copy
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase
from DTNLogFiles import DTNLogFiles

class RoutingEpidemic(RoutingBase):
    def __init__(self, theBufferNode):
        super(RoutingEpidemic, self).__init__(theBufferNode)


    # 作为relay, 接收a_id发来的i_pkt吗？
    def decideAddafterRece(self, a_id, i_pkt):
        isAdd = True
        return isAdd


    # 发送i_pkt给b_id 以后，决定要不要 从内存中删除
    def decideDelafterSend(self, b_id, i_pkt):
        isDel = False
        return isDel



