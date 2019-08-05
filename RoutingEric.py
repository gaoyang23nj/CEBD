from RoutingBase import RoutingBase
# 2018TONSM
# A Dynamic Trust Framework for Opportunistic Mobile Social Networks
# 2019-07-27



class RoutingEric(RoutingBase):
    def __init__(self, theBufferNode):
        super(RoutingEric, self).__init__(theBufferNode)
        self.node_id = self.theBufferNode.node_id

