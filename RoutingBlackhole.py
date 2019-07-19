from RoutingBase import RoutingBase


class RoutingBlackhole(RoutingBase):
    def __init__(self, theBufferNode):
        super(RoutingBlackhole, self).__init__(theBufferNode)


    # 根据对方node的pkt存储状态 和 自身存储状态, router 提供 准备传输的pktlist
    # 顺序地得到准备传输的list(b_id里没有的pkt), dst_id是b_id的pkt应该最先传
    def gettranpktlist(self, b_id, listb, a_id, lista):
        totran_pktlist = []
        for i_pkt in lista:
            isiexist = False
            # 如果pkt_id相等, 是同一个pkt 不必再传输
            for j_pkt in listb:
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


    def decideDelafterSend(self, b_id, i_pkt):
        isDel = False
        return isDel


    # 接收从a_id来的i_pkt以后, 决定要不要接收到内存里;
    # blackhole 永远不添加到自己的内存里
    def decideAddafterRece(self, a_id, i_pkt):
        isAdd = False
        return isAdd

