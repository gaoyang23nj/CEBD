import math
from DTNPkt import DTNPkt
from RoutingBase import RoutingBase

# 需要在pkt里加一个token属性
class DTNSWPkt(DTNPkt):
    def __init__(self, pkt_id, src_id, dst_id, gentime, pkt_size, token):
        super(DTNSWPkt, self).__init__(self, pkt_id, src_id, dst_id, gentime, pkt_size)
        self.token = token


    def __init__(self, dtnpkt, token):
        super(DTNSWPkt, self).__init__(dtnpkt.pkt_id, dtnpkt.src_id, dtnpkt.dst_id, dtnpkt.gentime, dtnpkt.pkt_size)
        self.token = token


class RoutingSparyandWait(RoutingBase):
    def __init__(self, theBufferNode, inittoken = 8):
        super(RoutingSparyandWait, self).__init__(theBufferNode)
        self.inittoken = inittoken


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
                # 作为relay 只有token>=2的时候 才转发
                else:
                    if i_pkt.token >= 2:
                        totran_pktlist.append(i_pkt)
        return totran_pktlist


    # 发送i_pkt给b_id 以后，决定要不要 从内存中删除
    # 依赖于 浅拷贝, 才能修改 a_id里的 token值
    def decideDelafterSend(self, b_id, i_pkt):
        isDel = False
        i_pkt.token = math.ceil(i_pkt.token / 2)
        return isDel


    # 作为relay, 接收a_id发来的i_pkt吗？
    # 依赖于 浅拷贝, 才能修改 a_id里的 token值
    def decideAddafterRece(self, a_id, target_pkt):
        isAdd = True
        if math.floor(target_pkt.token/2) >= 1:
            target_pkt.token = math.floor(target_pkt.token / 2)
        # 正常情况下 发起tran之前 a_id已经保证pkt有足够的token
        # 接收时 不会出现token不够的情况
        else:
            print('ERROR! SWRouting token补救措施!')
            isAdd = False
        return isAdd