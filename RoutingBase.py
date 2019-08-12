# -*- coding: UTF-8 -*-
class RoutingBase(object):
    def __init__(self, theBufferNode):
        self.theBufferNode = theBufferNode

    # 根据对方node的pkt存储状态 和 自身存储状态, router 提供 准备传输的pktlist
    # 顺序地得到准备传输的list(b_id里没有的pkt), dst_id是b_id的pkt应该最先传
    def gettranpktlist(self, runningtime, b_id, listb, a_id, lista, *args):
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

    # 响应 linkup linkdown事件
    def get_values_before_up(self, runningtime):
        pass

    def notify_link_up(self, running_time, b_id, *args):
        pass

    def get_values_before_down(self, runningtime):
        pass

    def notify_link_down(self, running_time, b_id, *args):
        pass

    def get_values_before_tran(self, runningtime):
        pass

    @classmethod
    def decideAddafterRece(cls, a_id, i_pkt):
        '''
        :return:
        '''

    @classmethod
    def decideDelafterSend(cls, b_id, i_pkt):
        '''

        :return:
        '''





