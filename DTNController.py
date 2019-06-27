import numpy as np
from RoutingEpidemic import RoutingEpidemic

class DTNController(object):
    def __init__(self, showtimes=100, com_range=100, genfreq_cnt=6000):
        # 一次刷新view 内部更新的timstep个数
        self.times_showtstep = showtimes
        # 通信范围
        self.range_comm = com_range

        # 保留node的list
        self.list_node = []
        self.nr_nodes = 0
        # link状态记录 以便检测linkdown事件
        self.mt_linkstate = np.empty((0,0),dtype='int')
        # 传输进度记录 以便确定上个timestep传输内容的量
        self.mt_tranprocess = np.empty((0, 0), dtype='int')

        # 全部生成报文的list
        self.list_genpkt = []
        # 生成报文的时间计数器 & 生成报文计算器的触发值
        self.cnt_genpkt = genfreq_cnt
        self.thr_genpkt = genfreq_cnt
        # 下一个pkt的id
        self.id_nextgenpkt = 1

    # attach node_list
    def attachnodelist(self, nodelist):
        assert(len(self.list_node)==0)
        self.list_node.extend(nodelist)
        self.nr_nodes = len(self.list_node)
        # 记录任何两个node之间的link状态
        self.mt_linkstate = np.zeros((self.nr_nodes,  self.nr_nodes),dtype='int')
        self.mt_tranprocess = np.zeros((self.nr_nodes,  self.nr_nodes),dtype='int')

    # View定时刷新机制

    #
    def run_onetimestep(self):
        # 报文生成计数器
        if self.cnt_genpkt == self.thr_genpkt:
            self.__routinggenpkt()
            self.cnt_genpkt = 1
        else:
            self.cnt_genpkt =  self.cnt_genpkt + 1

        # 节点移动一个timestep
        tunple_list = []
        for node in self.list_node:
            loc = node.run()
            tmp_tunple = (node.getNodeId(), loc, node.getNodeDest())
            tunple_list.append(tmp_tunple)
        # 检测linkdown事件
        self.detectlinkdown()
        # 检测相遇事件
        self.detectencounter()

    # 检测相遇事件
    def detectencounter(self):
        for a_index in range(len(self.list_node)):
            a_node = self.list_node[a_index]
            a_id = a_node.getNodeId()
            a_loc = a_node.getNodeLoc()
            for b_index in range(a_index + 1, len(self.list_node), 1):
                b_node = self.list_node[b_index]
                b_id = b_node.getNodeId()
                b_loc = b_node.getNodeLoc()
                # 如果在通信范围内 交换信息
                # 同时完成a->b b->a
                if np.sqrt(np.dot(a_loc - b_loc, a_loc - b_loc)) < self.range_comm:
                    self.mt_linkstate[a_id][b_id] = 1
                    self.__routingswap(a_id, b_id)
                    self.mt_linkstate[b_id][a_id] = 1
                    self.__routingswap(b_id, a_id)

    # 检测linkdown事件
    def detectlinkdown(self):
        for a_index in range(len(self.list_node)):
            a_node = self.list_node[a_index]
            a_id = a_node.getNodeId()
            a_loc = a_node.getNodeLoc()
            for b_index in range(a_index+1, len(self.list_node), 1):
                b_node = self.list_node[b_index]
                b_id = b_node.getNodeId()
                b_loc = b_node.getNodeLoc()
                # linkstate 的连通是相互的, linkdown事件也是
                if self.mt_linkstate[a_id][b_id] == 1:
                    if np.sqrt(np.dot(a_loc - b_loc, a_loc - b_loc)) >= self.range_comm:
                        self.mt_linkstate[a_id][b_id] = 0
                        self.__routinglinkdown(a_id, b_id)
                        self.mt_linkstate[b_id][a_id] = 0
                        self.__routinglinkdown(a_id, b_id)


    # def __routinginit(self):
        self.epidemicrouting = RoutingEpidemic(len(self.list_node))

    # 各个routing 生成报文
    def __routinggenpkt(self):
        src_index = np.random.randint(len(self.list_node))
        dst_index = np.random.randint(len(self.list_node))
        while dst_index==src_index:
            dst_index = np.random.randint(len(self.list_node))
        newpkt = (self.genfreq_pktid, src_index, dst_index)
        self.list_genpkt.append(newpkt)

        # 各routing生成pkt, pkt大小为100k
        self.epidemicrouting.gennewpkt(self.genfreq_pktid, src_index, dst_index, 0, 100)

        self.genfreq_pktid = self.genfreq_pktid + 1
        return

    # 各个routing开始交换报文
    def __routingswap(self, a_id, b_id):
        self.epidemicrouting.swappkt(a_id, b_id)
        self.epidemicrouting.swappkt(b_id, a_id)

    # 各个routing收到linkdown事件
    def __routinglinkdown(self, a_id, b_id):
        self.epidemicrouting.linkdown(a_id, b_id)
        self.epidemicrouting.linkdown(b_id, a_id)

    # 各个routing显示结果
    def __routingshowres(self):
        self.epidemicrouting.showres()
