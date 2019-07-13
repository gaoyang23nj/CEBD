import numpy as np
import threading
from RoutingEpidemic import RoutingEpidemic
from DTNLogFiles import DTNLogFiles

class DTNController(object):
    def __init__(self, dtnview, showtimes=100, com_range=100, genfreq_cnt=6000, totaltimes=36000):
        self.DTNView = dtnview
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
        self.pktid_nextgen = 1
        # 总的执行次数 设置
        self.RunningTime_Max = totaltimes
        self.RunningTime = 0
        self.timerisrunning = False
        # 启动log
        self.filelog = DTNLogFiles()
        self.filelog.initlog('eve')

    # attach node_list
    def attachnodelist(self, nodelist):
        assert(len(self.list_node)==0)
        self.list_node.extend(nodelist)
        self.nr_nodes = len(self.list_node)
        # 记录任何两个node之间的link状态
        self.mt_linkstate = np.zeros((self.nr_nodes,  self.nr_nodes),dtype='int')
        self.mt_tranprocess = np.zeros((self.nr_nodes,  self.nr_nodes),dtype='int')

    def closeApp(self):
        # 关闭log
        self.filelog.closelog()
        # 关闭定时器
        self.setTimerRunning(False)
        self.__routingshowres()
        self.t.cancel()


    def setTimerRunning(self, run):
        # 停止定时器
        self.timerisrunning = run

    # 执行指定的步数
    def updateOnce(self, updatetimesOnce):
        for i in range(updatetimesOnce):
            self.executeOnce()

    # View定时刷新机制
    def run(self, totaltime=36000):
        # 初始化各个routing
        self.__routinginit()
        # 定时器刷新位
        self.timerisrunning = True
        # 启动定时刷新机制
        self.t = threading.Timer(0.1, self.updateViewer)
        self.t.start()
        infotext = 'StepTime:'+ str(self.list_node[0].steptime)+' Times_showsteps:'+ str(self.times_showtstep) + ' Commu_range:'+ str(self.range_comm)
        self.DTNView.initshow(infotext)

    def updateViewer(self):
        # 是否收到停止的命令
        if not self.timerisrunning:
            self.t.cancel()
            self.__routingshowres()
            return
        # 按照times_showtstep的指定 执行showtimes次
        self.executeOnce()

        # 如果执行到最后的时刻，则停止下一次执行
        # self.runtimes_cur = self.runtimes_cur + 1
        if self.RunningTime < self.RunningTime_Max:
            # 没有到结束的时候, 设置定时器 下次更新视图
            self.t = threading.Timer(0.1, self.updateViewer)
            self.t.start()
            return
        else:
            # 到结束的时候, 打印routing结果
            self.__routingshowres()
            return

    # 完成self.showtimes规定次数的移动和计算 并更新界面
    def executeOnce(self):
        # 完成 self.showtimes 个 timestep的位置更新变化，routing变化
        for i in range(self.times_showtstep):
            self.run_onetimestep()
        # 获取self.DTNView指定的routing显示
        selected_routing = self.__getsrouting(self.DTNView.getroutingname())
        # 提供给Viewer显示Canvas
        tunple_list = []
        for node in self.list_node:
            tunple = (node.getNodeId(), node.getNodeLoc(), node.getNodeDest())
            tunple_list.append(tunple)
        self.DTNView.updateCanvaShow(tunple_list)
        # 提供给Viewer info
        info_nodelist = []
        for node in self.list_node:
            node_id = node.getNodeId()
            tunple = (node_id, selected_routing.getnodelist(node_id))
            info_nodelist.append(tunple)
        info_pktlist = self.list_genpkt
        self.DTNView.updateInfoShow((info_nodelist, info_pktlist))

    # 完成一个timestep的移动 计算 routing
    def run_onetimestep(self):
        # 报文生成计数器
        if self.cnt_genpkt == self.thr_genpkt:
            self.__routinggenpkt()
            self.cnt_genpkt = 1
        else:
            self.cnt_genpkt =  self.cnt_genpkt + 1
        # 节点移动一个timestep
        for node in self.list_node:
            node.run()
        # 检测linkdown事件
        self.detectlinkdown()
        # 检测相遇事件
        self.detectencounter()
        # 如果执行到最后的时刻，则停止下一次执行
        self.RunningTime = self.RunningTime + 1

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
                    if self.mt_linkstate[a_id][b_id] == 0:
                        self.filelog.insertlog('eve','[time_{}] [link_up] a(node_{})<->b(node_{})\n'.format(
                            self.RunningTime, a_id, b_id))
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
                        self.filelog.insertlog('eve','[time_{}] [link_down] a(node_{})<->b(node_{})\n'.format(
                                self.RunningTime, a_id, b_id))
                        self.mt_linkstate[a_id][b_id] = 0
                        self.__routinglinkdown(a_id, b_id)
                        self.mt_linkstate[b_id][a_id] = 0
                        self.__routinglinkdown(a_id, b_id)


    def __routinginit(self):
        self.epidemicrouting = RoutingEpidemic(len(self.list_node))

    # 各个routing 生成报文
    def __routinggenpkt(self):
        src_index = np.random.randint(len(self.list_node))
        dst_index = np.random.randint(len(self.list_node))
        while dst_index==src_index:
            dst_index = np.random.randint(len(self.list_node))
        newpkt = (self.pktid_nextgen, src_index, dst_index)
        self.list_genpkt.append(newpkt)
        # 各routing生成pkt, pkt大小为100k
        self.epidemicrouting.gennewpkt(self.pktid_nextgen, src_index, dst_index, 0, 100)
        # 生成报文生成的log
        self.filelog.insertlog('eve','[time_{}] [packet gen] pkt_{}:src(node_{})->dst(node_{})\n'.format(
                  self.RunningTime, self.pktid_nextgen, src_index, dst_index))
        self.pktid_nextgen = self.pktid_nextgen + 1
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

    def __getsrouting(self, routingname):
        if routingname == 'epidemicrouting':
            return self.epidemicrouting
        # elif routingname == 'prophetrouting':
        #     return self.prophetrouting
