import numpy as np
import threading
from DTNLogFiles import DTNLogFiles
from DTNScenario import DTNScenario

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

    def __getscenarioname(self, scenaname):
        if not scenaname in self.scenaDict.keys():
            print('ERROR! DTNController scenaDict 里 没有此key')
        else:
            return self.scenaDict[scenaname]


    def closeApp(self):
        # 关闭log
        self.filelog.closelog()
        # 关闭定时器
        self.__scenarioshowres()
        self.t.cancel()


    def setTimerRunning(self, run):
        # 停止定时器
        self.timerisrunning = run


    # 执行指定的步数
    def updateOnce(self, updatetimesOnce):
        for i in range(updatetimesOnce):
            self.executeOnce()


    # View定时刷新机制
    def run(self, totaltime = 36000):
        # 初始化各个routing
        list_scena = self.__scenarioinit()
        # 定时器刷新位
        self.timerisrunning = True
        infotext = 'StepTime:'+ str(self.list_node[0].steptime)+' Times_showsteps:'+ str(self.times_showtstep) + ' Commu_range:'+ str(self.range_comm)
        self.DTNView.initshow(infotext, list_scena)
        # 启动定时刷新机制
        self.t = threading.Timer(0.1, self.updateViewer)
        self.t.start()
        self.DTNView.mainloop()


    def updateViewer(self):
        # 是否收到停止的命令
        if not self.timerisrunning:
            self.t.cancel()
            self.__scenarioshowres()
            return
        # 按照times_showtstep的指定 执行showtimes次
        self.executeOnce()
        # 如果执行到最后的时刻，则停止下一次执行
        if self.RunningTime < self.RunningTime_Max:
            # 没有到结束的时候, 设置定时器 下次更新视图
            self.t = threading.Timer(0.1, self.updateViewer)
            self.t.start()
            return
        else:
            # 到结束的时候, 打印routing结果
            self.__scenarioshowres()
            self.DTNView.on_closing()
            return


    # 完成self.showtimes规定次数的移动和计算 并更新界面
    def executeOnce(self):
        # 完成 self.showtimes 个 timestep的位置更新变化，routing变化
        encounter_list = []
        for i in range(self.times_showtstep):
            encounter_list = self.run_onetimestep()
        # 获取self.DTNView指定的routing显示
        # 提供给Viewer显示Canvas
        tunple_list = []
        for node in self.list_node:
            tunple = (node.getNodeId(), node.getNodeLoc(), node.getNodeDest())
            tunple_list.append(tunple)
        self.DTNView.updateCanvaShow(tunple_list, encounter_list)
        # 提供给Viewer info
        scenaobj = self.__getscenarioname(self.DTNView.getscenaname())
        info_nodelist = []
        for node in self.list_node:
            node_id = node.getNodeId()
            # tmplist = []
            # for pkt in scenaobj.getnodelist(node_id):
            #     tmplist.append(pkt.pkt_id)
            tunple = (node_id, scenaobj.getnodelist(node_id))
            info_nodelist.append(tunple)
        info_pktlist = self.list_genpkt
        self.DTNView.updateInfoShow(info_nodelist, info_pktlist)

    # 完成一个timestep的移动 计算 routing
    def run_onetimestep(self):
        # 报文生成计数器
        if self.cnt_genpkt == self.thr_genpkt:
            self.__scenariogenpkt()
            self.cnt_genpkt = 1
        else:
            self.cnt_genpkt =  self.cnt_genpkt + 1
        # 节点移动一个timestep
        for node in self.list_node:
            node.run()
        # 检测linkdown事件
        self.detectlinkdown()
        # 检测相遇事件
        encounter_list = self.detectencounter()
        # 如果执行到最后的时刻，则停止下一次执行
        self.RunningTime = self.RunningTime + 1
        return encounter_list


    # 检测相遇事件
    def detectencounter(self):
        encounter_list = []
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
                    self.__scenarioswap(a_id, b_id)
                    self.mt_linkstate[b_id][a_id] = 1
                    self.__scenarioswap(b_id, a_id)
                    encounter_list.append((a_id, b_id, a_loc, b_loc))
        return encounter_list


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
                        self.__scenariolinkdown(a_id, b_id)
                        self.mt_linkstate[b_id][a_id] = 0
                        self.__scenariolinkdown(b_id, a_id)


    # =========提供给 scenario的接口, init gennewpkt swap linkdown showres============================
    # 各个scenario生成报文
    def __scenariogenpkt(self):
        src_index = np.random.randint(len(self.list_node))
        dst_index = np.random.randint(len(self.list_node))
        while dst_index == src_index:
            dst_index = np.random.randint(len(self.list_node))
        newpkt = (self.pktid_nextgen, src_index, dst_index)
        # controller记录这个pkt
        self.list_genpkt.append(newpkt)
        # 各scenario生成pkt, pkt大小为100k
        for key, value in self.scenaDict.items():
            value.gennewpkt(self.pktid_nextgen, src_index, dst_index, self.RunningTime, 100)
        # 生成报文生成的log
        self.filelog.insertlog('eve','[time_{}] [packet gen] pkt_{}:src(node_{})->dst(node_{})\n'.format(
                  self.RunningTime, self.pktid_nextgen, src_index, dst_index))
        self.pktid_nextgen = self.pktid_nextgen + 1
        return

    # 各个scenario开始交换报文
    def __scenarioswap(self, a_id, b_id):
        for key, value in self.scenaDict.items():
            value.swappkt(self.RunningTime, a_id, b_id)


    # 各个scenario收到linkdown事件
    def __scenariolinkdown(self, a_id, b_id):
        for key, value in self.scenaDict.items():
            value.linkdown(self.RunningTime, a_id, b_id)

    # 各个scenario显示结果
    def __scenarioshowres(self):
        for key, value in self.scenaDict.items():
            succ_num, stroutput = value.showres()
            print('\n{} succ_num:{}'.format(key, succ_num))
            print(stroutput)

   # 初始化各个路由场景 并返回 场景名的list
    def __scenarioinit(self):
        list_scena = ['scenario1', 'scenario2', 'scenario3', 'scenario4', 'scenario5']
        self.scenaDict = {}

        # ===============================场景1 全ep routing===================================
        list_idrouting = []
        for movenode in self.list_node:
            list_idrouting.append((movenode.node_id, 'RoutingEpidemic'))
        self.scenario1 = DTNScenario(list_scena[0], list_idrouting, 'DTNPkt')
        self.scenaDict.update({list_scena[0]: self.scenario1})

        # ===============================场景2 全sw routing===================================
        list_idrouting = []
        for movenode in self.list_node:
            list_idrouting.append((movenode.node_id, 'RoutingSparyandWait'))
        self.scenario2 = DTNScenario(list_scena[1], list_idrouting, 'DTNSWPkt')
        self.scenaDict.update({list_scena[1]: self.scenario2})

        # ===============================场景3 设置10%的dropping node===================================
        # 随机生成序列
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(0.5 * len(self.list_node))]
        normal_indices = indices[int(0.5 * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingSparyandWait'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init!')
            id = id + 1
        self.scenario3 = DTNScenario(list_scena[2], list_idrouting, 'DTNSWPkt')
        self.scenaDict.update({list_scena[2]: self.scenario3})

        # ===============================场景4 设置30%的dropping node===================================
        # 随机生成序列
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(0.5 * len(self.list_node))]
        normal_indices = indices[int(0.5 * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingSparyandWait'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init!')
            id = id + 1
        self.scenario4 = DTNScenario(list_scena[3], list_idrouting, 'DTNSWPkt')
        self.scenaDict.update({list_scena[3]: self.scenario4})

        # ===============================场景5 设置50%的dropping node===================================
        # 随机生成序列
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(0.5 * len(self.list_node))]
        normal_indices = indices[int(0.5 * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingSparyandWait'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init! id: ', id)
            id = id + 1
        self.scenario5 = DTNScenario(list_scena[4], list_idrouting, 'DTNSWPkt')
        self.scenaDict.update({list_scena[4]: self.scenario5})

        return list_scena
