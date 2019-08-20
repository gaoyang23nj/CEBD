import numpy as np
import threading
import datetime
from DTNLogFiles import DTNLogFiles
from DTNScenario import DTNScenario

class DTNControllerNoShow(object):
    def __init__(self, times_showtstep=100, range_comm=100, genfreq_cnt=6000, totaltimes=36000, isshowing=False):
        self.isshowing = isshowing
        # 一次刷新view 内部更新的timstep个数
        self.times_showtstep = times_showtstep
        # 通信范围
        self.range_comm = range_comm

        # 保留node的list
        self.list_node = []
        self.nr_nodes = 0
        # link状态记录 以便检测linkdown事件
        self.mt_linkstate = np.empty((0, 0), dtype='int')
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
        # self.filelog.initlog('eve')

    # attach node_list
    def attachnodelist(self, nodelist):
        assert(len(self.list_node) == 0)
        self.list_node.extend(nodelist)
        self.nr_nodes = len(self.list_node)
        # 记录任何两个node之间的link状态
        self.mt_linkstate = np.zeros((self.nr_nodes,  self.nr_nodes), dtype='int')
        self.mt_tranprocess = np.zeros((self.nr_nodes,  self.nr_nodes), dtype='int')

    def __getscenarioname(self, scenaname):
        if scenaname not in self.scenaDict.keys():
            print('ERROR! DTNController scenaDict 里 没有此key')
        else:
            return self.scenaDict[scenaname]

    # View定时刷新机制
    def run(self):
        short_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = 'result_'+short_time+'.tmp'
        # 初始化各个routing
        list_scena = self.__scenarioinit()
        # 按照times_showtstep的指定 执行showtimes次
        while True:
            self.run_onetimestep()
            # 没有到结束的时候, 继续
            if self.RunningTime < self.RunningTime_Max:
                # 每10s 显示一下临时结果
                if self.RunningTime % 100 == 0:
                    self.print_tmp_res()
                # 每1h 写入一下结果
                if self.RunningTime % 3600 == 0:
                    self.print_res(filename, ctstring='a+')
            else:
                # 到结束的时候, 打印routing结果
                break
        self.print_res(filename, ctstring='a+')
        return

    # 完成一个timestep的移动 计算 routing
    def run_onetimestep(self):
        # 报文生成计数器
        if self.cnt_genpkt == self.thr_genpkt:
            self.__scenariogenpkt()
            self.cnt_genpkt = 1
        else:
            self.cnt_genpkt = self.cnt_genpkt + 1
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
                        # self.filelog.insertlog('eve','[time_{}] [link_up] a(node_{})<->b(node_{})\n'.format(
                        #     self.RunningTime, a_id, b_id))
                        # 更新状态 交换 控制信息
                        self.__scenario_link_up(a_id, b_id)
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
                        # self.filelog.insertlog('eve','[time_{}] [link_down] a(node_{})<->b(node_{})\n'.format(
                        #         self.RunningTime, a_id, b_id))
                        self.mt_linkstate[a_id][b_id] = 0
                        self.mt_linkstate[b_id][a_id] = 0
                        self.__scenario_link_down(a_id, b_id)

    # ========================= 核心接口 调用scenario的接口, init gennewpkt swap linkdown showres============================
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
        # self.filelog.insertlog('eve','[time_{}] [packet gen cnt:{}] pkt_{}:src(node_{})->dst(node_{})\n'.format(
        #           self.RunningTime, self.cnt_genpkt, self.pktid_nextgen, src_index, dst_index))
        self.pktid_nextgen = self.pktid_nextgen + 1
        return

    # =========================== 核心接口 调用 scenario类的成员函数 执行linkup linkdown事件中控制信息的交换, 执行报文交换
    # 各个scenario收到linkup事件
    def __scenario_link_up(self, a_id, b_id):
        for key, value in self.scenaDict.items():
            value.linkup(self.RunningTime, a_id, b_id)

    # 各个scenario收到linkdown事件
    def __scenario_link_down(self, a_id, b_id):
        for key, value in self.scenaDict.items():
            value.linkdown(self.RunningTime, a_id, b_id)

    # 各个scenario开始交换报文
    def __scenarioswap(self, a_id, b_id):
        for key, value in self.scenaDict.items():
            value.swappkt(self.RunningTime, a_id, b_id)

    # =================== 场景初始化 ============================
    def __TestScienario1_init(self):
        self.scenaDict = {}
        index = 0
        # ===============================场景1 全ep routing===================================
        list_idrouting = []
        for movenode in self.list_node:
            list_idrouting.append((movenode.node_id, 'RoutingEpidemic'))
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景3 设置10%的dropping node===================================
        index += 1
        # 随机生成序列
        percent_selfish = 0.1
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(percent_selfish * len(self.list_node))]
        normal_indices = indices[int(percent_selfish * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingEpidemic'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init!')
            id = id + 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景4 设置30%的dropping node===================================
        index += 1
        # 随机生成序列
        percent_selfish = 0.3
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(percent_selfish * len(self.list_node))]
        normal_indices = indices[int(percent_selfish * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingEpidemic'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init!')
            id = id + 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景5 设置50%的dropping node===================================
        index += 1
        # 随机生成序列
        percent_selfish = 0.5
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(percent_selfish * len(self.list_node))]
        normal_indices = indices[int(percent_selfish * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingEpidemic'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init! id: ', id)
            id = id + 1
        tmp_senario_name = 'scenario'+str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景4 设置70%的dropping node===================================
        index += 1
        # 随机生成序列
        percent_selfish = 0.7
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(percent_selfish * len(self.list_node))]
        normal_indices = indices[int(percent_selfish * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingEpidemic'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init!')
            id = id + 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景4 设置90%的dropping node===================================
        index += 1
        # 随机生成序列
        percent_selfish = 0.9
        indices = np.random.permutation(len(self.list_node))
        malicious_indices = indices[: int(percent_selfish * len(self.list_node))]
        normal_indices = indices[int(percent_selfish * len(self.list_node)):]
        list_idrouting = []
        id = 0
        for movenode in self.list_node:
            if id in normal_indices:
                list_idrouting.append((movenode.node_id, 'RoutingEpidemic'))
            elif id in malicious_indices:
                list_idrouting.append((movenode.node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init!')
            id = id + 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})

        list_scena = list(self.scenaDict.keys())
        return list_scena

   # 初始化各个路由场景 并返回 场景名的list
    def __scenarioinit(self):
        list_scena = self.__TestScienario1_init()
        return list_scena

    # 打印出结果
    def print_tmp_res(self):
        gen_total_num = len(self.list_genpkt)
        print('\n range_comm:{} genfreq:{} RunningTime_Max:{} RunningTime:{} gen_num:{}'.format(
            self.range_comm, self.thr_genpkt, self.RunningTime_Max, self.RunningTime, gen_total_num))
        for key, value in self.scenaDict.items():
            total_succnum, normal_succnum, selfish_succnum, \
            total_delay, normal_delay, selfish_delay = value.showres()
            print('【{}】 total_succnum:{} normal_succnum:{} selfish_succnum:{} '
                  'total_delay:{} normal_delay:{} selfish_delay:{}'.format(
                key, total_succnum, normal_succnum, selfish_succnum, total_delay, normal_delay, selfish_delay))

    # 打印出结果
    def print_res(self, filename, ctstring):
        file_object = open(filename, ctstring, encoding="utf-8")
        gen_total_num = len(self.list_genpkt)
        file_object.write('\n')
        file_object.write('\n range_comm:{} genfreq:{} RunningTime_Max:{} RunningTime:{} gen_num:{}'.format(
            self.range_comm, self.thr_genpkt, self.RunningTime_Max, self.RunningTime, gen_total_num))
        for key, value in self.scenaDict.items():
            total_succnum, normal_succnum, selfish_succnum, \
            total_delay, normal_delay, selfish_delay = value.showres()
            file_object.write('\n【{}】 total_succnum:{} normal_succnum:{} selfish_succnum:{} '
                              'total_delay:{} normal_delay:{} selfish_delay:{}'.format(
                key, total_succnum, normal_succnum, selfish_succnum, total_delay, normal_delay, selfish_delay))
            gen_selfish_num = 0
            listselfishid = value.getselfishlist()
            file_object.write('\n normal_node:{} selfish_node:{}'.format(
                len(self.list_node) - len(listselfishid), len(listselfishid)))
            if len(listselfishid) > 0:
                for pkt in self.list_genpkt:
                    (id, src, dst) = pkt
                    if (src in listselfishid) or (dst in listselfishid):
                        gen_selfish_num += 1
                gen_normal_num = gen_total_num - gen_selfish_num
                if gen_normal_num > 0 and normal_succnum > 0:
                    file_object.write('\n normal_succratio:{} normal_avgdelay:{}'.format(
                        normal_succnum / gen_normal_num, normal_delay / normal_succnum))
            if gen_total_num > 0 and total_succnum > 0:
                    file_object.write('\n total_succratio:{} total_avgdelay:{}'.format(
                        total_succnum / gen_total_num, total_delay / total_succnum))
        file_object.close()

