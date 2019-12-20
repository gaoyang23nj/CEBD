# 本文件旨在:生成<相遇历史记录>，结果保存在 Simulation_ONE/EncoHistData/文件夹下
#

import numpy as np
# import tensorflow as tf
# import sys
# import threading
import datetime
# import _thread

from EnCoHistGenerator_Helsink.DTNNode import DTNNode
from EnCoHistGenerator_Helsink.WKTPathReader import WKTPathReader

# np.random.seed(1)
np.random.seed()
# tf.set_random_seed(1)

# 保存历史记录的位置
EncoHistDir = '../EncoHistData_Helsink/'


class RandomWalkGenerator(object):
    def __init__(self):
        # 节点个数默认100个, id 0~99
        self.MAX_NODE_NUM = 100
        # 通信范围100m
        self.RANGE_COMM = 50
        # 最大运行时间 执行时间 36000*12个间隔, 即12hour
        self.MAX_RUNNING_TIMES = 36000*24*2
        # 每个间隔的时间长度 0.1s
        self.sim_TimeStep = 0.1
        # # <仿真环境>的空间范围大小 2000m*2000m
        # self.sim_RealSize = 2000
        # 仿真环境 现在的时刻
        self.sim_TimeNow = 0
        # node所组成的list
        self.list_nodes = []
        pathreader = WKTPathReader()
        for node_id in range(self.MAX_NODE_NUM):
            # 每个timestep = <模拟>0.1s
            node = DTNNode('SPM', node_id, self.sim_TimeStep, pathreader)
            self.list_nodes.append(node)
        # link状态记录 记录任何两个node之间的link状态 以便检测linkdown事件
        self.mt_linkstate = np.zeros((self.MAX_NODE_NUM, self.MAX_NODE_NUM), dtype='int')
        # ====================== 仿真过程 ============================
        # 记录link事件状态：两个节点已经建立link 并且没有中断
        # 格式(a_id, b_id, a_loc, b_loc, time_linkup)
        self.link_event_list = []
        # 保存全部的encounter hist；格式为(time_linkup, time_linkdown, x_id, y_id, x_loc, y_loc, a_loc, b_loc)
        self.encounter_hist_list = []
        # 如果执行到最后的时刻，则停止下一次执行
        while self.sim_TimeNow < self.MAX_RUNNING_TIMES:
            # 节点移动一个timestep
            for node in self.list_nodes:
                node.run()
            # 检测相遇事件
            self.detect_encounter()
            self.sim_TimeNow = self.sim_TimeNow + 1
        # 过程结束 关闭所有link 并加入 encounter hist list 历史相遇记录
        self.close_each_link()
        # 按照发起时间顺序排列encounter list
        self.encounter_hist_list.sort()
        # 写入文件
        short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = EncoHistDir+'encohist_'+short_time+'.tmp'
        file_object = open(filename, 'a+', encoding="utf-8")
        file_object.write('Settings, MAX_NODE_NUM:{},RANGE_COMM:{},MAX_RUNNING_TIMES:{},sim_TimeStep:{}, Helsink'
                          .format(self.MAX_NODE_NUM, self.RANGE_COMM, self.MAX_RUNNING_TIMES, self.sim_TimeStep))
        # 总仿真事件 (hour)用小时表示
        total_running_time = (self.MAX_RUNNING_TIMES) / (3600 * 1 / (self.sim_TimeStep))
        file_object.write(
            'Settings, MAX_NODE_NUM:{},RANGE_COMM:{},MAX_RUNNING_TIMES:{},{}(h), sim_TimeStep:{}, vel:[{},{}]'.
                format(self.MAX_NODE_NUM, self.RANGE_COMM, self.MAX_RUNNING_TIMES, total_running_time, self.sim_TimeStep,
                       self.list_nodes[0].MovementModel.minspeed, self.list_nodes[0].MovementModel.maxspeed))
        for tunple in self.encounter_hist_list:
            (time_linkup, time_linkdown, x_id, y_id, x_loc, y_loc, a_loc, b_loc) = tunple
            file_object.write('\n{},{},{},{}'.format(time_linkup, time_linkdown, x_id, y_id))
        file_object.close()

    def detect_encounter(self):
        for a_id in range(self.MAX_NODE_NUM):
            for b_id in range(a_id + 1, self.MAX_NODE_NUM, 1):
                a_node = self.list_nodes[a_id]
                a_loc = a_node.getNodeLoc()
                b_node = self.list_nodes[b_id]
                b_loc = b_node.getNodeLoc()
                assert ((a_node.getNodeId() == a_id) and (b_node.getNodeId() == b_id))
                if np.sqrt(np.dot(a_loc - b_loc, a_loc - b_loc)) <= self.RANGE_COMM:
                    if self.mt_linkstate[a_id][b_id] == 0:
                        assert (self.mt_linkstate[b_id][a_id] == 0)
                        # 节点a和节点b 状态从0到1 发生linkup事件
                        # 更新状态 交换 控制信息
                        self.mt_linkstate[a_id][b_id] = 1
                        self.mt_linkstate[b_id][a_id] = 1
                        self.link_event_list.append((a_id, b_id, a_loc, b_loc, self.sim_TimeNow))
                    elif self.mt_linkstate[a_id][b_id] == 1:
                        assert (self.mt_linkstate[b_id][a_id] == 1)
                    else:
                        assert (True)
                else:
                    if self.mt_linkstate[a_id][b_id] == 0:
                        assert (self.mt_linkstate[b_id][a_id] == 0)
                    elif self.mt_linkstate[a_id][b_id] == 1:
                        assert (self.mt_linkstate[b_id][a_id] == 1)
                        # 节点a和节点b 状态从1到0 发生linkup事件
                        self.mt_linkstate[a_id][b_id] = 0
                        self.mt_linkstate[b_id][a_id] = 0
                        # 找到已经记录到 encounter_list 的相遇事件
                        for tunple in self.link_event_list:
                            (x_id, y_id, x_loc, y_loc, time_linkup) = tunple
                            if ((a_id == x_id) and (b_id == y_id)):
                                self.encounter_hist_list.append((time_linkup, self.sim_TimeNow, x_id, y_id,
                                                                 x_loc, y_loc, a_loc, b_loc))
                                print('encoutner event:{}~{},{}-{}'.format(time_linkup,self.sim_TimeNow, x_id,y_id))
                                self.link_event_list.remove(tunple)
                                break
                    else:
                        assert (True)
        return

    def close_each_link(self):
        for tunple in self.link_event_list:
            (x_id, y_id, x_loc, y_loc, time_linkup) = tunple
            a_loc = self.list_nodes[x_id].getNodeLoc()
            b_loc = self.list_nodes[y_id].getNodeLoc()
            assert ((self.mt_linkstate[x_id][y_id] == 1) and (self.mt_linkstate[y_id][x_id] == 1))
            self.mt_linkstate[x_id][y_id] = 0
            self.mt_linkstate[y_id][x_id] = 0
            self.encounter_hist_list.append((time_linkup, self.sim_TimeNow, x_id, y_id, x_loc, y_loc, a_loc, b_loc))
            # self.link_event_list.remove(tunple)
        # 确保没有漏项
        assert (np.sum(self.mt_linkstate) == 0)
        self.link_event_list.clear()

if __name__ == "__main__":
    the_generator = RandomWalkGenerator()