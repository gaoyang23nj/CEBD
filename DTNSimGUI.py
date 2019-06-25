import tkinter as tk
import threading
import numpy as np
from DTNSimBase import DTNSimBase
from DTNNode import DTNNode
from RoutingEpidemic import RoutingEpidemic

class DTNSimGUI(DTNSimBase):
    def __init__(self, maxsizeofCanvas, showtimes=100, com_range=100, genfreq_cnt=6000):
        self.MaxSize = maxsizeofCanvas
        self.showtimes = showtimes
        self.com_range = com_range
        self.genfreq_max = genfreq_cnt
        self.node_list = []
        self.__routinginit()
        # 全部生成报文的list
        self.genfreq_pktlist = []
        # 下一个pkt的id
        self.genfreq_pktid = 1
        # 生成报文的时间计数器
        self.genfreq_cnt = genfreq_cnt

        self.oval_size = 3
        self.window = tk.Tk()
        self.window.title('my win')
        self.window.geometry('1000x1000')

        frm_canvas = tk.Frame(self.window)
        frm_canvas.pack(side='left')
        frm_button = tk.Frame(self.window)
        frm_button.pack(side='right')
        # canvas
        tk.Label(frm_canvas, text='frm_canvas').pack()
        self.canvas = tk.Canvas(frm_canvas, bg='gray', height=self.MaxSize, width=self.MaxSize)
        self.canvas.pack()

    def attachDTNNode(self, node):
        self.node_list.append(node)

    def __drawPointandLine(self, node_id, loc, dest):
        node_id = str(node_id)
        self.canvas.delete('text' + '_' + node_id, 'oval' + '_' + node_id, 'doval' + '_' + node_id, 'line' + '_' + node_id)
        tmp_oval = self.canvas.create_oval(loc[0] - self.oval_size, loc[1] - self.oval_size,
                                           loc[0] + self.oval_size, loc[1] + self.oval_size,
                                           tag='oval' + '_' + node_id, fill='red')
        tmp_label = self.canvas.create_text(loc[0], loc[1] - (self.oval_size * 3), text='' + node_id,
                                            tag='text' + '_' + node_id)

        tmp_doval = self.canvas.create_oval(dest[0] - self.oval_size, dest[1] - self.oval_size,
                                            dest[0] + self.oval_size, dest[1] + self.oval_size,
                                            tag='doval' + '_' + node_id, fill='blue')
        tmp_line = self.canvas.create_line(loc[0], loc[1], dest[0], dest[1], fill="red", tags='line' + '_' + node_id)

    def run(self):
        # 节点个数
        self.numofnodes = len(self.node_list)
        # 记录任何两个node之间的link状态
        self.link_state = np.zeros(self.numofnodes,  self.numofnodes)

        self.t = threading.Timer(0.1, self.update)
        self.t.start()
        self.window.mainloop()

    def update(self):
        # 完成 self.showtimes 个 timestep的位置更新变化
        tunple_list = self.runonetimestep()
        for i in range(self.showtimes-1):
            tunple_list = self.runonetimestep()
        # 显示
        for tunple in tunple_list:
            node_id, loc, dest = tunple
            self.__drawPointandLine(node_id, loc, dest)
        # 下次更新视图
        self.t = threading.Timer(0.1, self.update)
        self.t.start()

    def runonetimestep(self):
        if self.genfreq_cnt == self.genfreq_max:
            # 报文生成
            self.__routinggenpkt()
            self.genfreq_cnt = 1
        else:
            self.genfreq_cnt =  self.genfreq_cnt + 1

        tunple_list = []
        # 节点移动一个timestep
        for node in self.node_list:
            node_id = node.node_id
            loc = node.run()
            tmp_tunple = (node.getNodeId(), loc, node.getNodeDest())
            # node_list.append(node_id)
            tunple_list.append(tmp_tunple)
        # print(tunple_list)
        # 是否有传输事件被中断
        for a_id in range(self.numofnodes):
            a_loc = self.node_list[a_id].getNodeLoc()
            for b_id in range(self.numofnodes):
                b_loc = self.node_list[b_id].getNodeLoc()
                if (self.link_state[a_id][b_id] == 1) and \
                        (np.multiply(a_loc - b_loc, a_loc - b_loc) > self.com_range):
                    self.link_state[a_id][b_id] == 0
                    self.__routinglinkdown(a_id, b_id)


        # 按照移动后的位置查找相遇事件
        for i in range(len(tunple_list)):
            a_id, a_loc, a_dest = tunple_list[i]
            for j in range(i+1, len(tunple_list), step=1):
                b_id, b_loc, b_dest = tunple_list[j]
                # 如果在通信范围内 交换信息
                # 同时完成a->b b->a
                if np.multiply(a_loc-b_loc, a_loc-b_loc) < self.com_range:
                    self.link_state[a_id][b_id] = 1
                    self.link_state[b_id][a_id] = 1
                    self.__routingswap(a_id, b_id)

        return tunple_list

    # 各个routing初始化
    def __routinginit(self):
        self.epidemicrouting = RoutingEpidemic(len(self.node_list))

    # 各个routing 生成报文
    def __routinggenpkt(self):
        src_index = np.random.randint(len(self.node_list))
        dst_index = np.random.randint(len(self.node_list))
        while dst_index==src_index:
            dst_index = np.random.randint(len(self.node_list))
        newpkt = (self.genfreq_pktid, src_index, dst_index)
        self.node_list.append(newpkt)

        # 各routing生成pkt, pkt大小为100k
        self.epidemicrouting.gennewpkt(self.genfreq_pktid, src_index, dst_index, 0, 100)

        self.genfreq_pktid = self.genfreq_pktid + 1
        return

    # 各个routing开始交换报文
    def __routingswap(self, a_id, b_id):
        self.epidemicrouting.swappkt(a_id, b_id)
        self.epidemicrouting.swappkt(b_id, a_id)

    def __routinglinkdown(self, a_id, b_id):
        self.epidemicrouting.linkdown(a_id, b_id)
        self.epidemicrouting.linkdown(b_id, a_id)





