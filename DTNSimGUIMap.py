import tkinter as tk
import numpy as np
import threading
import os
from DTNSimBase import DTNSimBase
from WKTPathReader import WKTPathReader
from DTNNode import DTNNode

class DTNSimGUIMap(DTNSimBase):
    def __init__(self, maxsize, pathreader, showtimes=100):
        self.MaxSize = maxsize
        self.pathreader = pathreader
        self.showtimes = showtimes
        self.node_list = []
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
        self.__drawmap()

    def attachDTNNode(self, node):
        self.node_list.append(node)

    # 画出地图 作为背景
    def __drawmap(self):
        alllines, allpoints, self.MinXY, self.scale = self.pathreader.getParams()
        # draw the lines （the roads in the map）
        for locs in alllines:
            # each two ad
            for loc_id in range(len(locs) - 1):
                loc = (locs[loc_id] - self.MinXY) * self.scale
                loc[1] = self.MaxSize - loc[1]
                dest = (locs[loc_id + 1] - self.MinXY) * self.scale
                dest[1] = self.MaxSize - dest[1]
                # 上下方位
                self.canvas.create_line(loc[0], loc[1], dest[0], dest[1], fill="white")

    def __drawPointandLine(self, node_id, loc, src, dest):
        node_id = str(node_id)
        # 坐标转换 到 绘图坐标系
        newloc = (loc - self.MinXY) * self.scale
        newloc[1] = self.MaxSize - newloc[1]
        newdest = (dest[0] - self.MinXY) * self.scale
        newdest[1] = self.MaxSize - newdest[1]
        # 删去canvas里面 之前的标识
        self.canvas.delete('oval' + '_' + node_id, 'text' + '_' + node_id, 'doval' + '_' + node_id, 'line' + '_' + node_id)
        # 绘制新的标识
        tmp_oval = self.canvas.create_oval(newloc[0] - self.oval_size, newloc[1] - self.oval_size,
                                           newloc[0] + self.oval_size, newloc[1] + self.oval_size,
                                           tag='oval' + '_' + node_id, fill='red')
        tmp_label = self.canvas.create_text(newloc[0], newloc[1] - (self.oval_size * 3), text=node_id, tag='text' + '_' + node_id)

        tmp_oval = self.canvas.create_oval(newdest[0] - self.oval_size/2, newdest[1] - self.oval_size/2,
                                           newdest[0] + self.oval_size/2, newdest[1] + self.oval_size/2,
                                           tag='doval' + '_' + node_id, fill='blue')
        tmp_line = self.canvas.create_line(newloc[0], newloc[1], newdest[0], newdest[1], fill="red",
                                           tags='line' + '_' + node_id)

    def __drawPath(self, locs):
        # each two ad
        for loc_id in range(len(locs) - 1):
            loc = (locs[loc_id] - self.MinXY) * self.scale
            loc[1] = self.MaxSize - loc[1]
            dest = (locs[loc_id + 1] - self.MinXY) * self.scale
            dest[1] = self.MaxSize - dest[1]
            # 上下方位
            self.canvas.create_line(loc[0], loc[1], dest[0], dest[1], fill="blue",  dash=(4, 4))

    def run(self):
        self.t = threading.Timer(0.1, self.update)
        self.t.start()
        self.window.mainloop()

    def update(self):
        tunple_list = self.runonetimestep()
        for i in range(self.showtimes-1):
            tunple_list = self.runonetimestep()

        for tmp_tunple in tunple_list:
            (node_id, loc, src, dest, path) = tmp_tunple
            self.__drawPointandLine(node_id, loc, src, dest)

        self.t = threading.Timer(0.1, self.update)
        self.t.start()

    def runonetimestep(self):
        tunple_list = []
        for node in self.node_list:
            loc = node.run()
            path = node.getPath()
            src, dest = node.getSrcDestPair()
            tmp_tunple = (node.getNodeId(), loc, src, dest, path)
            tunple_list.append(tmp_tunple)
            # 坐标转换
        # print(tunple_list)
        return tunple_list
