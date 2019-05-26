import tkinter as tk
import numpy as np
import threading
import os
from WKTPathReader import WKTPathReader
from DTNNodeMap import DTNNodeMap

class DTNSimGUIMap(object):
    def __init__(self):
        self.MaxSize = 800
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
        self.drawmap()

    def attach(self, node):
        self.node_list.append(node)


    # 画出地图 作为背景
    def drawmap(self):
        self.thePathReader = WKTPathReader(self.MaxSize)
        alllines, allpoints, self.MinXY, self.scale = self.thePathReader.getParams()
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
        # draw the points （the POIs in the map）
        # for poi in allpoints:
        #     loc = (poi - self.MinXY) * self.scale
        #     loc[1] = self.MaxSize - loc[1]
        #     self.canvas.create_oval(loc[1] - self.oval_size, loc[0] - self.oval_size,
        #                             loc[1] + self.oval_size, loc[0] + self.oval_size,
        #                             fill='red')
        # self.canvas.create_oval( - 5, - 5, + 5, + 5, fill='black')


    def run(self):
        self.t = threading.Timer(0.1, self.update)
        self.t.start()
        self.window.mainloop()

    def drawPointandLine(self, loc, src, dest, node_id):
        # 坐标转换  绘图坐标系
        newloc = (loc - self.MinXY) * self.scale
        newloc[1] = self.MaxSize - newloc[1]
        newdest = (dest[0] - self.MinXY) * self.scale
        newdest[1] = self.MaxSize - newdest[1]
        tmp_oval = self.canvas.create_oval(newloc[0] - self.oval_size, newloc[1] - self.oval_size,
                                           newloc[0] + self.oval_size, newloc[1] + self.oval_size,
                                           tag='oval' + '_' + node_id, fill='red')
        # src_a = (src[1] - self.MinXY) * self.scale
        # src_b = (src[2] - self.MinXY) * self.scale
        # src_a[1] = self.MaxSize - src_a[1]
        # src_b[1] = self.MaxSize - src_b[1]
        # tmp_oval = self.canvas.create_oval(src_a[0] - 2, src_a[1] - 2,
        #                                    src_a[0] + 2, src_a[1] + 2,
        #                                    tag='oval' + '_' + node_id, fill='black')
        # tmp_oval = self.canvas.create_oval(src_b[0] - 2, src_b[1] - 2,
        #                                    src_b[0] + 2, src_b[1] + 2,
        #                                    tag='oval' + '_' + node_id, fill='black')
        tmp_oval = self.canvas.create_oval(newdest[0] - self.oval_size/2, newdest[1] - self.oval_size/2,
                                           newdest[0] + self.oval_size/2, newdest[1] + self.oval_size/2,
                                           tag='doval' + '_' + node_id, fill='blue')
        # dest_a = (dest[1] - self.MinXY) * self.scale
        # dest_b = (dest[2] - self.MinXY) * self.scale
        # dest_a[1] = self.MaxSize - dest_a[1]
        # dest_b[1] = self.MaxSize - dest_b[1]
        # tmp_oval = self.canvas.create_oval(dest_a[0] - 2, dest_a[1] - 2,
        #                                    dest_a[0] + 2, dest_a[1] + 2,
        #                                    tag='oval' + '_' + node_id, fill='black')
        # tmp_oval = self.canvas.create_oval(dest_b[0] - 2, dest_b[1] - 2,
        #                                    dest_b[0] + 2, dest_b[1] + 2,
        #                                    tag='oval' + '_' + node_id, fill='black')
        tmp_line = self.canvas.create_line(newloc[0], newloc[1], newdest[0], newdest[1], fill="red",
                                           tags='line' + '_' + node_id)


    def drawPath(self, locs):
        # each two ad
        for loc_id in range(len(locs) - 1):
            loc = (locs[loc_id] - self.MinXY) * self.scale
            loc[1] = self.MaxSize - loc[1]
            dest = (locs[loc_id + 1] - self.MinXY) * self.scale
            dest[1] = self.MaxSize - dest[1]
            # 上下方位
            self.canvas.create_line(loc[0], loc[1], dest[0], dest[1], fill="blue",  dash=(4, 4))


    def update(self):
        # print('update ******************************************************************')
        loc, id, src, dest, path = self.runonetimestep()
        # for node in tunple_list:
        node_id = id
        node_id = str(node_id)

        # delete the old symbols
        for node in self.node_list:
            node_id = node.node_id
            node_id = str(node_id)
            self.canvas.delete('oval' + '_' + node_id, 'doval' + '_' + node_id, 'line' + '_' + node_id)
            # self.canvas.delete('text' + '_' + node_id, 'oval'+'_'+node_id, 'dtext' + '_' + node_id, 'doval'+'_'+node_id, 'line'+'_'+node_id)

        # create the new symbols
        self.drawPointandLine(loc, src, dest, node_id)
        # self.drawPath(path)

        self.t = threading.Timer(0.1, self.update)
        self.t.start()

    def runonetimestep(self):
        tunple_list = []
        for node in self.node_list:
            loc = node.run()
            path = node.getPath()
            id, src, dest = node.getSrcDest()
            # 坐标转换
        return loc, id, src, dest, path






if __name__ == "__main__":
    theGUI = DTNSimGUIMap()
    theNodes = []
    for node_id in range(1):
        node = DTNNodeMap(node_id, 0.1*100, theGUI.thePathReader)
        theNodes.append(node)
        theGUI.attach(node)
    theGUI.run()