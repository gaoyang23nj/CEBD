import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import threading
import ctypes
import time
from DTNSimBase import DTNSimBase
from WKTPathReader import WKTPathReader
from DTNNode import DTNNode
from RoutingSparyandWait import *

class DTNSimGUIMap(DTNSimBase):
    def __init__(self, pathreader, showsize, isshowconn=True):
        winapi = ctypes.windll.user32
        width = winapi.GetSystemMetrics(0)
        height = winapi.GetSystemMetrics(1)
        print(''+str(int(width))+'x'+str(int(height)))
        # geo = ''+str(int(width*3/4))+'x'+str(int(height*3/4))
        geo = '1000x600'
        # 读取参数
        self.PathReader = pathreader
        self.ShowSize = showsize
        self.isShowConn = isshowconn
        # 范围参数Range
        self.RangeHeight = np.array([-1.0, -1.0])
        self.RangeWidth = np.array([-1.0, -1.0])
        self.MinXY = np.array([-1.0, -1.0])
        self.scale = 1
        self.genScaleParams()
        # 定时器界面刷新相关
        self.timerisrunning = False
        self.oval_size = 3
        self.window = tk.Tk()
        self.window.title('my win')
        self.window.geometry(geo)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.inittkinter()
        self.__drawmap()


    def genScaleParams(self):
        (self.MinXY, self.RangeHeight, self.RangeWidth) = self.PathReader.getRangeParams()
        if (self.RangeWidth[1] - self.RangeWidth[0]) > (self.RangeHeight[1] - self.RangeHeight[0]):
            self.scale = self.ShowSize / (self.RangeWidth[1] - self.RangeWidth[0])
        else:
            self.scale = self.ShowSize / (self.RangeHeight[1] - self.RangeHeight[0])


    def attachController(self, dtncontroller):
        self.DTNController = dtncontroller


    # 初始化组件
    def inittkinter(self):
        frm_control = tk.Frame(self.window)
        frm_control.config(bg='White', height=40, width=1000)
        frm_control.place(x=0, y=0, anchor='nw')
        frm_canvas = tk.Frame(self.window)
        frm_canvas.config(bg='GhostWhite', height=600, width=600)
        # frm_canvas.pack(side='left')
        frm_canvas.place(x=0, y=40, anchor='nw')
        frm_infoshow = tk.Frame(self.window)
        frm_infoshow.config(bg='green', height=600, width=400)
        frm_infoshow.place(x=600, y=40, anchor='nw')

        # 按钮 控制执行进程
        tk.Button(frm_control, text='stop', command=self.on_clickstop).place(x=20, y=5, height=30, width=60, anchor='nw')
        tk.Button(frm_control, text='resume', command=self.on_clickresume).place(x=120, y=5, height=30, width=60, anchor='nw')
        self.nr_signlestep = tk.StringVar()
        self.comlist_nrstepsel = ttk.Combobox(frm_control, width=12, textvariable=self.nr_signlestep)
        self.comlist_nrstepsel['values'] = (1, 2, 3, 4, 5)  # 设置下拉列表的值
        self.comlist_nrstepsel.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
        self.comlist_nrstepsel.place(x=220, y=5, height=30, width=60, anchor='nw')
        tk.Button(frm_control, text='step', command=self.on_clickstep).place(x=320, y=5, height=30, width=60, anchor='nw')
        self.text_ctlinfo = tk.StringVar()
        self.text_ctlinfo.set('ctlinfo')
        tk.Label(frm_control, bg='White', textvariable=self.text_ctlinfo, height=2, width=500,justify='left').\
            place(x=420, y=5, height=30, width=500, anchor='nw')
        # 画布 绘制node移动
        self.canvas = tk.Canvas(frm_canvas, bg='gray', height=self.ShowSize, width=self.ShowSize)
        self.canvas.place(x=0, y=0, anchor='nw')
        frm_time = tk.Frame(frm_canvas)
        frm_time.config(bg='Black', height=20, width=120)
        frm_time.place(x=0, y=self.ShowSize, anchor='nw')
        self.text_time = tk.StringVar()
        self.text_time.set('text_time')
        tk.Label(frm_time, bg='Yellow', textvariable=self.text_time, height=2, width=100, justify='left').\
            place(x=0, y=0, height=20, width=100, anchor='nw')

        # 显示信息
        self.cbbox_scena = ttk.Combobox(frm_infoshow)
        self.cbbox_scena.pack()
        self.text_nodelist = tk.StringVar()
        self.text_nodelist.set('node_list:')
        self.text_pktlist = tk.StringVar()
        self.text_pktlist.set('pkt_list')
        tk.Label(frm_infoshow, bg='SkyBlue', textvariable=self.text_nodelist, height=12, width=40,justify='left').pack()
        tk.Label(frm_infoshow, bg='CadetBlue', textvariable=self.text_pktlist, height=12, width=40,justify='left').pack()
        # info_enorgen = tk.Label(frm_infoshow, bg='gray', text='info_enorgen:',height=11,width=40)
        # info_enorgen.pack()


    def initshow(self, infotext, list_scena):
        infotext = infotext + ' SimSize:['+ str(self.RangeWidth[0])+','+str(self.RangeWidth[1])+'],['+str(self.RangeHeight[0])+','+str(self.RangeHeight[1])+']'
        self.text_ctlinfo.set(infotext)
        self.cbbox_scena["values"] = list_scena
        self.cbbox_scena.current(1)


    def mainloop(self):
        self.window.mainloop()


    def on_clickstep(self):
        updatetimesOnce = self.comlist_nrstepsel.get()
        self.DTNController.updateOnce(int(updatetimesOnce))


    def on_clickstop(self):
        self.DTNController.setTimerRunning(False)


    def on_clickresume(self):
        self.DTNController.setTimerRunning(True)
        self.DTNController.updateViewer()


    def on_closing(self):
        # 点了closing  先暂停，如果确定就退出 否则就恢复更新
        self.DTNController.setTimerRunning(False)
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.DTNController.closeApp()
            # 删除窗口
            self.window.destroy()
        else:
            self.DTNController.setTimerRunning(True)
            self.DTNController.updateViewer()


    # 画出地图 作为背景
    def __drawmap(self):
        alllines, allpoints = self.PathReader.getMapParams()
        # draw the lines （the roads in the map）
        for locs in alllines:
            # each two ad
            for loc_id in range(len(locs) - 1):
                loc = (locs[loc_id] - self.MinXY) * self.scale
                # tkinter左上是(0,0)坐标, 地球坐标系则是左下
                loc[1] = self.ShowSize - loc[1]
                dest = (locs[loc_id + 1] - self.MinXY) * self.scale
                dest[1] = self.ShowSize - dest[1]
                # 上下方位
                self.canvas.create_line(loc[0], loc[1], dest[0], dest[1], fill="white")


    def getscenaname(self):
        return self.cbbox_scena.get()


    def updateCanvaShow(self, listtunple, encounter_list):
        # 显示
        for tunple in listtunple:
            node_id, loc, dest = tunple
            self.__drawPointandLine(node_id, loc, dest)
        if self.isShowConn:
            self.__drawConn(encounter_list)

    def updateInfoShow(self, node_list, pkt_list, info_time):
        strinfo_nodelist = '<info>pkt list in the node: \n'
        for node in node_list:
            (node_id, pktlistofnode) = node
            strinfo_nodelist = strinfo_nodelist + str(node_id) + ':'
            for pkt in pktlistofnode:
                strinfo_nodelist = strinfo_nodelist + ' p_' + str(pkt.pkt_id)
                if isinstance(pkt, DTNSWPkt):
                    strinfo_nodelist = strinfo_nodelist + '(t_{})'.format(pkt.token)
            strinfo_nodelist = strinfo_nodelist + '\n'
        strinfo_pktlist = '<info>pkt list: \n'
        cnt = 1
        for pkt in pkt_list:
            (pkt_id, src_id, dst_id) = pkt
            strinfo_pktlist = strinfo_pktlist + ' p'+str(pkt_id)+ ':'+str(src_id)+'->'+str(dst_id)+' '
            if cnt%5==0:
                strinfo_pktlist = strinfo_pktlist + '\n'
            cnt = cnt + 1
        self.text_nodelist.set(strinfo_nodelist)
        self.text_pktlist.set(strinfo_pktlist)
        self.text_time.set('Time:'+str(info_time))

    def __drawPointandLine(self, node_id, loc, dest):
        node_id = str(node_id)
        # 坐标转换 到 绘图坐标系
        newloc = (loc - self.MinXY) * self.scale
        newloc[1] = self.ShowSize - newloc[1]
        newdest = (dest - self.MinXY) * self.scale
        newdest[1] = self.ShowSize - newdest[1]
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


    # 绘制loc连成的折线
    def __drawPath(self, locs):
        # each two ad
        for loc_id in range(len(locs) - 1):
            loc = (locs[loc_id] - self.MinXY) * self.scale
            loc[1] = self.ShowSize - loc[1]
            dest = (locs[loc_id + 1] - self.MinXY) * self.scale
            dest[1] = self.ShowSize - dest[1]
            # 上下方位
            self.canvas.create_line(loc[0], loc[1], dest[0], dest[1], fill="blue",  dash=(4, 4))


    # 连接显示
    def __drawConn(self, encounter_list):
        self.canvas.delete('conline' + '_')
        for encounter_tunple in encounter_list:
            (a_id, b_id, a_loc, b_loc) = encounter_tunple
            newloc_a = (a_loc - self.MinXY) * self.scale
            newloc_a[1] = self.ShowSize - newloc_a[1]
            newloc_b = (b_loc - self.MinXY) * self.scale
            newloc_b[1] = self.ShowSize - newloc_b[1]
            tmp_line = self.canvas.create_line(newloc_a[0], newloc_a[1], newloc_b[0], newloc_b[1], fill="yellow",
                                               tags='conline' + '_')
        return

