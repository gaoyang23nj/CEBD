import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
import ctypes
from DTNSimBase import DTNSimBase
from DTNNode import DTNNode
from RoutingEpidemic import RoutingEpidemic

class DTNSimGUI(DTNSimBase):
    def __init__(self, showsize, realsize):
        winapi = ctypes.windll.user32
        width = winapi.GetSystemMetrics(0)
        height = winapi.GetSystemMetrics(1)
        print(''+str(int(width))+'x'+str(int(height)))
        # geo = ''+str(int(width*3/4))+'x'+str(int(height*3/4))
        geo = '1000x600'
        # 显示尺寸 真实尺寸
        self.ShowSize = showsize
        self.RealSize = realsize
        self.scale = showsize/realsize
        # 定时器界面刷新相关
        self.timerisrunning = False
        self.oval_size = 3
        self.window = tk.Tk()
        self.window.title('my win')
        self.window.geometry(geo)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.inittkinter()

    def attachController(self, dtncontroller):
        self.DTNController = dtncontroller

    # 初始化组件
    def inittkinter(self):
        frm_canvas = tk.Frame(self.window)
        frm_canvas.config(bg='GhostWhite', height=600, width=600)
        # frm_canvas.pack(side='left')
        frm_canvas.place(x=0, y=0, anchor='nw')
        frm_button = tk.Frame(self.window)
        frm_button.config(bg='green', height=600, width=400)
        frm_button.place(x=600, y=0, anchor='nw')
        self.canvas = tk.Canvas(frm_canvas, bg='gray', height=self.ShowSize, width=self.ShowSize)
        self.canvas.place(x=0, y=0, anchor='nw')

        self.text_routingname = tk.StringVar()
        self.text_routingname.set('routingname')
        self.text_nodelist = tk.StringVar()
        self.text_nodelist.set('node_list:')
        self.text_pktlist = tk.StringVar()
        self.text_pktlist.set('pkt_list')
        # info_routingname = 'epidemcirouting'
        info_routing = tk.Label(frm_button, bg='white', textvariable=self.text_routingname, height=2, width=40, justify='left')
        info_routing.pack()
        info_nodelist = tk.Label(frm_button, bg='SkyBlue', textvariable=self.text_nodelist, height=12, width=40,justify='left')
        info_nodelist.pack()
        info_pktlist = tk.Label(frm_button, bg='CadetBlue', textvariable=self.text_pktlist, height=12, width=40,justify='left')
        info_pktlist.pack()
        # info_enorgen = tk.Label(frm_button, bg='gray', text='info_enorgen:',height=11,width=40)
        # info_enorgen.pack()

    def initshow(self):
        self.window.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.DTNController.closetimer()
            # 删除窗口
            self.window.destroy()

    def getroutingname(self):
        return 'epidemicrouting'

    def updateCanvaShow(self, listtunple):
        # 显示
        for tunple in listtunple:
            node_id, loc, dest = tunple
            self.__drawPointandLine(node_id, loc, dest)

    def updateInfoShow(self, tunple):
        (node_list, pkt_list) = tunple
        strinfo_nodelist = '<info>pkt list in the node: \n'
        for node in node_list:
            (node_id, pktlistofnode) = node
            strinfo_nodelist = strinfo_nodelist+str(node_id)+':'
            for pkt_id in pktlistofnode:
                strinfo_nodelist = strinfo_nodelist + ' p_'+str(pkt_id)
            strinfo_nodelist = strinfo_nodelist+'\n'
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

    def __drawPointandLine(self, node_id, loc, dest):
        node_id = str(node_id)
        newloc = loc * self.scale
        newdest = dest * self.scale
        self.canvas.delete('text' + '_' + node_id, 'oval' + '_' + node_id, 'doval' + '_' + node_id, 'line' + '_' + node_id)
        tmp_oval = self.canvas.create_oval(newloc[0] - self.oval_size, newloc[1] - self.oval_size,
                                           newloc[0] + self.oval_size, newloc[1] + self.oval_size,
                                           tag='oval' + '_' + node_id, fill='red')
        tmp_label = self.canvas.create_text(newloc[0], newloc[1] - (self.oval_size * 3), text='' + node_id,
                                            tag='text' + '_' + node_id)

        tmp_doval = self.canvas.create_oval(newdest[0] - self.oval_size, newdest[1] - self.oval_size,
                                            newdest[0] + self.oval_size, newdest[1] + self.oval_size,
                                            tag='doval' + '_' + node_id, fill='blue')
        tmp_line = self.canvas.create_line(newloc[0], newloc[1], newdest[0], newdest[1], fill="red", tags='line' + '_' + node_id)



