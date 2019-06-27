import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
from DTNSimBase import DTNSimBase
from DTNNode import DTNNode
from RoutingEpidemic import RoutingEpidemic

class DTNSimGUI(DTNSimBase):
    def __init__(self, maxsizeofCanvas):
        self.MaxSize = maxsizeofCanvas
        # 定时器界面刷新相关
        self.timerisrunning = False
        self.oval_size = 3
        self.window = tk.Tk()
        self.window.title('my win')
        self.window.geometry('1000x1000')

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        frm_canvas = tk.Frame(self.window)
        frm_canvas.pack(side='left')
        frm_button = tk.Frame(self.window)
        frm_button.pack(side='right')
        # canvas
        tk.Label(frm_canvas, text='frm_canvas').pack()
        self.canvas = tk.Canvas(frm_canvas, bg='gray', height=self.MaxSize, width=self.MaxSize)
        self.canvas.pack()


    def attachController(self, dtncontroller):
        self.DTNController = dtncontroller

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.DTNController.closetimer()
            # 删除窗口
            self.window.destroy()


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


    def initshow(self):
        self.window.mainloop()


    def updateshow(self, listtunple):
        # 显示
        for tunple in listtunple:
            node_id, loc, dest = tunple
            self.__drawPointandLine(node_id, loc, dest)

