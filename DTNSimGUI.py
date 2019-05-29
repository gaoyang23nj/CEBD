import tkinter as tk
import threading
from DTNSimBase import DTNSimBase
from DTNNode import DTNNode

class DTNSimGUI(DTNSimBase):
    def __init__(self, maxsizeofCanvas):
        self.MaxSize = maxsizeofCanvas
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
        self.t = threading.Timer(0.1, self.update)
        self.t.start()
        self.window.mainloop()

    def update(self):
        tunple_list = self.runonetimestep()

        for tunple in tunple_list:
            node_id, loc, dest = tunple
            self.__drawPointandLine(node_id, loc, dest)

        self.t = threading.Timer(0.1, self.update)
        self.t.start()


    def runonetimestep(self):
        tunple_list = []
        for node in self.node_list:
            node_id = node.node_id
            loc = node.run()
            tmp_tunple = (node.getNodeId(), loc, node.getNodeDest())
            # node_list.append(node_id)
            tunple_list.append(tmp_tunple)
        # print(tunple_list)
        return tunple_list