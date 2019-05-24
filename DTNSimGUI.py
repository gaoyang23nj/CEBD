import tkinter as tk
import threading
# from DTNNode import DTNNode

class DTNSimGUI(object):
    def __init__(self):
        self.node_list = []
        self.oval_size = 3
        self.window = tk.Tk()
        self.window.title('my win')
        self.window.geometry('700x600')

        frm_canvas = tk.Frame(self.window)
        frm_canvas.pack(side='left')
        frm_button = tk.Frame(self.window)
        frm_button.pack(side='right')
        # canvas
        tk.Label(frm_canvas, text='frm_canvas').pack()
        self.canvas = tk.Canvas(frm_canvas, bg='gray', height=500, width=500)
        self.canvas.pack()
        # self.t = threading.Timer(0.1, self.run)
        # self.t.start()
        # self.window.mainloop()

    def attach(self, node):
        self.node_list.append(node)

    def run(self):
        self.t = threading.Timer(0.1, self.update)
        self.t.start()
        self.window.mainloop()

    def update(self):
        print('hello')
        for node in self.node_list:
            node_id = node.node_id
            node_id = str(node_id)
            self.canvas.delete('text' + '_' + node_id, 'oval'+'_'+node_id, 'dtext' + '_' + node_id, 'doval'+'_'+node_id, 'line'+'_'+node_id)

        tunple_list = self.runonetimestep()
        for node in tunple_list:
            node_id = node[0]
            node_id = str(node_id)
            loc = node[1]
            dest = node[2]
            tmp_oval = self.canvas.create_oval(loc[1]-self.oval_size, loc[0]-self.oval_size,
                                    loc[1]+self.oval_size, loc[0]+self.oval_size,
                                    tag='oval'+'_'+node_id, fill='red')
            tmp_label = self.canvas.create_text(loc[1], loc[0]-(self.oval_size*3),text = ''+node_id, tag='text' + '_' + node_id)

            tmp_doval = self.canvas.create_oval(dest[1]-self.oval_size, dest[0]-self.oval_size,
                                               dest[1]+self.oval_size, dest[0]+self.oval_size,
                                               tag='doval'+'_'+node_id, fill='blue')
            tmp_label = self.canvas.create_text(dest[1], dest[0] - (self.oval_size * 3), text='' + node_id,
                                                tag='dtext' + '_' + node_id)

            tmp_line = self.canvas.create_line(loc[1], loc[0], dest[1], dest[0], fill="red", tags='line'+'_'+node_id)
            # coord = 10, 50, 240, 210
            # arc = self.canvas.create_arc(coord, start=0, extent=150, fill="blue")
        self.t = threading.Timer(0.1, self.update)
        self.t.start()


    def runonetimestep(self):
        # node_list = []
        tunple_list = []
        for node in self.node_list:
            node_id = node.node_id
            loc = node.run()
            tmp_tunple = (node_id, loc, node.dest)
            # node_list.append(node_id)
            tunple_list.append(tmp_tunple)
        print(tunple_list)
        return tunple_list