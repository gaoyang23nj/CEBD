import tkinter as tk
import numpy as np
import threading
import os
# from DTNNode import DTNNode

DATA_DIR = 'data/'


class DTNSimGUI(object):
    def __init__(self):
        self.alllines = []
        self.allpoints = []
        self.RangeHeight = np.array([-1.0, -1.0])
        self.RangeWidth = np.array([-1.0, -1.0])
        self.MinXY = np.array([-1.0, -1.0])
        self.Scale = -1.0
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
        # self.t = threading.Timer(0.1, self.run)
        # self.t.start()
        # self.window.mainloop()

    def attach(self, node):
        self.node_list.append(node)

    def run(self):
        self.drawmap()
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

    def drawmap(self):
        files = os.listdir(DATA_DIR)
        for file in files:
            self.readwktfile(DATA_DIR+file)
        print(self.RangeWidth)
        print(self.RangeHeight)
        self.MinXY[0] = self.RangeWidth[0]
        self.MinXY[1] = self.RangeHeight[0]
        rangeL = np.array([self.RangeWidth[1] - self.RangeWidth[0], self.RangeHeight[1] - self.RangeHeight[0]])
        maxidx = np.argmax(rangeL)
        if (self.RangeWidth[1] - self.RangeWidth[0]) > (self.RangeHeight[1] - self.RangeHeight[0]):
            self.Scale = self.MaxSize / (self.RangeWidth[1] - self.RangeWidth[0])
        else:
            self.Scale = self.MaxSize / (self.RangeHeight[1] - self.RangeHeight[0])
        for locs in self.alllines:
            for loc_id in range(len(locs)-1):
                loc = (locs[loc_id] - self.MinXY) * self.Scale
                dest = (locs[loc_id+1] - self.MinXY) * self.Scale
                self.canvas.create_line(loc[0], self.MaxSize-loc[1], dest[0], self.MaxSize-dest[1], fill="white")
        self.window.mainloop()


    def readwktfile(self, filename):
        print(filename)
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                line = line.lstrip()
                if 'MULTILINESTRING ' in line:
                    tmp_str = line.replace('MULTILINESTRING ','').replace('\r','').replace('\n','')
                    while line:
                        if '))' in line:
                            break
                        line = f.readline().lstrip().replace('\r','').replace('\n','')
                        tmp_str = tmp_str+line
                    todrawlines = tmp_str.replace('((', '').replace('))', '').split('),(')
                    for todrawline in todrawlines:
                        self.recordlines(todrawline)
                    line = f.readline()
                elif 'LINESTRING' in line:
                    tmp_str = line.replace('LINESTRING ','').replace('\r','').replace('\n','')
                    while line:
                        if ')' in line:
                            break
                        line = f.readline().lstrip().replace('\r','').replace('\n','')
                        tmp_str = tmp_str+line
                    todrawline = tmp_str.replace('(', '').replace(')', '')
                    self.recordlines(todrawline)
                    line = f.readline()
                elif 'POINT' in line:
                    tmp_str = line.replace('POINT ', '').replace('\r', '').replace('\n', '')
                    # self.extractcontent_point(tmp_str)
                    coord = tmp_str.lstrip().rstrip().replace('(', '').replace(')', '')
                    self.recordpoint(coord, 'POINT')
                    line = f.readline()
                else:
                    line = f.readline()

    def recordpoint(self, coord, label):
        loc = np.array([0.0, 0.0])
        loc[0] = float(coord.split(' ')[0])
        loc[1] = float(coord.split(' ')[1])
        if self.RangeWidth[0] == -1:
            self.RangeWidth[0] = loc[0]
            self.RangeWidth[1] = loc[0]
            self.RangeHeight[0] = loc[1]
            self.RangeHeight[1] = loc[1]
        else:
            if loc[0] < self.RangeWidth[0]:
                self.RangeWidth[0] = loc[0]
            elif loc[0] > self.RangeWidth[1]:
                self.RangeWidth[1] = loc[0]
            if loc[1] < self.RangeHeight[0]:
                self.RangeHeight[0] = loc[1]
            elif loc[1] > self.RangeHeight[1]:
                self.RangeHeight[1] = loc[1]
        if label is 'POINT':
            self.allpoints.append(loc)
        return loc

    def recordlines(self, todrawline):
        print(todrawline)
        coords = todrawline.split(',')
        aline = []
        for coord in coords:
            coord = coord.lstrip().rstrip()
            loc = self.recordpoint(coord, 'NO_POINT')
            aline.append(loc)
        self.alllines.append(aline)




if __name__ == "__main__":
    theGUI = DTNSimGUI()
    # theNodes = []
    # for node_id in range(MAX_NODE_NUM):
    #     node = DTNNode(node_id, 500, 500, 0.1*100)
    #     theNodes.append(node)
    #     theGUI.attach(node)
    theGUI.drawmap()