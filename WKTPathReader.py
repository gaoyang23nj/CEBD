import numpy as np
import os

class WKTPathReader(object):
    def __init__(self, maxsize):
        self.data_dir = 'data/'
        self.allnodemap = []
        self.headofadjnodes = []
        self.alllines = []
        self.allpois = []
        self.RangeHeight = np.array([-1.0, -1.0])
        self.RangeWidth = np.array([-1.0, -1.0])
        self.MinXY = np.array([-1.0, -1.0])
        self.MaxSize = maxsize
        self.Scale = 1
        self.readthedata()
        self.genMapAdjNodes()
        self.genScaleParams()

    def readthedata(self):
        files = os.listdir(self.data_dir)
        for file in files:
            self.readwktfile(self.data_dir+file)
        self.MinXY[0] = self.RangeWidth[0]
        self.MinXY[1] = self.RangeHeight[0]


    def genScaleParams(self):
        if (self.RangeWidth[1] - self.RangeWidth[0]) > (self.RangeHeight[1] - self.RangeHeight[0]):
            self.Scale = self.MaxSize / (self.RangeWidth[1] - self.RangeWidth[0])
        else:
            self.Scale = self.MaxSize / (self.RangeHeight[1] - self.RangeHeight[0])

    def getParams(self):
        return self.alllines, self.allpois, self.MinXY, self.Scale

    def getListNode(self):
        return self.headofadjnodes.copy(), self.allnodemap.copy()

    def genMapAdjNodes(self):
        for tmpline in self.alllines:
            for i in range(len(tmpline)):
                tmpnode = tmpline[i]
                idx = -1
                for j in range(len(self.headofadjnodes)):
                    if np.dot(tmpnode-self.headofadjnodes[j], tmpnode-self.headofadjnodes[j]) == 0:
                        idx = j
                        break
                if idx == -1:
                    idx = len(self.headofadjnodes)
                    self.headofadjnodes.append(tmpnode)
                    anewnode = [tmpnode]
                    self.allnodemap.append(anewnode)
                if i > 0:
                    self.allnodemap[idx].append(tmpline[i - 1])
                if i < len(tmpline)-1:
                    self.allnodemap[idx].append(tmpline[i + 1])


    def readwktfile(self, filename):
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
            self.allpois.append(loc)
        return loc

    def recordlines(self, todrawline):
        # print(todrawline)
        coords = todrawline.split(',')
        aline = []
        for coord in coords:
            coord = coord.lstrip().rstrip()
            loc = self.recordpoint(coord, 'NO_POINT')
            aline.append(loc)
        self.alllines.append(aline)

    def getIdxInList(self, coord, listofcoords):
        idx = -1
        for j in range(len(listofcoords)):
            if np.dot(coord - listofcoords[j], coord - listofcoords[j]) == 0:
                idx = j
                break
        return idx

    def isSameCoord(self, coordA, coordB):
        return np.dot(coordA - coordB, coordA - coordB) == 0
