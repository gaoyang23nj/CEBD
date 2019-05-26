import numpy as np
from MoveShortestPathMap import MoveShortestPathMap
from WKTPathReader import WKTPathReader

class DTNNodeMap(object):
    def __init__(self, node_id, steptime, PathReader):
        self.MaxSize = 800
        self.thePathReader = PathReader
        self.theMoveShortestPathMap = MoveShortestPathMap(self.thePathReader)
        self.minspeed = 0.5
        self.maxspeed = 1.5
        self.minwaittime = 0
        self.maxwaittime = 5
        self.node_id = node_id
        # self.loc = np.array([self.loc_norm[0] * height, self.loc_norm[1] * width])
        # self.maxhw = np.array([height, width])
        self.steptime = steptime
        self.chooseSrc()
        self.choose_dest()


    def chooseSrc(self):
        self.src = self.theMoveShortestPathMap.chooseRandLoc()
        self.loc = self.src[0]


    def choose_dest(self):
        # 元素相乘法
        self.dest = self.theMoveShortestPathMap.chooseRandLoc()
        self.Path = self.theMoveShortestPathMap.getPath(self.src, self.dest)
        self.nextNode = 1
        self.speed = np.random.rand() * (self.maxspeed - self.minspeed) + self.minspeed
        self.waittime_Target = np.random.rand() * (self.maxwaittime - self.minwaittime) + self.minwaittime
        self.waittime_Have = 0


    def getSrcDest(self):
        return self.node_id, self.src, self.dest


    def run(self):
        # 如果不在dest 应该move一个steptime;  否则
        if not self.thePathReader.isSameCoord(self.loc, self.dest[0]):
            self.moveonestep(self.steptime, 'step_move')
        else:
            # 如果等的时候不够 继续等下去
            if self.waittime_Have + self.steptime < self.waittime_Target:
                self.waittime_Have = self.waittime_Have + self.steptime
            # 如果本次更新 等的时间足够了 就获得新的dest 并用剩余的时间move
            else:
                # [0, steptime]
                tmp_remaintime = self.waittime_Have + self.steptime - self.waittime_Target
                # 新的src, 注意 loc两端的路口也要更新
                self.src = self.dest
                self.choose_dest()
                self.moveonestep(tmp_remaintime, 'remain_move')
        if self.thePathReader.isSameCoord(self.loc, self.dest[0]):
            print('isInDest')
            print(self.loc)
        return self.loc

    def moveonestep(self, movetime, label):
        toNextdist = self.theMoveShortestPathMap.getdist(self.loc, self.Path[self.nextNode])
        onestep_dist = self.speed * movetime
        while self.nextNode <= len(self.Path)-1:
            if onestep_dist <= toNextdist:
                scale = onestep_dist / toNextdist
                self.loc = (self.Path[self.nextNode] - self.loc) * scale + self.loc
                break
            else:
                onestep_dist = onestep_dist - toNextdist
                self.loc = self.Path[self.nextNode]
                self.nextNode = self.nextNode + 1
                # next已经循环到最后了, 已经到达了目的
                if self.nextNode >= len(self.Path):
                    if not self.thePathReader.isSameCoord(self.loc, self.dest[0]):
                        print('ERROR!!!!!!!!!')
                    self.waittime_Have = onestep_dist/self.speed
                    # 如果waiting时间够了 更新目的地 接着跑；
                    if self.waittime_Have > self.waittime_Target:
                        tmp_remaintime = self.waittime_Have - self.waittime_Target
                        self.src = self.dest
                        self.choose_dest()
                        self.moveonestep(tmp_remaintime, 'still_remainmove')
                else:
                    toNextdist = self.theMoveShortestPathMap.getdist(self.Path[self.nextNode - 1], self.Path[self.nextNode])


    def getPath(self):
        return self.Path


    def getNodeId(self):
        return self.node_id


    def getNodeDest(self):
        return self.dest[0]