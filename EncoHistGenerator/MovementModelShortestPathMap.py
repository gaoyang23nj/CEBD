from EncoHistGenerator.MovementModelBase import MovementModelBase
import numpy as np

class MovementModelShortestPathMap(MovementModelBase):
    def __init__(self, steptime, pathreader):
        super(MovementModelShortestPathMap, self).__init__()
        self.pathreader = pathreader
        self.headofadjnodes, self.allnodemap = self.pathreader.getListNode()
        self.steptime = steptime
        self.choose_src()
        self.choose_dest()

    def getdist(self, a, b):
        c = a-b
        return np.sqrt(np.power(c[0],2)+np.power(c[1],2))

    def __chooseRandLoc(self):
        idx_a = np.random.randint(len(self.headofadjnodes))
        node_a = self.allnodemap[idx_a][0]
        idx_b = np.random.randint(len(self.allnodemap[idx_a]) - 1)
        node_b = self.allnodemap[idx_a][idx_b]
        totaldist = node_b - node_a
        randloc = np.random.rand() * totaldist + node_a
        return randloc, node_a, node_b

    def __computePath(self, src, dest):
        DestDist = -1
        DestPath = []
        djklist = []
        djkliststore = []
        djklistdist = []
        djklist.append(src[1])
        djkliststore.append([src[0], src[1]])
        djklistdist.append(self.getdist(src[0], src[1]))
        if not self.pathreader.isSameCoord(src[1], src[2]):
            djklist.append(src[2])
            djkliststore.append([src[0], src[2]])
            djklistdist.append(self.getdist(src[0], src[2]))
        isFinish = False
        i = 0
        while i < len(djklist):
            # for i in range(len(djklist)):
            # fromidx = -1
            fromidx = self.pathreader.getIdxInList(djklist[i], self.headofadjnodes)
            # djkliststore[i]
            for adj in range(1, len(self.allnodemap[fromidx])):
                # to visit the node
                targetnode = self.allnodemap[fromidx][adj]
                targetidx = self.pathreader.getIdxInList(targetnode, djklist)
                if targetidx == -1:
                    djklist.append(targetnode)
                    targetdist = djklistdist[i] + self.getdist(djklist[i], targetnode)
                    djklistdist.append(targetdist)
                    targetlist = djkliststore[i].copy()
                    targetlist.append(targetnode)
                    djkliststore.append(targetlist)
                else:
                    # 如果有新点也能到这里，那么它的距离是不是更短呢？
                    targetdist = djklistdist[i] + self.getdist(djklist[i], targetnode)
                    if targetdist < djklistdist[targetidx]:
                        djklistdist[targetidx] = targetdist
                        targetlist = djkliststore[i].copy()
                        targetlist.append(targetnode)
                        djkliststore[targetidx] = targetlist
                if self.pathreader.isSameCoord(targetnode, dest[1]) or self.pathreader.isSameCoord(targetnode,
                                                                                                         dest[2]):
                    tmpDestDist = djklistdist[targetidx] + self.getdist(targetnode, dest[0])
                    if (len(DestPath) == 0) or (tmpDestDist < DestDist):
                        DestDist = tmpDestDist
                        DestPath = djkliststore[targetidx].copy()
                        DestPath.append(dest[0])
            # ******************************
                    isBig = False
                    for j in range(i+1, targetidx):
                        if DestDist > djklistdist[j]:
                            isBig = True
                            break
                    if isBig == False:
                        isFinish = True
                        break

            if isFinish == True:
                break
            # ******************************
            i = i + 1
        return DestPath

    def choose_src(self):
        self.src = self.__chooseRandLoc()
        self.loc = self.src[0].copy()

    def choose_dest(self):
        # 元素相乘法
        self.dest = self.__chooseRandLoc()
        self.path = self.__computePath(self.src, self.dest)
        self.nextNode = 1
        self.speed = np.random.rand() * (self.maxspeed - self.minspeed) + self.minspeed
        self.waittime_Target = np.random.rand() * (self.maxwaittime - self.minwaittime) + self.minwaittime
        self.waittime_Have = 0

    def get_dest(self):
        return self.dest[0].copy()

    def get_src(self):
        return self.src[0].copy()

    def get_loc(self):
        return self.loc.copy()

    def get_path(self):
        return self.path

    def get_srcdestpair(self):
        return self.src, self.dest

    def __moveinDuringTime(self, movetime, label):
        toNextdist = self.getdist(self.loc, self.path[self.nextNode])
        onestep_dist = self.speed * movetime
        while self.nextNode <= len(self.path)-1:
            if onestep_dist <= toNextdist:
                scale = onestep_dist / toNextdist
                self.loc = (self.path[self.nextNode] - self.loc) * scale + self.loc
                break
            else:
                onestep_dist = onestep_dist - toNextdist
                self.loc = self.path[self.nextNode]
                self.nextNode = self.nextNode + 1
                # next已经循环到最后了, 已经到达了目的
                if self.nextNode >= len(self.path):
                    if not self.pathreader.isSameCoord(self.loc, self.dest[0]):
                        print('ERROR!!!!!!!!!')
                    self.waittime_Have = onestep_dist/self.speed
                    # 如果waiting时间够了 更新目的地 接着跑；
                    if self.waittime_Have > self.waittime_Target:
                        tmp_remaintime = self.waittime_Have - self.waittime_Target
                        self.src = self.dest
                        self.choose_dest()
                        self.__moveinDuringTime(tmp_remaintime, 'still_remainmove')
                else:
                    toNextdist = self.getdist(self.path[self.nextNode - 1], self.path[self.nextNode])

    def moveOneStep(self):
        # 如果不在dest 应该move一个steptime;  否则
        if not self.pathreader.isSameCoord(self.loc, self.dest[0]):
            self.__moveinDuringTime(self.steptime, 'step_move')
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
                self.__moveinDuringTime(tmp_remaintime, 'remain_move')
        if self.pathreader.isSameCoord(self.loc, self.dest[0]):
            print('isInDest')
            print(self.loc)
        return self.loc.copy()












