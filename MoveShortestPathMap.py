import numpy as np

from WKTPathReader import WKTPathReader

class MoveShortestPathMap(object):
    def __init__(self, thewtkpathreader):
        self.thePathReader = thewtkpathreader
        self.headofadjnodes, self.allnodemap = self.thePathReader.getListNode()


    def chooseRandLoc(self):
        idx_a = np.random.randint(len(self.headofadjnodes))
        node_a = self.allnodemap[idx_a][0]
        idx_b = np.random.randint(len(self.allnodemap[idx_a])-1)
        node_b = self.allnodemap[idx_a][idx_b]
        totaldist = node_b - node_a
        randloc = np.random.rand()*totaldist + node_a
        return randloc, node_a, node_b


    def getdist(self, a, b):
        c = a-b
        return np.sqrt(np.power(c[0],2)+np.power(c[1],2))


    def getPath(self, src, dest):
        DestDist = -1
        DestPath = []
        djklist = []
        djkliststore = []
        djklistdist = []

        djklist.append(src[1])
        djkliststore.append([src[0], src[1]])
        djklistdist.append(self.getdist(src[0], src[1]))

        if not self.thePathReader.isSameCoord(src[1], src[2]):
            djklist.append(src[2])
            djkliststore.append([src[0], src[2]])
            djklistdist.append(self.getdist(src[0], src[2]))

        isFinish = False
        i = 0
        while i < len(djklist):
        # for i in range(len(djklist)):
            # fromidx = -1
            fromidx = self.thePathReader.getIdxInList(djklist[i], self.headofadjnodes)
            # djkliststore[i]
            for adj in range(1, len(self.allnodemap[fromidx])):
                # to visit the node
                targetnode = self.allnodemap[fromidx][adj]
                targetidx = self.thePathReader.getIdxInList(targetnode, djklist)

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
                if self.thePathReader.isSameCoord(targetnode, dest[1]) or self.thePathReader.isSameCoord(targetnode, dest[2]):
                    tmpDestDist = djklistdist[targetidx] + self.getdist(targetnode, dest[0])
                    if (len(DestPath) == 0) or (tmpDestDist < DestDist):
                        DestDist = tmpDestDist
                        DestPath = djkliststore[targetidx].copy()
                        DestPath.append(dest[0])
            #         isBig = False
            #         for j in range(i+1, targetidx):
            #             if DestDist > djklistdist[j]:
            #                 isBig = True
            #                 break
            #         if isBig == False:
            #             isFinish = True
            #             break
            #
            # if isFinish == True:
            #     break
            i = i + 1

        return DestPath













