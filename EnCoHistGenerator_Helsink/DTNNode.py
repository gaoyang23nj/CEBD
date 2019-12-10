from EncoHistGenerator.MovementModelRandomWalk import MovementModelRandomWalk
from EncoHistGenerator.MovementModelShortestPathMap import MovementModelShortestPathMap


class DTNNode(object):
    def __init__(self, *args):
        if args[0] is 'RandomWalk':
            node_id, steptime, maxwidth, maxheight = args[1:]
            self.node_id = node_id
            self.steptime = steptime
            # self.MovementModel = MovementModelRandomWalk(steptime, *args)
            self.MovementModel = MovementModelRandomWalk(steptime, maxwidth=maxwidth, maxheight=maxheight)
        elif args[0] is 'SPM':
            node_id, steptime, pathreader = args[1:]
            self.node_id = node_id
            self.steptime = steptime
            # self.MovementModel = MovementModelShortestPathMap(steptime, *args)
            self.MovementModel = MovementModelShortestPathMap(steptime, pathreader)
        else:
            print('ERROR! DTNNode.__init__()')

    def getPath(self):
        if isinstance(self.MovementModel, MovementModelShortestPathMap):
            return self.MovementModel.get_path()
        else:
            print('ERROR! DTNNode.getPath() MovementModel is not SPM!')

    def getSrcDestPair(self):
        if isinstance(self.MovementModel, MovementModelShortestPathMap):
            return self.MovementModel.get_srcdestpair()
        else:
            print('ERROR! DTNNode.getSrcDestPair() MovementModel is not SPM!')

    def getNodeId(self):
        return self.node_id

    def getNodeDest(self):
        return self.MovementModel.get_dest()

    def getNodeLoc(self):
        return self.MovementModel.get_loc()

    def getStepTime(self):
        return self.steptime

    def run(self):
        loc = self.MovementModel.moveOneStep()
        return loc
