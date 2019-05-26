import numpy as np

class DTNNode(object):
    def __init__(self,node_id,height,width,steptime):
        self.minspeed = 0.5
        self.maxspeed = 1.5
        # self.minspeed = 50
        # self.maxspeed = 150
        self.minwaittime = 0
        self.maxwaittime = 5
        self.node_id = node_id
        self.loc_norm = np.random.rand(2)
        self.loc = np.array([self.loc_norm[0]*height, self.loc_norm[1]*width])
        self.maxhw = np.array([height, width])
        self.steptime = steptime
        self.choose_dest()

    def run(self):
        if (self.loc[0] != self.dest[0]) and (self.loc[1] != self.dest[1]):
            self.moveonestep(self.steptime, 'step_move')
        else:
            if self.waittime_Have + self.steptime < self.waittime_Target:
                self.waittime_Have = self.waittime_Have + self.steptime
            else:
                # [0, steptime]
                tmp_remaintime = self.waittime_Have + self.steptime - self.waittime_Target
                self.choose_dest()
                self.moveonestep(tmp_remaintime, 'remain_move')
        return self.loc

    def moveonestep(self, movetime, label):
        tmp_dist_vec = self.dest - self.loc
        tmp_dist = np.sqrt(pow(tmp_dist_vec[0], 2) + pow(tmp_dist_vec[1], 2))
        onestep_dist = self.speed * movetime
        if tmp_dist > onestep_dist:
            self.loc[0] = self.loc[0] + onestep_dist * tmp_dist_vec[0] / tmp_dist
            self.loc[1] = self.loc[1] + onestep_dist * tmp_dist_vec[1] / tmp_dist
        else:
            self.loc = self.dest
            self.waittime_Have = self.steptime - (onestep_dist - tmp_dist) / self.speed
            if self.waittime_Have > self.waittime_Target:
                tmp_remaintime = self.waittime_Have - self.waittime_Target
                self.choose_dest()
                self.moveonestep(tmp_remaintime, 'still_remainmove')


    def choose_dest(self):
        # 元素相乘法
        self.dest = np.multiply(np.random.rand(2), self.maxhw)
        self.speed = np.random.rand() * (self.maxspeed - self.minspeed) + self.minspeed
        self.waittime_Target = np.random.rand() * (self.maxwaittime - self.minwaittime) + self.minwaittime
        self.waittime_Have = 0


    def getNodeId(self):
        return self.node_id


    def getNodeDest(self):
        return self.dest