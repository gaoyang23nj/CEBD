from MovementModelBase import MovementModelBase
import numpy as np

class MovementModelRandomWalk(MovementModelBase):
    def __init__(self, steptime, maxwidth, maxheight):
        super(MovementModelRandomWalk, self).__init__()
        self.maxhw = np.array([maxwidth, maxheight])
        self.steptime = steptime
        self.choose_src()
        self.choose_dest()

    def choose_src(self):
        self.src = np.multiply(np.random.rand(2), self.maxhw)
        self.loc = self.src.copy()

    def choose_dest(self):
        self.dest = np.multiply(np.random.rand(2), self.maxhw)
        self.speed = np.random.rand() * (self.maxspeed - self.minspeed) + self.minspeed
        self.waittime_Target = np.random.rand() * (self.maxwaittime - self.minwaittime) + self.minwaittime
        self.waittime_Have = 0

    def get_dest(self):
        return self.dest.copy()

    def get_src(self):
        return self.src.copy()

    def get_loc(self):
        return self.loc.copy()

    def __moveinDuringTime(self, duringtime, label):
        # loc到dest的向量
        tmp_dist_vec = self.dest - self.loc
        # loc到dest的欧式距离
        tmp_dist = np.sqrt(pow(tmp_dist_vec[0], 2) + pow(tmp_dist_vec[1], 2))
        # 在duringtime内能行驶的距离
        duringtime_dist = self.speed * duringtime
        if tmp_dist > duringtime_dist:
            self.loc = self.loc + (duringtime_dist/tmp_dist)*tmp_dist_vec
        else:
            self.loc = self.dest.copy()
            self.waittime_Have = duringtime - (duringtime_dist - tmp_dist) / self.speed
            if self.waittime_Have > self.waittime_Target:
                tmp_remaintime = self.waittime_Have - self.waittime_Target
                self.choose_dest()
                self.__moveinDuringTime(tmp_remaintime, 'still_remainmove')

    def moveOneStep(self):
        if (self.loc[0] != self.dest[0]) and (self.loc[1] != self.dest[1]):
            self.__moveinDuringTime(self.steptime, 'step_move')
        else:
            if self.waittime_Have + self.steptime < self.waittime_Target:
                self.waittime_Have = self.waittime_Have + self.steptime
            else:
                tmp_remaintime = self.waittime_Have + self.steptime - self.waittime_Target
                self.choose_dest()
                self.__moveinDuringTime(tmp_remaintime, 'remain_move')
        return self.loc.copy()
