class MovementModelBase(object):
    def __init__(self):
        # 步行, 1.8km/h~ 5.4km/h
        # self.minspeed = 0.5
        # self.maxspeed = 1.5
        # 车, 10.8km/h (3m/s) ~ 20km/h (6m/s)
        self.minspeed = 3
        self.maxspeed = 6
        self.minwaittime = 0
        self.maxwaittime = 5

    @classmethod
    def choose_src(cls):
        '''

        :return:
        '''

    @classmethod
    def choose_dest(cls):
        '''

        :return:
        '''

    @classmethod
    def get_dest(cls):
        '''

        :return:
        '''

    @classmethod
    def get_src(cls):
        '''

        :return:
        '''

    @classmethod
    def moveOneStep(cls):
        '''

        :return:
        '''

