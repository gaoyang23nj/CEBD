import threading
# 通过字典 管理多个 log file
class DTNLogFiles(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        self.Dict = dict()
        pass

    # 做成 单例模式
    def __new__(cls, *args, **kwargs):
        if not hasattr(DTNLogFiles, "_instance"):
            with DTNLogFiles._instance_lock:
                if not hasattr(DTNLogFiles, "_instance"):
                    DTNLogFiles._instance = object.__new__(cls)
        return DTNLogFiles._instance

    # 初始化名为str的log文件 str.log, 以便于写入事件
    def initlog(self, strname='eve'):
        file_name = strname + '.log'
        filelog_object = open(file_name, "w+", encoding="utf-8")
        dict = {strname : filelog_object}
        # if not self.Dict.has_key(strname):
        if not strname in self.Dict.keys():
            self.Dict.update(dict)
        else:
            print('ERROR! 已有此key:({})'.format(strname))

    def insertlog(self, strname, str):
        if not strname in self.Dict.keys():
            print('ERROE! 没有此key')
        else:
            self.Dict[strname].write(str)
            self.Dict[strname].flush()

    def closelog(self):
        for tunple in self.Dict.items():
            (key, item) = tunple
            item.close()
