import numpy as np
# import tensorflow as tf
import sys
import time
import threading
import datetime

from WKTPathReader import WKTPathReader
from DTNNode import DTNNode
from DTNSimGUI import DTNSimGUI
from DTNSimGUIMap import DTNSimGUIMap
from DTNController import DTNController

np.random.seed(1)
# tf.set_random_seed(1)

MAX_NODE_NUM = 20
# 执行时间 36000*12个间隔, 即12hour
MAX_RUNNING_TIMES = 36000*12

def runRandomWalk():
    showsize = 500
    realsize = 2000
    # 每100个timestep(<模拟>10s)刷新一次界面, 通信范围100m, 每600个timestep(<模拟>60s)产生一次报文
    theViewer = DTNSimGUI(showsize, realsize)
    theController = DTNController(theViewer, showtimes=100, com_range=100, genfreq_cnt=150, totaltimes=MAX_RUNNING_TIMES)
    listNodes = []
    for node_id in range(MAX_NODE_NUM):
        # 每个timestep = <模拟>0.1s
        node = DTNNode('RandomWalk', node_id, 0.1, realsize, realsize)
        listNodes.append(node)
    theController.attachnodelist(listNodes)
    theViewer.attachController(theController)
    theController.run()


def runHelsinkSPM():
    showsize = 500
    realsize = 1000
    size = 800
    pathreader = WKTPathReader()
    # 100倍速度执行
    # theGUI = DTNSimGUIMap(size, pathreader, 100)
    theViewer = DTNSimGUIMap(pathreader, showsize, isshowconn=True)
    theController = DTNController(theViewer, showtimes=100, com_range=100, genfreq_cnt=6000, totaltimes=MAX_RUNNING_TIMES)
    listNodes = []
    for node_id in range(MAX_NODE_NUM):
        # 0.1s 一个间隔
        node = DTNNode('SPM', node_id, 0.1, pathreader)
        listNodes.append(node)
    theController.attachnodelist(listNodes)
    theViewer.attachController(theController)
    theController.run()


if __name__ == "__main__":
    beginTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    runRandomWalk()
    # runHelsinkSPM()

    endTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(beginTime)
    print(endTime)









