import numpy as np
# import tensorflow as tf
import sys
import time
import threading

from WKTPathReader import WKTPathReader
from DTNNode import DTNNode
from DTNSimGUI import DTNSimGUI
from DTNSimGUIMap import DTNSimGUIMap
from DTNController import DTNController

np.random.seed(1)
# tf.set_random_seed(1)

MAX_NODE_NUM = 10
MAX_TIME_INDEX = 10000

def runRandomWalk():
    size = 500
    # 每100个timestep(<模拟>10s)刷新一次界面, 通信范围100m, 每6000个timestep(<模拟>600s)产生一次报文
    theViewer = DTNSimGUI(size)
    theController = DTNController(theViewer, showtimes=100, com_range=100, genfreq_cnt=6000, totaltimes=36000)
    listNodes = []
    for node_id in range(MAX_NODE_NUM):
        # 每个timestep = <模拟>0.1s
        node = DTNNode('RandomWalk', node_id, 0.1, size, size)
        listNodes.append(node)
    theController.attachnodelist(listNodes)
    theViewer.attachController(theController)
    theController.run()


def runHelsinkSPM():
    size = 800
    pathreader = WKTPathReader(size)
    # 100倍速度执行
    theGUI = DTNSimGUIMap(size, pathreader, 100)
    theNodes = []
    for node_id in range(MAX_NODE_NUM):
        # 0.1s 一个间隔
        node = DTNNode('SPM', node_id, 0.1, size, pathreader)
        theNodes.append(node)
        theGUI.attachDTNNode(node)
    theGUI.run()


if __name__ == "__main__":
    runRandomWalk()
    # runHelsinkSPM()








