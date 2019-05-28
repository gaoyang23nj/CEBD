import numpy as np
# import tensorflow as tf
import sys
import time
import threading

from WKTPathReader import WKTPathReader
from DTNNode import DTNNode
from DTNSimGUI import DTNSimGUI
from DTNSimGUIMap import DTNSimGUIMap

np.random.seed(1)
# tf.set_random_seed(1)

MAX_NODE_NUM = 10
MAX_TIME_INDEX = 10000

def runRandomWalk():
    size = 800
    theGUI = DTNSimGUI(size)
    theNodes = []
    for node_id in range(MAX_NODE_NUM):
        node = DTNNode('RandomWalk', node_id, 0.1*100, size, size)
        theNodes.append(node)
        theGUI.attachDTNNode(node)
    theGUI.run()

def runHelsinkSPM():
    size = 800
    pathreader = WKTPathReader(size)
    theGUI = DTNSimGUIMap(size, pathreader)
    theNodes = []
    for node_id in range(MAX_NODE_NUM):
        node = DTNNode('SPM', node_id, 0.1*100, size, pathreader)
        theNodes.append(node)
        theGUI.attachDTNNode(node)
    theGUI.run()

if __name__ == "__main__":
    # runRandomWalk()
    runHelsinkSPM()








