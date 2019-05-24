import numpy as np
# import tensorflow as tf
import sys
import time
import threading

from DTNSimGUI import DTNSimGUI
from DTNNode import DTNNode


np.random.seed(1)
# tf.set_random_seed(1)

MAX_NODE_NUM = 10
MAX_TIME_INDEX = 10000

def runOnce(theNodes):
    node_list = []
    loc_list = []
    for node in theNodes:
        node_id = node.node_id
        loc = node.run()
        node_list.append(node_id)
        loc_list.append(loc)
    return node_list, loc_list

if __name__ == "__main__":
    theGUI = DTNSimGUI()
    theNodes = []
    for node_id in range(MAX_NODE_NUM):
        node = DTNNode(node_id, 500, 500, 0.1*100)
        theNodes.append(node)
        theGUI.attach(node)
    theGUI.run()
    # for timeindex in range(MAX_TIME_INDEX):
    #     node_list, loc_list = runOnce(theNodes)
    #     print(loc_list)






