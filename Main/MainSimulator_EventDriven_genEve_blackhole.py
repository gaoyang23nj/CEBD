import numpy as np
import datetime
import time
import sys
import os
import winsound

from Main.Multi_Scenario.DTNScenario_EP import DTNScenario_EP
from Main.Multi_Scenario.DTNScenario_Prophet import DTNScenario_Prophet

from Main.Scenario.DTNScenario_Prophet_Blackhole import DTNScenario_Prophet_Blackhole
from Main.Scenario_ToDetect.DTNScenario_Prophet_Blackhole_toDetect_time import DTNScenario_Prophet_Blackhole_toDetect_time


# 简化处理流程 传输速率无限

# 事件驱动
class Simulator(object):
    def __init__(self, enco_file, pktgen_freq):
        # 相遇记录文件
        self.ENCO_HIST_FILE = enco_file
        self.ENCO_ID = ((enco_file.split('\\')[-1]).split('.')[0]).split('_')[-1]
        # 节点个数默认100个, id 0~99
        self.MAX_NODE_NUM = 100
        # 最大运行时间 执行时间 36000*12个间隔, 即12hour; 应该根据 enco_hist 进行更新
        self.MAX_RUNNING_TIMES = 0
        # 每个间隔的时间长度 0.1s
        self.sim_TimeStep = 0.1
        # 仿真环境 现在的时刻
        self.sim_TimeNow = 0
        # 报文生成的间隔,即每10*60*20个时间间隔(10*60*20*0.1s 即20分钟)生成一个报文
        # self.THR_PKT_GEN_CNT = 10*30
        self.THR_PKT_GEN_CNT = pktgen_freq
        # # node所组成的list
        # self.list_nodes = []
        # 生成报文的时间计数器 & 生成报文计算器的触发值
        self.cnt_genpkt = self.THR_PKT_GEN_CNT
        self.thr_genpkt = self.THR_PKT_GEN_CNT
        # 下一个pkt的id
        self.pktid_nextgen = 0
        # 全部生成报文的list
        self.list_genpkt = []
        # 读取文件保存所有的相遇记录; self.mt_enco_hist.shape[0] 表示记录个数
        # self.mt_enco_hist = np.empty((0, 0), dtype='int')
        self.list_enco_hist = []
        self.list_gen_eve = []
        # 读取相遇记录
        self.read_enco_hist_file()
        print('read enco file end!')
        print(datetime.datetime.now())
        winsound.Beep(200, 500)
        self.build_gen_event()
        # 初始化各个场景 spamming节点的比例
        self.init_scenario()
        # 根据相遇记录执行 各场景分别执行路由
        short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = 'result_'+short_time+'.tmp'
        self.run()
        self.print_res(filename, ctstring='a+')

    def read_enco_hist_file(self):
        file_object = open(self.ENCO_HIST_FILE, 'r', encoding="utf-8")
        # 打印相遇记录的settings
        print(file_object.readline())
        tmp_all_lines = file_object.readlines()
        num_encos = len(tmp_all_lines)
        # self.mt_enco_hist = np.zeros((num_encos, 4), dtype='int')
        for index in range(len(tmp_all_lines)):
            # 读取相遇记录
            (linkup_time, linkdown_time, i_node, j_node) = tmp_all_lines[index].strip().split(',')
            linkup_time = int(linkup_time)
            linkdown_time = int(linkdown_time)
            i_node = int(i_node)
            j_node = int(j_node)
            if self.MAX_RUNNING_TIMES < linkdown_time:
                self.MAX_RUNNING_TIMES = linkdown_time
            isExist = False
            insert_index = -1
            for event_time_pair_index in range(len(self.list_enco_hist)):
                if self.list_enco_hist[event_time_pair_index][0] == linkup_time:
                    isExist = True
                    # 加入此事件
                    tunple = (linkup_time, linkdown_time, i_node, j_node)
                    self.list_enco_hist[event_time_pair_index].append(tunple)
                    break
                elif self.list_enco_hist[event_time_pair_index][0] > linkup_time:
                    insert_index = event_time_pair_index
                    break
            if isExist == False:
                if insert_index == -1:
                    self.list_enco_hist.append([linkup_time, (linkup_time, linkdown_time, i_node, j_node)])
                else:
                    self.list_enco_hist.insert(insert_index, [linkup_time, (linkup_time, linkdown_time, i_node, j_node)])
        file_object.close()

    def build_gen_event(self):
        i = 0
        while True:
            gen_time = i*self.THR_PKT_GEN_CNT
            if gen_time >= self.MAX_RUNNING_TIMES:
                break
            (src_index, dst_index) = self.__gen_pair_randint(self.MAX_NODE_NUM)
            self.list_gen_eve.append((gen_time, self.pktid_nextgen, src_index, dst_index))
            self.pktid_nextgen = self.pktid_nextgen + 1
            i = i+1

    def run(self):
        while self.sim_TimeNow < self.MAX_RUNNING_TIMES:
            if len(self.list_gen_eve)==0 and len(self.list_enco_hist)==0:
                break
            gen_time = sys.maxsize
            enco_time = sys.maxsize
            if len(self.list_gen_eve)>0:
                gen_time = self.list_gen_eve[0][0]
            if len(self.list_enco_hist)>0:
                enco_time = self.list_enco_hist[0][0]
            if gen_time <= enco_time:
                self.sim_TimeNow = gen_time
                # 执行报文生成
                # controller记录这个pkt
                self.list_genpkt.append((self.list_gen_eve[0][1], self.list_gen_eve[0][2], self.list_gen_eve[0][3]))
                # 各scenario生成pkt, pkt大小为100k
                print('time:{} pkt_id:{} src:{} dst:{}'.format(self.sim_TimeNow, self.list_gen_eve[0][1], self.list_gen_eve[0][2], self.list_gen_eve[0][3]))
                for key, value in self.scenaDict.items():
                    value.gennewpkt(self.list_gen_eve[0][1], self.list_gen_eve[0][2], self.list_gen_eve[0][3], self.sim_TimeNow, 500)
                # 删除这个生成事件 以便继续进行
                self.list_gen_eve.pop(0)
            if gen_time >= enco_time:
                self.sim_TimeNow = enco_time
                # 执行相遇事件list 同一时刻可能有多个相遇事件
                tmp_enc = self.list_enco_hist[0][1:]
                for enc_eve in tmp_enc:
                    for key, value in self.scenaDict.items():
                        value.swappkt(self.sim_TimeNow, enc_eve[2], enc_eve[3])
                        # value.swappkt(self.sim_TimeNow, enc_eve[3], enc_eve[2])
                self.list_enco_hist.pop(0)
        assert(len(self.list_gen_eve)==0 and len(self.list_enco_hist)==0)

    # # 随机决定 是否生成pkt
    # def pkt_generator_rand(self):
    #     # 生成 0~self.THR_PKT_GEN_CNT-1 的数字
    #     tmp_dec = np.random.randint(self.THR_PKT_GEN_CNT)
    #     if 0 == tmp_dec:
    #         self.__scenariogenpkt()

    def __gen_pair_randint(self, int_range):
        src_index = np.random.randint(int_range)
        dst_index = np.random.randint(int_range-1)
        if dst_index >= src_index:
            dst_index = dst_index + 1
        return (src_index, dst_index)

    # blackhole场景下 准备detect模型训练所需要的数据
    def init_scenario_blackhole_todetect(self):
        index = 0
        # ===============================场景0 Epidemic ===================================
        tmp_senario_name = 'scenario' + str(index)+'_Epidemic'
        tmpscenario = DTNScenario_EP(tmp_senario_name, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景1 Prophet ===================================
        index += 1
        tmp_senario_name = 'scenario' + str(index)+'_Prophet'
        tmpscenario = DTNScenario_Prophet(tmp_senario_name, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # 0.1 0.2 0.3 0.4 0.5
        for j in range(5):
            # ===============================场景2 Prophet + Blackhole + toDetect 0.1===================================
            # # 随机生成序列
            tmp = j + 1
            percent_selfish = 0.1 * tmp
            indices = np.random.permutation(self.MAX_NODE_NUM)
            malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
            normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]

            index += 1
            tmp_senario_name = 'scenario' + str(index)+'_blackhole_0_' + str(tmp)
            tmpscenario = DTNScenario_Prophet_Blackhole(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
            self.scenaDict.update({tmp_senario_name: tmpscenario})

            # 1*9 + 98*9 = 891个属性// 融合
            index += 1
            tmp_senario_name = 'traindata_' + str(self.ENCO_ID) + '_' + str(self.THR_PKT_GEN_CNT) + '_' + 'scenario' + str(index)+ '_blackhole_todetect_time_0_' + str(tmp)
            tmpscenario = DTNScenario_Prophet_Blackhole_toDetect_time(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM,
                                                                 20000, self.MAX_RUNNING_TIMES, True)
            self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景单个单个的实验吧===================================
        list_scena = list(self.scenaDict.keys())
        return list_scena

    def init_scenario(self):
        self.scenaDict = {}
        list_scena = self.init_scenario_blackhole_todetect()
        return list_scena

    # 打印出结果
    def print_res(self, filename, ctstring):
        file_object = open(filename, ctstring, encoding="utf-8")
        gen_total_num = len(self.list_genpkt)

        file_object.write('genfreq:{} RunningTime_Max:{} gen_num:{} nr_nodes:{}\n '.format(
            self.THR_PKT_GEN_CNT, self.MAX_RUNNING_TIMES, gen_total_num, self.MAX_NODE_NUM))
        for key, value in self.scenaDict.items():
            outstr, res, config = value.print_res(self.list_genpkt)
            file_object.write(outstr + '\n')
        file_object.close()

if __name__ == "__main__":
    t1 = datetime.datetime.now()
    print(datetime.datetime.now())

    simdurationlist = []
    encohistdir = '..\\EncoHistData\\train'
    filelist = os.listdir(encohistdir)

    # 1.真正的流程
    # 针对5个相遇记录 和 6个生成速率 分别进行实验（生成blackhole证据的实验）

    # genpkt_freqlist = [10*30, 10*60, 10*90, 10*120, 10*150, 10*180]
    genpkt_freqlist = [10 * 30, 10 * 60, 10 * 90, 10 * 120, 10 * 150]
    # for i_filename in range(len(filelist)):
    # 从指定为位置开始保存
    # i_filename = 0
    i_filename = 3
    while i_filename < len(filelist):
        filepath = os.path.join(encohistdir, filelist[i_filename])
        for genpkt_freq in genpkt_freqlist:
            print(filepath, genpkt_freq)
            t_start = time.time()
            theSimulator = Simulator(filepath, genpkt_freq)
            t_end = time.time()
            print('running time:{}'.format(t_end - t_start))
            simdurationlist.append(t_end - t_start)
        i_filename = i_filename + 1

    # or 2.简单测试的流程

    # genpkt_freqlist = 10 * 30
    # filepath = os.path.join(encohistdir, filelist[0])
    # theSimulator = Simulator(filepath, genpkt_freqlist)

    t2 = datetime.datetime.now()
    print(datetime.datetime.now())
    winsound.Beep(500, 2000)
    print(t1)
    print(t2)
    print('running time:{}'.format(t2 - t1))
    print(simdurationlist)