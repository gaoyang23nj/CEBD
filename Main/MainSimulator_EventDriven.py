import numpy as np
import datetime
import sys


from Main.DTNScenario_EP import DTNScenario_EP
from Main.DTNScenario_Prophet import DTNScenario_Prophet
from Main.DTNScenario_Prophet_Blackhole_toDetect import DTNScenario_Prophet_Blackhole_toDetect
from Main.DTNScenario_Prophet_Spam import DTNScenario_Prophet_Spam
from Main.DTNScenario_Prophet_SpamE import DTNScenario_Prophet_SpamE
from Main.DTNScenario_SandW import DTNScenario_SandW
from Main.DTNScenario_Prophet_Blackhole import DTNScenario_Prophet_Blackhole
# 简化处理流程 传输速率无限

# 事件驱动
class Simulator(object):
    def __init__(self):
        # 相遇记录文件
        # self.ENCO_HIST_FILE = '..\EncoHistData\encohist_20191203021550.tmp'
        # self.ENCO_HIST_FILE = '..\EncoHistData\encohist_20191205104415.tmp'
        self.ENCO_HIST_FILE = '..\EncoHistData\encohist_20191220182008.tmp'
        # 节点个数默认100个, id 0~99
        self.MAX_NODE_NUM = 100
        # 最大运行时间 执行时间 36000*12个间隔, 即12hour; 应该根据 enco_hist 进行更新
        self.MAX_RUNNING_TIMES = 0
        # 每个间隔的时间长度 0.1s
        self.sim_TimeStep = 0.1
        # 仿真环境 现在的时刻
        self.sim_TimeNow = 0
        # 报文生成的间隔,即每10*60*20个时间间隔(10*60*20*0.1s 即20分钟)生成一个报文
        self.THR_PKT_GEN_CNT = 10*10
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

    def init_scenario(self):
        self.scenaDict = {}
        index = 0
        # ===============================场景1 Epidemic ===================================
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_EP(tmp_senario_name, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景2 Prophet ===================================
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet(tmp_senario_name, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景3 Spary and Wait ===================================
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_SandW(tmp_senario_name, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景4 Prophet + Blackhole ===================================
        # # 随机生成序列
        percent_selfish = 0.3
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_Blackhole(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景5 Prophet + Blackhole ===================================
        # # 随机生成序列
        percent_selfish = 0.8
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_Blackhole(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景6 Prophet + Spam ===================================
        # # 随机生成序列
        percent_selfish = 0.3
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_Spam(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景7 Prophet + Spam ===================================
        # # 随机生成序列
        percent_selfish = 0.8
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_Spam(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景8 Prophet + Spam E ===================================
        # # 随机生成序列
        percent_selfish = 0.3
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_SpamE(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景9 Prophet + Spam E ===================================
        # # 随机生成序列
        percent_selfish = 0.8
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_SpamE(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景10 Prophet + Blackhole ===================================
        # # 随机生成序列
        percent_selfish = 0.3
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_Blackhole_toDetect(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景11 Prophet + Blackhole ===================================
        # # 随机生成序列
        percent_selfish = 0.8
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        index += 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario_Prophet_Blackhole_toDetect(tmp_senario_name, malicious_indices, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景单个单个的实验吧===================================
        list_scena = list(self.scenaDict.keys())
        return list_scena

    # 打印出结果
    def print_res(self, filename, ctstring):
        file_object = open(filename, ctstring, encoding="utf-8")
        gen_total_num = len(self.list_genpkt)

        file_object.write('genfreq:{} RunningTime_Max:{} gen_num:{} nr_nodes:{}\n '.format(
            self.THR_PKT_GEN_CNT, self.MAX_RUNNING_TIMES, gen_total_num, self.MAX_NODE_NUM))
        for key, value in self.scenaDict.items():
            str = value.print_res(self.list_genpkt)
            file_object.write(str+'\n')
        file_object.close()


if __name__ == "__main__":
    theSimulator = Simulator()
