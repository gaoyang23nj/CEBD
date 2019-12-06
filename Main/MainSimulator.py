import numpy as np
from Main.DTNScenario import DTNScenario

# 简化处理流程 传输速率无限

class Simulator():
    def Simulator(self):
        # 相遇记录文件
        self.ENCO_HIST_FILE = 'EncoHistData\encohist_20191202213935.tmp'
        # 节点个数默认100个, id 0~99
        self.MAX_NODE_NUM = 100
        # 最大运行时间 执行时间 36000*12个间隔, 即12hour
        self.MAX_RUNNING_TIMES = 36000*12
        # 每个间隔的时间长度 0.1s
        self.sim_TimeStep = 0.1
        # 仿真环境 现在的时刻
        self.sim_TimeNow = 0
        # 报文生成的间隔,即每6000个时间间隔(6000*0.1)生成一个报文
        self.THR_PKT_GEN_CNT = 6000
        # node所组成的list
        self.list_nodes = []
        # 生成报文的时间计数器 & 生成报文计算器的触发值
        self.cnt_genpkt = self.THR_PKT_GEN_CNT
        self.thr_genpkt = self.THR_PKT_GEN_CNT
        # 下一个pkt的id
        self.pktid_nextgen = 1
        # 全部生成报文的list
        self.list_genpkt = []
        # 读取文件保存所有的相遇记录; self.mt_enco_hist.shape[0] 表示记录个数
        # self.mt_enco_hist = np.empty((0, 0), dtype='int')
        self.list_enco_hist = []
        # 读取相遇记录
        self.read_enco_hist_file()
        # 初始化各个场景 spamming节点的比例
        self.init_scenario()
        # 根据相遇记录执行 各场景分别执行路由
        self.run()


    def read_enco_hist_file(self):
        file_object = open(self.ENCO_HIST_FILE, 'r', encoding="utf-8")
        # 打印相遇记录的settings
        print(file_object.readline())
        tmp_all_lines = file_object.readlines()
        num_encos = len(tmp_all_lines)
        # self.mt_enco_hist = np.zeros((num_encos, 5), dtype='int')
        for index in range(len(tmp_all_lines)):
            # 读取相遇记录
            (linkup_time, linkdown_time, i_node, j_node) = tmp_all_lines[index].strip().split(',')
            linkup_time = int(linkup_time)
            linkdown_time = int(linkdown_time)
            i_node = int(i_node)
            j_node = int(j_node)
            # 按照linkup的时间排序 因为我们假设传输瞬间完成
            insert_index = 0
            for tmp_i in range(len(self.list_enco_hist)):
                if self.list_enco_hist[tmp_i][0] > self.sim_TimeNow:
                    insert_index = tmp_i
                    break
            self.list_enco_hist.insert(insert_index, (linkup_time,linkdown_time,i_node,j_node))
        file_object.close()

    def run(self):
        while self.sim_TimeNow < self.MAX_RUNNING_TIMES:
            # 报文生成
            self.pkt_generator_constant()
            # **************************************************** 简化流程 只有在发起时刻 才尝试执行
            tmp_enc = []
            # 查找此时刻所有的连接事件 只要发起就执行; 因为是有序表 无需多查
            for index in range(len(self.list_enco_hist)):
                if (self.list_enco_hist[index][0] == self.sim_TimeNow):
                    tmp_enc.append((self.list_enco_hist[index][2], self.list_enco_hist[index][3]))
                    self.list_enco_hist.remove(index)
                else:
                    break
            # 执行所有这些相遇事件
            for enc_eve in tmp_enc:
                for key, value in self.scenaDict.items():
                    value.swappkt(self.sim_TimeNow, enc_eve[0], enc_eve[1])
                    value.swappkt(self.sim_TimeNow, enc_eve[1], enc_eve[0])
            # 时间增加 以判断是否终止
            self.sim_TimeNow = self.sim_TimeNow + 1

    # 均匀间隔 生成pkt <到时间就生成>
    def pkt_generator_constant(self):
        # 报文生成计数器
        if self.cnt_genpkt == self.thr_genpkt:
            self.__scenariogenpkt()
            self.cnt_genpkt = 1
        else:
            self.cnt_genpkt =  self.cnt_genpkt + 1

    # 随机决定 是否生成pkt
    def pkt_generator_rand(self):
        # 生成 0~self.THR_PKT_GEN_CNT-1 的数字
        tmp_dec = np.random.randint(self.THR_PKT_GEN_CNT)
        if 0 == tmp_dec:
            self.__scenariogenpkt()

    # 各个scenario生成报文
    def __scenariogenpkt(self):
        (src_index, dst_index) = self.__gen_pair_randint(self.MAX_NODE_NUM)
        newpkt = (self.pktid_nextgen, src_index, dst_index)
        # controller记录这个pkt
        self.list_genpkt.append(newpkt)
        # 各scenario生成pkt, pkt大小为100k
        for key, value in self.scenaDict.items():
            value.gennewpkt(self.pktid_nextgen, src_index, dst_index, self.sim_TimeNow, 100)
        self.pktid_nextgen = self.pktid_nextgen + 1
        return

    def __gen_pair_randint(self, int_range):
        src_index = np.random.randint(self.MAX_NODE_NUM)
        dst_index = np.random.randint(self.MAX_NODE_NUM-1)
        if dst_index >= src_index:
            dst_index = dst_index + 1
        return (src_index, dst_index)


    def init_scenario(self):
        self.scenaDict = {}
        index = 0
        # ===============================场景1 全ep routing===================================
        list_idrouting = []
        for node_id in range(self.MAX_NODE_NUM):
            list_idrouting.append((node_id, 'RoutingEpidemic'))
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景5 设置50%的dropping node===================================
        index += 1
        # 随机生成序列
        percent_selfish = 0.5
        indices = np.random.permutation(self.MAX_NODE_NUM)
        malicious_indices = indices[: int(percent_selfish * self.MAX_NODE_NUM)]
        normal_indices = indices[int(percent_selfish * self.MAX_NODE_NUM):]
        list_idrouting = []
        id = 0
        for node_id in range(self.MAX_NODE_NUM):
            if id in normal_indices:
                list_idrouting.append((node_id, 'RoutingEpidemic'))
            elif id in malicious_indices:
                list_idrouting.append((node_id, 'RoutingBlackhole'))
            else:
                print('ERROR! Scenario Init! id: ', id)
            id = id + 1
        tmp_senario_name = 'scenario' + str(index)
        tmpscenario = DTNScenario(tmp_senario_name, list_idrouting)
        self.scenaDict.update({tmp_senario_name: tmpscenario})
        # ===============================场景单个单个的实验吧===================================
        list_scena = list(self.scenaDict.keys())
        return list_scena

    # 打印出结果
    def print_tmp_res(self):
        gen_total_num = len(self.list_genpkt)
        print('\n range_comm:{} genfreq:{} RunningTime_Max:{} RunningTime:{} gen_num:{} nr_nodes:{}'.format(
            self.range_comm, self.thr_genpkt, self.RunningTime_Max, self.RunningTime, gen_total_num, self.nr_nodes))
        for key, value in self.scenaDict.items():
            total_succnum, normal_succnum, selfish_succnum, \
            total_delay, normal_delay, selfish_delay = value.showres()
            print('【{}】 total_succnum:{} normal_succnum:{} selfish_succnum:{} '
                  'total_delay:{} normal_delay:{} selfish_delay:{}'.format(
                key, total_succnum, normal_succnum, selfish_succnum, total_delay, normal_delay, selfish_delay))

    # 打印出结果
    def print_res(self, filename, ctstring):
        file_object = open(filename, ctstring, encoding="utf-8")
        gen_total_num = len(self.list_genpkt)
        file_object.write('\n')
        file_object.write('\n range_comm:{} genfreq:{} RunningTime_Max:{} RunningTime:{} gen_num:{} nr_nodes:{}'.format(
            self.range_comm, self.thr_genpkt, self.RunningTime_Max, self.RunningTime, gen_total_num, self.nr_nodes))
        for key, value in self.scenaDict.items():
            total_succnum, normal_succnum, selfish_succnum, \
            total_delay, normal_delay, selfish_delay = value.showres()
            file_object.write('\n【{}】 total_succnum:{} normal_succnum:{} selfish_succnum:{} '
                              'total_delay:{} normal_delay:{} selfish_delay:{}'.format(
                key, total_succnum, normal_succnum, selfish_succnum, total_delay, normal_delay, selfish_delay))
            gen_selfish_num = 0
            listselfishid = value.getselfishlist()
            file_object.write('\n normal_node:{} selfish_node:{}'.format(
                len(self.list_node) - len(listselfishid), len(listselfishid)))
            if len(listselfishid) > 0:
                for pkt in self.list_genpkt:
                    (id, src, dst) = pkt
                    if (src in listselfishid) or (dst in listselfishid):
                        gen_selfish_num += 1
                gen_normal_num = gen_total_num - gen_selfish_num
                if gen_normal_num > 0:
                    file_object.write('\n normal_succratio:{}'.format(normal_succnum / gen_normal_num))
                    if normal_succnum > 0:
                        file_object.write(' normal_avgdelay:{}'.format(normal_delay / normal_succnum))
            if gen_total_num > 0:
                file_object.write('\n total_succratio:{}'.format(total_succnum / gen_total_num))
                if total_succnum > 0:
                    file_object.write(' total_avgdelay:{}'.format(total_delay / total_succnum))
        file_object.close()


if __name__ == "__main__":
    theSimulator = Simulator()
