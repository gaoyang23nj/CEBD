# 本程序简化Simulation_ONE/EncoHistData/文件夹下的 encohist_20191220182008.tmp
# 把 linkuptime lindowntime node_i node_j
# 简化为 linkuptime node_i node_j并按照linkuptime的顺序 排列好
# 文件保存到 encohist_20191220182008.enc

import numpy as np
import datetime
import os
import re

# 保存历史记录的位置
EncoHistDir = '../EncoHistData/'

class ProcessENCFile(object):
    def __init__(self, input_tmp_filepath, output_enc_filepath):
        t1 = datetime.datetime.now()
        print(datetime.datetime.now())

        self.input_tmp_filepath = input_tmp_filepath
        self.output_enc_filepath = output_enc_filepath
        self.list_enco_hist = []
        # 最大运行时间 执行时间 36000*24个间隔, 即24hour; 应该根据 enco_hist 进行更新
        self.MAX_RUNNING_TIME_up = 0
        self.MAX_RUNNING_TIME_down = 0
        self.input_num_encos = 0
        self.output_num_encos = 0
        self.settings = ''
        self.read_enco_hist_file()
        self.write_enc_file()

        print('num_enc: before:{}, after:{}'.format(self.input_num_encos, self.output_num_encos))
        print('filename: before:{}, after:{}'.format(self.input_tmp_filepath, self.output_enc_filepath))

        t2 = datetime.datetime.now()
        print(datetime.datetime.now())
        print('running time:{}'.format(t2 - t1))

    def write_enc_file(self):
        file_object = open(self.output_enc_filepath, 'a+', encoding="utf-8")
        file_object.write(self.settings[:-1])
        for tunple in self.list_enco_hist:
            # 执行相遇事件list 同一时刻可能有多个相遇事件
            list_eve = tunple[1:]
            for enc_eve in list_eve:
                (time_linkup, x_id, y_id) = enc_eve
                file_object.write('{},{},{}\n'.format(time_linkup, x_id, y_id))
                self.output_num_encos = self.output_num_encos + 1
        file_object.close()

    def read_enco_hist_file(self):
        file_object = open(self.input_tmp_filepath, 'r', encoding="utf-8")
        # 打印相遇记录的settings
        self.settings = file_object.readline()
        print(self.settings)
        tmp_all_lines = file_object.readlines()
        self.num_encos = len(tmp_all_lines)
        for index in range(len(tmp_all_lines)):
            # 读取相遇记录
            (linkup_time, linkdown_time, i_node, j_node) = tmp_all_lines[index].strip().split(',')
            linkup_time = int(linkup_time)
            linkdown_time = int(linkdown_time)
            i_node = int(i_node)
            j_node = int(j_node)
            if self.MAX_RUNNING_TIME_up < linkup_time:
                self.MAX_RUNNING_TIME_up = linkup_time
            if self.MAX_RUNNING_TIME_down < linkdown_time:
                self.MAX_RUNNING_TIME_down = linkdown_time
            isExist = False
            insert_index = -1
            # 插入到有序list中(找到合适的位置)
            for event_time_pair_index in range(len(self.list_enco_hist)):
                if self.list_enco_hist[event_time_pair_index][0] == linkup_time:
                    isExist = True
                    # 加入此事件
                    tunple = (linkup_time, i_node, j_node)
                    self.list_enco_hist[event_time_pair_index].append(tunple)
                    break
                elif self.list_enco_hist[event_time_pair_index][0] > linkup_time:
                    # 找到合适的位置
                    insert_index = event_time_pair_index
                    break
                # 如果小于linkuptime 意味着 接着往下查
            if isExist == False:
                if insert_index == -1:
                    self.list_enco_hist.append([linkup_time, (linkup_time, i_node, j_node)])
                else:
                    self.list_enco_hist.insert(insert_index, [linkup_time, (linkup_time, i_node, j_node)])
        file_object.close()

if __name__ == "__main__":
    filelist = os.listdir(EncoHistDir)
    for filename in filelist:
        m = re.match(r'(encohist_\d*).tmp', filename)
        if m is None:
            continue
        enc_filename = filename.split('.')[0]+'.enc'
        input_tmp_filepath = os.path.join(EncoHistDir, filename)
        output_enc_filepath = os.path.join(EncoHistDir, enc_filename)
        ProcessENCFile(input_tmp_filepath, output_enc_filepath)

