import math
import os
import datetime
import numpy as np
shanghai_input_data = 'E:\\00-Trace\\01-shanghai_taxi\\taxi_100'
shanghai_interpolation = 'E:\\00-Trace\\01-shanghai_taxi\\taxi_100_interpolation'
EncoHistDir = '../EncoHistData_Shanghai/'

class EncoHistGenerator_Shanghai(object):
    def __init__(self):
        # self.num_nodes = 100
        self.input_dir = shanghai_input_data
        self.interpolation_dir = shanghai_interpolation
        # 每10s保留一个位置
        self.sim_TimeStep = 1
        # id 文件 真实id对应关系
        self.list_id = []
        # 各个节点的时空位置
        self.list_space_time_loc = []
        # 也是node_id
        self.num_nodes = 0
        self.__preprocess()
        assert self.num_nodes == 100
        # 每10s一个
        self.MAX_RUNNING_TIMES = int(24*60*60/(self.sim_TimeStep))
        for i in range(self.num_nodes):
            self.__interpolation(i)
        # self.__interpolation(0)
        self.dist_comm = 100
        assert len(self.list_space_time_loc) == self.num_nodes
        for i in range(self.num_nodes):
            print(self.list_space_time_loc[i][8041])
        print('*'*30)
        # print(self.list_space_time_loc[94][8041])
        # ====================== 仿真过程 ============================
        # 仿真环境 现在的时刻
        self.sim_TimeNow = 0
        # link状态记录 记录任何两个node之间的link状态 以便检测linkdown事件
        self.mt_linkstate = np.zeros((self.num_nodes, self.num_nodes), dtype='int')
        # 记录link事件状态：两个节点已经建立link 并且没有中断
        # 格式(a_id, b_id, a_loc, b_loc, time_linkup)
        self.link_event_list = []
        # 保存全部的encounter hist；格式为(time_linkup, time_linkdown, x_id, y_id, x_loc, y_loc, a_loc, b_loc)
        self.encounter_hist_list = []
        while self.sim_TimeNow < self.MAX_RUNNING_TIMES:
            # 节点移动一个timestep
            # 检测相遇事件
            self.detect_encounter()
            self.sim_TimeNow = self.sim_TimeNow + 1
        # 过程结束 关闭所有link 并加入 encounter hist list 历史相遇记录
        self.close_each_link()
        # 按照发起时间顺序排列encounter list
        self.encounter_hist_list.sort()
        # 写入相遇记录
        self.__write_encounter()

    def __write_encounter(self):
        # 写入文件
        short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = EncoHistDir + 'encohist_shanghai_' + short_time + '.tmp'
        file_object = open(filename, 'a+', encoding="utf-8")
        # 总仿真事件 (hour)用小时表示
        total_running_time = (self.MAX_RUNNING_TIMES) / (3600 * 1 / (self.sim_TimeStep))
        file_object.write('Settings, MAX_NODE_NUM:{},RANGE_COMM:{},MAX_RUNNING_TIMES:{},{}(h), '
                          'sim_TimeStep:{},'.format(self.num_nodes, self.dist_comm, self.MAX_RUNNING_TIMES,
                                                    total_running_time, self.sim_TimeStep))
        for tunple in self.encounter_hist_list:
            (time_linkup, time_linkdown, x_id, y_id, x1, x2, y1, y2) = tunple
            file_object.write('\n{},{},{},{}'.format(time_linkup, time_linkdown, x_id, y_id))
        file_object.close()

    def __preprocess(self):
        file_list = os.listdir(self.input_dir)
        for file_name in file_list:
            oldid = int(file_name.split('_')[1])
            filepath = os.path.join(self.input_dir, file_name)
            self.list_id.append((self.num_nodes, filepath, oldid))
            self.num_nodes = self.num_nodes + 1
        print(self.list_id)

    def __interpolation(self, node_id):
        # 节点id 时刻 lng lat
        this_node_loc = []
        real_node_id = self.list_id[node_id][2]
        # 插值后的文件
        filename = os.path.join(self.interpolation_dir, str(node_id)+'_taxi.i')
        if os.path.exists(filename):
            os.remove(filename)
        interpolfile_obj = open(filename, 'a+', encoding="utf-8")
        date_begin = datetime.datetime.strptime('2007-02-20 00:00:00', "%Y-%m-%d %H:%M:%S")
        date_step = self.sim_TimeStep
        # 预计总的时长是24h
        total_time_step = int(24*60*60/date_step)
        # print(self.list_id[node_id][1])
        oldfile_obj = open(self.list_id[node_id][1], 'r', encoding="utf-8")
        tmp_all_lines = oldfile_obj.readlines()
        num_lines = len(tmp_all_lines)
        last_time_index = 0
        last_lng = -1
        last_lat = -1
        current_count_time_step = 0
        index = 0
        last_real_time = datetime.datetime.strptime('2007-02-20 00:00:00', "%Y-%m-%d %H:%M:%S")
        last_real_delta_time = 0
        while index < num_lines:
            tmp_next_line = tmp_all_lines[index]
            tmp_next_date = datetime.datetime.strptime(tmp_next_line.split(',')[1], "%Y-%m-%d %H:%M:%S")
            tmp_next_delta_time = int((tmp_next_date - date_begin).total_seconds())
            # 第几个timestep
            next_true_time_index = math.floor(tmp_next_delta_time/date_step)
            next_lng = float(tmp_next_line.split(',')[2])
            next_lat = float(tmp_next_line.split(',')[3])
            if last_time_index <= next_true_time_index:
                tmp_ = last_time_index
                if index == 0:
                    # 第一次出现 没有位置 只能假设一直在这里
                    while tmp_ < next_true_time_index:
                        # print(node_id, tmp_ * date_step, next_lng, next_lat, tmp_next_date, tmp_next_delta_time)
                        output_str = '{},{},{},{},{},{},{}\n'.format(node_id, tmp_ * date_step, next_lng, next_lat,
                                                                  tmp_next_date, tmp_next_delta_time, real_node_id)
                        interpolfile_obj.write(output_str)
                        this_node_loc.append((node_id, tmp_ * date_step, next_lng, next_lat, tmp_next_date, real_node_id))
                        tmp_ = tmp_ + 1
                else:
                    # 线性插值 tmp_num_steps表示共分成几段
                    tmp_num_steps = next_true_time_index-last_time_index + 1
                    diff_lng = (next_lng - last_lng)/tmp_num_steps
                    diff_lat = (next_lat - last_lat)/tmp_num_steps
                    _tmp_count_step = 1
                    # print('from:{},{} to:{},{}... diff:{},{}'.format(last_lng, last_lat, next_lng, next_lat, diff_lng, diff_lat))
                    while tmp_ < next_true_time_index:
                        # print('{} in {}'.format(_tmp_count_step, tmp_num_steps))
                        new_lng = last_lng + diff_lng*_tmp_count_step
                        new_lat = last_lat + diff_lat*_tmp_count_step
                        # print('one step {},{}'.format(new_lng, new_lat))

                        # print(node_id, tmp_ * date_step, new_lng, new_lat, tmp_next_date, tmp_next_delta_time)
                        output_str = '{},{},{},{},{},{},{}\n'.format(node_id, tmp_ * date_step, new_lng, new_lat,
                                                                tmp_next_date, tmp_next_delta_time,real_node_id)
                        interpolfile_obj.write(output_str)
                        this_node_loc.append((node_id, tmp_ * date_step, new_lng, new_lat, tmp_next_date, real_node_id))
                        _tmp_count_step = _tmp_count_step + 1
                        tmp_ = tmp_ + 1

                assert(next_true_time_index == tmp_)

                # print('\033[42m', node_id, next_true_time_index * date_step, next_lng, next_lat,
                #       tmp_next_date, tmp_next_delta_time, '\033[0m')
                output_str = '{},{},{},{},{},{},{}\n'.format(node_id, next_true_time_index * date_step, next_lng, next_lat,
                                                          tmp_next_date, tmp_next_delta_time,real_node_id)
                interpolfile_obj.write(output_str)
                this_node_loc.append((node_id, next_true_time_index * date_step,
                                      next_lng, next_lat, tmp_next_date, real_node_id))

            last_time_index = next_true_time_index + 1
            index = index + 1
            last_lng = next_lng
            last_lat = next_lat
            last_real_time = tmp_next_date
            last_real_delta_time = tmp_next_delta_time
        # print(last_time_index, total_time_step)
        # 最后有没有到达时刻
        while last_time_index <= (total_time_step-1):
            # print(node_id, last_time_index * date_step, last_lng, last_lat, last_real_time, last_real_delta_time)
            output_str = '{},{},{},{},{},{},{}\n'.format(node_id, last_time_index * date_step, last_lng, last_lat, last_real_time,
                                                      last_real_delta_time, real_node_id)
            interpolfile_obj.write(output_str)
            this_node_loc.append((node_id, last_time_index * date_step, last_lng, last_lat,last_real_time,
                                  last_real_delta_time, real_node_id))
            last_time_index = last_time_index + 1

        # print(num_lines)
        oldfile_obj.close()
        interpolfile_obj.close()
        assert (len(this_node_loc) == total_time_step)
        self.list_space_time_loc.append(this_node_loc)


    def get_distance(self, origin_lat, origin_lng, destination_lat, destination_lng):
        # ($origin_lng, $origin_lat, $destination_lng, $destination_lat, $decimal = 2) {
        # // 地球半径系数
        earth_radius = 6370.996
        PI = 3.1415926
        def deg2rad(input):
            output = input * PI/180.
            return output
        radLat1 = deg2rad(origin_lat)
        radLat2 = deg2rad(destination_lat)
        radLng1 = deg2rad(origin_lng)
        radLng2 = deg2rad(destination_lng)

        a = abs(radLat1 - radLat2)
        b = abs(radLng1 - radLng2)

        distance = 2 * math.asin(
            math.sqrt(
                math.pow(math.sin(a / 2), 2) + math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b / 2), 2)))
        distance = distance * earth_radius * 1000
        return distance

    def detect_encounter(self):
        for a_id in range(self.num_nodes):
            for b_id in range(a_id + 1, self.num_nodes, 1):
                # 获得位置
                a_s_t_loc = self.list_space_time_loc[a_id][self.sim_TimeNow]
                b_s_t_loc = self.list_space_time_loc[b_id][self.sim_TimeNow]

                if a_s_t_loc[1] != b_s_t_loc[1]:
                    print(a_id, a_id, self.sim_TimeNow)
                    print(a_s_t_loc)
                    print(b_s_t_loc)
                assert a_s_t_loc[1] == b_s_t_loc[1]
                dist = self.get_distance(a_s_t_loc[2], a_s_t_loc[3], b_s_t_loc[2], b_s_t_loc[3])
                if dist <= self.dist_comm:
                    if self.mt_linkstate[a_id][b_id] == 0:
                        assert (self.mt_linkstate[b_id][a_id] == 0)
                        # 节点a和节点b 状态从0到1 发生linkup事件
                        # 更新状态 交换 控制信息
                        self.mt_linkstate[a_id][b_id] = 1
                        self.mt_linkstate[b_id][a_id] = 1
                        self.link_event_list.append((a_id, b_id, self.sim_TimeNow,
                                                     a_s_t_loc[2], a_s_t_loc[3], b_s_t_loc[2], b_s_t_loc[3]))
                    elif self.mt_linkstate[a_id][b_id] == 1:
                        assert (self.mt_linkstate[b_id][a_id] == 1)
                    else:
                        assert (True)
                else:
                    if self.mt_linkstate[a_id][b_id] == 0:
                        assert (self.mt_linkstate[b_id][a_id] == 0)
                    elif self.mt_linkstate[a_id][b_id] == 1:
                        assert (self.mt_linkstate[b_id][a_id] == 1)
                        # 节点a和节点b 状态从1到0 发生linkup事件
                        self.mt_linkstate[a_id][b_id] = 0
                        self.mt_linkstate[b_id][a_id] = 0
                        # 找到已经记录到 encounter_list 的相遇事件
                        for tunple in self.link_event_list:
                            (x_id, y_id, time_linkup, x1,x2,y1,y2) = tunple
                            if ((a_id == x_id) and (b_id == y_id)):
                                self.encounter_hist_list.append((time_linkup, self.sim_TimeNow, x_id, y_id,
                                                                 x1,x2,y1,y2))
                                print('encoutner event:{}~{},{}-{}'.format(time_linkup,self.sim_TimeNow, x_id,y_id))
                                self.link_event_list.remove(tunple)
                                break
                    else:
                        assert (True)

    def close_each_link(self):
        for tunple in self.link_event_list:
            (x_id, y_id, time_linkup, x1,x2,y1,y2) = tunple
            # a_s_t_loc = self.list_space_time_loc[x_id][self.sim_TimeNow]
            # b_s_t_loc = self.list_space_time_loc[y_id][self.sim_TimeNow]
            assert ((self.mt_linkstate[x_id][y_id] == 1) and (self.mt_linkstate[y_id][x_id] == 1))
            self.mt_linkstate[x_id][y_id] = 0
            self.mt_linkstate[y_id][x_id] = 0
            self.encounter_hist_list.append((time_linkup, self.sim_TimeNow, x_id, y_id,x1,x2,y1,y2))
            # self.link_event_list.remove(tunple)
        # 确保没有漏项
        assert (np.sum(self.mt_linkstate) == 0)
        self.link_event_list.clear()

if __name__ == "__main__":
    the = EncoHistGenerator_Shanghai()

    # dis = the.get_distance(118.822178,31.88837, 118.825582,31.888516)
    # print(dis)
