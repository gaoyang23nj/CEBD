import os
import time
import datetime
import operator

EncoHistDir = '../EncoHistData_Seu/'
InputDir = './MAC_trail_timeDiff_new/'
# MovementDir = './MovementRecord'

# 把AC接入记录变成相遇记录
class EncoHistGenerator_Seu(object):
    def __init__(self):
        # 人员总个数
        num_of_person = 0
        self.list_personfile = []
        self.list_movement = []
        for root, dirs, files in os.walk(InputDir):
            for file in files:
                filename = os.path.join(root, file)
                self.list_personfile.append((filename, file, num_of_person))
                num_of_person = num_of_person + 1
        for index in range(len(self.list_personfile)):
            # 处理一个person的移动记录
            tmp_list_move_record = self.convert_movement(self.list_personfile[index])
            self.list_movement.append(tmp_list_move_record)
        # 检查相遇 并生成相遇记录
        self.list_encounter = []
        self.detect_encounter()
        # 以相遇起始时间升序排列
        self.list_encounter.sort(key=operator.itemgetter(4))
        self.print_ecounter()

    # 补充移动记录
    def convert_movement(self, person_file_tunple):
        personfile = person_file_tunple[0]
        print('[{}]\t{}'.format(person_file_tunple[2], personfile))
        file_object = open(personfile, 'r', encoding="utf-8")
        tmp_all_lines = file_object.readlines()
        list_record = []
        to_time = datetime.datetime.now()
        loc = -1
        for index in range(len(tmp_all_lines)):
            # print(tmp_all_lines[index])
            values = tmp_all_lines[index].split('\t')
            person_id = int(values[1])
            person_mac = values[2]
            new_from_time = datetime.datetime.strptime(values[3], "%Y-%m-%d %H:%M:%S")
            new_to_time = datetime.datetime.strptime(values[4], "%Y-%m-%d %H:%M:%S")
            new_loc = values[5]
            # 如果上一条记录的结尾time == 本次记录的开始time 且 位置一样 应该合并记录
            if index > 0:
                if loc == new_loc:
                    list_record[-1][3] = new_to_time
                else:
                    if to_time != new_from_time:
                        list_record[-1][3] = new_from_time
                    list_record.append([person_id, person_mac, new_from_time, new_to_time, new_loc])
            else:
                list_record.append([person_id, person_mac, new_from_time, new_to_time, new_loc])
            from_time = new_from_time
            to_time = new_to_time
            loc = new_loc
        file_object.close()
        return list_record

    def detect_encounter(self):
        filename = os.path.join(EncoHistDir, 'Seu_Encounter_preview.tmp')
        file_object = open(filename, 'a+', encoding="utf-8")
        # 过滤 第i个人和 第j个人 的相遇记录
        for i in range(len(self.list_movement)):
            print('{}'.format(i))
            for j in range(i+1, len(self.list_movement)):
                person_i = self.list_movement[i]
                person_j = self.list_movement[j]
                # 遍历 person_i 的 移动记录 和 person_j的移动记录，检测相遇
                for m in range(len(person_i)):
                    for n in range(len(person_j)):
                        i_move = person_i[m]
                        j_move = person_j[n]
                        # 若处在同一个位置，求两个时间段的交集
                        if i_move[4] == j_move[4]:
                            # 开始时间取最大, 结束时间取最小
                            tmp_begin = i_move[2]
                            if i_move[2] < j_move[2]:
                                tmp_begin = j_move[2]
                            tmp_end = i_move[3]
                            if i_move[3] > j_move[3]:
                                tmp_end = j_move[3]
                            if tmp_begin < tmp_end:
                                encounter = (i_move[0], i_move[1], j_move[0], j_move[1], tmp_begin, tmp_end)
                                self.list_encounter.append(encounter)
                                str = '{},{},{},{},{},{}\n'.format(i_move[0], i_move[1], j_move[0], j_move[1], tmp_begin,
                                                                 tmp_end)
                                file_object.write(str)
        file_object.close()
        print('detect encounter completed!')
        return

    def print_ecounter(self):
        filename = os.path.join(EncoHistDir, 'Seu_Encounter.tmp')
        file_object = open(filename, 'a+', encoding="utf-8")
        tmp_all_lines = file_object.readlines()
        for enc in self.list_encounter:
            (person_iid, person_imac, person_jid, person_jmac, time_begin,time_end) = enc
            str = '{},{},{},{},{},{}\n'.format(person_iid,person_imac,person_jid,person_jmac,time_begin,time_end)
            file_object.write(str)
        file_object.close()

if __name__ == "__main__":
    EncoHistGenerator_Seu = EncoHistGenerator_Seu()
