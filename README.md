# Simulation_ONE
node移动与界面定时刷新
传输速度 0.1s 传输量 linkdown事件

v4.5
窗口关闭 则仿真停止

2019-07-17
下一步 适配Hel模型

2019-08-19
下一步 EPRouting & Balckhole RW模型
重复实验 保存 测试结果
读取各自的属性特征

2019-08-20
下一步 
1.简化处理流程；(1.传输tran 连续传输ctran; 2.计算传输序列)
2.查看 随时间变化 性能指标何时稳定？

2019-12-02
1.目前仿真过程太慢
准备：以两个main文件为核心(1)得到EncounterHist
(2)根据EncounterHist进行仿真

2020-02-03
balckhole_detect_and_ban 似乎很有效果
结果在2020-02-03.txt

2020-07-14
benchmark:1)MDS(1_hour);
combine训练模型:所有属性都作为(Nerual Network)NN的输入来训练; bk和gk都不错
time训练模型:小NN,得到p,求平均值P,检测认为的bk节点会复制一份保留在本地
refuse_all:小NN,得到p,求平均值P(p_x,98个去掉n_wch和n_un;p,本地观察),认为是bk就不交流

2020-07-24
collusion:设定0-0.1之间
<0.5部分的leave-one std, std的下降差值作为判断标准0.01, 考虑good-mouth共谋
莫非collusion的结果确实不好提升？--Eric



于社群的Routing  进行CTMC分析是个点？

带能源的Routing 进行CTMC分析是个点？