# -*-coding:utf-8-*-
import numpy as np
import csv
import random
import math
from tool.reliefF import *
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# from fitness_evaluate import *
from federate.fitness_evaluate import *
from federate.Markov_select import *

def calculate_3nn_acc(Bi_DataX, Bi_DataY, select, base=5):
    KF = StratifiedKFold(n_splits=base, shuffle=True, random_state=2)
    sum_acc = 0
    for train_idx, test_idx in KF.split(Bi_DataX, Bi_DataY):
        train_x, test_x = Bi_DataX[train_idx], Bi_DataX[test_idx]
        train_y, test_y = Bi_DataY[train_idx], Bi_DataY[test_idx]

        sub_train_x, sub_test_x = train_x[:, select], test_x[:, select]
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(sub_train_x, train_y)
        pred_y = clf.predict(sub_test_x)
        sum_acc += (metrics.accuracy_score(pred_y, test_y))
    return (1 / base) * sum_acc

class FPSO_FS():
    def GetParameter(self, Ori_Data, featuresNum, samplesNum, labelCount, tag, B_Num,
                     selectTimes, savePath):
        self.Ori_Data = Ori_Data
        self.featuresNum = featuresNum
        self.samplesNum = samplesNum
        self.labelCount = labelCount
        self.tag = tag
        self.B_Num = B_Num
        self.selectTimes = selectTimes
        self.savePath = savePath
        self.B_Data = [[] for i in range(self.B_Num)]
        self.B_Train_X = [[] for i in range(self.B_Num)]
        self.B_Train_Y = [[] for i in range(self.B_Num)]
        self.B_Test_X = [[] for i in range(self.B_Num)]
        self.B_Test_Y = [[] for i in range(self.B_Num)]

        self.R_max = 4  # 联邦学习最大轮回次数
        self.T_max = 25  # 群体最大迭代次数
        self.N = 20  # 种群规模大小
        self.num_max = 8  # 若全局最优连续num_max次都不发生改变或满足最大迭代次数则终止
        self.features_select_weight = 0.3  # 选择特征个数的权重
        # self.B_X = np.zeros((self.B_Num, self.N, self.featuresNum))  # 每个B参与者中每个粒子的特征被选中概率，范围在[0,1]
        # self.B_Z = np.zeros((self.B_Num, self.N, self.featuresNum))  # 每个B参与者中每个粒子的特征被选中情况，1表示选中，0表示没选中
        # self.A_X = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优概率
        # self.A_Z = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优特征子集
        # self.B_Pbest_Xi = np.zeros((self.B_Num, self.N, self.featuresNum))
        # self.B_Pbest_Zi = np.zeros((self.B_Num, self.N, self.featuresNum))
        # self.B_Pbest_Value = np.zeros((self.B_Num, self.N))
        # self.B_Gbest_Xi = np.zeros((self.B_Num, self.featuresNum))
        # self.B_Gbest_Zi = np.zeros((self.B_Num, self.featuresNum))
        # self.B_Gbest_Value = np.zeros(self.B_Num)
        # self.General_Best_Value = 0.0  # 装 配策略后最优的评估值
        # self.General_Best_Idx = 0  # 通用最优特征子集来自哪个B参与者
        # self.General_Best_Xi = np.zeros(self.featuresNum)  # 通用最优特征子集的特征被选中概率
        # self.General_Best_Zi = np.zeros(self.featuresNum)  # 通用最优特征子集
        self.alpha = 0.5  # 初始化控制参数
        self.Nl = 10  # 最优解引导的初始化粒子个数

        # self.change_count = 0  # 粒子群更新过程中用于统计全局最优连续几代不变化

    def Divided(self):
        if self.tag == 'cluster':
            y = np.copy(self.Ori_Data[:, self.featuresNum])
            for label in range(self.labelCount):
                lebel_idx = np.argwhere(y == label)
                lebel_idx = lebel_idx.flatten()
                random.seed(42)
                random.shuffle(lebel_idx)
                split_size = len(lebel_idx) // self.B_Num
                for i in range(self.B_Num):
                    idx = lebel_idx[i * split_size:(i + 1) * split_size]
                    self.B_Data[i].extend(self.Ori_Data[idx])
        if self.tag == 'random':
            sample_idx = [i for i in range(self.samplesNum)]
            random.seed(42)
            random.shuffle(sample_idx)
            # sample_idx = np.random.permutation(self.samplesNum)
            split_size = len(sample_idx) // self.B_Num
            for i in range(self.B_Num):
                idx = sample_idx[i * split_size:(i + 1) * split_size]
                self.B_Data[i].extend(self.Ori_Data[idx])

        self.B_Data = np.array(self.B_Data)

    def Get_Train_and_Test(self):
        KF = StratifiedKFold(n_splits=5, random_state=None)
        for i in range(self.B_Num):
            sample = self.B_Data[i][:, 0:self.featuresNum]
            label = self.B_Data[i][:, self.featuresNum]
            for train_index, test_index in KF.split(sample, label):
                self.B_Train_X[i].append(sample[train_index])
                self.B_Test_X[i].append(sample[test_index])
                self.B_Train_Y[i].append(label[train_index])
                self.B_Test_Y[i].append(label[test_index])

    def Initialized_A_and_B(self):
        self.B_X = np.zeros((self.B_Num, self.N, self.featuresNum))  # 每个B参与者中每个粒子的特征被选中概率，范围在[0,1]
        self.B_Z = np.zeros((self.B_Num, self.N, self.featuresNum))  # 每个B参与者中每个粒子的特征被选中情况，1表示选中，0表示没选中
        self.A_X = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优概率
        self.A_Z = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优特征子集
        self.A_Xi = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优概率
        self.A_Zi = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优特征子集
        self.A_Value = np.zeros((self.B_Num, self.B_Num))  # A参与者获取每个B参与者的最优适应度值
        self.B_Pbest_Xi = np.zeros((self.B_Num, self.N, self.featuresNum))
        self.B_Pbest_Zi = np.zeros((self.B_Num, self.N, self.featuresNum))
        self.B_Pbest_Value = np.zeros((self.B_Num, self.N))
        self.B_Gbest_Xi = np.zeros((self.B_Num, self.featuresNum))
        self.B_Gbest_Zi = np.zeros((self.B_Num, self.featuresNum))
        self.B_Gbest_Value = np.zeros(self.B_Num)
        self.General_Best_Value = 0.0  # 装 配策略后最优的评估值
        self.General_Best_Idx = 0  # 通用最优特征子集来自哪个B参与者
        self.General_Best_Xi = np.zeros(self.featuresNum)  # 通用最优特征子集的特征被选中概率
        self.General_Best_Zi = np.zeros(self.featuresNum)  # 通用最优特征子集

        self.Group_rand = np.zeros((self.B_Num, self.N))

        self.B_Train_SU_fi_and_Y = np.zeros((self.B_Num, self.featuresNum))
        self.B_Train_SU_fi_and_fj = np.zeros((self.B_Num, self.featuresNum, self.featuresNum))
        self.B_Train_MI_fi_and_Y = np.zeros((self.B_Num, self.featuresNum))
        self.B_Train_MI_fi_and_fj = np.zeros((self.B_Num, self.featuresNum, self.featuresNum))
        self.B_Train_CMI_fi_and_fj = np.zeros((self.B_Num, self.featuresNum, self.featuresNum))
        # self.B_mean_SU = np.zeros(self.B_Num)

    def Initialize_Particle_Swarm(self):
        # 初始化粒子群
        # 随机生成[0,1]之间的小数作为特征被选中的概率

        for i in range(self.B_Num):
            bt = np.random.rand()
            for j in range(self.N):
                for k in range(self.featuresNum):
                    bt = 4 * bt * (1 - bt)
                    if bt == 0.25 or bt == 0.5 or bt == 0.75 or bt == 0:
                        bt = bt + 0.1 * np.random.rand()
                    elif bt == 1:
                        bt = bt - 0.1 * np.random.rand()
                    self.B_X[i][j][k] = bt
                    if bt > np.random.rand():
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0

                # 对随机数进行排序并取出前selectNum个最大值的索引
                # top_indices = np.argsort(rand_Xi)[-self.selectNum:]
                # # 创建标记数组，并将前selectNum个最大值的索引位置标记为1，其他位置标记为0
                # labels = np.zeros(self.featuresNum)
                # labels[top_indices] = 1

                # 生成随机小数 rand
                # rand = np.random.rand()
                # # 标记数组
                # labels = np.zeros(self.featuresNum)
                # labels[rand_Xi > rand] = 1
                # self.B_Z[i][j] = labels

    def Update_Particle(self, i, change_count):
        # 更新粒子被选中概率
        for j in range(self.N):
            # rand = np.random.rand()
            # a1 = math.floor(rand * self.N)
            # rand = np.random.rand()
            # a2 = math.floor(rand * self.N)
            for k in range(self.featuresNum):
                rand = np.random.rand()

                if rand < 0.6:

                    a = (self.B_Pbest_Xi[i][j][k] + self.B_Gbest_Xi[i][k]) / 2
                    b = ((self.B_Pbest_Xi[i][j][k] - a) ** 2 + (self.B_Gbest_Xi[i][k] - a) ** 2) / 2
                    nor = np.random.normal(a, b)
                    self.B_X[i][j][k] = nor
                else:
                    self.B_X[i][j][k] = self.B_Pbest_Xi[i][j][k]

        p_c = 0.2 / (1 + math.exp(5 - (change_count + 1)))
        for j in range(self.N):
            rand = np.random.rand()
            if p_c > rand:
                # rand = np.random.rand()
                # ss = math.floor(rand * self.N)
                temp_list = [k for k in range(self.N)]
                temp_list.remove(j)
                ss = int(random.choice(temp_list))
                U = math.floor(p_c * self.featuresNum)
                for k in range(U):
                    # rand = np.random.rand()
                    # aa1 = math.floor(rand * self.featuresNum)
                    aa1 = random.randint(0, self.featuresNum - 1)
                    self.B_X[i][j][aa1] = self.B_Pbest_Xi[i][ss][aa1] + np.random.randn()
            for k in range(self.featuresNum):
                if self.B_X[i][j][k] < 0:
                    self.B_X[i][j][k] = 0
                elif self.B_X[i][j][k] > 1:
                    self.B_X[i][j][k] = 1

                rand = np.random.rand()
                if self.B_X[i][j][k] > rand:
                    self.B_Z[i][j][k] = 1
                else:
                    self.B_Z[i][j][k] = 0

    def Update(self, train_time):
        for i in range(self.B_Num):
            count = 0
            change_count = 0
            while count < self.T_max:
                if change_count >= self.num_max:
                    break
                for j in range(self.N):
                    select = np.argwhere(self.B_Z[i][j] == 1)
                    select = select.flatten()
                    if len(select) == 0:
                        # 没有特征被选中
                        fitness_value = 0
                    else:
                        fitness_value = calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                          select)

                    if count == 0:
                        self.B_Pbest_Value[i][j] = fitness_value
                        self.B_Pbest_Xi[i][j] = self.B_X[i][j]
                        self.B_Pbest_Zi[i][j] = self.B_Z[i][j]
                    else:
                        # print(fitness_value, self.B_Pbest_Value[i][j])
                        # 为了防止基因降解现象，采用强化记忆策略更新粒子的Pbest
                        if fitness_value > self.B_Pbest_Value[i][j]:
                            self.B_Pbest_Value[i][j] = fitness_value
                            self.B_Pbest_Xi[i][j] = 0.5 * (self.B_X[i][j] + self.B_Z[i][j])
                            # self.B_Pbest_Xi[i][j] = self.X[i][j]
                            self.B_Pbest_Zi[i][j] = self.B_Z[i][j]
                        elif fitness_value == self.B_Pbest_Value[i][j] and np.sum(self.B_Pbest_Zi[i][j] == 1) > np.sum(
                                self.B_Z[i][j] == 1):
                            self.B_Pbest_Value[i][j] = fitness_value
                            self.B_Pbest_Xi[i][j] = 0.5 * (self.B_X[i][j] + self.B_Z[i][j])
                            # self.B_Pbest_Xi[i][j] = self.X[i][j]
                            self.B_Pbest_Zi[i][j] = self.B_Z[i][j]

                # print('\n')
                max_Pbest_Value = np.argmax(self.B_Pbest_Value[i])
                if count == 0:
                    self.B_Gbest_Value[i] = self.B_Pbest_Value[i][max_Pbest_Value]
                    self.B_Gbest_Xi[i] = self.B_Pbest_Xi[i][max_Pbest_Value]
                    self.B_Gbest_Zi[i] = self.B_Pbest_Zi[i][max_Pbest_Value]
                else:
                    # print(self.B_Pbest_Value[i][max_Pbest_Value], self.B_Gbest_Value[i])
                    if self.B_Pbest_Value[i][max_Pbest_Value] > self.B_Gbest_Value[i]:
                        self.B_Gbest_Value[i] = self.B_Pbest_Value[i][max_Pbest_Value]
                        self.B_Gbest_Xi[i] = self.B_Pbest_Xi[i][max_Pbest_Value]
                        self.B_Gbest_Zi[i] = self.B_Pbest_Zi[i][max_Pbest_Value]
                        change_count = 0
                    elif self.B_Pbest_Value[i][max_Pbest_Value] == self.B_Gbest_Value[i] and np.sum(
                            self.B_Gbest_Zi[i] == 1) > np.sum(self.B_Pbest_Zi[i][max_Pbest_Value] == 1):
                        self.B_Gbest_Value[i] = self.B_Pbest_Value[i][max_Pbest_Value]
                        self.B_Gbest_Xi[i] = self.B_Pbest_Xi[i][max_Pbest_Value]
                        self.B_Gbest_Zi[i] = self.B_Pbest_Zi[i][max_Pbest_Value]
                        change_count = 0
                    else:
                        change_count += 1

                self.Update_Particle(i, change_count)

                count += 1

            print(count, change_count)

    def Send_Between_B_And_A(self, train_time):
        # 每个B参与者将各自的全局最优特征子集以及适应度评估值发给A参与者
        for i in range(self.B_Num):
            self.A_Value[i][i] = self.B_Gbest_Value[i]
            self.A_Xi[i] = self.B_Gbest_Xi[i]
            self.A_Zi[i] = self.B_Gbest_Zi[i]

        # A参与者再将其他B参与者的最优发送回去，进行评估，并且B参与者将评估值发送给A参与者
        for i in range(self.B_Num):
            # 第i个B参与者评估来自其他B参与者的最优特征子集
            for j in range(self.B_Num):
                if i != j:
                    other_Gbest_Zi = self.A_Zi[j]
                    select = np.argwhere(other_Gbest_Zi == 1)
                    select = select.flatten()
                    fitness_value = calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                      select)
                    self.A_Value[i][j] = fitness_value

    def Assemble_Strategy_A(self, federate_count):
        # A参与者对所有适应度评估值根据装配策略进行集成
        each_Gbest_mean = np.mean(self.A_Value, axis=0)  # 每一列为该下标私有最优特征子集在每一个B参与者的适应度评估值
        max_Gbest_idx = np.argmax(each_Gbest_mean)
        if federate_count == 0:
            self.General_Best_Value = each_Gbest_mean[max_Gbest_idx]
            self.General_Best_Xi = self.A_Xi[max_Gbest_idx]
            self.General_Best_Zi = self.A_Zi[max_Gbest_idx]
            self.General_Best_Idx = max_Gbest_idx
        else:
            if each_Gbest_mean[max_Gbest_idx] > self.General_Best_Value:
                self.General_Best_Value = each_Gbest_mean[max_Gbest_idx]
                self.General_Best_Xi = self.A_Xi[max_Gbest_idx]
                self.General_Best_Zi = self.A_Zi[max_Gbest_idx]
                self.General_Best_Idx = max_Gbest_idx
        # self.General_Best_Value = each_Gbest_mean[max_Gbest_idx]
        # self.General_Best_Xi = self.A_Xi[max_Gbest_idx]
        # self.General_Best_Zi = self.A_Zi[max_Gbest_idx]
        # self.General_Best_Idx = max_Gbest_idx

        # 根据通用最优特征子集更新每个B参与者的粒子群
        for i in range(self.B_Num):
            if i == self.General_Best_Idx:
                for j in range(self.Nl):
                    for k in range(self.featuresNum):
                        if self.General_Best_Zi[k] == 0:
                            self.B_X[i][j][k] = (1 - self.alpha) * np.random.rand()
                        elif self.General_Best_Zi[k] == 1:
                            self.B_X[i][j][k] = (1 - self.alpha) * np.random.rand() + self.alpha
                        # self.B_X[i][j][k] = (1 - self.alpha) * np.random.rand() + self.alpha * self.General_Best_Xi[k]
                for j in range(self.Nl, self.N):
                    rand_Xi = np.random.rand(self.featuresNum)
                    self.B_X[i][j] = rand_Xi
            else:
                for j in range(self.Nl):
                    for k in range(self.featuresNum):
                        if self.General_Best_Zi[k] == 0 and self.B_Gbest_Zi[i][k] == 0:
                            self.B_X[i][j][k] = (1 - self.alpha) * np.random.rand()
                        elif self.General_Best_Zi[k] == 1 and self.B_Gbest_Zi[i][k] == 0:
                            self.B_X[i][j][k] = (1 - self.alpha) * np.random.rand() + self.alpha
                        elif self.General_Best_Zi[k] == 0 and self.B_Gbest_Zi[i][k] == 1:
                            self.B_X[i][j][k] = np.random.rand()
                        else:
                            self.B_X[i][j][k] = 0.4 * np.random.rand() + 0.6
                        # if self.General_Best_Zi[k] == 0 and self.B_Gbest_Zi[i][k] == 1:
                        #     self.B_X[i][j][k] = np.random.rand()
                        # else:
                        #     self.B_X[i][j][k] = (1 - self.alpha) * np.random.rand() + self.alpha * self.General_Best_Xi[k]
                for j in range(self.Nl, self.N):
                    rand_Xi = np.random.rand(self.featuresNum)
                    self.B_X[i][j] = rand_Xi
            for j in range(self.N):
                for k in range(self.featuresNum):
                    if self.B_X[i][j][k] > np.random.rand():
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0


    def FPSO_Train(self, Train_times=5):
        acc_no_A = np.zeros((Train_times, self.B_Num))
        acc_with_A = np.zeros((Train_times, self.B_Num))
        for train_time in range(Train_times):
            self.Initialized_A_and_B()
            # self.calculate_B_SU(train_time)
            self.Initialize_Particle_Swarm()
            self.Update(train_time)

            # 没有A参与者的每个私有最优特征子集进行训练
            for i in range(self.B_Num):
                final_select = np.argwhere(self.B_Gbest_Zi[i] == 1)
                final_select = final_select.flatten()
                acc = self.calculate_test_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                              self.B_Test_X[i][train_time], self.B_Test_Y[i][train_time], final_select)
                # train_X = self.B_Train_X[i][train_time][:, final_select]
                # test_X = self.B_Test_X[i][train_time][:, final_select]
                # clf = KNeighborsClassifier(n_neighbors=3)
                # clf.fit(train_X, self.B_Train_Y[i][train_time])
                # pred_Y = clf.predict(test_X)
                # acc = metrics.accuracy_score(pred_Y, self.B_Test_Y[i][train_time])
                acc_no_A[train_time][i] = acc
            # print(accs_no_A)

            for federate_count in range(self.R_max):
                # 将一轮迭代后的每个B参与者全局最优特征子集以及适应度评估值发给A参与者
                # A参与者再将其他B参与者的最优发送回去，进行评估
                self.new_Send_Between_B_And_A(train_time)
                # A参与者对所有适应度评估值根据装配策略进行集成，并更新
                self.new_Assemble_Strategy_A(federate_count)
                if federate_count != self.R_max - 1:
                    # 前面已经有一次了，最后一次只需进行上面两步
                    self.Update(train_time)

            for i in range(self.B_Num):
                final_select = np.argwhere(self.B_Gbest_Zi[i] == 1)
                final_select = final_select.flatten()
                acc = self.calculate_test_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                              self.B_Test_X[i][train_time], self.B_Test_Y[i][train_time], final_select)
                # train_X = self.B_Train_X[i][train_time][:, final_select]
                # test_X = self.B_Test_X[i][train_time][:, final_select]
                # clf = KNeighborsClassifier(n_neighbors=3)
                # clf.fit(train_X, self.B_Train_Y[i][train_time])
                # pred_Y = clf.predict(test_X)
                # acc = metrics.accuracy_score(pred_Y, self.B_Test_Y[i][train_time])
                acc_with_A[train_time][i] = acc

        # print('FPSO_noA', np.mean(acc_no_A, axis=0))
        # print('FPSO_withA', np.mean(acc_with_A, axis=0))
        mean_acc_no_A = np.mean(acc_no_A, axis=0)
        mean_acc_with_A = np.mean(acc_with_A, axis=0)

        return mean_acc_no_A, mean_acc_with_A


    def Algo(self):
        self.Divided()
        self.Get_Train_and_Test()

        # self.filter_Train()

        each_time_acc_no_A = np.zeros((self.selectTimes, self.B_Num))
        each_time_acc_with_A = np.zeros((self.selectTimes, self.B_Num))
        for select_time in range(self.selectTimes):
            print('FPSO_FS')
            each_time_acc_no_A[select_time], each_time_acc_with_A[select_time] = self.FPSO_Train()

            # 将结果保存
            with open(self.savePath + '/FPSO_acc_no_A.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(each_time_acc_no_A[select_time])
                f.close()
            with open(self.savePath + '/FPSO_acc_with_A.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(each_time_acc_with_A[select_time])
                f.close()
