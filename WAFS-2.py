# -*-coding:utf-8-*-
import numpy as np
import csv
import random
import math
# from tool.reliefF import *
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
# from fitness_evaluate import *
# from federate.fitness_evaluate import *
# from federate.Markov_select import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def roulette_wheel_selection(probabilities):
    cumulative_prob = 0
    random_num = random.random()  # 生成一个随机数，范围在[0, 1)

    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if random_num < cumulative_prob:
            return i

    # 如果累积概率没有超过随机数，则默认选择最后一个
    return len(probabilities) - 1
class FPSO_FS():
    def GetParameter(self, Ori_Data, featuresNum, samplesNum, B_Num):
        self.Ori_Data = Ori_Data
        self.featuresNum = featuresNum
        self.samplesNum = samplesNum
        # self.labelCount = labelCount
        # self.tag = tag
        self.B_Num = B_Num

        # self.selectTimes = selectTimes
        # self.savePath = savePath
        self.straNum_SaPSOLSFS = 5


        ps = 20  # 假设这个值为 3，你可以根据实际情况修改

        # 使用 tile 函数进行复制
        # vSSP_SaPSOLSFS = np.tile(SSP_SaPSOLSFS, (ps, 1))




        self.B_Data = [[] for i in range(self.B_Num)]
        self.B_Train_X = [[] for i in range(self.B_Num)]
        self.B_Train_Y = [[] for i in range(self.B_Num)]
        self.B_Test_X = [[] for i in range(self.B_Num)]
        self.B_Test_Y = [[] for i in range(self.B_Num)]



        self.R_max = 4  # 联邦学习最大轮回次数
        self.T_max = 100  # 群体最大迭代次数
        self.N = 20  # 种群规模大小
        self.num_max = 8  # 若全局最优连续num_max次都不发生改变或满足最大迭代次数则终止
        self.features_select_weight = 0.3  # 选择特征个数的权重
        self.alpha = 0.5  # 初始化控制参数
        self.Nl = 10  # 最优解引导的初始化粒子个数
        self.count = 0
        self.flagiter = 0

#分数据是均匀分布还是聚类分布
    def Divided(self):
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
        self.B_V = np.zeros((self.B_Num, self.N, self.featuresNum))
        self.B_Z = np.zeros((self.B_Num, self.N, self.featuresNum))  # 每个B参与者中每个粒子的特征被选中情况，1表示选中，0表示没选中
        self.B_Value= np.zeros((self.B_Num, self.N))
        self.A_X = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优概率
        self.A_Z = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优特征子集
        self.A_Xi = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优概率
        self.A_Zi = np.zeros((self.B_Num, self.featuresNum))  # A参与者获取每个B参与者的最优特征子集
        self.A_Value = np.zeros((self.B_Num, self.B_Num))  # A参与者获取每个B参与者的最优适应度值
        self.B_Pbest_Xi = np.zeros((self.B_Num, self.N, self.featuresNum))
        self.B_Pbest_Vi = np.zeros((self.B_Num, self.N, self.featuresNum))
        self.B_Pbest_Zi = np.zeros((self.B_Num, self.N, self.featuresNum))
        self.B_Pbest_Value = np.zeros((self.B_Num, self.N))
        self.B_Gbest_Xi = np.zeros((self.B_Num, self.featuresNum))
        self.B_Gbest_Vi = np.zeros((self.B_Num, self.featuresNum))
        self.B_Gbest_Zi = np.zeros((self.B_Num, self.featuresNum))
        self.B_Gbest_Value = np.zeros(self.B_Num)
        self.General_Best_Value = 0.0  # 装 配策略后最优的评估值
        self.General_Best_Idx = 0  # 通用最优特征子集来自哪个B参与者
        self.General_Best_Xi = np.zeros(self.featuresNum)  # 通用最优特征子集的特征被选中概率
        self.General_Best_Zi = np.zeros(self.featuresNum)  # 通用最优特征子集

        self.Group_rand = np.zeros((self.B_Num, self.N))

        self.SSP_SaPSOLSFS = np.zeros(self.straNum_SaPSOLSFS)
        self.nsFlagSaPSOLSFS = np.zeros((self.N, self.straNum_SaPSOLSFS))
        self.nfFlagSaPSOLSFS = np.zeros((self.N, self.straNum_SaPSOLSFS))
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
                    self.B_V[i][j][k] = bt-0.5


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
                    if bt > 0.6:
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

    def Update_Particle(self, i,train_time):
        # 更新粒子被选中概率

        for j in range(self.N):
            selected_index = roulette_wheel_selection(self.SSP_SaPSOLSFS)
            if selected_index == 0:
                w = 0.7298
                c1 = 1.49618
                c2 = 1.49618
                r1 = random.random()
                r2 = random.random()

                self.B_V[i][j] = (w * self.B_X[i][j] +
                                c1 * r1 * (self.B_Pbest_Xi[i][j] - self.B_X[i][j]) +
                                c2 * r2 * (self.B_Gbest_Xi[i] - self.B_X[i][j]))

                #保证速度在-0.5 到0.5
                for k in range(self.featuresNum):
                    if self.B_V[i][j][k] < -0.5:
                        self.B_V[i][j][k] = -0.5
                    elif self.B_V[i][j][k] > 0.5:
                        self.B_V[i][j][k] = 0.5


                self.B_X[i][j] = self.B_X[i][j] + self.B_V[i][j]

                number_size = np.sum(self.B_Z[i][j] == 1)
                for k in range(self.featuresNum):
                    if self.B_X[i][j][k] < 0:
                        self.B_X[i][j][k] = 0
                    elif self.B_X[i][j][k] > 1:
                        self.B_X[i][j][k] = 1

                    if self.B_X[i][j][k] > 0.6:
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0

                select = np.argwhere(self.B_Z[i][j] == 1)
                select = select.flatten()
                if len(select) == 0:
                    # 没有特征被选中
                    self.nfFlagSaPSOLSFS[j][0] = 1
                else:
                    if self.B_Value[i][j] > calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    elif self.B_Value[i][j] == calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select) and number_size > np.sum(self.B_Z[i][j] == 1):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    else:
                        self.nfFlagSaPSOLSFS[j][0] = 1



                print("选项 0 被选中")
            elif selected_index == 1:

                r4 = np.random.permutation(self.N)
                a = r4[0]
                b = r4[1]

                for k in range(self.featuresNum):
                    self.B_X[i][j][k] = (random.random() * self.B_X[i][j][k]
                                         + random.random() * self.B_Pbest_Xi[i][j][k] +
                                         random.random() * (self.B_X[i][a][k] - self.B_X[i][b][k]))



                number_size = np.sum(self.B_Z[i][j] == 1)

                for k in range(self.featuresNum):
                    if self.B_X[i][j][k] < 0:
                        self.B_X[i][j][k] = 0
                    elif self.B_X[i][j][k] > 1:
                        self.B_X[i][j][k] = 1

                    if self.B_X[i][j][k] > 0.6:
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0

                select = np.argwhere(self.B_Z[i][j] == 1)
                select = select.flatten()
                if len(select) == 0:
                    # 没有特征被选中
                    self.nfFlagSaPSOLSFS[j][0] = 1
                else:
                    if self.B_Value[i][j] > calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    elif self.B_Value[i][j] == calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select) and number_size > np.sum(self.B_Z[i][j] == 1):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    else:
                        self.nfFlagSaPSOLSFS[j][0] = 1



                print("选项 1 被选中")
            elif selected_index == 2:

                means_X = np.mean(self.B_X[i], axis=0)

                r4 = np.random.permutation(self.N)
                a = r4[0]
                r1 = np.random.standard_cauchy()
                r2 = np.random.normal(0, 1)
                c = (self.featuresNum-1) * r2 / self.featuresNum + r1/self.featuresNum

                self.B_V[i][j] = (means_X - self.B_X[i][j]) + c / np.sqrt(3) * np.sqrt(
                    (self.B_Pbest_Xi[i][j] - means_X) ** 2 +
                    (self.B_X[i][j] - means_X) ** 2 + (self.B_X[i][a] - means_X) ** 2)



                #保证速度在-0.5 到0.5
                for k in range(self.featuresNum):
                    if self.B_V[i][j][k] < -0.5:
                        self.B_V[i][j][k] = -0.5
                    elif self.B_V[i][j][k] > 0.5:
                        self.B_V[i][j][k] = 0.5


                self.B_X[i][j] = self.B_X[i][j] + self.B_V[i][j]
                number_size = np.sum(self.B_Z[i][j] == 1)
                for k in range(self.featuresNum):
                    if self.B_X[i][j][k] < 0:
                        self.B_X[i][j][k] = 0
                    elif self.B_X[i][j][k] > 1:
                        self.B_X[i][j][k] = 1

                    if self.B_X[i][j][k] > 0.6:
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0

                select = np.argwhere(self.B_Z[i][j] == 1)
                select = select.flatten()
                if len(select) == 0:
                    # 没有特征被选中
                    self.nfFlagSaPSOLSFS[j][0] = 1
                else:
                    if self.B_Value[i][j] > calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    elif self.B_Value[i][j] == calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select) and number_size < np.sum(self.B_Z[i][j] == 1):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    else:
                        self.nfFlagSaPSOLSFS[j][0] = 1


                print("选项 2 被选中")
            elif selected_index == 3:
                w = 0.9 - 0.5 * self.count / 100
                c = 1.49445
                r1 = random.random()
                r2 = random.random()
                for k in range(self.featuresNum):
                    r4 = np.random.permutation(self.N)
                    a = r4[0]
                    self.B_V[i][j][k] = w * self.B_V[i][j][k] + c * random.random() * (self.B_Pbest_Xi[i][j][k] -
                                                                                       self.B_X[i][j][k])


                #保证速度在-0.5 到0.5
                for k in range(self.featuresNum):
                    if self.B_V[i][j][k] < -0.5:
                        self.B_V[i][j][k] = -0.5
                    elif self.B_V[i][j][k] > 0.5:
                        self.B_V[i][j][k] = 0.5


                self.B_X[i][j] = self.B_X[i][j] + self.B_V[i][j]
                number_size = np.sum(self.B_Z[i][j] == 1)
                for k in range(self.featuresNum):
                    if self.B_X[i][j][k] < 0:
                        self.B_X[i][j][k] = 0
                    elif self.B_X[i][j][k] > 1:
                        self.B_X[i][j][k] = 1

                    if self.B_X[i][j][k] > 0.6:
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0

                select = np.argwhere(self.B_Z[i][j] == 1)
                select = select.flatten()
                if len(select) == 0:
                    # 没有特征被选中
                    self.nfFlagSaPSOLSFS[j][0] = 1
                else:
                    if self.B_Value[i][j] > calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    elif self.B_Value[i][j] == calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select) and number_size < np.sum(self.B_Z[i][j] == 1):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    else:
                        self.nfFlagSaPSOLSFS[j][0] = 1



                print("选项 3 被选中")
            else:
                w = 0.9 - 0.5 * self.count / 100
                c = 1.49445
                for k in range(self.featuresNum):
                    r4 = np.random.permutation(self.N)
                    a = r4[0]
                    self.B_V[i][j][k] = w * self.B_V[i][j][k] + c * random.random() * (self.B_Pbest_Xi[i][j][k] -
                                                 self.B_X[i][j][k] + self.B_Gbest_Xi[i][k] - self.B_X[i][j][k])


                #保证速度在-0.5 到0.5
                for k in range(self.featuresNum):
                    if self.B_V[i][j][k] < -0.5:
                        self.B_V[i][j][k] = -0.5
                    elif self.B_V[i][j][k] > 0.5:
                        self.B_V[i][j][k] = 0.5


                self.B_X[i][j] = self.B_X[i][j] + self.B_V[i][j]
                number_size = np.sum(self.B_Z[i][j] == 1)
                for k in range(self.featuresNum):
                    if self.B_X[i][j][k] < 0:
                        self.B_X[i][j][k] = 0
                    elif self.B_X[i][j][k] > 1:
                        self.B_X[i][j][k] = 1

                    if self.B_X[i][j][k] > 0.6:
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0

                select = np.argwhere(self.B_Z[i][j] == 1)
                select = select.flatten()
                if len(select) == 0:
                    # 没有特征被选中
                    self.nfFlagSaPSOLSFS[j][0] = 1
                else:
                    if self.B_Value[i][j] > calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    elif self.B_Value[i][j] == calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                           select) and number_size < np.sum(self.B_Z[i][j] == 1):
                        self.nsFlagSaPSOLSFS[j][0] = 1
                    else:
                        self.nfFlagSaPSOLSFS[j][0] = 1





                print("选项 4 被选中")

        self.count = self.count + 1
        k = self.count - self.flagiter - 1

        self.SLPQ[k, :] = np.sum(self.nsFlagSaPSOLSFS, axis=0)
        self.FLPQ[k, :] = np.sum(self.nfFlagSaPSOLSFS, axis=0)

        self.nsFlagSaPSOLSFS = np.zeros((self.N, self.straNum_SaPSOLSFS))
        self.nfFlagSaPSOLSFS = np.zeros((self.N, self.straNum_SaPSOLSFS))

        if (self.count - self.flagiter) == 10:
            self.flagiter = self.count
            S3 = np.zeros(5)
            for j in range(5):
                S1 = np.sum(self.SLPQ[:, j])
                if S1 == 0:
                    S2 = 0.000000001
                else:
                    S2 = S1
                S3[j] = S1/(S2 + np.sum(self.FLPQ[:, j]))
            for j in range(5):
                self.SSP_SaPSOLSFS[j] = S3[j] / np.sum(S3)



            self.SLPQ = np.zeros((10, 5))
            self.FLPQ = np.zeros((10, 5))




    def calculate_test_acc(self, train_X, train_Y, test_X, test_Y, final_select):
        sub_train_X = train_X[:, final_select]
        sub_test_X = test_X[:, final_select]
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(sub_train_X, train_Y)
        pred_Y = clf.predict(sub_test_X)
        acc = metrics.accuracy_score(pred_Y, test_Y)
        # acc = kmean_res_test(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time], self.B_Test_X[i][train_time], self.B_Test_Y[i][train_time], final_select)
        return acc
    def Update(self, train_time):
        for i in range(self.B_Num):
            self.count = 0
            self.flagiter = 0
            for j in range(5):
                self.SSP_SaPSOLSFS[j] = 1 / 5
            self.SLPQ = np.zeros((10, 5))
            self.FLPQ = np.zeros((10, 5))
            while self.count < self.T_max:

                for j in range(self.N):
                    select = np.argwhere(self.B_Z[i][j] == 1)
                    select = select.flatten()
                    if len(select) == 0:
                        # 没有特征被选中
                        self.B_Value[i][j] = 0
                    else:
                        self.B_Value[i][j] = calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                          select)

                    if self.count == 0:
                        self.B_Pbest_Value[i][j] = self.B_Value[i][j]
                        self.B_Pbest_Xi[i][j] = self.B_X[i][j]
                        self.B_Pbest_Zi[i][j] = self.B_Z[i][j]
                        self.B_Pbest_Vi[i][j] = self.B_V[i][j]
                    else:
                        # print(fitness_value, self.B_Pbest_Value[i][j])
                        # 为了防止基因降解现象，采用强化记忆策略更新粒子的Pbest
                        if self.B_Value[i][j] > self.B_Pbest_Value[i][j]:
                            self.B_Pbest_Value[i][j] = self.B_Value[i][j]
                            self.B_Pbest_Xi[i][j] = self.B_X[i][j]
                            self.B_Pbest_Zi[i][j] = self.B_Z[i][j]
                            self.B_Pbest_Vi[i][j] = self.B_V[i][j]
                        elif self.B_Value[i][j] == self.B_Pbest_Value[i][j] and np.sum(self.B_Pbest_Zi[i][j] == 1) > np.sum(
                                self.B_Z[i][j] == 1):
                            self.B_Pbest_Value[i][j] = self.B_Value[i][j]
                            self.B_Pbest_Xi[i][j] = self.B_X[i][j]
                            self.B_Pbest_Zi[i][j] = self.B_Z[i][j]
                            self.B_Pbest_Vi[i][j] = self.B_V[i][j]

                # print('\n')
                max_Pbest_Value = np.argmax(self.B_Pbest_Value[i])
                if self.count == 0:
                    self.B_Gbest_Value[i] = self.B_Pbest_Value[i][max_Pbest_Value]
                    self.B_Gbest_Xi[i] = self.B_Pbest_Xi[i][max_Pbest_Value]
                    self.B_Gbest_Zi[i] = self.B_Pbest_Zi[i][max_Pbest_Value]
                    self.B_Gbest_Vi[i] = self.B_Pbest_Vi[i][max_Pbest_Value]
                else:
                    # print(self.B_Pbest_Value[i][max_Pbest_Value], self.B_Gbest_Value[i])
                    if self.B_Pbest_Value[i][max_Pbest_Value] > self.B_Gbest_Value[i]:
                        self.B_Gbest_Value[i] = self.B_Pbest_Value[i][max_Pbest_Value]
                        self.B_Gbest_Xi[i] = self.B_Pbest_Xi[i][max_Pbest_Value]
                        self.B_Gbest_Zi[i] = self.B_Pbest_Zi[i][max_Pbest_Value]
                        self.B_Gbest_Vi[i] = self.B_Pbest_Vi[i][max_Pbest_Value]
                    elif self.B_Pbest_Value[i][max_Pbest_Value] == self.B_Gbest_Value[i] and np.sum(
                            self.B_Gbest_Zi[i] == 1) > np.sum(self.B_Pbest_Zi[i][max_Pbest_Value] == 1):
                        self.B_Gbest_Value[i] = self.B_Pbest_Value[i][max_Pbest_Value]
                        self.B_Gbest_Xi[i] = self.B_Pbest_Xi[i][max_Pbest_Value]
                        self.B_Gbest_Zi[i] = self.B_Pbest_Zi[i][max_Pbest_Value]
                        self.B_Gbest_Vi[i] = self.B_Pbest_Vi[i][max_Pbest_Value]


                self.Update_Particle(i,train_time)

                # self.count = self.count + 1
                # change_count = count % 10
                # if change_count == 0:
                #
                #     SLPQ = np.zeros((10, 5))
                #     FLPQ = np.zeros((10, 5))

            # print(count, change_count)

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
        for train_time in range(Train_times):
            self.Initialized_A_and_B()
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
                # acc = metrics.accuracy_score(p red_Y, self.B_Test_Y[i][train_time])
                acc_no_A[train_time][i] = acc
                # print(acc)
            # print(accs_no_A)

        #     for federate_count in range(self.R_max):
        #         # 将一轮迭代后的每个B参与者全局最优特征子集以及适应度评估值发给A参与者
        #         # A参与者再将其他B参与者的最优发送回去，进行评估
        #         self.Send_Between_B_And_A(train_time)
        #         # A参与者对所有适应度评估值根据装配策略进行集成，并更新
        #         self.Assemble_Strategy_A(federate_count)
        #         if federate_count != self.R_max - 1:
        #             # 前面已经有一次了，最后一次只需进行上面两步
        #             self.Update(train_time)
        #
        #     for i in range(self.B_Num):
        #         final_select = np.argwhere(self.General_Best_Zi == 1)
        #         final_select = final_select.flatten()
        #         acc = self.calculate_test_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
        #                                       self.B_Test_X[i][train_time], self.B_Test_Y[i][train_time], final_select)
        #         # train_X = self.B_Train_X[i][train_time][:, final_select]
        #         # test_X = self.B_Test_X[i][train_time][:, final_select]
        #         # clf = KNeighborsClassifier(n_neighbors=3)
        #         # clf.fit(train_X, self.B_Train_Y[i][train_time])
        #         # pred_Y = clf.predict(test_X)
        #         # acc = metrics.accuracy_score(pred_Y, self.B_Test_Y[i][train_time])
        #         acc_with_A[train_time][i] = acc
        #
        # # print('FPSO_noA', np.mean(acc_no_A, axis=0))
        # # print('FPSO_withA', np.mean(acc_with_A, axis=0))
        mean_acc_no_A = np.mean(acc_no_A, axis=0)
        # mean_acc_with_A = np.mean(acc_with_A, axis=0)

        return mean_acc_no_A

#保存数据结果的
    def Algo(self):
        self.Divided()
        self.Get_Train_and_Test()

        print(self.FPSO_Train())
        # each_time_acc_no_A = np.zeros((self.selectTimes, self.B_Num))
        # each_time_acc_with_A = np.zeros((self.selectTimes, self.B_Num))
        # for select_time in range(self.selectTimes):
        #     print('FPSO_FS')
        #     each_time_acc_no_A[select_time], each_time_acc_with_A[select_time] = self.FPSO_Train()
        #
        #     # 将结果保存
        #     with open(self.savePath + '/FPSO_acc_no_A.csv', 'a', newline='', encoding='utf-8') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(each_time_acc_no_A[select_time])
        #         f.close()
        #     with open(self.savePath + '/FPSO_acc_with_A.csv', 'a', newline='', encoding='utf-8') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(each_time_acc_with_A[select_time])
        #         f.close()

# wine = load_wine()
# X, y = wine.data, wine.target
# est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform', subsample=None)
file_path = r'C:\Users\36978\PycharmProjects\mystudy\datasets\Wine.csv'
data = pd.read_csv(file_path, header=None)  # 告诉pandas没有列名

# 将数据标签分离给X和y
X = data.iloc[:, :-1].values  # 假设标签在最后一列
y = data.iloc[:, -1].values   # 假设标签在最后一列
# 对数据进行离散化处理
# X_discretized = est.fit_transform(X)
min_max_scaler = MinMaxScaler()
# 使用 MinMaxScaler 对特征变量(X)进行归一化处理
X_normalized_minmax = min_max_scaler.fit_transform(X)


featuresNum = X.shape[1]
samplesNum = X.shape[0]
Dis_Data = np.column_stack((X_normalized_minmax, y))
B_num = 3
func = FPSO_FS()
func.GetParameter(Ori_Data=Dis_Data, featuresNum=featuresNum, samplesNum=samplesNum, B_Num=B_num)
func.Algo()

# # 更新粒子被选中概率
# for j in range(self.N):
#     rand = np.random.rand()
#     a1 = math.floor(rand * self.N)
#     rand = np.random.rand()
#     a2 = math.floor(rand * self.N)
#     for k in range(self.featuresNum):
#         c1 = np.random.rand()
#         c2 = 1 - c1
#         if rand < 0.6:
#             dlt = np.random.rand() * np.abs(self.B_X[i][a1][k] - self.B_X[i][a2][k]) * np.exp(
#                 self.B_Pbest_Value[i][j] - self.B_Gbest_Value[i])
#             a = (c1 * self.B_Pbest_Xi[i][j][k] + c2 * self.B_Gbest_Xi[i][k])
#             b = np.abs(self.B_Pbest_Xi[i][j][k] + self.B_Gbest_Xi[i][k]) + dlt
#             nor = a + b * np.random.randn()
#             self.B_X[i][j][k] = nor
#         else:
#             self.B_X[i][j][k] = self.B_Pbest_Xi[i][j][k]