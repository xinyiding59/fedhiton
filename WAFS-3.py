# -*-coding:utf-8-*-
import numpy as np
import csv
import random
import math
from sklearn.neighbors import NearestNeighbors
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
        self.B_Data = [[] for i in range(self.B_Num)]
        self.B_Train_X = [[] for i in range(self.B_Num)]
        self.B_Train_Y = [[] for i in range(self.B_Num)]
        self.B_Test_X = [[] for i in range(self.B_Num)]
        self.B_Test_Y = [[] for i in range(self.B_Num)]

        self.T_max = 100  # 群体最大迭代次数
        self.N = 20  # 种群规模大小
        self.num_max = 8  # 若全局最优连续num_max次都不发生改变或满足最大迭代次数则终止
        self.features_select_weight = 0.3  # 选择特征个数的权重
        self.alpha = 0.5  # 初始化控制参数
        self.Nl = 10  # 最优解引导的初始化粒子个数



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
        self.B_Z = np.zeros((self.B_Num, self.N, self.featuresNum))  # 每个B参与者中每个粒子的特征被选中情况，1表示选中，0表示没选中
        self.B_Value = np.zeros((self.B_Num, self.N))  # 每个B参与者中每个粒子的特征被选中概率，范围在[0,1]
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
        self.clusters = np.zeros((self.B_Num, self.N))
        self.centers = np.zeros((self.B_Num, 2))



    def Initialize_Particle_Swarm(self,train_time):
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
                    if bt > 0.6:
                        self.B_Z[i][j][k] = 1
                    else:
                        self.B_Z[i][j][k] = 0
                select = np.argwhere(self.B_Z[i][j] == 1)
                select = select.flatten()
                if len(select) == 0:
                    # 没有特征被选中
                    self.B_Value[i][j] = 0
                else:
                    self.B_Value[i][j] = calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                      select)




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
                if self.B_X[i][j][k] > 0.6:
                    self.B_Z[i][j][k] = 1
                else:
                    self.B_Z[i][j][k] = 0

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
            count = 0
            while count < self.T_max:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(self.B_X[i])
                labels = kmeans.labels_
                cluster_sizes = np.bincount(labels)
                p_0 = cluster_sizes[0] / cluster_sizes[0] + cluster_sizes[1]
                p_1 = cluster_sizes[1] / cluster_sizes[0] + cluster_sizes[1]
                cluster_centers = []
                cluster_centers_fit = np.zeros(2)
                best_individual_idx = np.zeros(2)
                cluster_centers_fit_best = 0
                idx_best = 0
                random_integer = random.randint(0, 1)
                for j in range(2):  # 遍历每个簇
                    cluster_indices = np.where(labels == j)[0]
                    cluster_fitness_scores = [self.B_Value[i][idx] for idx in cluster_indices]

                    best_individual_idx[j] = cluster_indices[np.argmax(cluster_fitness_scores)]

                    idx = best_individual_idx[j]
                    idx = int(idx)

                    cluster_centers.append(self.B_X[i][idx])
                    cluster_centers_fit[j] = self.B_Value[i][idx]
                if random.random() <0.2:
                    idx = best_individual_idx[random_integer]
                    idx = int(idx)
                    self.B_X[i][idx] = np.random.rand(self.featuresNum)
                    for k in range(self.featuresNum):
                        if self.B_X[i][idx][k] < 0:
                            self.B_X[i][idx][k] = 0
                        elif self.B_X[i][idx][k] > 1:
                            self.B_X[i][idx][k] = 1
                        if self.B_X[i][idx][k] > 0.6:
                            self.B_Z[i][idx][k] = 1
                        else:
                            self.B_Z[i][idx][k] = 0
                    select = np.argwhere(self.B_Z[i][idx] == 1)
                    select = select.flatten()
                    if len(select) == 0:
                        # 没有特征被选中
                        self.B_Value[i][idx] = 0
                    else:

                        self.B_Value[i][idx] = calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                          select)

                for j in range(self.N):
                    idx_0 = best_individual_idx[0]
                    idx_0 = int(idx_0)
                    idx_1 = best_individual_idx[1]
                    idx_1 = int(idx_1)
                    x_new = np.zeros(self.featuresNum)
                    z_new = np.zeros(self.featuresNum)
                    r = random.random()
                    if r < 0.8:
                        #Local optimal strategy
                        if r < 0.4:
                            for k in range(2):
                                # 找出属于簇j的所有索引
                                cluster_indices = np.where(labels == k)[0]

                                # 检查cluster_indices是否包含索引k
                                if k in cluster_indices:
                                    x_new = self.B_X[i][j] + (self.B_X[i][idx_0]-self.B_X[i][j]) * np.random.normal(0, 1)
                                else:
                                    x_new = self.B_X[i][j] + (self.B_X[i][idx_1]-self.B_X[i][j]) * np.random.normal(0, 1)
                                break
                        #Nearest neighbor strategy:
                        else:
                            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(self.B_X[i])
                            # 找到每个粒子的最近邻
                            distances, indices = nbrs.kneighbors(self.B_X[i])
                            iddx =indices[j][1]
                            print(iddx)
                            x_near = self.B_X[i][iddx]
                            x_new = self.B_X[i][j] + (x_near - self.B_X[i][j]) * np.random.normal(
                                0, 1)
                    # Global optimal strategy
                    else:
                        if self.B_Value[i][idx_0] > self.B_Value[i][idx_1]:
                            x_new = self.B_X[i][j] + (self.B_X[i][idx_0] - self.B_X[i][j]) * np.random.normal(0, 1)
                        else:
                            x_new = self.B_X[i][j] + (self.B_X[i][idx_1] - self.B_X[i][j]) * np.random.normal(
                                0, 1)
                    for k in range(self.featuresNum):
                        if x_new[k] < 0:
                            x_new[k] = 0
                        elif x_new[k] > 1:
                            x_new[k] = 1

                        if x_new[k] > 0.6:
                            z_new[k] = 1
                        else:
                            z_new[k] = 0



                    select = np.argwhere(z_new == 1)
                    select = select.flatten()
                    if len(select) == 0:
                        # 没有特征被选中
                        fitness_value = 0
                    else:

                        fitness_value = calculate_3nn_acc(self.B_Train_X[i][train_time], self.B_Train_Y[i][train_time],
                                                          select)

                    if fitness_value > self.B_Pbest_Value[i][j]:
                        self.B_Value[i][j] = fitness_value
                        self.B_X[i][j] = x_new
                        self.B_Z[i][j] = z_new


                # print('\n')
                max_Pbest_Value = np.argmax(self.B_Value[i])
                if count == 0:
                    self.B_Gbest_Value[i] = self.B_Value[i][max_Pbest_Value]
                    self.B_Gbest_Xi[i] = self.B_X[i][max_Pbest_Value]
                    self.B_Gbest_Zi[i] = self.B_Z[i][max_Pbest_Value]
                else:
                    # print(self.B_Pbest_Value[i][max_Pbest_Value], self.B_Gbest_Value[i])
                    if self.B_Pbest_Value[i][max_Pbest_Value] > self.B_Gbest_Value[i]:
                        self.B_Gbest_Value[i] = self.B_Value[i][max_Pbest_Value]
                        self.B_Gbest_Xi[i] = self.B_X[i][max_Pbest_Value]
                        self.B_Gbest_Zi[i] = self.B_Z[i][max_Pbest_Value]





                count += 1








    def FPSO_Train(self, Train_times=5):
        acc_no_A = np.zeros((Train_times, self.B_Num))
        for train_time in range(Train_times):
            self.Initialized_A_and_B()
            self.Initialize_Particle_Swarm(train_time)
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