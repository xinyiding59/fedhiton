import numpy as np
import pandas as pd
from scipy.stats import chi2
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from collections import defaultdict
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsClassifier  # 修改点1：导入KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score  # 修改点2：分类指标
from sklearn.preprocessing import LabelEncoder

# ====================== 联邦学习组件 ======================
class Server:
    def __init__(self, n_features, n_random_features=200):
        self.n_features = n_features
        self.n_random_features = n_random_features
        self.W_features = None
        self.W_target = None
        self.b_base = None
        self.client_stats = defaultdict(list)
        self.global_scaler = None
        self.phi_x = None
        self.phi_y = None
        self.phi_z = {}

    def generate_global_params(self, data_sources):
        global_mean = np.mean(self.client_stats['mean'], axis=0)
        global_var = np.mean(self.client_stats['var'], axis=0)
        self.global_scaler = (global_mean, np.sqrt(global_var + 1e-8))

        valid_sources = [X for X, _ in data_sources if len(X) > 0]
        try:
            dummy_data = np.vstack([np.quantile(X, np.linspace(0.1, 0.9, 100), axis=0) for X in valid_sources])
            pairwise_dist = np.sqrt(((dummy_data[:, None] - dummy_data) ** 2).sum(axis=2) + 1e-8)
            sigma = np.median(pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)])
            sigma = max(sigma, 0.1)
        except Exception as e:
            print(f"参数生成异常: {e}, 使用默认参数")
            sigma = 1.0

        np.random.seed(0)
        scale = 1.0 / sigma
        self.W_features = np.random.normal(scale=scale, size=(self.n_features, self.n_random_features))
        self.W_target = np.random.normal(scale=scale, size=(1, self.n_random_features))
        self.b_base = np.random.uniform(0, 2*np.pi, size=(1, self.n_random_features))

    def collect_statistics(self, mean, var):
        self.client_stats['mean'].append(mean)
        self.client_stats['var'].append(var)

    def collect_client_data(self, phi_z, phi_y):
        if self.phi_x is None:
            self.phi_x = np.vstack(list(phi_z.values()))
            self.phi_y = phi_y
            self.phi_z = phi_z
        else:
            self.phi_x = np.vstack((self.phi_x, np.vstack(list(phi_z.values()))))
            self.phi_y = np.vstack((self.phi_y, phi_y))
            for idx in phi_z:
                self.phi_z[idx] = np.vstack((self.phi_z[idx], phi_z[idx]))

    def global_ci_test(self, X_idx, cond_set=[], n_perm=100):
        X = self.phi_z[X_idx]
        y = self.phi_y
        Z = np.hstack([self.phi_z[z] for z in cond_set]) if cond_set else None

        if Z is not None and Z.shape[1] > 0:
            alphas = np.logspace(-3, 3, 10)
            ridge_X = RidgeCV(alphas=alphas, cv=5).fit(Z, X)
            res_X = X - ridge_X.predict(Z)
            ridge_y = RidgeCV(alphas=alphas, cv=5).fit(Z, y)
            res_y = y - ridge_y.predict(Z)
        else:
            res_X, res_y = X, y

        hsic_obs = self._hsic(res_X, res_y)
        hsic_null = np.zeros(n_perm)
        for i in range(n_perm):
            perm_idx = np.random.permutation(res_y.shape[0])
            hsic_null[i] = self._hsic(res_X, res_y[perm_idx])

        p_value = (np.sum(hsic_null >= hsic_obs) + 1) / (n_perm + 1)
        print(p_value)
        return p_value

    def _hsic(self, X, Y):
        """HSIC计算（优化版）"""
        n = X.shape[0]
        res_X_centered = X - np.mean(X, axis=0)  # 显式中心化
        res_Y_centered = Y - np.mean(Y, axis=0)
        T = np.linalg.norm(res_X_centered.T @ res_Y_centered, 'fro') ** 2
        return T / (n ** 2)

class Client(BaseEstimator):
    def __init__(self, server):
        self.server = server
        self.local_b = None
        self.phi_z = {}

    def process_data(self, X, y, phase=1):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        if phase == 1:
            self.server.collect_statistics(np.mean(X, axis=0), np.var(X, axis=0))
        else:
            global_mean, global_std = self.server.global_scaler
            X = (X - global_mean) / (global_std + 1e-8)

            y_mean = np.mean(y)
            y_std = np.std(y)
            if np.isclose(y_std, 0):
                y_std = 1.0
            y = (y - y_mean) / (y_std + 1e-8)

            base_b = self.server.b_base
            delta_b = np.random.uniform(low=0, high=np.pi, size=base_b.shape)
            self.local_b = base_b + delta_b

            self.phi_z = {}
            for feat_idx in range(X.shape[1]):
                feat_data = X[:, feat_idx].reshape(-1, 1)
                W = self.server.W_features[feat_idx].reshape(1, -1)
                self.phi_z[feat_idx] = np.sqrt(2 / self.server.n_random_features) * np.cos(feat_data @ W + self.local_b)

            self.phi_y = np.sqrt(2 / self.server.n_random_features) * np.cos(
                y.reshape(-1, 1) @ self.server.W_target + self.local_b)
            self.server.collect_client_data(self.phi_z, self.phi_y)
        return self

class FederalHitonPC:
    def __init__(self, server, alpha=0.05, max_cond_size=10):
        self.server = server
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.pc_ = []
        self.ci_test_count = 0

    def fit(self):
        candidate_pc = []
        for X_idx in range(self.server.n_features):
            p_val = self.server.global_ci_test(X_idx, cond_set=[])
            self.ci_test_count += 1
            if p_val < self.alpha:
                candidate_pc.append(X_idx)

        prev_size = len(candidate_pc)
        stop_counter = 0
        l = 0

        while l <= self.max_cond_size and stop_counter < 2:
            to_remove = set()
            sorted_features = sorted(candidate_pc,
                                     key=lambda x: abs(np.corrcoef(self.server.phi_z[x],
                                                                   self.server.phi_y)[0, 1]),
                                     reverse=True)

            for X_idx in sorted_features:
                cond_candidates = [z for z in sorted_features if z != X_idx]
                for k in range(min(l, len(cond_candidates)) + 1):
                    for cond_set in combinations(cond_candidates[:20], k):
                        p_val = self.server.global_ci_test(X_idx, list(cond_set))
                        self.ci_test_count += 1
                        if p_val >= self.alpha:
                            to_remove.add(X_idx)
                            break
                    if X_idx in to_remove:
                        break

            candidate_pc = [x for x in candidate_pc if x not in to_remove]
            if len(to_remove) == 0:
                stop_counter += 1
            else:
                stop_counter = 0
            l += 1

        self.pc_ = candidate_pc
        return self

def evaluate_features(server, hiton, original_data):
    """修改后的评估函数（使用KNN分类器）"""
    try:
        X = original_data.iloc[:, :-1].values
        y = original_data.iloc[:, -1].values
        le = LabelEncoder()
        y = le.fit_transform(y)  # 确保标签为数值型

        global_mean, global_std = server.global_scaler
        X_normalized = (X - global_mean) / (global_std + 1e-8)

        selected_features = hiton.pc_
        if not selected_features:
            raise ValueError("未选择任何特征")

        X_all = X_normalized
        X_selected = X_normalized[:, selected_features]
        np.random.seed(42)
        random_features = np.random.choice(X.shape[1], size=len(selected_features), replace=False)
        X_random = X_normalized[:, random_features]

        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

        def train_evaluate(X_train_part, X_test_part):
            model = KNeighborsClassifier(n_neighbors=5)  # 使用KNN分类器
            model.fit(X_train_part, y_train)
            y_pred = model.predict(X_test_part)
            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average='weighted')
            }

        metrics_all = train_evaluate(X_train, X_test)
        metrics_selected = train_evaluate(X_train[:, selected_features], X_test[:, selected_features])
        metrics_random = train_evaluate(X_train[:, random_features], X_test[:, random_features])

        print("\n=== 特征评估结果（KNN分类）===")
        print(f"特征总数: {X.shape[1]}, 选出特征数: {len(selected_features)}")
        print("全特征 - 准确率: {:.3f}, F1: {:.3f}".format(metrics_all['Accuracy'], metrics_all['F1']))
        print("选择特征 - 准确率: {:.3f}, F1: {:.3f}".format(metrics_selected['Accuracy'], metrics_selected['F1']))
        print("随机特征 - 准确率: {:.3f}, F1: {:.3f}".format(metrics_random['Accuracy'], metrics_random['F1']))

    except Exception as e:
        print(f"评估错误: {e}")

# ====================== 执行示例 ======================
if __name__ == "__main__":
    #加载数据集
    file_path = r'C:\Users\36978\PycharmProjects\mystudy\datasets\Sonar.csv'
    data = pd.read_csv(file_path)  # 告诉pandas没有列名

    # 将数据标签分离给X和y
    X = data.iloc[:, :-1].values  # 假设标签在最后一列
    y = data.iloc[:, -1].values  # 假设标签在最后一列
    feature_names = data.columns[:-1].tolist()
    # 提取特征和目标


    # 数据分割（动态调整保证非空）
    n_samples = X.shape[0]
    print(f"\n总样本数: {n_samples}")
    split_indices = np.linspace(0, n_samples, 4, dtype=int)
    split_indices[-1] = n_samples  # 确保最后一位正确
    data_sources = []
    for i in range(3):
        start, end = split_indices[i], split_indices[i + 1]
        if end > start:
            data_sources.append((X[start:end], y[start:end]))
            print(f"客户端{i + 1}样本数: {end - start}")
        else:
            print(f"警告: 客户端{i + 1}无数据，跳过")

    # 初始化服务器
    server = Server(n_features=X.shape[1])

    # 阶段1: 统计量收集
    for X_part, y_part in data_sources:
        if len(X_part) > 0:
            Client(server).process_data(X_part, y_part, phase=1)

    # 生成全局参数
    server.generate_global_params(data_sources)

    # 阶段2: 特征映射
    for X_part, y_part in data_sources:
        if len(X_part) > 0:
            Client(server).process_data(X_part, y_part, phase=2)

    # 因果发现
    hiton = FederalHitonPC(server, alpha=0.05)
    hiton.fit()

    # 结果展示
    print("\n=== 因果特征分析结果 ===")
    print("重要特征索引:", hiton.pc_)
    print("对应特征名称:", [f"Feature_{i+1}" for i in hiton.pc_])

    # 执行特征评估
    evaluate_features(server, hiton, data)