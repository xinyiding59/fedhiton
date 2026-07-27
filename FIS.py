import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载wine数据集
wine = load_wine()
X, y = wine.data, wine.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 计算类内均值
def class_mean(X, y, class_label):
    X_class = X[y == class_label]
    return np.mean(X_class, axis=0)


# 计算类内方差
def class_variance(X, y, class_label):
    X_class = X[y == class_label]
    return np.var(X_class, axis=0)


# 计算Fisher Score
def fisher_score(X, y):
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    fisher_scores = np.zeros(num_features)

    for i in range(num_features):
        within_class_variance = 0
        between_class_variance = 0
        for c in range(num_classes):
            class_mean_value = class_mean(X, y, c)
            class_variance_value = class_variance(X, y, c)
            within_class_variance += class_variance_value[i]
            between_class_variance += (class_mean_value[i] - np.mean(X[:, i])) ** 2
        fisher_scores[i] = between_class_variance / within_class_variance

    return fisher_scores


# 计算Fisher Score
fisher_scores = fisher_score(X_train, y_train)
print(fisher_scores)
# 输出Fisher Score排名
print("Fisher Score排名:")
print(np.argsort(fisher_scores)[::-1])

# 选择具有最高Fisher Score的特征
selected_features = np.argsort(fisher_scores)[::-1][:5]  # 假设选择前5个特征
print("选择的特征:")
print(selected_features)
