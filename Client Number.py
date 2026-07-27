import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗
plt.rcParams['font.weight'] = 'bold'  # 全局字体加粗
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# 移除seaborn样式，使用matplotlib默认样式
plt.style.use('default')
#准备数据F1分数
data = {
    'Sonar': {
        'KNN': {
            'Fed FiS': [69.85, 68.43, 70.12, 71.25, 72.08],
            'FPSO-FS': [73.28, 72.18, 73.85, 74.36, 73.98],
            'FedCIFL': [72.82, 72.33, 73.47, 74.77, 75.16],
            'FedHITON-PC': [73.79, 72.90, 74.15, 75.81, 74.40],
        },
        'SVM': {
            'Fed FiS': [70.89, 69.91, 71.33, 72.16, 73.21],
            'FPSO-FS': [72.21, 73.61, 74.38, 75.43, 75.43],
            'FedCIFL': [73.13, 73.30, 73.53, 75.98, 74.77],
            'FedHITON-PC': [73.77, 73.30, 73.67, 75.26, 74.50]
        }
    },
    'Dnatest': {
        'KNN': {
            'Fed FiS': [86.54, 85.47, 87.49, 85.58, 86.22],
            'FPSO-FS': [87.57, 88.47, 88.05, 88.50, 87.51],
            'FedCIFL': [88.70, 88.09, 88.83, 90.73, 89.91],
            'FedHITON-PC': [89.54, 88.36, 90.34, 88.54, 88.01],
        },
        'SVM': {
            'Fed FiS': [87.83, 86.08, 88.34, 87.31, 88.25],
            'FPSO-FS': [89.19, 88.23, 89.53, 88.84, 88.31],
            'FedCIFL': [89.29, 90.09, 90.84, 88.93, 90.19],
            'FedHITON-PC': [89.91, 90.15, 89.53, 89.18, 88.18]
        }
    },
    'Madelon': {
        'KNN': {
            'Fed FiS': [46.13, 47.94, 45.59, 48.93, 47.40],
            'FPSO-FS': [53.74, 54.46, 52.60, 52.65, 52.89],
            'FedCIFL': [52.55, 53.22, 52.05, 53.71, 52.98],
            'FedHITON-PC': [53.29, 53.65, 52.19, 54.20, 53.27],
        },
        'SVM': {
            'Fed FiS': [47.09, 48.82, 47.01, 48.88, 47.50],
            'FPSO-FS': [53.69, 52.16, 51.65, 52.86, 54.51],
            'FedCIFL': [54.20, 52.81, 51.63, 52.73, 54.19],
            'FedHITON-PC': [54.23, 52.71, 52.39, 53.53, 52.08]
        }
    },
    'Bankruptcy': {
        'KNN': {
            'Fed FiS': [48.28, 49.12, 48.20, 47.23, 48.21],
            'FPSO-FS': [49.76, 50.49, 51.22, 50.42, 49.92],
            'FedCIFL': [50.18, 50.83, 51.59, 51.05, 49.00],
            'FedHITON-PC': [51.26, 51.25, 52.32, 52.54, 50.53],
        },
        'SVM': {
            'Fed FiS': [49.26, 50.47, 48.76, 50.86, 49.19],
            'FPSO-FS': [50.29, 52.05, 51.48, 50.61, 49.94],
            'FedCIFL': [51.93, 52.57, 50.71, 51.21, 49.83],
            'FedHITON-PC': [51.63, 53.24, 52.75, 51.63, 50.34]
        }
    }
}
# # 准备数据准确率
data = {
    'Sonar': {
        'KNN': {
            'Fed FiS': [72.80, 70.50, 72.25, 73.60, 73.90],
            'FPSO-FS': [70.85, 69.80, 71.05, 71.35, 71.80],
            'FedCIFL': [75.10, 73.85, 75.18, 76.20, 75.95],
            'FedHITON-PC': [76.70, 75.65, 76.49, 76.45, 76.90]
        },
        'SVM': {
            'Fed FiS': [73.10, 70.75, 72.48, 73.90, 74.10],
            'FPSO-FS': [71.15, 70.05, 71.32, 71.60, 72.00],
            'FedCIFL': [75.60, 74.25, 75.67, 76.50, 76.40],
            'FedHITON-PC': [77.45, 76.35, 77.21, 77.20, 77.60]
        }
    },
    'Dnatest': {
        'KNN': {
            'Fed FiS': [88.10, 87.16, 87.80, 89.00, 88.55],
            'FPSO-FS': [85.75, 85.53, 85.70, 86.20, 86.15],
            'FedCIFL': [87.70, 87.21, 87.60, 88.90, 88.15],
            'FedHITON-PC': [88.85, 88.34, 88.50, 88.85, 88.55]
        },
        'SVM': {
            'Fed FiS': [88.65, 87.52, 88.15, 89.20, 88.95],
            'FPSO-FS': [86.35, 85.98, 86.15, 86.60, 86.60],
            'FedCIFL': [88.05, 87.79, 88.20, 89.30, 88.75],
            'FedHITON-PC': [89.20, 88.96, 89.15, 89.15, 89.45]
        }
    },
    'Madelon': {
        'KNN': {
            'Fed FiS': [58.20, 57.20, 56.56, 59.50, 57.70],
            'FPSO-FS': [58.10, 57.60, 56.87, 58.80, 58.75],
            'FedCIFL': [63.45, 62.25, 61.71, 64.20, 63.00],
            'FedHITON-PC': [63.65, 62.75, 62.17, 63.65, 63.75]
        },
        'SVM': {
            'Fed FiS': [58.50, 57.45, 56.81, 59.80, 58.00],
            'FPSO-FS': [58.45, 58.00, 57.25, 59.20, 59.10],
            'FedCIFL': [63.85, 62.85, 62.39, 64.60, 63.70],
            'FedHITON-PC': [64.30, 63.35, 62.74, 64.30, 64.15]
        }
    },
    'Bankruptcy': {
        'KNN': {
            'Fed FiS': [81.20, 82.95, 81.75, 80.20, 79.50],
            'FPSO-FS': [80.10, 82.45, 81.70, 79.90, 78.05],
            'FedCIFL': [81.80, 84.55, 83.60, 81.30, 80.65],
            'FedHITON-PC': [82.40, 84.25, 83.75, 82.40, 81.75]
        },
        'SVM': {
            'Fed FiS': [81.50, 83.40, 82.15, 80.50, 78.80],
            'FPSO-FS': [80.35, 82.95, 82.15, 80.25, 79.50],
            'FedCIFL': [82.10, 85.10, 84.15, 81.80, 81.10],
            'FedHITON-PC': [82.95, 84.90, 84.50, 82.95, 82.35]
        }
    }
}

# 创建图形和子图
fig, axes = plt.subplots(2, 4, figsize=(24, 12),
                         facecolor='white',  # 设置图形背景为白色
                         edgecolor='white')

# 特征选择方法列表、颜色和条纹
methods = ['Fed FiS', 'FPSO-FS', 'FedCIFL', 'FedHITON-PC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
patterns = ['/', '\\', '-', 'x']

client_labels = ['1', '2', ' 3', '4', '5']

# 设置柱状图参数
x = np.arange(len(client_labels))
width = 0.2  # 柱子的宽度


# 计算每个数据集的Y轴范围函数
def calculate_ylim(dataset_data, classifier):
    """计算合理的Y轴范围，基于数据的最小值和最大值"""
    all_values = []
    for method in methods:
        all_values.extend(dataset_data[classifier][method])

    min_val = min(all_values)
    max_val = max(all_values)

    # 添加一些边距
    margin = (max_val - min_val) * 0.1
    y_min = max(0, min_val - margin)  # 确保不低于0
    y_max = max_val + margin

    return y_min, y_max


# 子图标题序号
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

# 获取图例句柄和标签（从第一个子图）
handles, labels = None, None

# 绘制KNN分类器的四个子图（第一行）
datasets_knn = ['Sonar', 'Dnatest', 'Madelon', 'Bankruptcy']
for i, dataset in enumerate(datasets_knn):
    ax = axes[0, i]

    # 设置子图背景为白色
    ax.set_facecolor('white')

    # 绘制每个方法的柱状图
    for j, method in enumerate(methods):
        positions = x + j * width - (len(methods) - 1) * width / 2
        accuracies = data[dataset]['KNN'][method]
        bars = ax.bar(positions, accuracies, width,
                      label=method, color=colors[j], alpha=0.8,
                      edgecolor='black', linewidth=0.8,
                      hatch=patterns[j] * 3)

    # 设置子图属性
    ax.set_xlabel('Client Id', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

    # 添加序号和标题在子图下方
    title_text = f'{subplot_labels[i]} {dataset} - MLP'
    ax.text(0.5, -0.2, title_text, transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='center', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(client_labels, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=11)

    # 设置坐标轴线
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # 设置动态Y轴范围
    y_min, y_max = calculate_ylim(data[dataset], 'KNN')
    ax.set_ylim(y_min, y_max)

    # 添加水平网格线
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # 将网格线放在数据后面

    # 保存第一个子图的图例句柄和标签
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# 绘制SVM分类器的四个子图（第二行）
for i, dataset in enumerate(datasets_knn):
    ax = axes[1, i]

    # 设置子图背景为白色
    ax.set_facecolor('white')

    for j, method in enumerate(methods):
        positions = x + j * width - (len(methods) - 1) * width / 2
        accuracies = data[dataset]['SVM'][method]
        bars = ax.bar(positions, accuracies, width,
                      label=method, color=colors[j], alpha=0.8,
                      edgecolor='black', linewidth=0.8,
                      hatch=patterns[j] * 3)

    # 设置子图属性
    ax.set_xlabel('Client Id', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

    # 添加序号和标题在子图下方
    title_text = f'{subplot_labels[i + 4]} {dataset} - SVM'
    ax.text(0.5, -0.2, title_text, transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='center', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(client_labels, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=11)

    # 设置坐标轴线
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # 设置动态Y轴范围
    y_min, y_max = calculate_ylim(data[dataset], 'SVM')
    ax.set_ylim(y_min, y_max)

    # 添加水平网格线
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # 将网格线放在数据后面

# 在图形顶部中央添加图例
if handles and labels:
    # 创建自定义的图例
    legend = fig.legend(handles, labels,
                        loc='upper center',
                        bbox_to_anchor=(0.5, 0.96),  # 居中，在图形顶部
                        fontsize=12,
                        ncol=4,
                        frameon=True,
                        fancybox=False,
                        shadow=False,
                        edgecolor='black',
                        borderpad=0.6,
                        handlelength=1.5,
                        handletextpad=0.5,
                        columnspacing=1.0)

    # 设置图例中文本的加粗
    for text in legend.get_texts():
        text.set_fontweight('bold')

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.12, hspace=0.4, wspace=0.2, left=0.05, right=0.98)

# 保存为PDF格式
plt.savefig('Federated_Feature_Selection_Comparison.pdf',
            dpi=300,
            bbox_inches='tight',
            format='pdf',
            facecolor='white',  # 保存时背景为白色
            edgecolor='white')

# 显示图形
plt.show()