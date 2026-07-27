import matplotlib.pyplot as plt
import numpy as np

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'  # 设置全局加粗
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['font.size'] = 18  # 设置全局字体大小

# 创建示例数据
# 4种特征选择方法
feature_selectors = ['Fed FiS', 'FPSO-FS', 'FedCIFL', 'FedHITON-PC']

# 不同客户端数量
client_counts = [3, 5, 7, 10]

# 4个数据集名称
datasets = ['Sonar', 'Dnatest', 'Madelon', 'Bankruptcy']

# 从Excel表格中提取的真实数据
# 更新后的数据格式: {数据集: {评价指标: {特征选择方法: [不同客户端数量的数据]}}}
data = {
    'Sonar': {
        'Accuracy': {
            'Fed FiS': [73.25, 72.61, 72.08, 71.54],
            'FPSO-FS': [74.12, 70.97, 70.25, 69.58],
            'FedCIFL': [75.85, 75.56, 75.12, 74.78],
            'FedHITON-PC': [77.35, 76.64, 76.15, 75.68]
        },
        'F1-Score': {
            'Fed FiS': [72.45, 70.35, 69.78, 69.12],
            'FPSO-FS': [74.28, 73.53, 72.58, 71.85],
            'FedCIFL': [72.85, 73.37, 72.88, 72.45],
            'FedHITON-PC': [74.15, 74.33, 73.85, 73.42]
        }
    },
    'Dnatest': {
        'Accuracy': {
            'Fed FiS': [88.15, 88.12, 87.68, 87.15],
            'FPSO-FS': [87.15, 85.87, 85.28, 84.48],
            'FedCIFL': [87.35, 87.91, 87.52, 87.08],
            'FedHITON-PC': [88.35, 88.62, 88.18, 87.85]
        },
        'F1-Score': {
            'Fed FiS': [85.85, 86.62, 85.95, 85.38],
            'FPSO-FS': [86.78, 88.02, 87.18, 86.38],
            'FedCIFL': [86.78, 88.25, 87.58, 87.15],
            'FedHITON-PC': [87.35, 88.96, 88.35, 87.92]
        }
    },
    'Madelon': {
        'Accuracy': {
            'Fed FiS': [58.85, 57.83, 57.45, 56.98],
            'FPSO-FS': [58.82, 58.02, 57.45, 56.82],
            'FedCIFL': [61.35, 62.92, 62.45, 61.98],
            'FedHITON-PC': [61.65, 63.19, 62.75, 62.28]
        },
        'F1-Score': {
            'Fed FiS': [40.45, 47.20, 46.28, 45.55],
            'FPSO-FS': [52.68, 53.27, 52.48, 51.75],
            'FedCIFL': [49.15, 52.70, 52.05, 51.48],
            'FedHITON-PC': [51.85, 53.32, 52.65, 52.05]
        }
    },
    'Bankruptcy': {
        'Accuracy': {
            'Fed FiS': [82.45, 81.12, 80.68, 80.15],
            'FPSO-FS': [84.12, 80.44, 79.78, 79.12],
            'FedCIFL': [82.85, 82.38, 81.88, 81.38],
            'FedHITON-PC': [85.15, 82.91, 82.35, 81.78]
        },
        'F1-Score': {
            'Fed FiS': [44.15, 48.21, 47.15, 46.45],
            'FPSO-FS': [45.58, 50.36, 49.38, 48.55],
            'FedCIFL': [43.45, 50.53, 49.65, 48.98],
            'FedHITON-PC': [47.85, 51.58, 50.75, 50.05]
        }
    }
}

# 转换数据为百分比格式（除以100）
for dataset in data:
    for metric in data[dataset]:
        for selector in data[dataset][metric]:
            data[dataset][metric][selector] = [x / 100 for x in data[dataset][metric][selector]]

# 定义颜色和线型
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 四种不同的颜色
linestyles = ['-', '--', '-.', ':']  # 不同的线型
markers = ['o', 's', '^', 'D']  # 不同的标记

# 创建一个大图，横向排列8个子图（2行×4列）
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 子图标签
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

# 绘制每个子图
for row in range(2):  # 两行：Accuracy和F1 Score
    for col in range(4):  # 四列：四个数据集
        ax = axes[row, col]
        dataset_idx = col
        metric = 'Accuracy' if row == 0 else 'F1-Score'
        dataset = datasets[dataset_idx]
        subplot_idx = row * 4 + col

        for idx, selector in enumerate(feature_selectors):
            ax.plot(client_counts,
                    data[dataset][metric][selector],
                    color=colors[idx],
                    linestyle=linestyles[idx % len(linestyles)],
                    marker=markers[idx],
                    markersize=9,  # 稍微增大标记大小
                    linewidth=2.5,
                    label=selector)

        # 设置坐标轴标签
        ax.set_xlabel('Number of Clients', fontsize=18, fontweight='bold')
        ax.set_ylabel(metric, fontsize=18, fontweight='bold')

        # 根据数据集和指标设置不同的y轴范围
        if dataset == 'Madelon':
            if metric == 'Accuracy':
                ax.set_ylim(0.50, 0.70)
            else:  # F1 Score
                ax.set_ylim(0.35, 0.60)
        elif dataset == 'Bankruptcy' and metric == 'F1-Score':
            ax.set_ylim(0.40, 0.55)
        else:
            ax.set_ylim(0.65, 0.95)

        ax.grid(True, alpha=0.3, linestyle='--')

        # 设置x轴刻度
        ax.set_xticks(client_counts)
        ax.set_xticklabels(client_counts, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18, width=2)

        # 设置坐标轴为加粗
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # 在底部添加标题
        title_text = f'{subplot_labels[subplot_idx]} {dataset} ({metric})'
        ax.text(0.5, -0.25, title_text,
                fontsize=18, fontweight='bold',
                fontfamily='Times New Roman',
                ha='center', va='center', transform=ax.transAxes)

# 调整布局，为底部标题和顶部图例留出空间
plt.subplots_adjust(top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)

# 创建大图的图例
handles, labels = axes[0, 0].get_legend_handles_labels()

# 创建大图的图例
legend = fig.legend(handles, labels,
                    loc='upper center',
                    ncol=len(feature_selectors),
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    borderpad=1,
                    prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 18})

# 设置图例标题的字体
for text in legend.get_texts():
    text.set_fontproperties(plt.matplotlib.font_manager.FontProperties(
        family='Times New Roman', weight='bold', size=18))
plt.savefig('results_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()