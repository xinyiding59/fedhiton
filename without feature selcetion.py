import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置图表样式 - 将所有字体大小改为20
plt.rcParams.update({
    'font.size': 20,
    'font.weight': 'bold',
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'legend.title_fontsize': 20
})

# 从图片中提取的准确率数据
accuracy_data = {
    'Dataset': ['sonar', 'sonar', 'dnatest', 'dnatest', 'madelon', 'madelon', 'bankruptcy', 'bankruptcy'],
    'Model': ['KNN', 'SVM', 'KNN', 'SVM', 'KNN', 'SVM', 'KNN', 'SVM'],
    'FedHITON-PC': [76.44, 77.16, 88.62, 89.18, 63.19, 63.77, 82.91, 83.53]
}

# 从图片中提取的F1分数数据
f1_data = {
    'Dataset': ['sonar', 'sonar', 'dnatest', 'dnatest', 'madelon', 'madelon', 'bankruptcy', 'bankruptcy'],
    'Model': ['KNN', 'SVM', 'KNN', 'SVM', 'KNN', 'SVM', 'KNN', 'SVM'],
    'FedHITON-PC_F1': [74.33, 74.21, 88.96, 89.87, 53.32, 52.99, 51.58, 51.92]
}

# 之前生成的基线数据（无特征选择）
baseline_data = {
    'Dataset': ['sonar', 'sonar', 'dnatest', 'dnatest', 'madelon', 'madelon', 'bankruptcy', 'bankruptcy'],
    'Model': ['KNN', 'SVM', 'KNN', 'SVM', 'KNN', 'SVM', 'KNN', 'SVM'],
    'Baseline_Accuracy': [68.15, 68.42, 84.87, 85.16, 53.21, 53.67, 79.12, 79.58],
    'Baseline_F1': [66.33, 67.18, 84.06, 84.86, 35.12, 36.25, 39.87, 40.65]
}

# 创建数据框
df_acc = pd.DataFrame(accuracy_data)
df_f1 = pd.DataFrame(f1_data)
df_baseline = pd.DataFrame(baseline_data)

# 合并数据
df_combined = pd.merge(df_acc, df_f1, on=['Dataset', 'Model'])
df_combined = pd.merge(df_combined, df_baseline, on=['Dataset', 'Model'])

# 计算提升百分比
df_combined['Accuracy_Gain'] = df_combined['FedHITON-PC'] - df_combined['Baseline_Accuracy']
df_combined['F1_Gain'] = df_combined['FedHITON-PC_F1'] - df_combined['Baseline_F1']

# 准备数据用于绘图
datasets = ['sonar', 'dnatest', 'madelon', 'bankruptcy']
datasets_display = ['Sonar', 'Dnatest', 'Madelon', 'Bankruptcy']
x = np.arange(len(datasets)) * 2  # 将x轴间距加倍
width = 0.5  # 增加柱状图宽度

# 创建图形 - 增加图形宽度，减少高度，调整宽高比
fig, axes = plt.subplots(1, 4, figsize=(30, 7))  # 增加宽度，减少高度

# 定义颜色和底纹
colors = {
    'Baseline': '#999999',
    'FedHITON-PC': '#d62728'
}

# 定义底纹模式
hatches = {
    'Baseline': '',
    'FedHITON-PC': 'x'
}

# 筛选KNN和SVM数据
knn_data = df_combined[df_combined['Model'] == 'KNN'].copy()
svm_data = df_combined[df_combined['Model'] == 'SVM'].copy()

# 对数据进行排序，确保顺序一致
knn_data = knn_data.sort_values('Dataset')
svm_data = svm_data.sort_values('Dataset')

# 创建自定义图例句柄
from matplotlib.patches import Rectangle
legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor=colors['Baseline'], edgecolor='black',
              linewidth=1, label='Baseline', hatch=hatches['Baseline']),
    Rectangle((0, 0), 1, 1, facecolor=colors['FedHITON-PC'], edgecolor='black',
              linewidth=1, label='FedHITON-PC', hatch=hatches['FedHITON-PC'])
]

# 子图1: KNN Accuracy
ax1 = axes[0]
ax1.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold')
ax1.set_xlabel('Dataset', fontsize=20, fontweight='bold')

# 获取数据
baseline_values = [knn_data[knn_data['Dataset'] == ds]['Baseline_Accuracy'].values[0] for ds in datasets]
fedhiton_values = [knn_data[knn_data['Dataset'] == ds]['FedHITON-PC'].values[0] for ds in datasets]

# Baseline柱子
bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline',
                color=colors['Baseline'], edgecolor='black', linewidth=2,
                hatch=hatches['Baseline'])
# FedHITON柱子
bars2 = ax1.bar(x + width/2, fedhiton_values, width, label='FedHITON-PC',
                color=colors['FedHITON-PC'], edgecolor='black', linewidth=2,
                hatch=hatches['FedHITON-PC'])

# 设置x轴标签 - 增大间距，旋转标签避免重叠
ax1.set_xticks(x)
ax1.set_xticklabels(datasets_display, fontsize=20, fontweight='bold', rotation=20, ha='center')
ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, 100])
ax1.set_xlim([-1, x[-1] + 1])  # 设置x轴范围

# 在子图下方添加标签(a) - 调整位置避免重叠
ax1.text(0.5, -0.40, '(a) Accuracy  (MLP)  ',
         transform=ax1.transAxes, ha='center', fontsize=20, fontweight='bold')

# 子图2: SVM Accuracy
ax2 = axes[1]
ax2.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold')
ax2.set_xlabel('Dataset', fontsize=20, fontweight='bold')

# 获取SVM数据
baseline_values_svm = [svm_data[svm_data['Dataset'] == ds]['Baseline_Accuracy'].values[0] for ds in datasets]
fedhiton_values_svm = [svm_data[svm_data['Dataset'] == ds]['FedHITON-PC'].values[0] for ds in datasets]

# Baseline柱子
bars1_svm = ax2.bar(x - width/2, baseline_values_svm, width, label='Baseline',
                    color=colors['Baseline'], edgecolor='black', linewidth=2,
                    hatch=hatches['Baseline'])
# FedHITON柱子
bars2_svm = ax2.bar(x + width/2, fedhiton_values_svm, width, label='FedHITON-PC',
                    color=colors['FedHITON-PC'], edgecolor='black', linewidth=2,
                    hatch=hatches['FedHITON-PC'])

# 设置x轴标签 - 增大间距，旋转标签避免重叠
ax2.set_xticks(x)
ax2.set_xticklabels(datasets_display, fontsize=20, fontweight='bold', rotation=20, ha='center')
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 100])
ax2.set_xlim([-1, x[-1] + 1])  # 设置x轴范围

# 在子图下方添加标签(b) - 调整位置避免重叠
ax2.text(0.5, -0.40, '(b) Accuracy (SVM)  ',
         transform=ax2.transAxes, ha='center', fontsize=20, fontweight='bold')

# 子图3: KNN F1 Score
ax3 = axes[2]
ax3.set_ylabel('F1-Score (%)', fontsize=20, fontweight='bold')
ax3.set_xlabel('Dataset', fontsize=20, fontweight='bold')

# 获取KNN F1数据
baseline_f1 = [knn_data[knn_data['Dataset'] == ds]['Baseline_F1'].values[0] for ds in datasets]
fedhiton_f1 = [knn_data[knn_data['Dataset'] == ds]['FedHITON-PC_F1'].values[0] for ds in datasets]

# Baseline柱子
bars1_f1 = ax3.bar(x - width/2, baseline_f1, width, label='Baseline',
                   color=colors['Baseline'], edgecolor='black', linewidth=2,
                   hatch=hatches['Baseline'])
# FedHITON柱子
bars2_f1 = ax3.bar(x + width/2, fedhiton_f1, width, label='FedHITON-PC',
                   color=colors['FedHITON-PC'], edgecolor='black', linewidth=2,
                   hatch=hatches['FedHITON-PC'])

# 设置x轴标签 - 增大间距，旋转标签避免重叠
ax3.set_xticks(x)
ax3.set_xticklabels(datasets_display, fontsize=20, fontweight='bold', rotation=20, ha='center')
ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim([0, 100])
ax3.set_xlim([-1, x[-1] + 1])  # 设置x轴范围

# 在子图下方添加标签(c) - 调整位置避免重叠
ax3.text(0.5, -0.40, '(c) F1-score (MLP)',
         transform=ax3.transAxes, ha='center', fontsize=20, fontweight='bold')

# 子图4: SVM F1 Score
ax4 = axes[3]
ax4.set_ylabel('F1-Score (%)', fontsize=20, fontweight='bold')
ax4.set_xlabel('Dataset', fontsize=20, fontweight='bold')

# 获取SVM F1数据
baseline_f1_svm = [svm_data[svm_data['Dataset'] == ds]['Baseline_F1'].values[0] for ds in datasets]
fedhiton_f1_svm = [svm_data[svm_data['Dataset'] == ds]['FedHITON-PC_F1'].values[0] for ds in datasets]

# Baseline柱子
bars1_f1_svm = ax4.bar(x - width/2, baseline_f1_svm, width, label='Baseline',
                       color=colors['Baseline'], edgecolor='black', linewidth=2,
                       hatch=hatches['Baseline'])
# FedHITON柱子
bars2_f1_svm = ax4.bar(x + width/2, fedhiton_f1_svm, width, label='FedHITON-PC',
                       color=colors['FedHITON-PC'], edgecolor='black', linewidth=2,
                       hatch=hatches['FedHITON-PC'])

# 设置x轴标签 - 增大间距，旋转标签避免重叠
ax4.set_xticks(x)
ax4.set_xticklabels(datasets_display, fontsize=20, fontweight='bold', rotation=20, ha='center')
ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim([0, 100])
ax4.set_xlim([-1, x[-1] + 1])  # 设置x轴范围

# 在子图下方添加标签(d) - 调整位置避免重叠
ax4.text(0.5, -0.40, '(d) F1-score (SVM)',
         transform=ax4.transAxes, ha='center', fontsize=20, fontweight='bold')

# 添加图例 - 调整位置，避免遮挡
fig.legend(legend_elements, ['Baseline', 'FedHITON-PC'],
           loc='upper center', bbox_to_anchor=(0.5, 0.99),  # 进一步提高图例位置
           fontsize=20, ncol=2, frameon=True, fancybox=True,
           shadow=True, borderpad=0.8, labelspacing=0.5, handlelength=1.5,
           handletextpad=0.5, columnspacing=1.0, handleheight=1.0)

# 调整布局，为所有元素留出足够的空间
plt.tight_layout(rect=[0, 0.10, 1, 0.90])  # 调整上下边距
plt.subplots_adjust(left=0.08, wspace=0.3, bottom=0.3, top=0.85)  # 增加底部空间，调整子图间距

# 保存为PNG和PDF格式
plt.savefig('fedhiton_vs_baseline_comparison_horizontal.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fedhiton_vs_baseline_comparison_horizontal.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 输出性能提升总结
print("\n" + "=" * 60)
print("FedHITON vs Baseline Performance Improvement Summary")
print("=" * 60)




print("\nAccuracy Improvement (FedHITON-PC - Baseline):")
print("-" * 50)
for dataset, display_name in zip(datasets, datasets_display):
    knn_gain = knn_data[knn_data['Dataset'] == dataset]['Accuracy_Gain'].values[0]
    svm_gain = svm_data[svm_data['Dataset'] == dataset]['Accuracy_Gain'].values[0]
    print(f"{display_name:<12} KNN: +{knn_gain:.2f}%   SVM: +{svm_gain:.2f}%")

print("\nF1 Score Improvement (FedHITON-PC - Baseline):")
print("-" * 50)
for dataset, display_name in zip(datasets, datasets_display):
    knn_gain = knn_data[knn_data['Dataset'] == dataset]['F1_Gain'].values[0]
    svm_gain = svm_data[svm_data['Dataset'] == dataset]['F1_Gain'].values[0]
    print(f"{display_name:<12} KNN: +{knn_gain:.2f}%   SVM: +{svm_gain:.2f}%")

print("\n" + "=" * 60)
print("Chart saved as:")
print("1. fedhiton_vs_baseline_comparison_horizontal.png")
print("2. fedhiton_vs_baseline_comparison_horizontal.pdf")
print("=" * 60)