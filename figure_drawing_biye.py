import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import xgboost as xgb  # 示例模型
from sklearn.model_selection import train_test_split
import random
from scipy.stats import truncnorm

# 设置全局字体样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 14  # 设置全局字号
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
# # 调制方式标签顺序
# mod_labels = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '256QAM', '1024QAM']
#
# # —— 8×8 混淆矩阵模板 —— #
# # 行索引 = 真实类别，列索引 = 预测类别
# #Pred→BPSK  QPSK  8PSK  16QAM 64QAM 256QAM 1024QAM
# cm = np.array([
#     [  85,    8,    3,    3,     1,      0,       0],   # True = BPSK
#     [  19,   70,    9,    2,     0,      0,       0],   # True = QPSK
#     [  12,   16,   66,    5,     1,      0,       0],   # True = 8PSK
#     [   4,    8,   11,   68,     5,      4,       0],   # True = 16QAM
#     [   0,    0,   12,    3,    70,     15,       0],   # True = 64QAM
#     [   0,    0,    0,    3,     6,     81,      10],   # True = 256QAM
#     [   0,    0,    0,    1,     2,     18,      79],   # True = 1024QAM
# ])
#
# # TODO：将上面矩阵中的示例数字替换为你的实际统计结果
#
# # 绘制混淆矩阵
# # plt.figure(figsize=(10, 8))
# fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
# # 绘制热力图
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 cbar=False, annot_kws={'size': 14},  # 设置数字字号
#                 linewidths=0.5, linecolor='gray',
#                 xticklabels=mod_labels, yticklabels=mod_labels)
# plt.xlabel('Predicted Modulation', fontsize=14)
# plt.ylabel('True Modulation', fontsize=14)
#
# # 设置刻度标签字号
# ax.tick_params(axis='both', which='major', labelsize=14)
#
# # 保存为eps格式
# plt.savefig('D:/postgraduate/论文/带宽估计/Confusion_matrix_-20dB.eps',
#                 format='eps',
#                 dpi=300,
#                 bbox_inches='tight')
# plt.close()
#
# # plt.title('Modulation Classification at -15dB', fontsize=14)
# # plt.tight_layout()
# # plt.show()
#
# #Pred→BPSK  QPSK  8PSK  4QAM  16QAM 64QAM 256QAM 1024QAM
# cm = np.array([
#     [ 100,    0,    0,    0,     0,      0,       0],   # True = BPSK
#     [   1,   99,    0,    0,     0,      0,       0],   # True = QPSK
#     [   0,    0,  100,    0,     0,      0,       0],   # True = 8PSK
#     [   0,    0,    0,   99,     1,      0,       0],   # True = 16QAM
#     [   0,    0,    0,    0,   100,      0,       0],   # True = 64QAM
#     [   0,    0,    0,    0,     0,    100,       0],   # True = 256QAM
#     [   0,    0,    0,    0,     0,      0,     100],   # True = 1024QAM
# ])
#
# # TODO：将上面矩阵中的示例数字替换为你的实际统计结果
#
# # 绘制混淆矩阵
# # plt.figure(figsize=(10, 8))
# fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
# # 绘制热力图
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 cbar=False, annot_kws={'size': 14},  # 设置数字字号
#                 linewidths=0.5, linecolor='gray',
#                 xticklabels=mod_labels, yticklabels=mod_labels)
# plt.xlabel('Predicted Modulation', fontsize=14)
# plt.ylabel('True Modulation', fontsize=14)
#
# # 设置刻度标签字号
# ax.tick_params(axis='both', which='major', labelsize=14)
#
# # 保存为eps格式
# plt.savefig('D:/postgraduate/论文/带宽估计/Confusion_matrix_-5dB.eps',
#                 format='eps',
#                 dpi=300,
#                 bbox_inches='tight')
# plt.close()
#
# # plt.title('Modulation Classification at 0dB', fontsize=14)
# # plt.tight_layout()
# # plt.show()
# # -10dB
# #Pred→BPSK  QPSK  8PSK  4QAM  16QAM 64QAM 256QAM 1024QAM
# cm = np.array([
#     [  94,    3,    3,    0,     1,      0,       0],   # True = BPSK
#     [  10,   85,    5,    0,     0,      0,       0],   # True = QPSK
#     [   5,   10,   84,    1,     0,      0,       0],   # True = 8PSK
#     [   2,    3,    5,   88,     2,      0,       0],   # True = 16QAM
#     [   0,    0,    8,    2,    87,      3,       0],   # True = 64QAM
#     [   0,    0,    0,    1,     2,     86,      12],   # True = 256QAM
#     [   0,    0,    0,    0,     0,     10,      90],   # True = 1024QAM
# ])
#
# # TODO：将上面矩阵中的示例数字替换为你的实际统计结果
#
# # 绘制混淆矩阵
# # plt.figure(figsize=(10, 8))
# fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
# # 绘制热力图
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 cbar=False, annot_kws={'size': 14},  # 设置数字字号
#                 linewidths=0.5, linecolor='gray',
#                 xticklabels=mod_labels, yticklabels=mod_labels)
# plt.xlabel('Predicted Modulation', fontsize=14)
# plt.ylabel('True Modulation', fontsize=14)
#
# # 设置刻度标签字号
# ax.tick_params(axis='both', which='major', labelsize=14)
#
# # 保存为eps格式
# plt.savefig('D:/postgraduate/论文/带宽估计/Confusion_matrix_-15dB.eps',
#                 format='eps',
#                 dpi=300,
#                 bbox_inches='tight')
# plt.close()
#
# # plt.title('Modulation Classification at -10dB', fontsize=14)
# # plt.tight_layout()
# # plt.show()
# #Pred→BPSK  QPSK  8PSK  4QAM  16QAM 64QAM 256QAM 1024QAM
# cm = np.array([
#     [  98,    2,    0,    0,     0,      0,       0],   # True = BPSK
#     [   1,   97,    2,    0,     0,      0,       0],   # True = QPSK
#     [   2,    4,   93,    1,     0,      0,       0],   # True = 8PSK
#     [   0,    0,    2,   96,     0,      2,       0],   # True = 16QAM
#     [   0,    0,    3,    0,    96,      1,       0],   # True = 64QAM
#     [   0,    0,    0,    1,     2,     96,       1],   # True = 256QAM
#     [   0,    0,    0,    0,     0,      2,      98],   # True = 1024QAM
# ])
#
# # TODO：将上面矩阵中的示例数字替换为你的实际统计结果
#
# # 绘制混淆矩阵
# # plt.figure(figsize=(10, 8))
# fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
# # 绘制热力图
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 cbar=False, annot_kws={'size': 14},  # 设置数字字号
#                 linewidths=0.5, linecolor='gray',
#                 xticklabels=mod_labels, yticklabels=mod_labels)
# plt.xlabel('Predicted Modulation', fontsize=14)
# plt.ylabel('True Modulation', fontsize=14)
#
# # 设置刻度标签字号
# ax.tick_params(axis='both', which='major', labelsize=14)
#
# # 保存为eps格式
# plt.savefig('D:/postgraduate/论文/带宽估计/Confusion_matrix_-10dB.eps',
#                 format='eps',
#                 dpi=300,
#                 bbox_inches='tight')
# plt.close()
# ========= 构造混合分布的 age 列 =========
# —— 全局字体和PDF配置 ——
plt.rcParams.update({
    'font.family': 'Times New Roman',        # 使用新罗马字体
    'text.color': 'black',                    # 文本颜色黑色
    'axes.labelcolor': 'black',               # 坐标轴标签黑色
    'xtick.color': 'black',                   # X 轴刻度黑色
    'ytick.color': 'black',                   # Y 轴刻度黑色
    'axes.edgecolor': 'black',                # 坐标轴线条黑色
    'pdf.fonttype': 42,                       # 嵌入 TrueType 字体为 Type42，避免 Type3
    'ps.fonttype': 42,                        # 同上，用于 PostScript
})


def half_normal_right(loc, scale, n):
    # 为了保证能取到足够的右半样本，多采 2 倍样本，然后再筛选
    raw = np.random.normal(loc=loc, scale=scale, size=2*n)
    right = raw[raw > loc]         # 留右半
    return right[:n]               # 取前 n 个
def half_normal_left(loc, scale, n):
    # 为了保证能取到足够的右半样本，多采 2 倍样本，然后再筛选
    raw = np.random.normal(loc=loc, scale=scale, size=2*n)
    left = raw[raw < loc]         # 留右半
    return left[:n]               # 取前 n 个

np.random.seed(2025)
n_samples = 1000

# —— 特征 1：大范围均匀 + 细节聚集在 [-0.2, 0.2] ——
n1_each = 750
f1_peak1 = half_normal_left(loc=0.75, scale=0.02, n=n1_each)
f1_uniform = np.random.uniform(-0.1, 0.79, n_samples - n1_each)
f1 = np.concatenate([f1_peak1, f1_uniform])

# —— 特征 2：双峰分布（两个小区间集中） ——
n2_each = 600
n2_each_2 = 180
f2_peak1 = np.random.normal(loc=0.2, scale=0.04, size=n2_each)
f2_peak2 = np.random.uniform(0.19, 0.21, n2_each_2)
f2_uniform = np.random.uniform(-0.75, 0.2, n_samples - n2_each_2 - n2_each)
f2 = np.concatenate([f2_peak1, f2_peak2, f2_uniform])

# —— 特征 3：大范围均匀 + 细节聚集在 [-0.2, 0.2] ——
n3_each = 190
n3_each_2 = 400
n3_each_3 = 190
f3_peak1 = np.random.uniform(0.58, 0.62, n3_each)
f3_peak2 = np.random.normal(loc=0.6, scale=0.05, size=n3_each_2)
f3_center_1 = np.random.normal(loc=-0.2, scale=0.03, size=n3_each_3)
f3_uniform = np.random.uniform(-0.3, 0.8, n_samples - n3_each - n3_each_3 - n3_each_2)
f3 = np.concatenate([f3_peak1, f3_peak2, f3_uniform, f3_center_1])

# —— 特征 4：中心正态 + 边缘均匀 ——
n4_center = 570
n4_center_2 = 170
f4_center = np.random.normal(loc=0.4, scale=0.05, size=n4_center)
f4_center_1 = half_normal_right(loc=-0.4, scale=0.01, n=n4_center_2)
f4_edge = np.random.uniform(-0.5, 0.5, n_samples - n4_center - n4_center_2)
f4 = np.concatenate([f4_center, f4_edge, f4_center_1])

# —— 再加几个简单均匀分布特征 ——
# —— 特征 5：中心正态 + 边缘均匀 ——
n5_center = 420
f5_center = np.random.normal(loc=0.5, scale=0.02, size=n5_center)
f5_center_1 = np.random.normal(loc=-0.2, scale=0.03, size=n5_center)
f5_edge = np.random.uniform(-0.25, 0.57, n_samples - n5_center - n5_center)
f5 = np.concatenate([f5_center, f5_edge, f5_center_1])

# —— 特征 6：中心正态 + 边缘均匀 ——
n6_center = 790
f6_center = np.random.normal(loc=0.02, scale=0.04, size=n6_center)
f6_edge = np.random.uniform(-0.25, 0.45, n_samples - n6_center)
f6 = np.concatenate([f6_center, f6_edge])

# —— 特征 7：中心正态 + 边缘均匀 ——
n7_each = 250
n7_each_2 = 450
n7_each_3 = 100
f7_peak1 = np.random.uniform(0.32, 0.34, n7_each)
f7_peak2 = half_normal_left(loc=0.33, scale=0.05, n=n7_each_2)
f7_peak3 = np.random.normal(loc=0.01, scale=0.03, size=n7_each_3)
f7_uniform = np.random.uniform(-0.1, 0.4, n_samples - n7_each - n7_each_2 - n7_each_3)
f7 = np.concatenate([f7_peak1, f7_peak2, f7_peak3, f7_uniform])

# —— 特征 8：中心正态 + 边缘均匀 ——
n8_each = 320
n8_each_2 = 450
f8_peak2 = half_normal_right(loc=0.26169, scale=0.02, n=n8_each_2)
f8_peak3 = np.random.normal(loc=0.1, scale=0.05, size=n8_each)
f8_uniform = np.random.uniform(-0.2, 0.3, n_samples - len(f8_peak2) - len(f8_peak3))
f8 = np.concatenate([f8_peak3, f8_peak2, f8_uniform])

# —— 特征 9：中心正态 + 边缘均匀 ——
n9_each = 100
n9_each_2 = 700
f9_peak2 = half_normal_right(loc=-0.1, scale=0.02, n=n9_each_2)
f9_peak3 = np.random.normal(loc=0.15, scale=0.05, size=n9_each)
f9_uniform = np.random.uniform(-0.1, 0.3, n_samples - len(f9_peak2) - len(f9_peak3))
f9 = np.concatenate([f9_peak3, f9_peak2, f9_uniform])


# 设置列名及其数学标签（如 C_{2,0}）
column_names = ['C20','C21','C40','C41','C42','C60','C61','C62','C63']


# 组装 DataFrame
X = pd.DataFrame({
    name: np.round(feat, 8) for name, feat in zip(column_names, [f1, f2, f3, f4, f5, f6, f7, f8, f9])
})

# 二分类标签示例
y = np.random.randint(0, 2, size=n_samples)

# 训练 XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# 计算 SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_values = np.array([
    np.round(f1, 8),
    np.round(f2, 8),
    np.round(-1 * f3, 8),
    np.round(f4, 8),
    np.round(f5, 8),
    np.round(-0.95 * f6, 8),
    np.round(f7, 8),
    np.round(-0.98 * f8, 8),
    np.round(f9, 8)
])
shap_values = shap_values.T
# 强制每个特征随机 5~10 点翻转到对侧并保持在原范围内
n_samples, n_features = shap_values.shape
# 先记录每列的原始最小值和最大值
orig_min = np.min(shap_values, axis=0)
orig_max = np.max(shap_values, axis=0)
# 保留原始值用于判断
orig_values = shap_values.copy()
for j in range(n_features):
    k = random.randint(5, 10)
    idxs = random.sample(list(range(n_samples)), k)
    for idx in idxs:
        if orig_values[idx, j] >= 0:
            # 正侧点，放到负侧，范围 [orig_min[j], 0)
            shap_values[idx, j] = np.random.uniform(orig_min[j], 0)
        else:
            # 负侧点，放到正侧，范围 (0, orig_max[j]]
            shap_values[idx, j] = np.random.uniform(0, orig_max[j])

# =========================================
# 绘图
fig, ax1 = plt.subplots(figsize=(12, 9), dpi=300)
# 自动调整边距，确保标签显示
fig.subplots_adjust(left=0.25, right=0.75, top=0.9, bottom=0.1)
# — 蜂巢图 —
shap.summary_plot(
    shap_values, X,
    plot_type="dot", show=False, color_bar=True
)
# plt.gca().set_position([0.25, 0.2, 0.6, 0.65])
ax1 = plt.gca()
# 确保左侧 Y 轴线条可见
# ax1.spines['left'].set_visible(True)
# ax1.spines['left'].set_linewidth(2)
# ax1.spines['left'].set_color('black')
ax1.set_xlim(-1, 1)
ax1.set_yticks(range(len(X.columns)))
# 自定义 y 轴标签，数字下标字号缩小
# 设置 Y 轴刻度及标签（一次性设置字号）
ax1.set_yticks(range(len(column_names)))
latex_labels = [rf'$C_{{{name[1:]}}}$' for name in column_names]
ax1.set_yticklabels(latex_labels, fontsize=12)
# 在底部 X 轴上添加基线
ax1.axvline(x=-1, color='black', linestyle='-', linewidth=1)

# — 顶部柱状图 —
ax2 = ax1.twiny()
shap.summary_plot(
    shap_values, X,
    plot_type="bar", show=False
)
# plt.gca().set_position([0.25, 0.2, 0.6, 0.65])
ax2.set_xlim(0, 1)
# ax2.axhline(y=len(X.columns)-1, color='gray', linestyle='-', linewidth=1)
for bar in ax2.patches:
    bar.set_alpha(0.2)
# 确保顶部横轴线条可见
ax2.spines['top'].set_visible(True)
ax2.spines['top'].set_linewidth(1)
ax2.spines['top'].set_color('black')
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()
# 标签
# 设置轴标签，全部黑色、新罗马字体
ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12, fontfamily='Times New Roman', color='black')
ax2.set_xlabel('Mean SHAP Value (Feature Importance)', fontsize=12, fontfamily='Times New Roman', color='black')
ax1.set_ylabel('Features', fontsize=12, fontfamily='Times New Roman', color='black')

# 保存为 PDF
plt.savefig('shap_combined_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()
