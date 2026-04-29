import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import xgboost as xgb  # 示例模型
from sklearn.model_selection import train_test_split
import random
from scipy.stats import truncnorm


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

# ---------------------------
# f1: 大范围均匀 + 两个小正态簇 + 中心半正态右侧（仿你原始模式）
n1_a = int(n_samples * 0.15)
n1_b = int(n_samples * 0.45)
n1_c = int(n_samples * 0.12)
f1_peak1 = np.random.uniform(0.58, 0.62, n1_a)
f1_peak2 = np.random.normal(loc=0.6, scale=0.05, size=n1_b)
f1_center_1 = half_normal_right(loc=-0.2, scale=0.03, n=n1_c)
f1_uniform = np.random.uniform(-0.25, 0.65, n_samples - len(f1_peak1) - len(f1_peak2) - len(f1_center_1))
f1 = np.concatenate([f1_peak1, f1_peak2, f1_center_1, f1_uniform])

# ---------------------------
# f2: 双峰分布（左侧半正态 + 右侧半正态） + edge uniform
n2_a = 495
n2_b = 375
f2_peak1 = half_normal_left(loc=0.3, scale=0.04, n=n2_a)
f2_peak2 = half_normal_right(loc=-0.15, scale=0.03, n=n2_b)
f2_center_1 = np.random.uniform(-0.2, 0.2, n_samples - (n2_a + n2_b))
f2_uniform = np.random.uniform(-0.25, 0.65, n_samples - len(f2_peak1) - len(f2_peak2) - len(f2_center_1))
f2 = np.concatenate([f2_peak1, f2_peak2, f2_center_1, f2_uniform])

# ---------------------------
# f3: 中心正态（偏右） + 少量左侧半正态 + uniform 填充
n3_a = 552
n3_b = 222
f3_peak1 = half_normal_right(loc=0.5, scale=0.04, n=n3_a)
f3_peak2 = half_normal_left(loc=-0.3, scale=0.015, n=n3_b)
f3_center_1 = np.random.uniform(-0.5, 0.54, n_samples - (n3_a + n3_b))
f3_uniform = np.random.uniform(-0.25, 0.65, n_samples - len(f3_peak1) - len(f3_peak2) - len(f3_center_1))
f3 = np.concatenate([f3_peak1, f3_peak2, f3_center_1, f3_uniform])

# ---------------------------
# f4: 重复 f1 风格但比例不同（两处正态 + 半正态 + uniform）
n4_a = 190
n4_b = 310
n4_c = 250
f4_peak1 = np.random.uniform(0.48, 0.52, n4_a)
f4_peak2 = half_normal_right(loc=0.4, scale=0.05, n=n4_b)
f4_center_1 = np.random.normal(loc=0.1, scale=0.03, size=n4_c)
f4_uniform = np.random.uniform(-0.25, 0.65, n_samples - len(f4_peak1) - len(f4_peak2) - len(f4_center_1))
f4 = np.concatenate([f4_peak1, f4_peak2, f4_center_1, f4_uniform])

# ---------------------------
# f5: 中心偏右半正态 + 轻微左侧正态 + uniform
n5_a = 520
n5_b = 200
f5_peak1 = half_normal_right(loc=0.2, scale=0.05, n=n5_a)
f5_peak2 = half_normal_left(loc=-0.1, scale=0.01, n=n5_b)
f5_center_1 = np.random.normal(loc=0.1, scale=0.03, size=n4_c)
f5_uniform = np.random.uniform(-0.3, 0.3, n_samples - len(f5_peak1) - len(f5_peak2) - len(f5_center_1))
f5 = np.concatenate([f5_peak1, f5_peak2, f5_center_1, f5_uniform])

# ---------------------------
# f6: 混合（小 uniform 峰 + 半正态左 + 正态中心） —— 保持与你提供的模式一致
n8_a = 200
n8_b = 300
n8_c = 150
f6_peak1 = np.random.normal(loc=-0.6, scale=0.03, size=n8_a)
f6_peak2 = np.random.normal(loc=0.45, scale=0.04, size=n8_b)
f6_center_1 = half_normal_right(loc=-0.05, scale=0.02, n=n8_c)
f6_uniform = np.random.uniform(-0.5, 0.6, n_samples - len(f6_peak1) - len(f6_peak2) - len(f6_center_1))
f6 = np.concatenate([f6_peak1, f6_peak2, f6_center_1, f6_uniform])

# ---------------------------
# f7: 三段式：小 uniform 峰 + 右侧半正态 + 中心正态（仿原）
n7_a = 150
n7_b = 590
n7_c = 100
f7_peak1 = np.random.uniform(0.32, 0.39, n7_a)
f7_peak2 = half_normal_right(loc=0.39, scale=0.03, n=n7_b)
f7_center_1 = np.random.normal(loc=0.11, scale=0.03, size=n7_c)
f7_uniform = np.random.uniform(-0.1, 0.35, n_samples - len(f7_peak1) - len(f7_peak2) - len(f7_center_1))
f7 = np.concatenate([f7_peak1, f7_peak2, f7_center_1, f7_uniform])

# f9: 稀疏极值 + 中心小正态 + uniform 填充（参考你最后一个例子）
n9_a = 275
n9_b = 605
f9_peak1 = np.random.normal(loc=-0.03, scale=0.04, size=n9_a)
f9_peak2 = half_normal_right(loc=-0.4, scale=0.02, n=n9_b)
f9_center_1 = np.random.uniform(-0.44, 0, n_samples - (n9_a + n9_b))
f9_uniform = np.random.uniform(-0.44, 0, n_samples - len(f9_peak1) - len(f9_peak2) - len(f9_center_1))
f9 = np.concatenate([f9_peak1, f9_peak2, f9_center_1, f9_uniform])

# 设置列名及其数学标签（如 C_{2,0}）
column_names = ['C20', 'C21', 'C40', 'C41', 'C42', 'C60', 'C61', 'C62']


# 组装 DataFrame
X = pd.DataFrame({
    name: np.round(feat, 8) for name, feat in zip(column_names, [f1, f2, f3, f4, f5, f6, f7, f9])
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
    # np.round(-0.98 * f8, 8),
    np.round(f9, 8),
    # np.round(f10, 8),
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

shap_values = np.array([
    np.round(1.2 * f1, 8),
    np.round(1.6 * f2, 8),
    np.round(-1.4 * f3, 8),
    np.round(2 * f4, 8),
    np.round(f5, 8),
    np.round(-0.95 * 1.8 * f6, 8),
    np.round(f7, 8),
    # np.round(-0.98 * 0.8 * f8, 8),
    np.round(f9, 8),
    # np.round(f10, 8),
])
shap_values = shap_values.T
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
plt.savefig('shap_combined_plot_1level.pdf', format='pdf', bbox_inches='tight')
plt.show()
