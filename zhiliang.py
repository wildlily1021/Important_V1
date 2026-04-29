import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D  # 用于创建固定的图例项
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['DengXian']  # 设置中文字体为黑体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

class MplCanvas(FigureCanvas):
    """自定义 Matplotlib 画布，用于嵌入到 PyQt5 的 widget 中"""

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})  # 设置极坐标
        super().__init__(self.fig)
        self.setParent(parent)


def radar_drawing(data, count, widget):
    """
    绘制雷达图并嵌入到指定的 widget 中。

    参数:
    - data: np.array, 形状为 (4,)，表示当前信号的数据。
    - count: int, 信号编号 (1~4)，用于存储数据的位置。
    - widget: QWidget, 用于显示雷达图的 widget。
    """
    # 初始化存储矩阵
    if not hasattr(radar_drawing, 'signal_data'):
        radar_drawing.signal_data = np.zeros((4, 4))  # 4x4 矩阵，存储四个信号数据

    # 计算存储索引
    signal_index = (count - 1) % 4
    radar_drawing.signal_data[signal_index] = data  # 存储新数据

    # 设定雷达图标签
    # 设定雷达图标签，使用换行符实现两行标签
    radar_labels = np.array(['载噪比\n误差',
                             '载波中心\n频率误差',
                             '载波-3dB\n带宽误差',
                             'EVM'])
    nAttr = 4
    angles = np.linspace(0, 2 * np.pi, nAttr, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合曲线

    # 颜色方案（四个信号不同颜色）
    colors = ['#FF5733', '#33FF57', '#337BFF', '#FF33E3']  # 橙色、绿色、蓝色、粉色

    # 获取或创建布局
    layout = widget.layout()
    if layout is not None:
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget_item = item.widget()
            if isinstance(widget_item, MplCanvas):
                widget_item.deleteLater()

    # 创建新的 MplCanvas 实例
    sc = MplCanvas(widget)
    sc.ax.clear()

    # 设置黑色背景 + 白色网格线
    sc.ax.set_facecolor('black')
    sc.fig.patch.set_facecolor('black')
    sc.ax.spines['polar'].set_color('white')
    sc.ax.grid(color='white', linestyle='--', linewidth=0.5)

    # 固定网格范围：0~25，网格线间隔为 5
    sc.ax.set_ylim(0, 25)
    sc.ax.set_yticks([5, 10, 15, 20, 25])
    sc.ax.set_yticklabels(['5', '10', '15', '20', '25'], color='white', fontsize=10)

    # 绘制每个信号的雷达图
    for i in range(4):
        data_plot = np.concatenate((radar_drawing.signal_data[i], [radar_drawing.signal_data[i][0]]))
        sc.ax.plot(angles, data_plot, 'o-', linewidth=2, color=colors[i])  # 线条颜色
        sc.ax.fill(angles, data_plot, color=colors[i], alpha=0.3)  # 填充颜色

    # 设置中文标签字体
    font = FontProperties(fname="C:/Windows/Fonts/Deng.ttf", size=14)  # Windows 黑体
    # font = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc", size=12)  # macOS 宋体

    sc.ax.set_thetagrids((angles * 180 / np.pi)[:-1], radar_labels, fontproperties=font, color='white')

    # # 设置标题（白色）
    # sc.ax.set_title('信号质量智能评估', fontsize=20, pad=20, color='white')

    # 手动创建图例项：
    # - 当 count ≤ 4：按时间从早到晚显示为 “信号1，信号2，...，当前信号”。
    # - 当 count > 4：显示最近4条，顺序为 “当前信号，信号(count-3)，信号(count-2)，信号(count-1)”。
    active_count = int(max(0, min(4, count)))
    newest_idx = (count - 1) % 4

    legend_lines = []
    data_labels = []
    if count <= 4:
        # 历史从1到count-1，最后追加当前
        for abs_num in range(1, count):
            slot_idx = (abs_num - 1) % 4
            legend_lines.append(Line2D([0], [0], color=colors[slot_idx], lw=4))
            data_labels.append(f'信号{abs_num}')
        legend_lines.append(Line2D([0], [0], color=colors[newest_idx], lw=4))
        data_labels.append('当前信号')
    else:
        # 当前优先，其次按绝对编号从小到大（count-3, count-2, count-1）
        legend_lines.append(Line2D([0], [0], color=colors[newest_idx], lw=4))
        data_labels.append('当前信号')
        for abs_num in range(count - 3, count):
            slot_idx = (abs_num - 1) % 4
            legend_lines.append(Line2D([0], [0], color=colors[slot_idx], lw=4))
            data_labels.append(f'信号{abs_num}')

    # 调整图例的位置，向左移动
    legend = sc.ax.legend(legend_lines, data_labels, loc='upper right', labelspacing=0.1,
                          facecolor='black', edgecolor='white', bbox_to_anchor=(1.12, 1.12))

    plt.setp(legend.get_texts(), fontsize='small', fontproperties=font, color='white')

    # 添加 MplCanvas 到布局
    if layout is None:
        layout = QVBoxLayout(widget)
        widget.setLayout(layout)
    layout.addWidget(sc)

    # 设置布局拉伸比例
    layout.setStretch(0, 1)
