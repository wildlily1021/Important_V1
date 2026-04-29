import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# 导入之前定义的 plot_radar 函数
def plot_radar(data, count, widget_quality):
    """
    绘制雷达图并嵌入到指定的 widget 中。

    参数:
    - data: np.array, 四维数组，表示当前信号的数据。
    - count: int, 当前信号的计数，用于确定信号存储的位置。
    - widget_quality: QWidget, 用于显示雷达图的 widget。
    """
    # 初始化信号存储矩阵
    if not hasattr(plot_radar, 'signal_data'):
        plot_radar.signal_data = np.zeros((4, 4))  # 4x4 的零矩阵，用于存储信号数据

    # 根据 count 计算信号存储的位置
    signal_index = (count - 1) % 4  # 余数 0 对应信号4，1对应信号1，2对应信号2，3对应信号3
    plot_radar.signal_data[signal_index] = data  # 更新信号数据

    # 雷达图标签
    radar_labels = np.array(['载噪比误差', '载波中心频率误差', '载波-3dB带宽误差', 'EVM'])
    nAttr = 4
    angles = np.linspace(0, 2 * np.pi, nAttr, endpoint=False)
    data_plot = np.concatenate((plot_radar.signal_data, [plot_radar.signal_data[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # 创建图形
    fig = plt.figure(facecolor="white")
    plt.subplot(111, polar=True)
    plt.plot(angles, data_plot, 'o-', linewidth=1.5, alpha=0.2)
    plt.fill(angles, data_plot, alpha=0.25)
    plt.thetagrids((angles * 180 / np.pi)[:-1], radar_labels)
    plt.figtext(0.52, 0.95, '信号质量智能评估', ha='center', size=20)
    legend = plt.legend([f'信号{i+1}' for i in range(4)], loc=(0.94, 0.80), labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='small')
    plt.grid(True)

    # 将图形嵌入到指定的 widget 中
    canvas = FigureCanvas(fig)
    layout = widget_quality.layout()
    if layout is not None:
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
    widget_quality.setLayout(layout)
    layout.addWidget(canvas)
    canvas.draw()

# 测试窗口
class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口布局
        self.setWindowTitle('雷达图测试')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        # 添加一个按钮
        self.button = QPushButton('生成随机信号', self)
        self.button.clicked.connect(self.generate_random_signal)
        layout.addWidget(self.button)

        # 添加一个用于显示雷达图的 widget
        self.widget_quality = QWidget(self)
        layout.addWidget(self.widget_quality)

        # 设置布局
        self.setLayout(layout)

        # 初始化计数器
        self.count = 0

    def generate_random_signal(self):
        # 生成一个随机的四维数组
        random_data = np.random.rand(4)  # 生成 4 个 0~1 之间的随机数
        self.count += 1  # 更新计数器

        # 调用 plot_radar 函数
        plot_radar(random_data, self.count, self.widget_quality)

# 运行测试
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())