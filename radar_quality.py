import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from zhiliang import radar_drawing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

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
        random_data = 25 * (np.random.rand(4))  # 生成 4 个 0~1 之间的随机数
        # Center_Fre_quality = 25 - ((abs(set_Center - true_Center) / set_Center) * 100 - 1)
        # -3dB_band_quality = 25 - ((abs(set_band - true_band) / set_band) * 50 - 1)
        # SNR_quality = 25 - ((abs(set_SNR - true_SNR) / set_SNR) * 100 - 1)
        # if Modulation == 1:
        #     EVM_quality = 25 - ((true_EVM / 17.5) * 5 - 1) * 5
        # elif Modulation == 2:
        #     EVM_quality = 25 - ((true_EVM / 12) * 5 - 1) * 5
        # elif Modulation == 3:
        #     EVM_quality = 25 - ((true_EVM / 12.5) * 5 - 1) * 5
        # elif Modulation == 4:
        #     EVM_quality = 25 - ((true_EVM / 8) * 5 - 1) * 5
        # else:
        #     EVM_quality = 15
        # center_Fre_quality = 25 - ((abs(set_Center - true_Center) / set_Center) * 100 - 1)

        self.count += 1  # 更新计数器
        factors = np.array([1,1,1,1])
        radom_data_final = random_data * factors
        # 调用 radar_drawing 函数
        radar_drawing(radom_data_final, self.count, self.widget_quality)

# 运行测试
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())