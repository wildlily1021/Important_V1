# caller.py
# from image_processing import process_image
from matplotlib import pyplot as plt
from scipy.stats import stats

from EVM_Ana import signal_ideal
from image_processing_41 import process_image
#处理STFT图像
import cv2
import sys
from PyQt5 import QtWidgets
import numpy as np
from PyQt5.QtWidgets import QToolTip, QMessageBox, QFileDialog, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, \
    QDesktopWidget
from PyQt5.QtGui import QFont, QPixmap, QBrush, QColor, QPainter, QPolygon, QIcon

from photo_save import get_amplitude_from_rgb
from win_v1 import Ui_Signal_analysis
from signal_photo_save import signal_create, signal_read
from state_process import load_excel_signal
from visualization import Constellation_dreawing, FFT_dreawing, STFT_dreawing, Eyediagram_dreawing, FFT_dreawing_dynamic
from PyQt5.QtCore import Qt, QPoint
from test import bandwidth_predict
from EVM_Ana import signal_ideal
from image_processing_41 import process_image
from zhiliang import radar_drawing
from parameter_tune import flag1_tune, flag2_tune, flag3_tune
import time
from PIL import Image
from unet import Unet
import os
import matplotlib
from stft_pipeline import (
    load_stft_metadata,
    pixel_height_to_bandwidth_hz,
    pixel_to_frequency_hz,
    resolve_recognition_image_path,
)

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

# from bandwith_model import load_and_predict

global Fs_input, Fc_input, Rs_input, SNR_input
#Fs采样频率
#Fc载波频率
#Rs符号速率
#SNR载噪比
global count
count = 0

def safe_detect(unet_obj, image):
#处理UNet模型返回值的兼容性函数
    """
    兼容调用 unet.detect_image 的返回值：
    - 如果返回 (r_image, rect_height, flag_mask) 则直接返回
    - 如果返回 (r_image, rect_height) 则补 None 作为 flag_mask 返回
    - 否则返回 (result, None, None)
    """
    _res = unet_obj.detect_image(image)
    if isinstance(_res, (tuple, list)):
        if len(_res) == 3:
            return _res[0], _res[1], _res[2]
        elif len(_res) == 2:
            return _res[0], _res[1], None
    return _res, None, None


def map_center_frequency_from_stft(image_path, image_height, center_frequency_output, fs_hz):
    stft_metadata = load_stft_metadata(image_path)
    if stft_metadata:
        return (
            pixel_to_frequency_hz(center_frequency_output, stft_metadata, image_height_px=image_height),
            stft_metadata,
        )
    fallback = fs_hz / image_height * (image_height - center_frequency_output) - fs_hz / 2
    return fallback, None


def map_bandwidth_from_stft(rect_height, image_height, image_path, stft_metadata=None):
    stft_metadata = stft_metadata or load_stft_metadata(image_path)
    if stft_metadata:
        return pixel_height_to_bandwidth_hz(rect_height, stft_metadata, image_height_px=image_height) / 1e9
    return rect_height / image_height * 10

class Signal_analysis_form(QtWidgets.QWidget, Ui_Signal_analysis):
    def __init__(self):
        super(Signal_analysis_form, self).__init__()
        self.setupUi(self)
        #初始化主窗口：集成QtWidgets.QWidget和Ui_Signal_analysis
        # Wrap large designer content with a scroll area for smaller displays
        #setupUi函数是Qt Designer生成的UI文件的初始化函数，用于将UI文件中的组件加载到窗口中
        self.scrollArea = QtWidgets.QScrollArea(self) #创建一个滚动区域，用于显示UI文件中的组件
        self.scrollArea.setWidgetResizable(False)     #设置滚动区域不可调整大小：若设置为True，则滚动区域可以调整大小
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded) #设置水平滚动条的显示策略：若设置为Qt.ScrollBarAsNeeded，则水平滚动条只有在内容超出窗口宽度时才会显示
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded) #设置垂直滚动条的显示策略：若设置为Qt.ScrollBarAsNeeded，则垂直滚动条只有在内容超出窗口高度时才会显示
        # Move the existing tabWidget into the scroll area
        self.tabWidget.setParent(None) #将tabWidget设置为滚动区域的父亲，以便滚动区域可以显示tabWidget中的内容
        # Keep original designed size as content size so scrollbars appear when needed
        self.tabWidget.setMinimumSize(self.tabWidget.size()) #设置tabWidget的最小尺寸为tabWidget的尺寸
        self.scrollArea.setWidget(self.tabWidget)
        self.scrollArea.setGeometry(self.rect()) #设置滚动区域的几何形状为窗口的几何形状

        # Ensure we only fit and center once at first show
        self._fittedOnShow = False #初始化一个标志，用于确保窗口只被适配和居中一次

        # Remove the extra tab labeled "信号质量评估" and hide the tab bar labels entirely
        try:
            # Remove tab_2 if present
            if hasattr(self, 'tab_2'):
                idx = self.tabWidget.indexOf(self.tab_2)
                if idx != -1:
                    self.tabWidget.removeTab(idx)
            # Hide tab bar so labels are not visible
            # if self.tabWidget.tabBar() is not None:
            #     self.tabWidget.tabBar().setVisible(False)
        except Exception:
            pass

        # 设置TabWidget的背景颜色
        self.tabWidget.setStyleSheet("QWidget { background-color: #2d2d30; }")

        # 初始化全局变量 file_path
        self.file_path = None  # 初始化为 None 或者其他默认值

        # 设置提示信息
        self.center()  # 将窗口居中
        self.Fc_txt.setToolTip("1G ~ 50G")
        self.Fs_txt.setToolTip("0.25 ~ 0.5*Fs")
        self.Rs_txt.setToolTip("0.25 ~ 0.5*Fs")
        self.SNR_txt.setToolTip("-13dB ~ ∞")
        self.progressBar.hide()
        # Initialize attributes for tracking progress
        self.total_steps = 10
        self.current_step = 0
        self.progress_max = 100
        # self.count = 0  # 初始化计数变量

        # 设置提示信息的字体和大小（可选）
        QToolTip.setFont(QFont('Times New Roman', 9))

        # 初始化QGraphicsView
        self.scene = QGraphicsScene()
        self.STFT_View.setScene(self.scene)

        # 获取 QGraphicsView 对象
        self.graphics_view = self.findChild(QGraphicsView, 'STFT_View')

        # 设置初始边框为透明
        self.graphics_view.setStyleSheet('border: none;')

        # 创建 QGraphicsScene 对象并设置背景颜色
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor(249, 249, 249)))  # 设置背景颜色为 RGB (249, 249, 249)

        # 将 QGraphicsScene 设置到 QGraphicsView 中
        self.graphics_view.setScene(self.scene)

        # 获取 QGraphicsView 对象
        self.graphics_view = self.findChild(QGraphicsView, 'Cyclic_View')

        # 设置初始边框为透明
        self.graphics_view.setStyleSheet('border: none;')

        # 创建 QGraphicsScene 对象并设置背景颜色
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor(249, 249, 249)))  # 设置背景颜色为 RGB (249, 249, 249)

        # 将 QGraphicsScene 设置到 QGraphicsView 中
        self.graphics_view.setScene(self.scene)

        # 初始化控件状态
        self.Signal_file_Btn.setVisible(False)  # 初始不可视
        self.Signal_file_Btn.setEnabled(False)  # 初始不可用
        self.Cyclic_View.setVisible(False)  # 初始不可视
        self.STFT_View.setVisible(False)  # 初始不可视
        # 开始评估按钮（在 UI 中定义），初始隐藏并禁用，等待外部文件模式时启用
        try:
            self.start_eval_btn.setVisible(False)
            self.start_eval_btn.setEnabled(False)
        except Exception:
            # 如果 UI 中尚未创建 start_eval_btn，则忽略
            pass

        # 存储外部文件模式下待评估的数据（当用户点击“开始评估”时再展示质量评估）
        self._pending_quality_data = None
        # 开始计算按钮（在 UI 中定义），初始隐藏并禁用，等待外部文件模式时启用
        self._pending_calculation_data = None
        self._confirmed_Fs_input = None
        self._confirmed_Modulation_input = None
        self._file_sample_rate = None
        self._fs_input_source = None
        try:
            if not hasattr(self, 'start_input_btn') and hasattr(self, 'start_eval_btn_2'):
                self.start_input_btn = self.start_eval_btn_2
                self.start_input_btn.clicked.connect(self.start_input)
        except Exception:
            pass

        # 记录原始几何，用于最大化/恢复来回切换
        try:
            self._orig_geos = {
                'groupBox_signa_lanalysisi': self.groupBox_signa_lanalysisi.geometry(),
                'groupBox': self.groupBox.geometry(),
                'groupBox_5': self.groupBox_5.geometry(),
                'groupBox_parameter': self.groupBox_parameter.geometry(),
                'groupBox_star': self.groupBox_star.geometry(),
                'groupBox_signalquality': self.groupBox_signalquality.geometry(),
                'groupBox_signal_test': self.groupBox_signal_test.geometry(),
                'STFT_View': self.STFT_View.geometry(),
                'Cyclic_View': self.Cyclic_View.geometry(),
                'widget_FFT': self.widget_FFT.geometry(),
                'widget_STFT': self.widget_STFT.geometry(),
                'widget_Constellation': self.widget_Constellation.geometry(),
                'widget_quality': self.widget_quality.geometry(),
            }
        except Exception:
            self._orig_geos = {}

        # 设置不同QGroupBox的样式
        self.groupBox.setObjectName("groupbox1")
        self.groupBox_5.setObjectName("groupbox2")
        self.groupBox_signa_lanalysisi.setObjectName("groupbox3")
        self.groupBox_parameter.setObjectName("groupbox4")
        self.groupBox_star.setObjectName("groupbox5")
        self.groupBox_signalquality.setObjectName("groupbox6")
        self.groupBox_signal_test.setObjectName("groupbox7")

        # self.setGroupBoxStyle()

        # 设置不同QGroupBox的样式
        self.Bandwidth_ES.setObjectName("textedit1")
        self.EVM_TXT.setObjectName("textedit2")
        self.Fc_ES.setObjectName("textedit3")
        self.PARP_TXT.setObjectName("textedit4")
        self.Rs_ES.setObjectName("textedit5")
        self.SNR_ES.setObjectName("textedit6")

        self.label10_1.setObjectName("label10_1")
        self.label10_3.setObjectName("label10_3")
        self.label10_5.setObjectName("label10_5")
        self.label5_1.setObjectName("label5_1")
        self.label5_2.setObjectName("label5_2")
        self.label6_1.setObjectName("label6_1")
        self.label6_2.setObjectName("label6_2")
        self.label7_1.setObjectName("label7_1")
        self.label7_2.setObjectName("label7_2")
        self.label7_4.setObjectName("label7_4")
        self.label7_5.setObjectName("label7_5")

        # 设置标签和文本编辑器样式
        self.setStyle()
        # Add Prev/Next page buttons for the parameter analysis stacked widget.
        # Create them as children of groupBox_parameter but defer precise placement
        # to resizeEvent so they won't be hidden by later geometry adjustments.
        try:
            # Create small triangular icon buttons for prev/next (drawn dynamically so no external assets needed)
            def _make_triangle_icon(direction='left', size=20, color=QColor(0, 0, 0)):
                pix = QPixmap(size, size)
                pix.fill(Qt.transparent)
                painter = QPainter(pix)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)
                if direction == 'left':
                    pts = [QPoint(int(size * 0.7), int(size * 0.2)),
                           QPoint(int(size * 0.7), int(size * 0.8)),
                           QPoint(int(size * 0.2), int(size * 0.5))]
                else:
                    pts = [QPoint(int(size * 0.3), int(size * 0.2)),
                           QPoint(int(size * 0.3), int(size * 0.8)),
                           QPoint(int(size * 0.8), int(size * 0.5))]
                poly = QPolygon(pts)
                painter.drawPolygon(poly)
                painter.end()
                return QIcon(pix)

            self.prev_page_btn = QtWidgets.QPushButton("", self.groupBox_parameter)
            self.prev_page_btn.setToolTip("上一页")
            self.prev_page_btn.setFixedSize(24, 24)
            prev_icon = _make_triangle_icon('left', size=20, color=QColor(0, 0, 0))
            self.prev_page_btn.setIcon(prev_icon)
            self.prev_page_btn.setIconSize(self.prev_page_btn.size() * 0.7)
            self.prev_page_btn.setFlat(True)
            self.prev_page_btn.setStyleSheet("background: transparent; border: none;")

            self.next_page_btn = QtWidgets.QPushButton("", self.groupBox_parameter)
            self.next_page_btn.setToolTip("下一页")
            self.next_page_btn.setFixedSize(24, 24)
            next_icon = _make_triangle_icon('right', size=20, color=QColor(0, 0, 0))
            self.next_page_btn.setIcon(next_icon)
            self.next_page_btn.setIconSize(self.next_page_btn.size() * 0.7)
            self.next_page_btn.setFlat(True)
            self.next_page_btn.setStyleSheet("background: transparent; border: none;")

            # initial placement in groupBox title area (top-right), will be refined by showEvent
            try:
                gb_w = self.groupBox_parameter.width()
                self.prev_page_btn.setGeometry(gb_w - 58, 5, 24, 24)
                self.next_page_btn.setGeometry(gb_w - 30, 5, 24, 24)
            except Exception:
                self.prev_page_btn.setGeometry(249, 5, 24, 24)
                self.next_page_btn.setGeometry(277, 5, 24, 24)

            # Connect signals
            self.prev_page_btn.clicked.connect(self._go_prev_parameter_page)
            self.next_page_btn.clicked.connect(self._go_next_parameter_page)

            # Ensure visible
            self.prev_page_btn.show()
            self.next_page_btn.show()
        except Exception:
            pass

    def showEvent(self, event):
        super(Signal_analysis_form, self).showEvent(event)
        # On first show, clamp window to available screen and center
        if not self._fittedOnShow:
            try:
                desktop = QDesktopWidget()
                # Use the screen where the window appears
                available_geo = desktop.availableGeometry(self)
                # Leave a small margin so OS taskbars/frames are respected
                margin_w = 40
                margin_h = 60
                max_w = max(800, available_geo.width() - margin_w)
                max_h = max(600, available_geo.height() - margin_h)
                new_w = min(self.width(), max_w)
                new_h = min(self.height(), max_h)
                self.resize(new_w, new_h)
                # Center after resizing
                self.center()
            except Exception:
                # Fallback to existing center logic
                self.center()
            # Place buttons in groupBox title area (top-right), mirroring Qt Designer arrows
            try:
                if hasattr(self, 'groupBox_parameter') and hasattr(self, 'prev_page_btn') and hasattr(self, 'next_page_btn'):
                    gb_w = self.groupBox_parameter.width()
                    self.prev_page_btn.setGeometry(gb_w - 58, 5, 24, 24)
                    self.next_page_btn.setGeometry(gb_w - 30, 5, 24, 24)
                    self.prev_page_btn.raise_()
                    self.next_page_btn.raise_()
            except Exception:
                pass
            self._fittedOnShow = True

    def resizeEvent(self, event):
        # Ensure scroll area always fills the window
        if hasattr(self, 'scrollArea') and self.scrollArea is not None:
            self.scrollArea.setGeometry(self.rect())
        # 仅在最大化时进行左右方向的自适应调整
        try:
            is_max = self.isMaximized() or (self.windowState() & Qt.WindowMaximized) == Qt.WindowMaximized
        except Exception:
            is_max = False

        if is_max:
            try:
                # 目标总宽度基于 tabWidget 所在内容区
                total_w = max(self.tabWidget.width(), self.width())
                # 固定左列宽度保持不变
                right_geo = self._orig_geos.get('groupBox_parameter')
                center_geo = self._orig_geos.get('groupBox_signa_lanalysisi')
                if left_geo is not None and right_geo is not None and center_geo is not None:
                    left_w = left_geo.width()
                    right_w = right_geo.width()
                    # 中列起始 x 取原始的 x
                    center_x = center_geo.x()
                    # 右列锚定到最右侧
                    right_x = total_w - right_w
                    # 中列宽度 = 右列左边缘 - 中列 x - 微调边距(10)
                    center_w = max(200, right_x - center_x - 10)

                    # 应用到三行对应组
                    g1 = self.groupBox_signa_lanalysisi.geometry()
                    self.groupBox_signa_lanalysisi.setGeometry(center_x, g1.y(), center_w, g1.height())
                    g2 = self.groupBox.geometry()
                    self.groupBox.setGeometry(center_x, g2.y(), center_w, g2.height())
                    g3 = self.groupBox_5.geometry()
                    self.groupBox_5.setGeometry(center_x, g3.y(), center_w, g3.height())

                    # 右列三块对齐到最右侧
                    p1 = self.groupBox_parameter.geometry()
                    self.groupBox_parameter.setGeometry(right_x, p1.y(), right_w, p1.height())
                    p2 = self.groupBox_star.geometry()
                    self.groupBox_star.setGeometry(right_x, p2.y(), right_w, p2.height())
                    p3 = self.groupBox_signalquality.geometry()
                    self.groupBox_signalquality.setGeometry(right_x, p3.y(), right_w, p3.height())

                    # 调整参数估计组内两个视图按 1:1 宽度分配，保持原始边距与间距
                    stft_geo = self._orig_geos.get('STFT_View')
                    cyc_geo = self._orig_geos.get('Cyclic_View')
                    if stft_geo is not None and cyc_geo is not None:
                        left_margin = stft_geo.x()
                        top_margin = stft_geo.y()
                        inter_gap = cyc_geo.x() - (stft_geo.x() + stft_geo.width())
                        right_margin = max(10, left_margin)  # 近似取与左边距一致
                        usable_w = max(0, center_w - left_margin - right_margin - inter_gap)
                        each_w = max(50, usable_w // 2)
                        height_avail = stft_geo.height()  # 保持高度不变
                        self.STFT_View.setGeometry(left_margin, top_margin, each_w, height_avail)
                        self.Cyclic_View.setGeometry(left_margin + each_w + inter_gap, top_margin, each_w, height_avail)

                    # 调整中列另外两块内部绘图区域宽度（高度不变）
                    wf_geo = self._orig_geos.get('widget_FFT')
                    if wf_geo is not None:
                        # widget_FFT 左右边距按原始 (x 与 group 宽度差)
                        inner_left = wf_geo.x()
                        inner_right = wf_geo.x()  # 假设对称
                        inner_w = max(50, center_w - inner_left - inner_right)
                        self.widget_FFT.setGeometry(inner_left, wf_geo.y(), inner_w, wf_geo.height())
                    ws_geo = self._orig_geos.get('widget_STFT')
                    if ws_geo is not None:
                        inner_left = ws_geo.x()
                        inner_right = ws_geo.x()
                        inner_w = max(50, center_w - inner_left - inner_right)
                        self.widget_STFT.setGeometry(inner_left, ws_geo.y(), inner_w, ws_geo.height())
            except Exception:
                pass
        else:
            # 恢复原始几何
            try:
                if getattr(self, '_orig_geos', None):
                    self.groupBox_signa_lanalysisi.setGeometry(self._orig_geos['groupBox_signa_lanalysisi'])
                    self.groupBox.setGeometry(self._orig_geos['groupBox'])
                    self.groupBox_5.setGeometry(self._orig_geos['groupBox_5'])
                    self.groupBox_parameter.setGeometry(self._orig_geos['groupBox_parameter'])
                    self.groupBox_star.setGeometry(self._orig_geos['groupBox_star'])
                    self.groupBox_signalquality.setGeometry(self._orig_geos['groupBox_signalquality'])
                    self.groupBox_signal_test.setGeometry(self._orig_geos['groupBox_signal_test'])
                    self.STFT_View.setGeometry(self._orig_geos['STFT_View'])
                    self.Cyclic_View.setGeometry(self._orig_geos['Cyclic_View'])
                    self.widget_FFT.setGeometry(self._orig_geos['widget_FFT'])
                    self.widget_STFT.setGeometry(self._orig_geos['widget_STFT'])
                    self.widget_Constellation.setGeometry(self._orig_geos['widget_Constellation'])
                    self.widget_quality.setGeometry(self._orig_geos['widget_quality'])
            except Exception:
                pass
        # Reposition Prev/Next buttons in groupBox title area (top-right), mirroring Qt Designer arrows
        try:
            if hasattr(self, 'groupBox_parameter') and hasattr(self, 'prev_page_btn') and hasattr(self, 'next_page_btn'):
                gb_w = self.groupBox_parameter.width()
                self.prev_page_btn.setGeometry(gb_w - 58, 5, 24, 24)
                self.next_page_btn.setGeometry(gb_w - 30, 5, 24, 24)
                self.prev_page_btn.show()
                self.next_page_btn.show()
                self.prev_page_btn.raise_()
                self.next_page_btn.raise_()
        except Exception:
            pass
        super(Signal_analysis_form, self).resizeEvent(event)

    # def setGroupBoxStyle(self):
    #     # 使用十六进制颜色代码设置不同QGroupBox的样式
    #     self.setStyleSheet("""
    #         QGroupBox#groupbox1 {
    #             background-color: #000000;  # 黑色背景
    #             border: 2px solid red;  # 边框颜色为橙色
    #             color: white;
    #         }
    #         QGroupBox#groupbox2 {
    #             background-color: #000000;  # 黑色背景
    #             border: 10px solid #ffff00;  # 边框颜色为绿色
    #             color: white;
    #         }
    #         QGroupBox#groupbox3 {
    #             background-color: #000000;  # 黑色背景
    #             border: 2px solid #00caca;  # 边框颜色为蓝色
    #             color: white;
    #         }
    #         QGroupBox#groupbox4 {
    #             background-color: #000000;  # 黑色背景
    #             border: 2px solid #33ff00;  # 边框颜色为蓝色
    #             color: white;
    #         }
    #         QGroupBox#groupbox5 {
    #             background-color: #000000;  # 黑色背景
    #             border: 2px solid #6fe9fe;  # 边框颜色为蓝色
    #             color: white;
    #         }
    #         QGroupBox#groupbox6 {
    #             background-color: #000000;  # 黑色背景
    #             border: 2px solid #ff6020;  # 边框颜色为蓝色
    #             color: white;
    #         }
    #         QGroupBox#groupbox7 {
    #             background-color: #000000;  # 黑色背景
    #             border: 2px solid #da07da;  # 边框颜色为蓝色
    #             color: white;
    #         }
    #     """)
    #
    # def setWidgetsStyle(self):
    #     # 设置全局样式
    #     self.setStyleSheet("""
    #         /* 全局样式：设置通用控件样式 */
    #         QWidget {
    #             font-size: 12pt;
    #             font-family: Segoe UI;
    #         }
    #
    #         QLabel {
    #             color: white;
    #         }
    #
    #         QTextEdit {
    #             background-color: #2b2a34;
    #             color: white;
    #             border: 1px solid #444;
    #             border-radius: 5px;
    #             padding: 5px;
    #         }
    #
    #         QPushButton {
    #             background-color: #2b2a34;  # 所有QPushButton背景为#ff6020
    #             color: white;  # 所有QPushButton字体为白色
    #             border-radius: 5px;  # 设置圆角
    #             padding: 8px 15px;  # 设置内边距
    #         }
    #
    #         /* 针对特定控件的个性化样式 */
    #         QLabel#label10_1 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label10_3 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label10_5 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label5_1 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label5_2 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label6_1 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label6_2 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label7_1 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label7_2 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label7_4 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #          QLabel#label7_5 {
    #             color: #ff0067;  # 将my_label的字体颜色设置为#ff6020
    #             font-size: 12pt;  # 设置my_label的字体大小
    #         }
    #
    #         QTextEdit#textedit1 {
    #             color: #ff0067;  # 所有QTextEdit字体为白色
    #             background-color: #000000;  # 将my_textedit的背景色修改为#333
    #             border: 2px solid #000000;  # 设置边框颜色为#ff6020
    #             padding: 10px;  # 设置内边距
    #         }
    #         QTextEdit#textedit2 {
    #             color: #ff0067;  # 所有QTextEdit字体为白色
    #             background-color: #000000;  # 将my_textedit的背景色修改为#333
    #             border: 2px solid #000000;  # 设置边框颜色为#ff6020
    #             padding: 10px;  # 设置内边距
    #         }
    #         QTextEdit#textedit3 {
    #             color: #ff0067;  # 所有QTextEdit字体为白色
    #             background-color: #000000;  # 将my_textedit的背景色修改为#333
    #             border: 2px solid #000000;  # 设置边框颜色为#ff6020
    #             padding: 10px;  # 设置内边距
    #         }
    #         QTextEdit#textedit4 {
    #             color: #ff0067;  # 所有QTextEdit字体为白色
    #             background-color: #000000;  # 将my_textedit的背景色修改为#333
    #             border: 2px solid #000000;  # 设置边框颜色为#ff6020
    #             padding: 10px;  # 设置内边距
    #         }
    #         QTextEdit#textedit5 {
    #             color: #ff0067;  # 所有QTextEdit字体为白色
    #             background-color: #000000;  # 将my_textedit的背景色修改为#333
    #             border: 2px solid #000000;  # 设置边框颜色为#ff6020
    #             padding: 10px;  # 设置内边距
    #         }
    #         QTextEdit#textedit6 {
    #             color: #ff0067;  # 所有QTextEdit字体为白色
    #             background-color: #000000;  # 将my_textedit的背景色修改为#333
    #             border: 2px solid #000000;  # 设置边框颜色为#ff6020
    #             padding: 10px;  # 设置内边距
    #         }
    #
    #         # QPushButton#my_button {
    #         #     background-color: #444;  # 设置my_button的背景颜色为#444
    #         #     border: 1px solid #ff6020;  # 设置边框颜色为#ff6020
    #         # }
    #     """)

    def setStyle(self):
        self.setStyleSheet("""
            QGroupBox#groupbox1 {
                background-color: #000000;
                border: 5px solid #ffff00;
                color: white;
            }
            QGroupBox#groupbox2 {
                background-color: #000000;
                border: 5px solid #00caca;
                color: white;
            }
            QGroupBox#groupbox3 {
                background-color: #000000;
                border: 5px solid #33ff00;
                color: white;
            }
            QGroupBox#groupbox4 {
                background-color: #000000;
                border: 5px solid #6fe9fe;
                color: white;
            }
            QGroupBox#groupbox5 {
                background-color: #000000;
                border: 5px solid #ff6020;
                color: white;
            }
            QGroupBox#groupbox6 {
                background-color: #000000;
                border: 5px solid #ff0067;
                color: white;
            }
            QGroupBox#groupbox7 {
                background-color: #000000;
                border: 5px solid red;
                color: white;
            }

            QLabel {
                color: white;
            }

            QComboBox{
                color: white;
            }
                QComboBox QAbstractItemView {
                color: white;
                background-color: black;
                selection-background-color: grey;
                selection-color: black;
            }

            QTextEdit {
                background-color: #2b2a34;
                color: white; 
                border: 1px solid #444;
                border-radius: 5px;
                padding: 1px;
                font-family: "DengXian";
            }

            QPushButton {
                background-color: #2b2a34;
                color: white;
                border-radius: 1px;
                padding: 8px 15px;
            }

            QLabel#label10_1, QLabel#label10_3, QLabel#label10_5,
            QLabel#label5_1, QLabel#label5_2, QLabel#label6_1, QLabel#label6_2,
            QLabel#label7_1, QLabel#label7_2, QLabel#label7_4, QLabel#label7_5 {
                color: #6fe9fe;
            }

            QTextEdit#textedit1, QTextEdit#textedit2, QTextEdit#textedit3,
            QTextEdit#textedit4, QTextEdit#textedit5, QTextEdit#textedit6 {
                color: #6fe9fe;
                background-color: #000000;
                border: 1px solid #000000;
                padding: 1px;
                font-family: "DengXian";
            }
        """)

    def center(self):
        # 获取主屏幕的几何信息
        screen_geometry = QDesktopWidget().availableGeometry()
        window_geometry = self.frameGeometry()

        # 计算窗口应该放置的位置 (屏幕中心)
        window_geometry.moveCenter(screen_geometry.center())

        # 移动窗口到该位置
        self.move(window_geometry.topLeft())

    def update_progress(self):
        # 更新进度条的值
        self.current_step += 1
        progress = min(self.current_step * 10, 100)  # 每步增加 10%
        self.progressBar.setValue(progress)

        # 检查任务是否完成
        if self.current_step >= self.total_steps:
            self.progressBar.setValue(self.progress_max)

    # Navigation helpers for parameter analysis stacked widget
    def _go_prev_parameter_page(self):
        try:
            idx = self.stackedWidget.currentIndex()
            count = self.stackedWidget.count() if hasattr(self, 'stackedWidget') else 0
            if count > 0:
                self.stackedWidget.setCurrentIndex((idx - 1) % count)
        except Exception:
            pass

    def _go_next_parameter_page(self):
        try: 
            idx = self.stackedWidget.currentIndex()
            count = self.stackedWidget.count() if hasattr(self, 'stackedWidget') else 0
            if count > 0:
                self.stackedWidget.setCurrentIndex((idx + 1) % count)
        except Exception:
            pass

    # 文件占用报错
    def file_error(self, filed_flag):
        if filed_flag == 1:
            QMessageBox.warning(self, "文件占用", "目标文件被占用，请保存并关闭！")
            return

    # combobox的读取部分
    def Fs_level_combo(self):
        selection = self.Fs_level.currentText()
        # 根据选择返回对应的因子
        if selection == "G":
            return 1e9
        elif selection == "M":
            return 1e6
        elif selection == "K":
            return 1e3
        else:
            return 1  # 默认因子为1

    def Fc_level_combo(self):
        selection = self.Fc_level.currentText()
        # 根据选择返回对应的因子
        if selection == "G":
            return 1e9
        elif selection == "M":
            return 1e6
        elif selection == "K":
            return 1e3
        else:
            return 1  # 默认因子为1

    def Rs_level_combo(self):
        selection = self.Rs_level.currentText()
        # 根据选择返回对应的因子
        if selection == "G":
            return 1e9
        elif selection == "M":
            return 1e6
        elif selection == "K":
            return 1e3
        else:
            return 1  # 默认因子为1

    def modulation_level_combo(self):
        selection = self.modulation_level.currentText()
        # 根据选择返回对应的因子
        if selection == "QPSK":
            return 1
        elif selection == "8PSK":
            return 2
        elif selection == "16QAM":
            return 3
        elif selection == "64QAM":
            return 4
        else:
            return 100  # 默认因子为1

    def _read_fs_from_ui(self):
        text = self.Fs_txt.toPlainText().strip()
        if text == "":
            raise ValueError("采样频率未输入")
        return float(text) * self.Fs_level_combo()

    def _active_fs_input(self):
        confirmed_fs = getattr(self, '_confirmed_Fs_input', None)
        if confirmed_fs is not None:
            return confirmed_fs
        return self._read_fs_from_ui()

    def _active_modulation_input(self):
        confirmed_modulation = getattr(self, '_confirmed_Modulation_input', None)
        if confirmed_modulation is not None:
            return confirmed_modulation
        return self.modulation_level_combo()

    def _set_fs_controls_from_hz(self, fs_value):
        if fs_value is None or not np.isfinite(fs_value) or fs_value <= 0:
            return

        for unit, factor in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
            scaled = fs_value / factor
            if abs(scaled) >= 1:
                self.Fs_txt.setPlainText(f"{scaled:.12g}")
                index = self.Fs_level.findText(unit)
                if index >= 0:
                    self.Fs_level.setCurrentIndex(index)
                return

        self.Fs_txt.setPlainText(f"{fs_value:.12g}")
        if self.Fs_level.count() > 3:
            self.Fs_level.setCurrentIndex(3)

    def confirm_signalquality_input(self, checked=False):
        try:
            self._confirmed_Fs_input = self._read_fs_from_ui()
            self._confirmed_Modulation_input = self.modulation_level_combo()
            self._fs_input_source = 'ui'
            print(
                "Confirmed signal input: "
                f"Fs={self._confirmed_Fs_input:.6e} Hz, "
                f"Modulation={self._confirmed_Modulation_input}"
            )
        except Exception as e:
            QMessageBox.warning(self, "未输入参数", f"确认参数失败：{e}")
            return

    def start_input(self):
        return self.confirm_signalquality_input()

    # FFT图像放大后还原模块
    def reset_view(self):
        # 还原视图
        self.canvas.ax.set_xlim([self.xf[0], self.xf[-1]])  # 恢复到全频率范围
        self.canvas.ax.set_ylim([min(self.yf), max(self.yf)])  # 恢复到全幅度范围
        self.canvas.draw()

    def save_image(self, image, path):
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"Error saving image: {e}")

    global count, file_path

    # 文件读取模块
    def file_signal_get(self):
        # 打开文件选择对话框
        options = QFileDialog.Options()
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select Signal File", "",
                                                        "Signal Files (*.txt *.csv *.xlsx *.xls);;Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)",
                                                        options=options)

        if self.file_path:
            self.File_path_Btn.setVisible(False)  # 设置为可视
            self.File_path_Btn.setEnabled(False)  # 设置为可用
            self.Signal_file_Btn.setVisible(True)  # 设置为不可视
            self.Signal_file_Btn.setEnabled(True)  # 设置为不可用
            self.signal_test.setVisible(False)  # 设置为不可视
            self.signal_test.setEnabled(False)  # 设置为不可用
            # 选择文件后，清除待评估数据并隐藏开始评估按钮（需要在读取信号并显示结果后才启用）
            self._pending_quality_data = None
            self.start_eval_btn.setVisible(False)
            self.start_eval_btn.setEnabled(False)
            self._file_sample_rate = None
            if getattr(self, '_fs_input_source', None) == 'file':
                self._confirmed_Fs_input = None
                self._fs_input_source = None
            # 重置计数与雷达历史数据，确保外部导入信号的第一次评估为“信号1”
            try:
                global count
                count = 0
                # 清空 radar_drawing 内部缓存
                from zhiliang import radar_drawing as _radar
                if hasattr(_radar, 'signal_data'):
                    _radar.signal_data = np.zeros((4, 4))
            except Exception:
                pass
        else:
            self.File_path_Btn.setVisible(True)  # 设置为可视
            self.File_path_Btn.setEnabled(True)  # 设置为可用
            self.Signal_file_Btn.setVisible(False)  # 设置为不可视
            self.Signal_file_Btn.setEnabled(False)  # 设置为不可用
            self.signal_test.setVisible(True)  # 设置为不可视
            self.signal_test.setEnabled(True)  # 设置为不可用
            QMessageBox.warning(self, "文件选择失败", "未选择文件，请重新选择")
            return

    def read_signal(self):
        plt.clf()
        plt.close("all")

        # # 后面记得删除！！！
        # self.widget_Constellation.setVisible(False)  # 设置不可见
        # self.widget_Eye.setVisible(False)  # 设置不可见
        # 空缺输入控制
        global Fs_input, Fc_input, Rs_input, SNR_input
        file_can_supply_fs = bool(getattr(self, 'file_path', None) and self.file_path.lower().endswith('.csv'))
        if self.Fs_txt.toPlainText().strip() == "" and not file_can_supply_fs:
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "采样频率未输入，请输入参数")
            return
        elif self.Fc_txt.toPlainText().strip() == "":
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "载波频率未输入，请输入参数")
            return
        elif self.Rs_txt.toPlainText().strip() == "":
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "码元速率未输入，请输入参数")
            return
        elif self.SNR_txt.toPlainText().strip() == "":
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "信噪比未输入，请输入参数")
            return

        else:
            # 将参数从文本框中读取并转化成float变量（后续根据需要选择是否要修改成别的类型）
            global rec_wave, Fs_input
            try:
                Fs_input = self._active_fs_input()
            except Exception:
                if file_can_supply_fs:
                    Fs_input = None
                else:
                    QMessageBox.warning(self, "未输入参数", "采样频率未输入，请输入参数")
                    return
            Fc_input = float(self.Fc_txt.toPlainText()) * self.Fc_level_combo()
            Rs_input = float(self.Rs_txt.toPlainText()) * self.Rs_level_combo()
            SNR_input = float(self.SNR_txt.toPlainText())
            Modulation_input = self._active_modulation_input()
            self.progressBar.setValue(0)
            self.progressBar.show()

            plt.clf()
            plt.close("all")
            # def select_file(self):
            # # 打开文件选择对话框
            # options = QFileDialog.Options()
            # self.file_path, _ = QFileDialog.getOpenFileName(self, "Select TXT File", "", "Text Files (*.txt);;All Files (*)",
            #                                                options=options)
            current_path = self.file_path
            file_sample_rate = None
            if current_path:
                # 根据文件扩展名选择读取方式
                if current_path.lower().endswith(('.xlsx', '.xls', '.csv')):
                    # Excel/CSV文件：使用load_excel_signal函数读取
                    try:
                        rec_wave, file_sample_rate = load_excel_signal(current_path, normalize=True, return_fs=True)
                        if file_sample_rate is not None:
                            Fs_input = file_sample_rate
                            self._file_sample_rate = file_sample_rate
                            self._confirmed_Fs_input = file_sample_rate
                            self._fs_input_source = 'file'
                            self._set_fs_controls_from_hz(file_sample_rate)
                        print(f"成功从文件加载信号数据，共 {len(rec_wave)} 个采样点")
                    except Exception as e:
                        QMessageBox.warning(self, "文件读取失败", f"读取文件出错：{str(e)}")
                        return
                else:
                    # TXT文件：读取复数字符串（原有逻辑）
                    with open(current_path, 'r') as file:
                        lines = file.readlines()
                    # 去掉每行末尾的换行符
                    lines = [line.strip() for line in lines]
                    # 将内容存储到数组中
                    rec_wave = [complex(line) for line in lines if line]
                    # 将rec_wave列表转换为numpy数组
                    rec_wave = np.array(rec_wave)
            else:
                QMessageBox.warning(self, "文件选择失败", "未选择文件，请重新选择")
                return
            self.update_progress()
            if file_sample_rate is None and self.Fs_txt.toPlainText().strip() == "":
                # 文本为空，显示错误对话框
                QMessageBox.warning(self, "未输入参数", "采样频率未输入，请输入参数")
                return
            else:
                if file_sample_rate is not None:
                    Fs_input = file_sample_rate
                else:
                    Fs_input = self._active_fs_input()
                [Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI] = signal_read(rec_wave, Fs_input)
                # 将数据输入到STFT处理过程，得到图片
                print("rec_wave_picture:", rec_wave[0:3])
                self.update_progress()
                self.canvas_stft = STFT_dreawing(Fs, rec_wave, self.widget_STFT)  # 绘制STFT图像
                plt.clf()
                plt.close("all")

                # print("rec_wave_picture:", rec_wave[0:3])
                self.canvas, self.xf, self.yf = FFT_dreawing_dynamic(
                    Fs,
                    rec_wave,
                    self.widget_FFT,
                    seconds_per_chunk=0.1,
                    strategy='sliding',
                    max_samples_per_chunk=2048,
                    update_interval_ms=300,
                    step_fraction=0.5,
                )
                plt.clf()
                plt.close("all")
                #
                #
                # downsample_factor = Eyediagram_dreawing(Fs, RS_GUJI, rec_wave, self.widget_Eye)
                # plt.clf()
                # plt.close("all")
                self.update_progress()
                Magenitude_estimate = magnitude_GUJI
                SNR_estimate = SNR_GUJI
                RS_estimate = RS_GUJI
                Rs_process = RS_GUJI / 1e9
                Fs_process = Fs / 1e9
                # 加载图像,一张是原始STFT，一张是标注了中心频点和高度的。
                image_path_STFT = resolve_recognition_image_path('./signal_ana/STFT_Org_txt.jpg')
                [height, center_frequency_output] = process_image(Rs_process, 2, rec_wave, Fs, SNR_estimate, image_path_STFT)
                self.update_progress()

                # 加载图像,一张是原始STFT，一张是标注了中心频点和高度的。
                image_path_STFT = resolve_recognition_image_path('./signal_ana/STFT_Org_txt.jpg')
                image_path_STFT_ana = './signal_ana/STFT_Ana_txt.jpg'
                image_grad_CAM = 'signal_ana/bandwidth_Grad_Cam.jpg'
                # 图像识别求Fc（采用原始程序的计算方式，将像素行映射到频率）
                center_frequency_estimate, stft_metadata = map_center_frequency_from_stft(
                    image_path_STFT,
                    height,
                    center_frequency_output,
                    Fs_input,
                )
                Fc_process = center_frequency_estimate / 1e9
                self.update_progress()

                # 神经网络求bandwidth
                image_path = image_path_STFT
                dir_save_path = './signal_ana/'
                # img_names = os.listdir(dir_origin_path)
                # for img_name in tqdm(img_names):
                #     if img_name.lower().endswith(
                #             ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                #         image_path = os.path.join(dir_origin_path, img_name)
                image = cv2.imread(image_path_STFT)
                height, width, channels = image.shape

                unet = Unet()
                image = Image.open(image_path)
                r_image, rect_height, flag_mask = safe_detect(unet, image)
                img_name = 'street_mask.jpg'
                r_image.save(os.path.join(dir_save_path, img_name))
                bandwidth_estimate_1 = map_bandwidth_from_stft(rect_height, height, image_path_STFT, stft_metadata)

                # 神经网络求bandwidth
                image_path = image_path_STFT
                dir_save_path = './signal_ana/'
                # img_names = os.listdir(dir_origin_path)
                # for img_name in tqdm(img_names):
                #     if img_name.lower().endswith(
                #             ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                #         image_path = os.path.join(dir_origin_path, img_name)
                image = cv2.imread(image_path_STFT)
                height, width, channels = image.shape

                unet = Unet()
                image = Image.open(image_path)
                r_image, rect_height, flag_mask = safe_detect(unet, image)
                img_name = 'street_mask.jpg'
                r_image.save(os.path.join(dir_save_path, img_name))
                bandwidth_estimate_2 = map_bandwidth_from_stft(rect_height, height, image_path_STFT, stft_metadata)

                bandwidth_true = (1 + 0.35 / 2) * Rs_input
                print("真实带宽：", bandwidth_true)

                if (round(Rs_process, 2) < 0.745):
                    Band_flag = 0.753124
                elif (0.745 <= round(Rs_process, 2) < 0.99):
                    Band_flag = 0.755928
                elif (0.99 <= round(Rs_process, 2) < 1.20):
                    Band_flag = 0.733674
                elif (1.20 <= round(Rs_process, 2) < 1.35):
                    Band_flag = 0.727199
                elif (1.35 <= round(Rs_process, 2) < 1.55):
                    Band_flag = 0.6722442
                elif (1.55 <= round(Rs_process, 2) < 1.85):
                    Band_flag = 0.619473
                elif (1.85 <= round(Rs_process, 2) < 2.05):
                    Band_flag = 0.610236
                elif (2.05 <= round(Rs_process, 2) < 2.65):
                    Band_flag = 0.695199
                else:
                    Band_flag = 1

                bandwidth_estimate = (bandwidth_estimate_1 + bandwidth_estimate_2) / 2 * flag3_tune(
                    Rs_process) / Band_flag * 1e9 / flag1_tune(Rs_process, SNR_GUJI, Fc_process) / flag2_tune(
                    Rs_process,
                    SNR_GUJI,
                    Fc_process)
                # # 信号-3dB占用带宽
                # bandwidth_estimate = bandwidth_estimate / Band_flag / (1 + 0.35 / 2) * (1 - 0.35 / 2) * 2
                # 数据转化成科学计数法
                scientific_Fc_input = "{:.2e}".format(center_frequency_estimate)
                scientific_Rs_input = "{:.2e}".format(RS_estimate)
                scientific_bandwidth_input = "{:.2e}".format(bandwidth_estimate)

                # 数据显示在文本框内
                self.Fc_ES.setPlainText(scientific_Fc_input)
                self.Rs_ES.setPlainText(scientific_Rs_input)
                self.Bandwidth_ES.setPlainText(scientific_bandwidth_input)
                self.SNR_ES.setPlainText(str(f'{SNR_estimate:.2f}'))
                # self.A_ES.setPlainText(str(f'{Magenitude_estimate:.2f}'))
                self.update_progress()
                [evm_percentage, evm_db, PARP, signal_star] = signal_ideal(Fs, center_frequency_estimate, RS_GUJI,
                                                                           rec_wave,
                                                                           Modulation_input, SNR_GUJI)
                print("signal_star:", signal_star[0:8])
                self.canvas_star = Constellation_dreawing(Fs, signal_star, self.widget_Constellation, self.I_phase,
                                                          self.N_phase)
                print(f'EVM: {evm_percentage};dB_evm:{evm_db};PARP:{PARP}')
                self.EVM_TXT.setPlainText(str(round(evm_percentage, 2)))
                # self.EVM_dB_TXT.setPlainText(str(round(evm_db, 2)))
                self.PARP_TXT.setPlainText(str(round(PARP, 2)))

                plt.clf()
                plt.close("all")
                # Eyediagram_dreawing(Fs, RS_GUJI, rec_wave, self.widget_Eye)
                # plt.clf()
                # plt.close("all")
                self.update_progress()
                image = cv2.imread(image_path_STFT)
                height, width, channels = image.shape

                # 绘制信号段的边框与中点
                output_image = image.copy()
                pt1 = (20, int(center_frequency_output - 0.5 * rect_height))
                pt2 = (width - 20, int(center_frequency_output + 0.5 * rect_height))
                cv2.rectangle(output_image, pt1, pt2, (0, 255, 0), 8)
                cv2.circle(output_image, (int(width * 0.5), int(center_frequency_output)), 5, (255, 0, 0), -1)
                self.save_image(output_image, image_path_STFT_ana)
                self.update_progress()

                # 将图像显示在对应的图像框内
                pixmap_STFT = QPixmap(image_grad_CAM)
                pixmap_STFT_ana = QPixmap(image_path_STFT_ana)
                # 获取 STFT_View 的大小
                view_size_STFT = self.STFT_View.size()
                view_size_STFT_Ana = self.Cyclic_View.size()
                # 将图像缩放到 QGraphicsView 控件的大小
                scaled_pixmap_STFT = pixmap_STFT.scaled(view_size_STFT, aspectRatioMode=Qt.IgnoreAspectRatio)
                scaled_pixmap_STFT_ana = pixmap_STFT_ana.scaled(view_size_STFT_Ana,
                                                                aspectRatioMode=Qt.IgnoreAspectRatio)
                # 创建 QGraphicsScene
                scene_STFT = QGraphicsScene()
                scene_STFT_ana = QGraphicsScene()
                # 创建 QGraphicsPixmapItem
                pixmap_item_STFT = QGraphicsPixmapItem(scaled_pixmap_STFT)
                pixmap_item_STFT_ana = QGraphicsPixmapItem(scaled_pixmap_STFT_ana)
                # 将 pixmap_item 添加到 scene 中
                scene_STFT.addItem(pixmap_item_STFT)
                scene_STFT_ana.addItem(pixmap_item_STFT_ana)
                # 将 scene 设置到 QGraphicsView 中
                self.STFT_View.setScene(scene_STFT)
                self.Cyclic_View.setScene(scene_STFT_ana)
                self.update_progress()
                self.progressBar.hide()

                Center_Fre_quality = 25 - ((abs(center_frequency_estimate - Fc_input) / Rs_input) * 20 - 1)
                band_quality = 25 - ((abs(bandwidth_estimate - bandwidth_true) / Rs_input) * 10 - 1)
                SNR_quality = 25 - (abs(SNR_estimate - SNR_input) * 2.5)
                if Modulation_input == 1:
                    EVM_quality = 25 - ((evm_percentage / 17.5) * 10 - 1) * 5
                elif Modulation_input == 2:
                    EVM_quality = 25 - ((evm_percentage / 12) * 10 - 1) * 5
                elif Modulation_input == 3:
                    EVM_quality = 25 - ((evm_percentage / 12.5) * 10 - 1) * 5
                elif Modulation_input == 4:
                    EVM_quality = 25 - ((evm_percentage / 8) * 10 - 1) * 5
                else:
                    EVM_quality = 15

                quality_data = np.array([SNR_quality, Center_Fre_quality, band_quality, EVM_quality])
                print("quality_data", quality_data)
                if Center_Fre_quality > 25:
                    Center_Fre_quality = 25
                if band_quality > 25:
                    band_quality = 25
                if SNR_quality > 25:
                    SNR_quality = 25
                if EVM_quality > 25:
                    EVM_quality = 25
                # 外部文件情况：不立即显示信号质量评估，保存待评估数据，等待用户点击“开始评估”
                self._pending_quality_data = {
                    "Center_Fre_quality": Center_Fre_quality,
                    "band_quality": band_quality,
                    "SNR_quality": SNR_quality,
                    "EVM_quality": EVM_quality,
                    "Center_Fre_est": center_frequency_estimate,
                    "bandwidth_est": bandwidth_estimate,
                    "bandwidth_true": bandwidth_true,
                    "SNR_est": SNR_estimate,
                    "SNR_input": SNR_input,
                    "evm_percentage": evm_percentage,
                    "Modulation_input": Modulation_input,
                    "Fs_input": Fs_input,
                    "file_sample_rate": file_sample_rate,
                }
                # 保持质量评估区始终可见（只清空 widget_quality 的绘图内容，保留参数与结构）
                try:
                    # 确保 groupBox_signalquality 始终显示（模块结构与参数始终可见）
                    try:
                        self.groupBox_signalquality.setVisible(True)
                    except Exception:
                        pass
                    # 仅清除 widget_quality 内的绘图/子控件（不隐藏整个 groupbox）
                    for child in list(self.widget_quality.children()):
                        try:
                            child.setParent(None)
                        except Exception:
                            pass
                    try:
                        self.widget_quality.update()
                    except Exception:
                        pass
                except Exception:
                    pass
                # 使能开始评估按钮，提示用户在“数据载入”配置完参数后手动评估
                self.start_eval_btn.setVisible(True)
                self.start_eval_btn.setEnabled(True)
                self.File_path_Btn.setVisible(True)  # 设置为可视
                self.File_path_Btn.setEnabled(True)  # 设置为可用
                self.Signal_file_Btn.setVisible(False)  # 设置为不可视
                self.Signal_file_Btn.setEnabled(False)  # 设置为不可用
                self.signal_test.setVisible(True)  # 设置为不可视
                self.signal_test.setEnabled(True)  # 设置为不可用
                self.Cyclic_View.setVisible(True)  # 初始不可视
                self.STFT_View.setVisible(True)  # 初始不可视

                # # 在绘图时显示边框
                # self.STFT_View.setStyleSheet('border: 2px solid black;')
                # self.Cyclic_View.setStyleSheet('border: 2px solid black;')
            i = 6

    def signal_count(self, number):
        global count
        # 增加调试输出以便定位何处对计数进行了修改（有时会导致“跳过一次”）
        try:
            import traceback
            stack = traceback.extract_stack()
            caller = stack[-2].name if len(stack) >= 2 else "<unknown>"
        except Exception:
            caller = "<unknown>"
        count += number
        try:
            print(f"signal_count called by {caller}: add {number}, new count = {count}")
        except Exception:
            print("signal_count updated")
        return count

    # 自设信号——参数输入模块
    def parameter_input(self):
        plt.clf()
        plt.close("all")

        # # 后面记得删除！！！
        # self.widget_Constellation.setVisible(False)  # 设置不可见
        # self.widget_Eye.setVisible(False)  # 设置不可见

        self.progressBar.setValue(0)
        self.progressBar.show()
        # 空缺输入控制
        global Fs_input, Fc_input, Rs_input, SNR_input
        if self.Fs_txt.toPlainText().strip() == "":
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "采样频率未输入，请输入参数")
            return
        elif self.Fc_txt.toPlainText().strip() == "":
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "载波频率未输入，请输入参数")
            return
        elif self.Rs_txt.toPlainText().strip() == "":
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "码元速率未输入，请输入参数")
            return
        elif self.SNR_txt.toPlainText().strip() == "":
            # 文本为空，显示错误对话框
            QMessageBox.warning(self, "未输入参数", "信噪比未输入，请输入参数")
            return

        else:
            # 将参数从文本框中读取并转化成float变量（后续根据需要选择是否要修改成别的类型）

            Fs_input = self._active_fs_input()
            Fc_input = float(self.Fc_txt.toPlainText()) * self.Fc_level_combo()
            Rs_input = float(self.Rs_txt.toPlainText()) * self.Rs_level_combo()
            SNR_input = float(self.SNR_txt.toPlainText())
            Modulation_input = self._active_modulation_input()
            # signal_create(Fs_input, Fc_input, Rs_input, SNR_input, Modulation_input)
            # 输出界面的内容，包含幅度，信噪比，符号速率
            [Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI, min_val, max_val] = signal_create(Fs_input, Fc_input,
                                                                                                Rs_input, SNR_input,
                                                                                                Modulation_input)
            print("rec_wave_picture:", rec_wave[0:3])
            self.update_progress()
            self.canvas_stft = STFT_dreawing(Fs, rec_wave, self.widget_STFT)  # 绘制STFT图像
            plt.clf()
            plt.close("all")
            self.update_progress()
            # print("rec_wave_picture:", rec_wave[0:3])
            self.canvas, self.xf, self.yf = FFT_dreawing_dynamic(
                Fs,
                rec_wave,
                self.widget_FFT,
                seconds_per_chunk=0.1,
                strategy='sliding',
                max_samples_per_chunk=2048,
                update_interval_ms=300,
                step_fraction=0.5,
            )
            plt.clf()
            plt.close("all")
            self.update_progress()

            Magenitude_estimate = magnitude_GUJI
            SNR_estimate = SNR_GUJI
            RS_estimate = RS_GUJI
            Rs_process = RS_GUJI / 1e9
            Fs_process = Fs / 1e9
            image_path_STFT = resolve_recognition_image_path('./signal_ana/STFT_Org.jpg')
            [height, center_frequency_output] = process_image(Rs_process, 1, rec_wave, Fs, SNR_estimate, image_path_STFT)
            self.update_progress()

            # 加载图像,一张是原始STFT，一张是标注了中心频点和高度的。
            image_path_STFT = resolve_recognition_image_path('./signal_ana/STFT_Org.jpg')
            image_path_STFT_ana = './signal_ana/STFT_Ana.jpg'
            image_grad_CAM = './signal_ana/bandwidth_Grad_Cam.jpg'

            # 图像识别求Fc
            center_frequency_estimate, stft_metadata = map_center_frequency_from_stft(
                image_path_STFT,
                height,
                center_frequency_output,
                Fs_input,
            )
            Fc_process = center_frequency_estimate / 1e9
            self.update_progress()

            # 神经网络求bandwidth
            image_path = image_path_STFT
            dir_save_path = './signal_ana/'
            # img_names = os.listdir(dir_origin_path)
            # for img_name in tqdm(img_names):
            #     if img_name.lower().endswith(
            #             ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            #         image_path = os.path.join(dir_origin_path, img_name)
            image = cv2.imread(image_path_STFT)
            height, width, channels = image.shape

            unet = Unet()
            image = Image.open(image_path)
            r_image, rect_height, flag_mask = safe_detect(unet, image)
            img_name = 'street_mask.jpg'
            r_image.save(os.path.join(dir_save_path, img_name))
            bandwidth_estimate_1 = map_bandwidth_from_stft(rect_height, height, image_path_STFT, stft_metadata)

            # 神经网络求bandwidth
            image_path = image_path_STFT
            dir_save_path = './signal_ana/'
            # img_names = os.listdir(dir_origin_path)
            # for img_name in tqdm(img_names):
            #     if img_name.lower().endswith(
            #             ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            #         image_path = os.path.join(dir_origin_path, img_name)
            image = cv2.imread(image_path_STFT)
            height, width, channels = image.shape

            unet = Unet()
            image = Image.open(image_path)
            r_image, rect_height, flag_mask = safe_detect(unet, image)
            img_name = 'street_mask.jpg'
            r_image.save(os.path.join(dir_save_path, img_name))
            bandwidth_estimate_2 = map_bandwidth_from_stft(rect_height, height, image_path_STFT, stft_metadata)

            bandwidth_true = (1 + 0.35) * Rs_input
            # 载波-3dB带宽计算
            print("真实带宽：", bandwidth_true)

            if (round(Rs_process, 2) < 0.745):
                Band_flag = 0.753124
            elif (0.745 <= round(Rs_process, 2) < 0.99):
                Band_flag = 0.755928
            elif (0.99 <= round(Rs_process, 2) < 1.20):
                Band_flag = 0.733674
            elif (1.20 <= round(Rs_process, 2) < 1.35):
                Band_flag = 0.727199
            elif (1.35 <= round(Rs_process, 2) < 1.55):
                Band_flag = 0.6722442
            elif (1.55 <= round(Rs_process, 2) < 1.85):
                Band_flag = 0.619473
            elif (1.85 <= round(Rs_process, 2) < 2.05):
                Band_flag = 0.610236
            elif (2.05 <= round(Rs_process, 2) < 2.65):
                Band_flag = 0.695199
            else:
                Band_flag = 1

            bandwidth_estimate = (bandwidth_estimate_1 + bandwidth_estimate_2) / 2 * flag3_tune(
                Rs_process) / Band_flag * 1e9 / flag1_tune(Rs_process, SNR_GUJI, Fc_process) / flag2_tune(Rs_process,
                                                                                                        SNR_GUJI,
                                                                                                        Fc_process)
            # # 信号-3dB占用带宽
            # bandwidth_estimate = bandwidth_estimate / Band_flag / (1 + 0.35 / 2) * (1 - 0.35 / 2) * 2
            # 数据转化成科学计数法
            scientific_Fc_input = "{:.2e}".format(center_frequency_estimate)
            scientific_Rs_input = "{:.2e}".format(RS_estimate)
            scientific_bandwidth_input = "{:.2e}".format(bandwidth_estimate)

            # 数据显示在文本框内
            self.Fc_ES.setPlainText(scientific_Fc_input)
            self.Rs_ES.setPlainText(scientific_Rs_input)
            self.Bandwidth_ES.setPlainText(scientific_bandwidth_input)
            self.SNR_ES.setPlainText(str(f'{SNR_estimate:.2f}'))
            # self.A_ES.setPlainText(str(f'{Magenitude_estimate:.2f}'))
            
            self.update_progress()
            [evm_percentage, evm_db, PARP, signal_star] = signal_ideal(Fs, center_frequency_estimate, RS_GUJI, rec_wave,
                                                                    Modulation_input, SNR_GUJI)
            print("signal_star:", signal_star[0:8])
            self.canvas_star = Constellation_dreawing(Fs, signal_star, self.widget_Constellation, self.I_phase,
                                                    self.N_phase)
            print(f'EVM: {evm_percentage};dB_evm:{evm_db};PARP:{PARP}')
            self.EVM_TXT.setPlainText(str(round(evm_percentage, 2)))
            # self.EVM_dB_TXT.setPlainText(str(round(evm_db, 2)))
            self.PARP_TXT.setPlainText(str(round(PARP, 2)))

            plt.clf()
            plt.close("all")
            # Eyediagram_dreawing(Fs, RS_GUJI, rec_wave, self.widget_Eye)
            # plt.clf()
            # plt.close("all")
            self.update_progress()
            image = cv2.imread(image_path_STFT)
            height, width, channels = image.shape

            # 绘制信号段的边框与中点
            output_image = image.copy()
            pt1 = (20, int(center_frequency_output - 0.5 * rect_height))
            pt2 = (width - 20, int(center_frequency_output + 0.5 * rect_height))
            cv2.rectangle(output_image, pt1, pt2, (0, 255, 0), 8)
            cv2.circle(output_image, (int(width * 0.5), int(center_frequency_output)), 5, (255, 0, 0), -1)
            self.save_image(output_image, image_path_STFT_ana)
            self.update_progress()

            # 将图像显示在对应的图像框内
            pixmap_STFT = QPixmap(image_grad_CAM)
            pixmap_STFT_ana = QPixmap(image_path_STFT_ana)
            # 获取 STFT_View 的大小
            view_size_STFT = self.STFT_View.size()
            view_size_STFT_Ana = self.Cyclic_View.size()
            # 将图像缩放到 QGraphicsView 控件的大小
            scaled_pixmap_STFT = pixmap_STFT.scaled(view_size_STFT, aspectRatioMode=Qt.IgnoreAspectRatio)
            scaled_pixmap_STFT_ana = pixmap_STFT_ana.scaled(view_size_STFT_Ana, aspectRatioMode=Qt.IgnoreAspectRatio)
            # 创建 QGraphicsScene
            scene_STFT = QGraphicsScene()
            scene_STFT_ana = QGraphicsScene()
            # 创建 QGraphicsPixmapItem
            pixmap_item_STFT = QGraphicsPixmapItem(scaled_pixmap_STFT)
            pixmap_item_STFT_ana = QGraphicsPixmapItem(scaled_pixmap_STFT_ana)
            # 将 pixmap_item 添加到 scene 中
            scene_STFT.addItem(pixmap_item_STFT)
            scene_STFT_ana.addItem(pixmap_item_STFT_ana)
            # 将 scene 设置到 QGraphicsView 中
            self.STFT_View.setScene(scene_STFT)
            self.Cyclic_View.setScene(scene_STFT_ana)
            self.update_progress()
            self.progressBar.hide()

            Center_Fre_quality = 25 - ((abs(center_frequency_estimate - Fc_input) / Rs_input) * 20 - 1)
            band_quality = 25 - ((abs(bandwidth_estimate - bandwidth_true) / Rs_input) * 10 - 1)
            SNR_quality = 25 - (abs(SNR_estimate - SNR_input) * 2.5)
            if Modulation_input == 1:
                EVM_quality = 25 - ((evm_percentage / 17.5) * 10 - 1) * 5
            elif Modulation_input == 2:
                EVM_quality = 25 - ((evm_percentage / 12) * 10 - 1) * 5
            elif Modulation_input == 3:
                EVM_quality = 25 - ((evm_percentage / 12.5) * 10 - 1) * 5
            elif Modulation_input == 4:
                EVM_quality = 25 - ((evm_percentage / 8) * 10 - 1) * 5
            else:
                EVM_quality = 15

            quality_data = np.array([SNR_quality, Center_Fre_quality, band_quality, EVM_quality])
            print("quality_data", quality_data)
            if Center_Fre_quality > 25:
                Center_Fre_quality = 25
            if band_quality > 25:
                band_quality = 25
            if SNR_quality > 25:
                SNR_quality = 25
            if EVM_quality > 25:
                EVM_quality = 25
            count_need = self.signal_count(1)
            factors = np.array([1, 1, 1, 1])
            quality_data = np.array([SNR_quality, Center_Fre_quality, band_quality, EVM_quality])

            radom_data_final = quality_data * factors
            # 调用质量评估并显示（自生成信号，点击“信号测试”时直接显示质量评估）
            try:
                self.evaluate_signal_quality(center_frequency_estimate, bandwidth_estimate, bandwidth_true,
                                            SNR_estimate, SNR_input, evm_percentage,  Modulation_input,
                                            count_need=count_need)
                self.groupBox_signalquality.setVisible(True)
            except Exception:
                # 保持向后兼容，若调用失败则继续流程但不阻塞
                pass
            self.Cyclic_View.setVisible(True)  # 初始可视
            self.STFT_View.setVisible(True)  # 初始可视

        j = 7

    def evaluate_signal_quality(self, center_frequency_estimate, bandwidth_estimate, bandwidth_true,
                                SNR_estimate, SNR_input, evm_percentage, Modulation_input, count_need=None):
        """
        根据传入的参数计算信号质量各项评分并在 widget_quality 上绘制雷达图。
        可被自生成信号直接调用，也可被“开始评估”按钮调用（外部信号模式）。
        """
        Center_Fre_quality = 25 - ((abs(center_frequency_estimate - Fc_input) / Rs_input) * 20 - 1)
        band_quality = 25 - ((abs(bandwidth_estimate - bandwidth_true) / Rs_input) * 10 - 1)
        SNR_quality = 25 - (abs(SNR_estimate - SNR_input) * 2.5)
        if Modulation_input == 1:
            EVM_quality = 25 - ((evm_percentage / 17.5) * 10 - 1) * 5
        elif Modulation_input == 2:
            EVM_quality = 25 - ((evm_percentage / 12) * 10 - 1) * 5
        elif Modulation_input == 3:
            EVM_quality = 25 - ((evm_percentage / 12.5) * 10 - 1) * 5
        elif Modulation_input == 4:
            EVM_quality = 25 - ((evm_percentage / 8) * 10 - 1) * 5
        else:
            EVM_quality = 15

        # 限幅
        if Center_Fre_quality > 25:
            Center_Fre_quality = 25
        if band_quality > 25:
            band_quality = 25
        if SNR_quality > 25:
            SNR_quality = 25
        if EVM_quality > 25:
            EVM_quality = 25

        quality_data = np.array([SNR_quality, Center_Fre_quality, band_quality, EVM_quality])
        # 如果调用者未传入 count_need，则在此处增加一次计数并使用（向后兼容）
        if count_need is None:
            count_need = self.signal_count(1)
        factors = np.array([1, 1, 1, 1])
        radom_data_final = quality_data * factors
        radar_drawing(radom_data_final, count_need, self.widget_quality)

    # UI 按钮需要的别名（连接到 win_v1.py 中的信号）
    def start_eval(self):
        return self.start_evaluation_clicked()

    def start_cal(self):
        return self.start_calculation_clicked()

    def start_evaluation_clicked(self):
        """
        点击“开始评估”后，如果存在待评估的数据，则计算并显示信号质量评估结果。
        """
        if not self._pending_quality_data:
            QMessageBox.warning(self, "无待评估数据", "当前没有外部信号的待评估数据，请先执行信号测试。")
            return

        pdata = self._pending_quality_data or {}
        try:
            # 禁用按钮防止重复点击，显示等待
            try:
                self.start_eval_btn.setEnabled(False)
            except Exception:
                pass

            # 从 UI 读取当前参数（用户可能在读取后修改了参数）
            try:
                SNR_input_ui = float(self.SNR_txt.toPlainText())
            except Exception:
                SNR_input_ui = pdata.get("SNR_input", None)
            try:
                Modulation_input_ui = self._active_modulation_input()
            except Exception:
                Modulation_input_ui = pdata.get("Modulation_input", None)

            # 清除 widget_quality 旧内容以防旧图显示
            try:
                for child in list(self.widget_quality.children()):
                    try:
                        child.setParent(None)
                    except Exception:
                        pass
                try:
                    self.widget_quality.update()
                except Exception:
                    pass
            except Exception:
                pass

            # 为评估获取新的计数（避免跳过/重复），并调用评估函数
            count_need = self.signal_count(1)
            self.evaluate_signal_quality(pdata["Center_Fre_est"], pdata["bandwidth_est"], pdata["bandwidth_true"],
                                        pdata.get("SNR_est", None), SNR_input_ui, pdata.get("evm_percentage", None),
                                        Modulation_input_ui, count_need=count_need)

            # 确保界面显示评估区域
            try:
                self.groupBox_signalquality.setVisible(True)
            except Exception:
                pass

            # 不在此处清空 _pending_quality_data，保留给“开始计算”使用；仅在 start_calculation_clicked 中清空
            try:
                self.start_eval_btn.setEnabled(False)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.warning(self, "评估失败", f"信号质量评估失败：{e}")
            # 如果失败，允许用户重试
            try:
                self.start_eval_btn.setEnabled(True)
            except Exception:
                pass

    # 计算指标结果（由 UI 的“开始计算”按钮触发）
    def start_calculation_clicked(self):
        """
        点击“开始计算”后，使用保存在 self._pending_quality_data 的估计值和当前 UI 参数
        计算误差并把结果写回到右侧“参数分析结果”页的文本框中。
        兼容自生信号和外部信号：外部信号需先通过文件读取产生 _pending_quality_data。
        """
        pdata = getattr(self, '_pending_quality_data', None)
        if not pdata:
            QMessageBox.warning(self, "无待评估数据", "当前没有外部信号的待评估数据，请先执行信号测试或生成信号。")
            return

        try:
            # 从 pending 数据读取估计值；若缺失则退回到 None（后续判断）
            center_frequency_estimate = pdata.get("Center_Fre_est", None)
            bandwidth_estimate = pdata.get("bandwidth_est", None)
            bandwidth_true = pdata.get("bandwidth_true", None)
            SNR_estimate = pdata.get("SNR_est", None)
            # 从 UI 读取当前用户输入的参考参数（用户可能已修改）
            try:
                SNR_input_ui = float(self.SNR_txt.toPlainText())
            except Exception:
                SNR_input_ui = pdata.get("SNR_input", None)
            try:
                Fc_input_ui = float(self.Fc_txt.toPlainText()) * self.Fc_level_combo()
            except Exception:
                Fc_input_ui = None
            try:
                Rs_input_ui = float(self.Rs_txt.toPlainText()) * self.Rs_level_combo()
            except Exception:
                Rs_input_ui = None

            if None in (center_frequency_estimate, bandwidth_estimate, bandwidth_true, SNR_estimate, Rs_input_ui):
                QMessageBox.warning(self, "数据不足", "计算所需数据不完整，请先执行信号测试或确认输入参数。")
                return

            center_Error = abs((center_frequency_estimate - Fc_input_ui) / Rs_input_ui * 100) if Fc_input_ui is not None else float('nan')
            bandwidth_Error = abs((bandwidth_estimate - bandwidth_true) / Rs_input_ui * 100)
            SNR_Error = abs(SNR_estimate - SNR_input_ui) if SNR_input_ui is not None else float('nan')

            # 数据转换为科学计数法
            scientific_center_Error = "{:.2e}".format(center_Error)
            scientific_bandwidth_Error = "{:.2e}".format(bandwidth_Error)
            scientific_SNR_Error = "{:.2e}".format(SNR_Error)

            try:
                if hasattr(self, 'Bandwidth_Err_Txt'):
                    self.Bandwidth_Err_Txt.setPlainText(f"{scientific_center_Error} %" if scientific_center_Error is not None else "N/A")
                if hasattr(self, 'Bandwidth_ES_Err'):
                    self.Bandwidth_ES_Err.setPlainText(f"{scientific_bandwidth_Error} %" if scientific_bandwidth_Error is not None else "N/A")
                if hasattr(self, 'SNR_Err_Txt'):
                    self.SNR_Err_Txt.setPlainText(f"{scientific_SNR_Error} %" if scientific_SNR_Error is not None else "N/A")
            except Exception as e:
                QMessageBox.warning(self, "显示失败", f"计算结果显示失败：{e}")
            # 计算后清空 pending 数据，要求下一次新数据需要重新测试
            self._pending_quality_data = None
        except Exception as e:
            QMessageBox.warning(self, "计算失败", f"开始计算失败：{e}")


if __name__ == '__main__':
    # Enable high-DPI scaling for sharper fonts and proper sizing on high-DPI monitors
    try:
        # Prefer automatic per-monitor scaling when available
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        # Pass-through rounding avoids odd font size jumps on some Windows setups
        if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    signal_analysis_form = Signal_analysis_form()
    signal_analysis_form.show()

    sys.exit(app.exec_())
