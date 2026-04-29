import numpy as np
from PyQt5.QtGui import QFont
# from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.fftpack import fft, fftfreq
from PyQt5.QtWidgets import QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer
from eyediagram.mpl import eyediagram
import matplotlib.pyplot as plt
from signal_photo_save import signal_create_test
import numpy as np
import re


def format_scientific_notation(value):
    """
    将科学计数法格式化为×10^9的上标格式
    例如：1.23e+09 -> 1.23×10⁹
    """
    # 使用科学计数法格式化
    scientific_str = "{:.2e}".format(value)
    
    # 使用正则表达式解析科学计数法
    match = re.match(r'([+-]?\d+\.?\d*)e([+-]?\d+)', scientific_str)
    if match:
        mantissa = match.group(1)
        exponent = int(match.group(2))
        
        # 将指数转换为上标字符
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '+': '⁺', '-': '⁻'
        }
        
        # 转换指数为上标
        exponent_str = str(exponent)
        superscript_exponent = ''.join(superscript_map.get(char, char) for char in exponent_str)
        
        return f"{mantissa}×10{superscript_exponent}"
    else:
        return scientific_str


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

        # 自适应大小
        FigureCanvas.updateGeometry(self)
def update_constellation_dynamic(sc1, scatter_plot, rec_wave, points_per_second, current_index, total_points):
    """动态更新星座图的函数"""
    # 计算这一帧要显示的点数
    end_index = min(current_index + points_per_second, total_points)
    
    # 确保索引不超出边界
    if current_index < total_points:
        # 获取当前要显示的数据点
        current_real = rec_wave.real[current_index:end_index]
        current_imag = rec_wave.imag[current_index:end_index]
        
        # 只有当有数据时才更新散点图
        if len(current_real) > 0:
            scatter_plot.set_offsets(np.column_stack((current_real, current_imag)))
            sc1.draw()
    
    # 更新索引
    new_index = end_index
    
    # 如果所有点都显示完了，从头开始
    if new_index >= total_points:
        new_index = 0
    
    return new_index

def Constellation_dreawing(Fs, rec_wave, widget, textbox1, textbox2):
    layout = widget.layout()
    if layout is not None:
        # Iterate over layout items and remove MplCanvas instances
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget_item = item.widget()
            if isinstance(widget_item, MplCanvas):
                widget_item.deleteLater()

    sc1 = MplCanvas(widget)

    # 清除当前绘图内容
    sc1.ax.clear()

    # 设置背景颜色为黑色,图像为黄色，字体为Segoe UI，12号
    sc1.ax.set_facecolor('black')
    
    # 初始化散点图对象
    scatter_plot = sc1.ax.scatter([], [], color='#ff6020', alpha=0.5, s=2)
    
    # sc1.ax.set_title('Constellation Diagram', color='white')
    sc1.ax.set_xlabel('In-phase', color='white')
    sc1.ax.set_ylabel('Quadrature', color='white')
    font_prop = FontProperties(family='Segoe UI', size=12)
    # sc2.ax.set_title('Frequency Spectrum', color='white')

    # 设置x，y轴字体样式
    sc1.ax.set_xlabel('I_Phase', color='white', fontproperties=font_prop)
    sc1.ax.set_ylabel('Q_Phase', color='white', fontproperties=font_prop)

    # 设置坐标轴刻度字体
    sc1.ax.tick_params(colors='white')
    for label in sc1.ax.get_xticklabels() + sc1.ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    sc1.ax.set_facecolor('black')
    sc1.ax.figure.patch.set_facecolor('black')

    # 设置白色虚线网格
    sc1.ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5)

    # 设置坐标轴范围
    max_val = np.max([np.abs(rec_wave.real), np.abs(rec_wave.imag)])
    sc1.ax.set_xlim(-max_val, max_val)
    sc1.ax.set_ylim(-max_val, max_val)

    sc1.figure.subplots_adjust(left=0.12, right=0.95, top=0.975, bottom=0.1)

    # Create new MplCanvas instance
    # Add MplCanvas to the existing layout
    if layout is None:
        layout = QVBoxLayout(widget)
        widget.setLayout(layout)
    layout.addWidget(sc1)
    
    # 动态显示参数设置
    points_per_second = min(1000, len(rec_wave) // 10)  # 每秒显示的点数，最多1000个点
    current_index = 0  # 当前显示到的索引
    total_points = len(rec_wave)  # 总点数
    
    # 创建定时器用于动态更新
    timer = QTimer()
    
    def timer_callback():
        nonlocal current_index
        current_index = update_constellation_dynamic(sc1, scatter_plot, rec_wave, 
                                                   points_per_second, current_index, total_points)
    
    # 设置定时器每1000ms（1秒）更新一次
    timer.timeout.connect(timer_callback)
    timer.start(1000)
    
    # 将定时器保存到widget中，以便在需要时停止
    widget.constellation_timer = timer
    
    # 添加鼠标点击事件
    sc1.mpl_connect('button_press_event', lambda event: on_click_star(event, widget, rec_wave, textbox1, textbox2, sc1))
    return sc1  # 返回 canvas 和数据以便在主程序中使用

def FFT_dreawing(Fs, rec_wave, widget):
    # 绘制频谱图
    N = len(rec_wave)
    T = 1.0 / Fs
    yf = fft(rec_wave)
    # 计算双边频率
    xf = fftfreq(N, T)
    # 调整频谱顺序
    yf_shifted = np.fft.fftshift(yf)
    xf_shifted = np.fft.fftshift(xf)

    layout = widget.layout()
    if layout is not None:
        # Iterate over layout items and remove MplCanvas instances
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget_item = item.widget()
            if isinstance(widget_item, MplCanvas):
                widget_item.deleteLater()

    sc2 = MplCanvas(widget)

    # 清除当前绘图内容
    sc2.ax.clear()

    # 绘制频谱图，去掉空余部分
    sc2.ax.plot(xf_shifted, 2.0 / N * np.abs(yf_shifted), color='#ffff00')
    # 仅显示非负频率（显示部分修改，不改变数据）
    sc2.ax.set_xlim(0, Fs / 2)  # 设置频率轴范围为非负部分
    font_prop = FontProperties(family='Segoe UI', size=12)
    # sc2.ax.set_title('Frequency Spectrum', color='white')

    # 设置x，y轴字体样式
    sc2.ax.set_xlabel('Frequency (Hz)', color='white', fontproperties=font_prop)
    sc2.ax.set_ylabel('Amplitude', color='white', fontproperties=font_prop)

    # 设置y轴使用科学计数法
    sc2.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    sc2.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    # 设置y轴使用科学计数法
    sc2.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    sc2.ax.xaxis.get_major_formatter().set_powerlimits((0, 1))


    # 设置坐标轴刻度字体
    sc2.ax.tick_params(colors='white')
    for label in sc2.ax.get_xticklabels() + sc2.ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 绘制网格线
    sc2.ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    sc2.ax.set_facecolor('black')
    sc2.ax.figure.patch.set_facecolor('black')
    sc2.ax.tick_params(colors='white')
    sc2.figure.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.15)

    # Add MplCanvas to the existing layout
    if layout is None:
        layout = QVBoxLayout(widget)
        widget.setLayout(layout)
    # layout1 = QVBoxLayout(widget)  # 使用指定的 widget
    # layout1.addWidget(sc1)
    layout.addWidget(sc2)

    # layout2 = QVBoxLayout(widget)
    # layout2.addWidget(sc2)
    layout.setStretch(0, 1)  # 让绘图区随着窗口变化自适应

    # 保存数据和图表对象，用于点击事件
    widget.xf = xf
    widget.yf = 2.0 / N * np.abs(yf_shifted)
    widget.sc2 = sc2

    # # 后面记得删掉！！！注释
    # # 添加鼠标点击事件
    sc2.mpl_connect('button_press_event', lambda event: on_click(event, widget))

    return sc2, xf_shifted, widget.yf  # 返回 canvas 和数据以便在主程序中使用

def FFT_dreawing_dynamic(
    Fs,
    rec_wave,
    widget,
    seconds_per_chunk=0.1,
    strategy='sliding',
    max_samples_per_chunk=2048,
    update_interval_ms=300,
    step_fraction=1.0,
    fixed_y_axis=True,
    initial_y_max=0.03,
    y_increment=1e-2,
):
    # 动态绘制频谱图：每秒抽取一段数据做FFT，并更新曲线
    # 参数
    #  - seconds_per_chunk: 每次FFT的时间长度（秒）
    #  - strategy: 'sliding' 按窗口滑动，'random' 随机起点

    # 如已有定时器，先停止，避免重复启动
    if hasattr(widget, 'fft_timer') and isinstance(widget.fft_timer, QTimer):
        try:
            widget.fft_timer.stop()
        except Exception:
            pass

    N_total = len(rec_wave)
    samples_per_chunk = max(16, int(Fs * seconds_per_chunk))
    if max_samples_per_chunk is not None:
        samples_per_chunk = min(samples_per_chunk, int(max_samples_per_chunk))
    samples_per_chunk = min(samples_per_chunk, N_total)
    T = 1.0 / Fs

    layout = widget.layout()
    if layout is not None:
        # 清理所有布局项，包括MplCanvas和控制面板（QHBoxLayout）
        while layout.count() > 0:
            item = layout.takeAt(0)  # 从布局中移除第一个项
            if item is not None:
                # 如果是控件，删除控件
                widget_item = item.widget()
                if widget_item is not None:
                    widget_item.deleteLater()
                # 如果是布局，删除布局及其所有子项
                layout_item = item.layout()
                if layout_item is not None:
                    # 清理布局中的所有控件
                    while layout_item.count() > 0:
                        sub_item = layout_item.takeAt(0)
                        if sub_item is not None:
                            sub_widget = sub_item.widget()
                            if sub_widget is not None:
                                sub_widget.deleteLater()
                            # 如果子项也是布局，递归清理（虽然这里不太可能）
                            sub_layout = sub_item.layout()
                            if sub_layout is not None:
                                sub_layout.deleteLater()
                    layout_item.deleteLater()
                # 删除item本身
                del item

    sc2 = MplCanvas(widget)
    sc2.ax.clear()

    # 初始窗口与FFT
    start_index = 0
    segment = rec_wave[start_index:start_index + samples_per_chunk]
    N = len(segment)
    yf = fft(segment)
    xf = fftfreq(N, T)
    yf_shifted = np.fft.fftshift(yf)
    xf_shifted = np.fft.fftshift(xf)

    # 绘制初始化曲线对象以便后续更新
    (line,) = sc2.ax.plot(xf_shifted, 2.0 / N * np.abs(yf_shifted), color='#ffff00')
    # 仅显示非负频率
    sc2.ax.set_xlim(0, Fs / 2)
    
    # 根据fixed_y_axis参数决定是否固定纵坐标
    if fixed_y_axis:
        sc2.ax.set_ylim(0, initial_y_max)
    
    font_prop = FontProperties(family='Segoe UI', size=12)
    sc2.ax.set_xlabel('Frequency (Hz)', color='white', fontproperties=font_prop)
    sc2.ax.set_ylabel('Amplitude', color='white', fontproperties=font_prop)
    sc2.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    sc2.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    sc2.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    sc2.ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    sc2.ax.tick_params(colors='white')
    for label in sc2.ax.get_xticklabels() + sc2.ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    sc2.ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    sc2.ax.set_facecolor('black')
    sc2.ax.figure.patch.set_facecolor('black')
    sc2.ax.tick_params(colors='white')
    sc2.figure.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.15)

    if layout is None:
        layout = QVBoxLayout(widget)
        widget.setLayout(layout)
    layout.addWidget(sc2)
    layout.setStretch(0, 1)

    # 保存数据和图表对象，用于点击事件和后续更新
    widget.xf = xf_shifted
    widget.yf = 2.0 / N * np.abs(yf_shifted)
    widget.sc2 = sc2
    sc2.mpl_connect('button_press_event', lambda event: on_click(event, widget))

    # 如果启用固定y轴，添加控制按钮
    if fixed_y_axis:
        from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QLineEdit
        from PyQt5.QtCore import Qt
        
        # 创建控制面板
        control_layout = QHBoxLayout()
        
        # 减按钮
        minus_btn = QPushButton("-")
        minus_btn.setFixedSize(30, 30)
        minus_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border: 1px solid #666666;
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
        """)
        
        # 当前y轴最大值显示
        y_max_display = QLineEdit(f"{initial_y_max:.3f}")
        y_max_display.setFixedSize(60, 30)
        y_max_display.setAlignment(Qt.AlignCenter)
        y_max_display.setStyleSheet("""
            QLineEdit {
                background-color: #333333;
                color: white;
                border: 1px solid #666666;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        y_max_display.setReadOnly(True)
        
        # 加按钮
        plus_btn = QPushButton("+")
        plus_btn.setFixedSize(30, 30)
        plus_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border: 1px solid #666666;
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
        """)
        
        # 添加按钮到控制面板
        control_layout.addWidget(minus_btn)
        control_layout.addWidget(y_max_display)
        control_layout.addWidget(plus_btn)
        control_layout.addStretch()  # 添加弹性空间
        
        # 将控制面板添加到主布局
        layout.addLayout(control_layout)
        
        # 当前y轴最大值
        current_y_max = initial_y_max
        
        # 按钮事件处理
        def decrease_y_max():
            nonlocal current_y_max
            current_y_max = max(y_increment, current_y_max - y_increment)
            sc2.ax.set_ylim(0, current_y_max)
            y_max_display.setText(f"{current_y_max:.3f}")
            sc2.draw()
        
        def increase_y_max():
            nonlocal current_y_max
            current_y_max += y_increment
            sc2.ax.set_ylim(0, current_y_max)
            y_max_display.setText(f"{current_y_max:.3f}")
            sc2.draw()
        
        minus_btn.clicked.connect(decrease_y_max)
        plus_btn.clicked.connect(increase_y_max)

    # 定时器更新逻辑
    timer = QTimer()

    # 滑动步长，默认为一个窗口长度，可用 step_fraction 调整（例如 0.5 表示半窗步进）
    step = max(1, int(samples_per_chunk * float(step_fraction)))

    def update_fft_frame():
        nonlocal start_index
        # 选择新起点
        if strategy == 'random':
            if N_total > samples_per_chunk:
                start_index = int(np.random.randint(0, N_total - samples_per_chunk))
            else:
                start_index = 0
        else:
            start_index += step
            if start_index >= N_total:
                start_index = 0

        seg = rec_wave[start_index:start_index + samples_per_chunk]
        if len(seg) < samples_per_chunk:
            # 不足则进行环绕补齐
            wrap = samples_per_chunk - len(seg)
            seg = np.concatenate([seg, rec_wave[:wrap]])

        N_seg = len(seg)
        yf_local = fft(seg)
        xf_local = fftfreq(N_seg, T)
        yf_shift = np.fft.fftshift(yf_local)
        xf_shift = np.fft.fftshift(xf_local)

        y_plot = 2.0 / N_seg * np.abs(yf_shift)
        # 更新曲线与点击查询数据
        line.set_data(xf_shift, y_plot)
        widget.xf = xf_shift
        widget.yf = y_plot

        # 根据fixed_y_axis参数决定是否自适应y轴范围
        if not fixed_y_axis:
            # 自适应y轴范围但固定x轴范围
            ymin = float(np.min(y_plot)) if y_plot.size > 0 else 0.0
            ymax = float(np.max(y_plot)) if y_plot.size > 0 else 1.0
            if ymax <= ymin:
                ymax = ymin + 1.0
            sc2.ax.set_ylim(ymin, ymax * 1.05)
        
        sc2.draw()

    timer.timeout.connect(update_fft_frame)
    # 刷新周期（毫秒）
    timer.start(int(update_interval_ms))
    widget.fft_timer = timer

    return sc2, xf_shifted, widget.yf

def STFT_dreawing(Fs, rec_wave, widget):
    layout = widget.layout()
    if layout is not None:
        # Iterate over layout items and remove MplCanvas instances
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget_item = item.widget()
            if isinstance(widget_item, MplCanvas):
                widget_item.deleteLater()
    # 创建绘图画布
    sc_stft = MplCanvas(widget)

    # 清除当前绘图内容
    sc_stft.ax.clear()
    # 使用 spectrogram 函数绘制频谱图
    window = np.hamming(256)  # 窗函数
    noverlap = 220  # 窗口重叠
    nfft = 256  # FFT 点数
    # 设置图像大小和DPI

    # 去除图像外围的黑色边框或将其变成其他颜色
    sc_stft.ax.set_facecolor('black')  # 改变绘图区的背景颜色
    # 计算STFT
    p, f, t, im = sc_stft.ax.specgram(rec_wave, NFFT=nfft, Fs=Fs, Fc=0, window=window, noverlap=noverlap, cmap='jet',
                              xextent=None, pad_to=None, sides='twosided', scale_by_freq=None,
                              mode='psd', scale='default')

    # 将时间和频率轴进行转置
    p = p.T

    # dB
    p = 10 * np.log10(p)  # 加上1e-10以避免对数为负无穷大
    print("p_STFT:", p[0:10])

    # 仅在显示层面保留非负频率（不改原始计算逻辑）
    try:
        # 当前 p 的形状为 (len(t), len(f))，因此按列切片
        if np.any(f < 0):
            mask_pos = f >= 0
            f_pos = f[mask_pos]
            p = p[:, mask_pos]
        else:
            f_pos = f
    except Exception:
        # 若出现异常，则退回到原始值（保持鲁棒性）
        f_pos = f

    # 去除图像外围的黑色边框或将其变成其他颜色
    sc_stft.ax.set_facecolor('black')  # 改变绘图区的背景颜色
    # 绘制转置后的频谱图
    sc_stft.ax.imshow(p, aspect='auto', origin='lower', cmap='jet',
                      extent=[f_pos.min(), f_pos.max(), t.min(), t.max()])

    font_prop_STFT = FontProperties(family='Segoe UI', size=12)

    # sc_stft.ax.set_title('STFT (Spectrogram)')
    sc_stft.ax.set_xlabel('Frequency (Hz)', color='white', fontproperties=font_prop_STFT, labelpad=-1)  # x轴设置为频率
    sc_stft.ax.set_ylabel('Time (s)', color='white', fontproperties=font_prop_STFT)  # y轴设置为时间

    # 设置y轴使用科学计数法
    sc_stft.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    sc_stft.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    # 设置x轴使用科学计数法
    sc_stft.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    sc_stft.ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    # 设置黑色
    sc_stft.ax.set_facecolor('black')
    sc_stft.ax.figure.patch.set_facecolor('black')
    # 设置坐标轴刻度字体
    sc_stft.ax.tick_params(colors='white')
    for label in sc_stft.ax.get_xticklabels() + sc_stft.ax.get_yticklabels():
        label.set_fontproperties(font_prop_STFT)

    # 增加颜色条（colorbar）
    cax = inset_axes(sc_stft.ax,
                     width="100%",  # 长度占据原始图像的90%
                     height="2.5%",  # 高度为原始图像的5%
                     loc='lower center',  # 颜色条位置
                     borderpad=-3.5)  # 边距调整，确保颜色条更靠近图像
    cbar = sc_stft.figure.colorbar(im, cax=cax, orientation='horizontal', pad=0.2)
    # cbar.set_label('Amplitude (dB)', color='black', fontproperties=font_prop_STFT)  # 设置颜色条的标签
    cbar.ax.tick_params(labelsize=10, colors='white')  # 设置颜色条刻度的字体大小和颜色

    # 调整边距，使图像紧贴widget的边界
    # sc_stft.figure.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1, rect=[0, 0, 1, 1])
    sc_stft.figure.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.2)

    # Add MplCanvas to the existing layout
    if layout is None:
        layout = QVBoxLayout(widget)
        widget.setLayout(layout)
    # layout1 = QVBoxLayout(widget)  # 使用指定的 widget
    # layout1.addWidget(sc1)
    layout.addWidget(sc_stft)

    # layout = QVBoxLayout(widget)
    # layout.addWidget(sc_stft)
    layout.setStretch(0, 1)  # 让绘图区随着窗口变化自适应

def Eyediagram_dreawing(Fs, Rs, rec_wave, widget):
    # 分离实部和虚部
    real_part = np.real(rec_wave)
    imag_part = np.imag(rec_wave)

    # 计算幅度
    magnitude = np.abs(rec_wave)

    num_symbols = len(rec_wave)

    # 每符号的采样点数
    samples_per_symbol = int(Fs // Rs)

    # 计算总的采样符号数
    total_symbols = len(real_part) // samples_per_symbol
    downsample_factor = 10
    # 对信号进行下采样以提高绘图效率
    if downsample_factor > 1:
        signal = real_part[::downsample_factor]
        samples_per_symbol //= downsample_factor  # 更新每符号的采样点数

    # 计算眼图的步长和截取数据
    step_size = samples_per_symbol
    truncated_signal = signal[:step_size * total_symbols]  # 截取整数个符号的信号
    N = len(rec_wave)
    y = truncated_signal  # [:(samples_per_symbol * 1000)]
    eyediagram(y, 4, widget, offset=samples_per_symbol, cmap=plt.cm.coolwarm)

def on_click(event, widget):
    x_click = event.xdata
    y_click = event.ydata

    if x_click is None or y_click is None:
        return

    # 找到距离最近的频率点
    idx = (np.abs(widget.xf - x_click)).argmin()
    freq = widget.xf[idx]
    amp = widget.yf[idx]

    # 使用新的格式化函数
    formatted_freq = format_scientific_notation(freq)
    formatted_amp = format_scientific_notation(amp)

    # 设置QLabel的字体为等线字体，14号字
    font = QFont("DengXian", 14)  # DengXian 是等线字体的名称


    # 显示点击点的频率和幅度信息
    label = QLabel(f'Frequency: {formatted_freq} Hz\nAmplitude: {formatted_amp}', widget)
    label.setFont(font)
    label.setStyleSheet("QLabel { color: yellow; background-color: black; }")
    # 设置 QLabel 的固定宽度和高度
    label.setFixedWidth(200)  # 根据需要调整宽度
    label.setFixedHeight(50)  # 根据需要调整高度

    # 确定 QLabel 的大小
    label_width = label.width()
    label_height = label.height()
    widget_height = widget.height()

    # 计算QLabel的位置
    label_x = event.x
    label_y = event.y

    # 防止QLabel超出窗口范围
    if label_x + label_width > widget.width():
        label_x = widget.width() - label_width
    if label_y + label_height > widget.height():
        label_y = widget.height() - label_height
    label_y = widget_height - label_y
    label.move(label_x, label_y)
    label.show()

    # 自动隐藏信息标签
    QTimer.singleShot(2000, label.hide)  # 2秒后自动隐藏

    # 更新文本框内容，显示频率和幅度
    # 数据转化成科学计数法
    # formatted_freq_input = format_scientific_notation(freq)
    formatted_amp_input = format_scientific_notation(amp)
    # textbox1.setText(formatted_freq_input)
    # textbox2.setText(formatted_amp_input)


def on_click_star(event, widget, rec_wave, textbox1, textbox2, sc1):
    # 确保点击事件发生在坐标轴区域内
    if event.inaxes is not None:
        # 获取点击点的坐标
        click_x, click_y = event.xdata, event.ydata

        # 计算点击点的相位角
        click_phase = np.arctan2(click_y, click_x)

        # 确定点击点所在的象限
        if click_x >= 0 and click_y >= 0:
            quadrant = 1
        elif click_x < 0 and click_y >= 0:
            quadrant = 2
        elif click_x < 0 and click_y < 0:
            quadrant = 3
        else:
            quadrant = 4

        # 过滤星座点，只保留在相同象限内的点
        if quadrant == 1:
            mask = (rec_wave.real >= 0) & (rec_wave.imag >= 0)
        elif quadrant == 2:
            mask = (rec_wave.real < 0) & (rec_wave.imag >= 0)
        elif quadrant == 3:
            mask = (rec_wave.real < 0) & (rec_wave.imag < 0)
        else:
            mask = (rec_wave.real >= 0) & (rec_wave.imag < 0)

        filtered_points = rec_wave[mask]

        if len(filtered_points) == 0:
            return
        # 计算距离
        distances = np.sqrt((filtered_points.real - click_x) ** 2 + (filtered_points.imag - click_y) ** 2)
        # 找到最近的点
        closest_point_index = np.argmin(distances)
        closest_x = filtered_points.real[closest_point_index]
        closest_y = filtered_points.imag[closest_point_index]
        # closest_x = rec_wave.real[closest_point_index]
        # closest_y = rec_wave.imag[closest_point_index]
        textbox1.setText(str(f'{closest_x:.2f}'))
        textbox2.setText(str(f'{closest_y:.2f}'))

        # 将星座图坐标转换为窗口中的像素坐标
        display_coords = sc1.ax.transData.transform((closest_x, closest_y))
        label_x, label_y = display_coords[0], display_coords[1]

        # 设置QLabel的字体为等线字体，14号字
        font = QFont("DengXian", 14)  # DengXian 是等线字体的名称

        # 显示点击点的频率和幅度信息
        label = QLabel(f'I: {closest_x:.2f} \nN: {closest_y:.2f}', widget)
        label.setFont(font)
        label.setStyleSheet("QLabel { color: #ff6020; background-color: black; }")

        # 设置 QLabel 的固定宽度和高度
        label.setFixedWidth(125)  # 根据需要调整宽度
        label.setFixedHeight(50)  # 根据需要调整高度

        # 确定 QLabel 的大小
        label_width = label.width()
        label_height = label.height()
        widget_height = widget.height()

        # 防止QLabel超出窗口范围
        if label_x + label_width > widget.width():
            label_x = widget.width() - label_width
        if label_y + label_height > widget.height():
            label_y = widget.height() - label_height

        label_y = widget_height - int(label_y)
        label.move(int(label_x), int(label_y))
        label.show()

        # 自动隐藏信息标签
        QTimer.singleShot(2000, label.hide)  # 2秒后自动隐藏


def reset_view(self):
    # 还原视图到初始状态
    if self.xf is not None and self.yf is not None:
        self.ax.set_xlim([self.xf[0], self.xf[-1]])  # 恢复到全频率范围
        self.ax.set_ylim([min(self.yf), max(self.yf)])  # 恢复到全幅度范围
        self.draw()  # 重绘图像