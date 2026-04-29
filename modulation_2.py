import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import shap
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import convolve
import joblib

def rcosdesign(beta: float, span: float, sps: float, shape='sqrt'):
    if beta < 0 or beta > 1:
        raise ValueError("parameter beta must be float between 0 and 1, got {}"
                         .format(beta))

    if span < 0:
        raise ValueError("parameter span must be positive, got {}"
                         .format(span))

    if sps < 0:
        raise ValueError("parameter sps must be positive, got {}".format(span))

    if ((sps * span) % 2) == 1:
        raise ValueError("rcosdesign:OddFilterOrder {}, {}".format(sps, span))

    if shape != 'normal' and shape != 'sqrt':
        raise ValueError("parameter shape must be either 'normal' or 'sqrt'")

    eps = np.finfo(float).eps

    # design the raised cosine filter

    delay = span * sps / 2
    t = np.arange(-delay, delay)

    if len(t) % 2 == 0:
        t = np.concatenate([t, [delay]])
    t = t / sps
    b = np.empty(len(t))

    if shape == 'normal':
        # design normal raised cosine filter

        # find non-zero denominator
        denom = (1 - np.power(2 * beta * t, 2))
        idx1 = np.nonzero(np.fabs(denom) > np.sqrt(eps))[0]

        # calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1]) * (np.cos(np.pi * beta * t[idx1]) / denom[idx1]) / sps

        # fill in the zeros denominator indices
        idx2 = np.arange(len(t))
        idx2 = np.delete(idx2, idx1)

        b[idx2] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)

    else:
        # design a square root raised cosine filter

        # find mid-point
        idx1 = np.nonzero(t == 0)[0]
        if len(idx1) > 0:
            b[idx1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)

        # find non-zero denominator indices
        idx2 = np.nonzero(np.fabs(np.fabs(4 * beta * t) - 1) < np.sqrt(eps))[0]
        if idx2.size > 0:
            b[idx2] = 1 / (2 * np.pi * sps) * (
                    np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
                    - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
                    + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta))
            )

        # fill in the zeros denominator indices
        ind = np.arange(len(t))
        idx = np.unique(np.concatenate([idx1, idx2]))
        ind = np.delete(ind, idx)
        nind = t[ind]

        b[ind] = -4 * beta / sps * (np.cos((1 + beta) * np.pi * nind) +
                                    np.sin((1 - beta) * np.pi * nind) / (4 * beta * nind)) / (
                         np.pi * (np.power(4 * beta * nind, 2) - 1))

    # normalize filter energy
    b = b / np.sqrt(np.sum(np.power(b, 2)))
    return b
def demodulate_signal(rx_symbols, M):
    if M == 1:  # QPSK
        demodulated_symbols = demodulate_psk(rx_symbols, 4)
    elif M == 2:  # 8PSK
        demodulated_symbols = demodulate_psk(rx_symbols, 8)
    elif M == 3:  # 16QAM
        demodulated_symbols = demodulate_16qam(rx_symbols)  # 16QAM解调
    elif M == 4:  # 64QAM
        demodulated_symbols = demodulate_64qam(rx_symbols)  # 64QAM解调
    elif M == 5:  # 256QAM
        demodulated_symbols = demodulate_256qam(rx_symbols)  # 256QAM解调
    elif M == 6:  # 16PSK
        demodulated_symbols = demodulate_psk(rx_symbols, 16)  # 16PSK解调
    elif M == 7:  # 16PSK
        demodulated_symbols = demodulate_4qam(rx_symbols)  # 16PSK解调
    elif M == 8:  # 32QAM
        demodulated_symbols = demodulate_32qam(rx_symbols)  # 16PSK解调
    elif M == 9:  # BPSK
        demodulated_symbols = demodulate_bpsk(rx_symbols)
        # 16PSK解调
    else:
        demodulated_symbols = 0
    return demodulated_symbols.astype(int)

def demodulate_bpsk(rx):
    # BPSK: 实部>0判1，否则判0
    bits = (np.real(rx) > 0).astype(int)
    return bits
def demodulate_psk(rx, order):
    """
    通用PSK解调：将相位分区并映射到 [0, order-1]
    """
    angles = np.angle(rx)                    # [-pi, pi)
    # 归一化到 [0, 2pi)
    ang = (angles + 2*np.pi) % (2*np.pi)
    # 每个符号的相位宽度
    step = 2*np.pi / order
    # 对每个符号做量化
    symbols = np.round(ang / step + 0.5) % order
    return symbols.astype(int)

def demodulate_16qam(rx_symbols):
    C = get_qam_constellation(16)
    return _demodulate_by_constellation(rx_symbols, C)

def demodulate_4qam(rx_symbols):
    C = get_qam_constellation(4)
    return _demodulate_by_constellation(rx_symbols, C)

def demodulate_32qam(rx_symbols):
    C = get_qam_constellation(32)
    return _demodulate_by_constellation(rx_symbols, C)

def demodulate_64qam(rx_symbols):
    C = get_qam_constellation(64)
    return _demodulate_by_constellation(rx_symbols, C)

def demodulate_256qam(rx_symbols):
    C = get_qam_constellation(256)
    return _demodulate_by_constellation(rx_symbols, C)

def _demodulate_by_constellation(rx, constellation):
    """
    通用QAM解调：对每个接收点，找最近的星座点索引
    """
    # 计算距离矩阵：len(rx) x len(constellation)
    dists = np.abs(rx[:, None] - constellation[None, :])
    # 取最小距离对应的索引
    idx = np.argmin(dists, axis=1)
    return idx.astype(int)

def get_qam_constellation(M):
    """
    生成归一化的M-QAM星座点，支持4,16,32,64,256
    """
    if M == 4:
        # 4QAM == QPSK幅度版
        levels = np.array([-1, 1])
    elif M == 16:
        levels = np.array([-3, -1, 1, 3])
    elif M == 32:
        # 简单取5x7网格前32点
        Ix = np.array([-5, -3, -1, 1, 3])
        Qx = np.array([-3, -1, 1, 3, 5, 7, 9])
        pts = np.array([i+1j*q for i in Ix for q in Qx])
        levels = None
        constellation = pts[:32]
    elif M == 64:
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    elif M == 256:
        levels = np.arange(-15, 16, 2)
    else:
        raise ValueError(f"Unsupported QAM size M={M}")

    if M in (4, 16, 64, 256):
        # 笛卡尔积生成星座
        C = np.array([r + 1j*q for r in levels for q in levels])
    else:
        C = constellation  # M==32

    # 均值归一化功率为1
    norm = np.sqrt(np.mean(np.abs(C)**2))
    return C / norm

def signal_create_test(Fs, Fc, Rs, SNR, Modulation):
    # """ Raised cosine FIR filter design
    # Calculates square root raised cosine FIR
    # filter coefficients with a rolloff factor of `beta`. The filter is
    # truncated to `span` symbols and each symbol is represented by `sps`
    # samples. rcosdesign designs a symmetric filter. Therefore, the filter
    # order, which is `sps*span`, must be even. The filter energy is one.
    # Keyword arguments:
    # beta  -- rolloff factor of the filter (0 <= beta <= 1)
    # span  -- number of symbols that the filter spans
    # sps   -- number of samples per symbol
    # shape -- `normal` to design a normal raised cosine FIR filter or
    #          `sqrt` to design a sqre root raised cosine filter
    # """
    global data_modu, SNR_GUJI, RS_GUJI
    plt.clf()
    plt.close("all")
    def rcosdesign(beta: float, span: float, sps: float, shape='sqrt'):
        if beta < 0 or beta > 1:
            raise ValueError("parameter beta must be float between 0 and 1, got {}"
                            .format(beta))

        if span < 0:
            raise ValueError("parameter span must be positive, got {}"
                            .format(span))

        if sps < 0:
            raise ValueError("parameter sps must be positive, got {}".format(span))

        if ((sps * span) % 2) == 1:
            raise ValueError("rcosdesign:OddFilterOrder {}, {}".format(sps, span))

        if shape != 'normal' and shape != 'sqrt':
            raise ValueError("parameter shape must be either 'normal' or 'sqrt'")

        eps = np.finfo(float).eps

        # design the raised cosine filter

        delay = span * sps / 2
        t = np.arange(-delay, delay)

        if len(t) % 2 == 0:
            t = np.concatenate([t, [delay]])
        t = t / sps
        b = np.empty(len(t))

        if shape == 'normal':
            # design normal raised cosine filter

            # find non-zero denominator
            denom = (1 - np.power(2 * beta * t, 2))
            idx1 = np.nonzero(np.fabs(denom) > np.sqrt(eps))[0]

            # calculate filter response for non-zero denominator indices
            b[idx1] = np.sinc(t[idx1]) * (np.cos(np.pi * beta * t[idx1]) / denom[idx1]) / sps

            # fill in the zeros denominator indices
            idx2 = np.arange(len(t))
            idx2 = np.delete(idx2, idx1)

            b[idx2] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)

        else:
            # design a square root raised cosine filter

            # find mid-point
            idx1 = np.nonzero(t == 0)[0]
            if len(idx1) > 0:
                b[idx1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)

            # find non-zero denominator indices
            idx2 = np.nonzero(np.fabs(np.fabs(4 * beta * t) - 1) < np.sqrt(eps))[0]
            if idx2.size > 0:
                b[idx2] = 1 / (2 * np.pi * sps) * (
                        np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
                        - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
                        + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta))
                )

            # fill in the zeros denominator indices
            ind = np.arange(len(t))
            idx = np.unique(np.concatenate([idx1, idx2]))
            ind = np.delete(ind, idx)
            nind = t[ind]

            b[ind] = -4 * beta / sps * (np.cos((1 + beta) * np.pi * nind) +
                                        np.sin((1 - beta) * np.pi * nind) / (4 * beta * nind)) / (
                             np.pi * (np.power(4 * beta * nind, 2) - 1))

        # normalize filter energy
        b = b / np.sqrt(np.sum(np.power(b, 2)))
        return b
    def AWGN(x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x * np.conj(x)) / len(x)
        npower = xpower / snr
        if isinstance(x[0], complex):
            noise = (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * np.sqrt(0.5 * npower)  # Complex Number
        else:
            noise = np.random.randn(len(x)) * np.sqrt(npower)  # Real Number
        return x + noise

    def Modulation_select(Modulation, symbol_num, rate):
        """
        通用调制选择函数，使用冲激+零插值上采样方式。
        """
        mod_order_map = {
            1: 4,  # QPSK
            2: 8,  # 8PSK
            3: 16,  # 16QAM
            4: 64,  # 64QAM
            5: 256,  # 256QAM
            6: 16,  # 16PSK
            7: 4,  # 4QAM
            8: 2,  # BPSK
            9: 2,  # OOK
            10: 4,  # 4ASK
            11: 8,  # 8ASK
            12: 16,  # 16PSK
            13: 32,  # 32PSK
            14: 16,  # 16APSK
            15: 32,  # 32APSK
            16: 64,  # 64APSK
            17: 128,  # 128APSK
            18: 32,  # 32QAM
            19: 128,  # 128QAM
            20: 2,  # GMSK
            21: 4,  # OQPSK
            22: 256  # 256QAM (alias, optional)
        }

        if Modulation not in mod_order_map:
            raise ValueError("Unsupported Modulation")

        M = mod_order_map[Modulation]
        data_rand = np.random.randint(0, M, size=symbol_num)
        up_len = symbol_num * rate
        data_modu = np.zeros(up_len, dtype=complex)

        if Modulation in [8]:  # BPSK
            mapped = 2 * data_rand - 1
            data_modu[::rate] = mapped.astype(complex)

        elif Modulation in [1, 7, 21]:  # QPSK, 4QAM, OQPSK
            mapped = np.exp(1j * (np.pi / 4 + data_rand * (np.pi / 2)))
            if Modulation == 21:
                i = np.real(mapped)
                q = np.imag(mapped)
                q = np.roll(q, 1)
                data_modu[::rate] = i + 1j * q
            else:
                data_modu[::rate] = mapped

        elif Modulation in [2, 6, 12, 13]:  # PSK
            phase_map = {2: 8, 6: 16, 12: 16, 13: 32}
            Mpsk = phase_map[Modulation]
            mapped = np.exp(1j * data_rand * (2 * np.pi / Mpsk))
            data_modu[::rate] = mapped

        elif Modulation in [3, 4, 5, 18, 19, 22]:  # QAM
            idx = data_rand ^ (data_rand >> 1)
            data_modu[::rate] = qammod(idx, M)

        elif Modulation in [9]:  # OOK
            data_modu[::rate] = data_rand.astype(float)

        elif Modulation in [10, 11]:  # ASK
            mapped = 2 * data_rand / (M - 1) - 1
            data_modu[::rate] = mapped.astype(float)

        elif Modulation in [14, 15, 16, 17]:  # APSK
            data_modu[::rate] = apskmod(data_rand, M)


        elif Modulation == 20:  # GMSK

            bit_seq = np.random.randint(0, 2, symbol_num)

            data_modu = np.zeros(symbol_num * rate, dtype=complex)

            gmsk_sig = gmskmod(bit_seq, bt=0.3, samples_per_symbol=rate)

            data_modu[:len(gmsk_sig)] = gmsk_sig

        print(f"Mod={Modulation} upsampled power=", np.mean(np.abs(data_modu) ** 2))
        return data_modu

    def qammod(index, M):
        m = int(np.sqrt(M))
        re = 2 * (index % m) - m + 1
        im = 2 * (index // m) - m + 1
        symbols = re + 1j * im
        return symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

    def apskmod(index, M):
        ring_map = {
            16: [4, 12],
            32: [4, 12, 16],
            64: [4, 12, 16, 32],
            128: [8, 16, 32, 72]
        }
        ring = ring_map[M]
        total = sum(ring)
        index = np.asarray(index).flatten()  # 确保是 1D array

        # 为每个输入符号分配所在的 ring
        symbol = np.zeros(index.shape, dtype=complex)
        base = 0
        for r_idx, n in enumerate(ring):
            ring_mask = (index >= base) & (index < base + n)
            local_index = index[ring_mask] - base
            radius = 1 + 0.3 * r_idx
            angle = 2 * np.pi * local_index / n
            symbol[ring_mask] = radius * np.exp(1j * angle)
            base += n

        # 功率归一化
        symbol /= np.sqrt(np.mean(np.abs(symbol) ** 2))
        return symbol

    def gmskmod(bits, bt=0.3, samples_per_symbol=8):
        """
        GMSK调制器。
        bits: 输入比特流 (0,1)
        bt: 带宽-时间乘积（越小越平滑）
        samples_per_symbol: 每个符号对应的采样数
        return: GMSK调制后的复数基带信号
        """
        # 将bits从{0,1}映射到{-1,1}
        bit_symbols = 2 * bits - 1

        # 插值，使每个比特占据 samples_per_symbol 个采样点
        data_upsampled = np.repeat(bit_symbols, samples_per_symbol)

        # 生成高斯滤波器（脉冲整形）
        span = 4  # 滤波器跨度为4个符号
        N = span * samples_per_symbol
        t = np.linspace(-span / 2, span / 2, N)
        alpha = np.sqrt(np.log(2)) / (bt * np.sqrt(2))
        h = np.exp(- (2 * np.pi * alpha * t) ** 2)
        h = h / np.sum(h)  # 单位能量归一化

        # 将输入比特流进行高斯滤波（平滑频率偏移）
        phase = np.pi / 2 * np.cumsum(convolve(data_upsampled, h, mode='same')) / samples_per_symbol

        # 生成 GMSK 信号
        gmsk_signal = np.exp(1j * phase)
        return gmsk_signal

    # 参数配置
    N = 5e4
    plt.figure()
    # fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 增加图像大小和DPI
    rate = round(Fs / Rs)  # 每符号采样点数
    symbol_num = int(N // rate)  # 符号数
    N_index = symbol_num * rate
    t = np.arange(0, N_index) / Fs  # 采样时间矢量
    rolloff = 0.35  # 滚降系数
    span = 6  # 冲激响应截断符号数
    # # 根升余弦滤波器
    # filter1 = rcosdesign(rolloff, span, rate)
    # # print(data_modu[:6])
    # # 信号成型——不加噪声
    data_modu = Modulation_select(Modulation, symbol_num, rate)
    # # print("符号能量：", np.mean(np.abs(data_modu) ** 2))
    # # print(np.unique(np.angle(data_modu)) * 180 / np.pi)  # 查看相位角度是否为 0°, 45°, ..., 315°
    # data_tran = signal.lfilter(filter1, 1.0, data_modu)
    # # print("成型后符号能量：", np.mean(np.abs(data_tran) ** 2))
    # # data_tran = signal.convolve(data_modu, filter1, mode='same')
    # signal_amp = 1
    # signal_wave = signal_amp * data_tran * np.exp(1j * 2 * np.pi * Fc * t)
    signal_wave = data_modu
    signal_noise = AWGN(signal_wave, SNR)
    # 图像生成与原始数据载入
    rec_wave = signal_noise
    return [Fs, rec_wave]


def cumulant_order2(x):
    """
    二阶中心累积量（协方差）
    C20 = E[x^2] - (E[x])^2
    C21 = E[x * x.conj()] - |E[x]|^2  -> 实际是功率
    """
    x = np.asarray(x)
    mean_x = np.mean(x)
    c20 = np.mean(x ** 2) - mean_x ** 2
    c21 = np.mean(x * np.conj(x)) - np.abs(mean_x) ** 2
    return c20, c21


def cumulant_order4(x):
    """
    典型四阶复累积量：
    C40 = E[x^4] - 3*(E[x^2])^2
    C41 = E[x^3 * x*] - 3*E[x^2]*E[x*x*]
    C42 = E[x^2 * (x*)^2] - |E[x^2]|^2 - 2*(E[x * x*])^2
    """
    x = np.asarray(x)
    mean_x = np.mean(x)

    x2 = x ** 2
    x_conj = np.conj(x)
    x2_mean = np.mean(x2)
    x_conj2_mean = np.mean(x_conj ** 2)
    xx_conj_mean = np.mean(x * x_conj)

    # 四阶累积量
    c40 = np.mean(x ** 4) - 3 * x2_mean ** 2
    c41 = np.mean(x ** 3 * x_conj) - 3 * x2_mean * xx_conj_mean
    c42 = np.mean(x2 * (x_conj ** 2)) - np.abs(x2_mean) ** 2 - 2 * (xx_conj_mean ** 2)

    return c40, c41, c42


def cumulant_order6(x):
    """
    计算六阶复高阶累积量：C60, C61, C62, C63
    """
    x = np.asarray(x)
    x_conj = np.conj(x)

    # 一些必要的矩估计
    m2 = np.mean(x ** 2)
    m3 = np.mean(x ** 3)
    m4 = np.mean(x ** 4)
    m5 = np.mean(x ** 5)
    m6 = np.mean(x ** 6)

    m_conj = np.mean(x_conj)
    m_conj2 = np.mean(x_conj ** 2)

    m_xx = np.mean(x * x_conj)
    m_x2_xc2 = np.mean((x ** 2) * (x_conj ** 2))
    m_x3_xc3 = np.mean((x ** 3) * (x_conj ** 3))
    m_x2_xc = np.mean((x ** 2) * x_conj)
    m_x4_xc2 = np.mean((x ** 4) * (x_conj ** 2))

    # C60
    C60 = m6 - 15 * m4 * m2 + 30 * (m2 ** 3)

    # C61
    m_x5_xc = np.mean((x ** 5) * x_conj)
    C61 = m_x5_xc - 10 * m3 * m_x2_xc - 5 * m4 * m_xx + 30 * (m2 ** 2) * m_xx

    # C62
    C62 = m_x4_xc2 - 6 * m2 * m_x2_xc2 - 4 * m4 * (m_conj ** 2) \
          + 12 * (m2 ** 2) * (m_conj ** 2) + 6 * m2 * abs(m_conj) ** 2 * m_xx

    # C63
    C63 = m_x3_xc3 - 9 * m_xx * m_x2_xc2 + 12 * (m_xx ** 3)

    return C60, C61, C62, C63

def cumulant_order8(x):
    """
    计算八阶复高阶累积量 C80（中心累积量）：

    C80 = E[x^8]
         - 28 E[x^6] E[x^2]
         + 35 E[x^4]^2
         + 168 E[x^4] E[x^2]^2
         - 630 E[x^2]^4

    其中所有期望都由样本均值近似。
    """
    x = np.asarray(x)

    # 计算必要的原始矩
    m2 = np.mean(x ** 2)
    m4 = np.mean(x ** 4)
    m6 = np.mean(x ** 6)
    m8 = np.mean(x ** 8)

    # 中心累积量 C80
    C80 = (
            m8
            - 28 * m6 * m2
            + 35 * (m4 ** 2)
            + 168 * m4 * (m2 ** 2)
            - 630 * (m2 ** 4)
    )
    return C80

def predict_modulation(rx_samps):
    """
    给定已成型、下变频、匹配滤波、抽取并剔零后的复数基带符号 rx_samps，
    自动计算特征并调用 clf_rf 预测调制方式（1~8）。
    返回：预测的调制编号（int）
    """
    pipeline = joblib.load('modulation_recognizer.pkl')
    # 1) 计算各阶累积量（取模后展平为标量）
    #    这里只示例 C20, C21, C40, C41, C42, C60, C61, C62, C63, C80
    C20 = np.abs(np.mean(rx_samps**2) - np.mean(rx_samps)**2)
    C21 = np.abs(np.mean(rx_samps * np.conj(rx_samps)) - np.abs(np.mean(rx_samps))**2)
    # 四阶
    m2 = np.mean(rx_samps**2)
    m4 = np.mean(rx_samps**4)
    C40 = np.abs(m4 - 3 * m2**2)
    xx = rx_samps * np.conj(rx_samps)
    m_xx = np.mean(xx)
    m2_xx = np.mean(rx_samps**2 * np.conj(rx_samps))
    C41 = np.abs(np.mean(rx_samps**3 * np.conj(rx_samps)) - 3*m2*m_xx)
    C42 = np.abs(np.mean(rx_samps**2 * np.conj(rx_samps)**2) - np.abs(m2)**2 - 2*m_xx**2)
    # 六阶（示例 C60）
    m6 = np.mean(rx_samps**6)
    # 这里可按需要继续计算 C61, C62, C63、C80...
    # 假设后续 C61, C62, C63, C80 函数已定义：
    C60, C61, C62, C63 = cumulant_order6(rx_samps)
    C80 = cumulant_order8(rx_samps)

    # 2) 构造比值特征
    # 注意都要保证分母不为零
    eps = 1e-9
    f1 = np.abs(C40) / (np.abs(C21))
    f2 = np.abs(C42) / (np.abs(C21))
    f3 = np.abs(C41) / ((np.abs(C42))**2)
    f4 = np.abs(C42) / (np.abs(C21)) ** 2
    f5 = np.abs(C40) / (np.abs(C41))
    f6 = np.abs(C61) / (np.abs(C42))
    f7 = np.abs(C61) / (np.abs(C20))

    # 3) 堆成一维特征向量
    feats = np.array([
        np.abs(C20), np.abs(C21), np.abs(C40), np.abs(C41), np.abs(C42),
        np.abs(C60), np.abs(C61), np.abs(C62), np.abs(C63),
        f1, f2, f3, f4, f5, f6, f7
    ]).reshape(1, -1)

    # # 4) 缺失值填充
    # feats_imp = imputer.transform(feats)
    #
    # # 5) 标准化
    # feats_norm = scaler.transform(feats_imp)
    #
    # # 6) 选取 RFECV 保留的特征
    # feats_sel = feats_norm[:, selected_idx]
    #
    # # 7) 调用模型预测
    # pred = clf_rf.predict(feats_sel)

    # 2) 一行代码调用整个 pipeline
    pred = pipeline.predict(feats)

    return int(pred[0])


Modulations = np.arange(1, 23)  # 8种调制方式
snr_range = np.arange(-10, 11, 0.5)  # -10 到 10 dB
cum20_matrix = np.zeros((22, 42), dtype=complex)
cum21_matrix = np.zeros((22, 42), dtype=complex)
cum40_matrix = np.zeros((22, 42), dtype=complex)
cum41_matrix = np.zeros((22, 42), dtype=complex)
cum42_matrix = np.zeros((22, 42), dtype=complex)
cum60_matrix = np.zeros((22, 42), dtype=complex)
cum61_matrix = np.zeros((22, 42), dtype=complex)
cum62_matrix = np.zeros((22, 42), dtype=complex)
cum63_matrix = np.zeros((22, 42), dtype=complex)
cum80_matrix = np.zeros((22, 42), dtype=complex)

for i, Modulation in enumerate(Modulations):
    for j, SNR in enumerate(snr_range):
        [Fs, rec_wave] = signal_create_test(10e9, 3e9, 2e9, SNR, Modulation)
        N = len(rec_wave)
        rate = round(10e9 / 2e9)  # 每符号采样点数
        t = np.arange(0, N) / 10e9  # 采样时间矢量
        DC_out = rec_wave #  * np.exp(-1j * 2 * np.pi * 3e9 * t)
        #
        # # 根升余弦滤波器
        # rolloff = 0.35  # 滚降系数
        # span = 6  # 冲激响应截断符号数
        # filter1 = rcosdesign(rolloff, span, rate)
        # rx_filt = signal.lfilter(filter1, 1, DC_out)
        # # 7) 延迟补偿 & 抽取
        # delay = (len(filter1) - 1) // 2
        # rx_samps = rx_filt[delay::rate]
        # rx_samps_clean = rx_samps[np.abs(rx_samps) > 2e-3]
        rx_samps_clean = DC_out
        # 计算四阶累积量
        c20, c21 = cumulant_order2(rx_samps_clean)
        cum20_matrix[i, j] = c20
        cum21_matrix[i, j] = c21

        # 计算四阶累积量
        c40, c41, c42 = cumulant_order4(rx_samps_clean)
        cum40_matrix[i, j] = c40
        cum41_matrix[i, j] = c41
        cum42_matrix[i, j] = c42

        c60, c61, c62, c63 = cumulant_order6(rx_samps_clean)
        cum60_matrix[i, j] = c60
        cum61_matrix[i, j] = c61
        cum62_matrix[i, j] = c62
        cum63_matrix[i, j] = c63

        c80 = cumulant_order8(rx_samps_clean)
        cum80_matrix[i, j] = c80

# 1. 调制方式名称映射
mod_names = {
    1: 'QPSK',
    2: '8PSK',
    3: '16QAM',
    4: '64QAM',
    5: '256QAM',
    6: '16PSK',
    7: '4QAM',
    8: 'BPSK',
    9: 'OOK',
   10: '4ASK',
   11: '8ASK',
   12: '16PSK',
   13: '32PSK',
   14: '16APSK',
    15: '32APSK',
    16: '64APSK',
    17: '128APSK',
    18: '32QAM',
    19: '128QAM',
    20: 'GMSK',
    21: 'OQPSK',
    22: '256QAM',
}

# 2. 颜色和点样式映射
psk_color = 'tab:blue'
qam_color = 'tab:red'
marker_map = {
    1: 'o',  # QPSK
    2: 's',  # 8PSK
    6: '^',  # 16PSK
    8: 'd',  # BPSK
    3: 'v',  # 16QAM
    4: '>',  # 64QAM
    5: '<',  # 256QAM
    7: 'p',  # 4QAM
}

# 假设已有这些数据：
# Modulations = [1,2,3,4,5,6,7,8]
# snr_range = np.arange(-10, 11)
# cum20_matrix, cum21_matrix, cum40_matrix, cum41_matrix, cum42_matrix
# 形状均为 (len(Modulations), len(snr_range))
# 先取模，确保所有矩阵都是实数
C20 = np.abs(cum20_matrix)
C21 = np.abs(cum21_matrix)
C40 = np.abs(cum40_matrix)
C41 = np.abs(cum41_matrix)
C42 = np.abs(cum42_matrix)
C60 = np.abs(cum60_matrix)
C61 = np.abs(cum61_matrix)
C62 = np.abs(cum62_matrix)
C63 = np.abs(cum63_matrix)
C80 = np.abs(cum80_matrix)
# 构造比值矩阵
ratio1_C40_C21 = C40 / C21                          # C40 / C21
ratio2_C42_C21 = C42 / C21                          # C42 / C21
ratio3_C40_C21_C21 = C40 / (C21 ** 2)                   # C40 / (C21**2)
ratio4_C40_C41 = C40 / C41                  # (C40 + C42) / C21
ratio6_C60_C42 = C60 / C42
ratio6_C60_C42_C42 = C60 / (C42 ** 2)
ratio6_C61_C42 = C61 / C42
ratio6_C61_C20 = C61 / C20
ratio6_C61_C20_C20 = C61 / C20 / C20
ratio6_C60_C21 = C60 / C21
ratio8_C80_C21 = C80 / C21
ratio8_C80_C42 = C80 / C42
ratio8_C80_C60 = C80 / C60

# 3. 将要绘图的矩阵和名称放入列表
cum_list = [
    (cum20_matrix, 'C20'),
    (cum21_matrix, 'C21'),
    (cum40_matrix, 'C40'),
    (cum41_matrix, 'C41'),
    (cum42_matrix, 'C42'),
    (cum60_matrix, 'C60'),
    (cum61_matrix, 'C61'),
    (cum62_matrix, 'C62'),
    (cum63_matrix, 'C63'),
    (cum80_matrix, 'C80'),
]

# 将四个比值也加入
cum_list += [
    (ratio1_C40_C21, 'C40_C21'),
    (ratio2_C42_C21, 'C42_C21'),
    (ratio3_C40_C21_C21, 'C40_C21^2'),
    (ratio4_C40_C41, 'C40_C41'),
    (ratio6_C60_C42, 'C60_C42'),
    (ratio6_C60_C42_C42, 'C60_C42^2'),
    (ratio6_C61_C42, 'C61_C42'),
    (ratio6_C61_C20, 'C61_C20'),
    (ratio6_C61_C20_C20, 'C61_C20^2'),
    (ratio6_C60_C21, 'C60_C21'),
    (ratio8_C80_C21, 'C80_C21'),
    (ratio8_C80_C42, 'C80_C42'),
    (ratio8_C80_C60, 'C80_C60'),
]
# 1) 构造数据集 X, y
num_mod = len(Modulations)
num_snr = len(snr_range)
N = num_mod * num_snr

# 把 5 种累积量按样本堆叠
features = []
for mat in [cum20_matrix, cum21_matrix, cum40_matrix, cum41_matrix, cum42_matrix]:
    features.append(mat.flatten())
# shape = (5, N) -> (N,5)
X_raw = np.vstack(features).T
# 标签 y：每个调制方式重复 len(snr_range) 次
y = np.repeat(Modulations, num_snr)

# 2) 构造比值特征
# 我们选取以下候选比值（可自行增删）：
# C40/C21, C42/C21, C40/(C21**2), C42/(C21**2), (C40+C42)/C21
# 假设 cum20_matrix, cum21_matrix, … 都是 complex64/128
C20 = np.abs(cum20_matrix).flatten()    # 先取模再展平
C21 = np.abs(cum21_matrix).flatten()
C40 = np.abs(cum40_matrix).flatten()
C41 = np.abs(cum41_matrix).flatten()
C42 = np.abs(cum42_matrix).flatten()
C60 = np.abs(cum60_matrix).flatten()
C61 = np.abs(cum61_matrix).flatten()
C62 = np.abs(cum62_matrix).flatten()
C63 = np.abs(cum63_matrix).flatten()

feat_list = [
    C20,
    C21,
    C40,
    C41,
    C42,
    C60,
    C61,
    C62,
    C63,
    C40 / C21,
    C42 / C21,
    C41 / (C42**2),
    C42 / (C21**2),
    C40 / C41,
    C61 / C42,
    C61 / C20,
]
feature_names = [
    'C20',
    'C21',
    'C40',
    'C41',
    'C42',
    'C60',
    'C61',
    'C62',
    'C63',
    'C40/C21',
    'C42/C21',
    'C41/(C42^2)',
    'C42/(C21^2)',
    'C40/C41',
    'C61/C42',
    'C61/C20',
]

X = np.vstack(feat_list).T  # shape (N, 5_candidate)
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_imputed = np.nan_to_num(X_imputed, nan=np.nanmean(X_imputed), posinf=np.nanmax(X_imputed), neginf=np.nanmin(X_imputed))

# 3) 标准化
scaler = StandardScaler().fit(X)
Xn = scaler.transform(X_imputed)

# 4) LASSO 初筛（L1 多项逻辑回归）

# 1) RFECV + Random Forest feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(
    estimator=rf,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    n_jobs=-1
)
rfecv.fit(Xn, y)
selected_idx = np.where(rfecv.support_)[0]
print("RFECV selected feature indices:", selected_idx)
print("Optimal number of features:", rfecv.n_features_)

# Figure 1: CV accuracy vs. number of features
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
    rfecv.cv_results_['mean_test_score'],
    marker='o'
)
plt.xlabel("Number of Features Selected", fontsize=12)
plt.ylabel("Cross-Validation Accuracy", fontsize=12)
plt.title("RFECV: Accuracy vs. Number of Features", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 2) Feature importance of the final RF on selected features
plt.figure(figsize=(10, 5))
importances = rfecv.estimator_.feature_importances_
plt.bar(range(len(selected_idx)), importances, tick_label=selected_idx)
plt.xlabel("Feature Index (after RFECV selection)", fontsize=12)
plt.ylabel("Feature Importance", fontsize=12)
plt.title("Feature Importances from Random Forest", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3) Retrain RF on selected features and plot confusion matrix
X_selected = Xn[:, selected_idx]
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_selected, y)
y_pred = clf_rf.predict(X_selected)

acc = accuracy_score(y, y_pred)
print(f"Training Accuracy: {acc:.4f}")
cm = confusion_matrix(y, y_pred, labels=clf_rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_rf.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Training Set)", fontsize=14)
plt.tight_layout()
# plt.show()

# 1) 准备好 Xn, y, 以及原始候选特征矩阵 X_imputed
#    以及你在训练时用到的 selected_idx 也可以内嵌在 RFECV

# 2) 构建 Pipeline
pipeline = Pipeline([
    ('imputer',    KNNImputer(n_neighbors=5)),
    ('scaler',     StandardScaler()),
    ('feature_sel', RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        n_jobs=-1
    )),
    ('clf',        RandomForestClassifier(n_estimators=100, random_state=42))
])

# 3) 拟合整个管道
pipeline.fit(X, y)  # 这里 X 是原始的候选特征（未缩放、未选特）矩阵

# 4) 保存到文件
joblib.dump(pipeline, 'modulation_recognizer.pkl')
print("模型已保存到 modulation_recognizer.pkl")
# 使用SHAP解释模型
# 获取特征选择后的数据和模型
clf = pipeline.named_steps['clf']
feature_selector = pipeline.named_steps['feature_sel']
selected_mask = feature_selector.support_
selected_indices = np.where(selected_mask)[0]
selected_feature_names = [feature_names[i] for i in selected_indices]

# 抽取部分数据用于SHAP分析（避免计算过载）
X_sample_raw = X[:500]  # 取前500个样本的原始特征数据
# 获取预处理管道（排除最后的分类器）
preprocessor = pipeline[:-1]

# 预处理数据：填充缺失值 → 标准化 → 特征选择
X_sample_processed = preprocessor.transform(X_sample_raw)

# 获取被选中的特征名称
feature_selector = preprocessor.named_steps['feature_sel']
selected_mask = feature_selector.support_
selected_feature_names = [feature_names[i] for i in np.where(selected_mask)[0]]

plt.close('all')
# 创建SHAP解释器
explainer = shap.TreeExplainer(clf)

# 计算SHAP值（可能较慢）
shap_values = explainer.shap_values(X_sample_processed)

# 绘制SHAP摘要图（所有类别）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample_processed, feature_names=selected_feature_names, class_names=list(mod_names.values()))
plt.title("SHAP Summary Plot for All Modulations")
plt.show()

# 绘制SHAP摘要图（特征重要性条形图）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample_processed, feature_names=selected_feature_names, plot_type="bar")
plt.title("SHAP Feature Importance")
plt.show()

# 可选：绘制单个样本的SHAP决策图（例如第一个样本）
sample_idx = 0
plt.figure(figsize=(10, 4))
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][sample_idx, :],
    X_sample_processed[sample_idx, :],
    feature_names=selected_feature_names,
    matplotlib=True
)
plt.title(f"SHAP Force Plot for Sample {sample_idx}")
plt.show()
# 生成优化后的SHAP摘要图
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_sample_processed,
    feature_names=selected_feature_names,
    class_names=list(mod_names.values()),
    plot_type="bar",
    color=plt.get_cmap("tab10"),  # 使用更鲜明的色系
    show=False
)

# 调整样式
plt.title("Optimized SHAP Feature Importance by Modulation Type", fontsize=14, pad=20)
plt.xlabel("Mean(|SHAP Value|)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)
sns.despine(left=True)  # 去除边框线
plt.tight_layout()

# 保存为EPS文件
plt.savefig("shap_summary.eps", format='eps', dpi=300, bbox_inches='tight')
plt.show()

# 假设 feature_names 是长度 p_1 的列表，对应 X^{(1)} 的列名
ranking = rfecv.ranking_    # array of length p_1
plt.figure(figsize=(6, max(4, len(ranking)*0.3)))
plt.barh(feature_names, ranking, color='skyblue')
plt.xlabel('RFECV Elimination Rank (1 = Final Retained)')
plt.title('Feature Elimination Order by RFECV')
plt.gca().invert_yaxis()  # 最重要的特征排在最上面
plt.tight_layout()
plt.show()

# M = np.array([support for support in support_path])  # shape (T, p_1)
#
# plt.figure(figsize=(10, 6))
# sns.heatmap(M, cmap='Greys', cbar=False,
#             yticklabels=np.arange(1, M.shape[0]+1),
#             xticklabels=feature_names)
# plt.xlabel('Features')
# plt.ylabel('RFECV Iteration')
# plt.title('Feature Retention Path during RFECV')
# plt.tight_layout()
# plt.show()

Modulations_test = np.arange(0.52, 8.52, 0.1)  # 8种调制方式
snr_range = np.arange(-15, 6, 0.5)  # -10 到 10 dB

# for i, Modulation in enumerate(Modulations_test):
#     for j, SNR in enumerate(snr_range):
#         [Fs, rec_wave] = signal_create_test(10e9, 3e9, 2e9, SNR, round(Modulation))
#         N = len(rec_wave)
#         rate = round(10e9 / 2e9)  # 每符号采样点数
#         t = np.arange(0, N) / 10e9  # 采样时间矢量
#         DC_out = rec_wave * np.exp(-1j * 2 * np.pi * 3e9 * t)
#
#         # 根升余弦滤波器
#         rolloff = 0.35  # 滚降系数
#         span = 6  # 冲激响应截断符号数
#         filter1 = rcosdesign(rolloff, span, rate)
#         rx_filt = signal.lfilter(filter1, 1, DC_out)
#         # 7) 延迟补偿 & 抽取
#         delay = (len(filter1) - 1) // 2
#         rx_samps = rx_filt[delay::rate]
#         rx_samps_clean = rx_samps[np.abs(rx_samps) > 2e-3]
#         # --- 使用示例 ---
#         # 假设 rx_samps_clean 是你新采集并预处理好的符号序列
#         mod_type = predict_modulation(rx_samps_clean)
#         print(f"Predicted Modulation Type: {mod_type} ({mod_names[mod_type]})")
# # 训练好 RFECV 后，又训练了最终的随机森林 clf_rf, 并用 selected_idx 筛了特征：
# X_selected = Xn[:, selected_idx]
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_rf.fit(X_selected, y)
#
# # 1) 构造对齐的 snr_array
# snr_array = np.tile(snr_range, len(Modulations))  # (8*21=168,)
#
# # 2) 提取被选中特征
# X_selected = Xn[:, selected_idx]
#
# # 3) 重新训练（如果需要）
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_rf.fit(X_selected, y)
#
# # 4) 针对不同 SNR 画混淆矩阵
# target_snrs = [-10, -5, 0]
# for snr in target_snrs:
#     # 筛出该 SNR 下的样本
#     mask = (snr_array == snr)
#     X_s = X_selected[mask]
#     y_s = y[mask]
#
#     # 预测与准确率
#     y_pred = clf_rf.predict(X_s)
#     acc = accuracy_score(y_s, y_pred)
#     print(f"SNR = {snr} dB, Accuracy = {acc:.3f}")
#
#     # 混淆矩阵
#     cm = confusion_matrix(y_s, y_pred, labels=clf_rf.classes_)
#     disp = ConfusionMatrixDisplay(cm, display_labels=clf_rf.classes_)
#
#     plt.figure(figsize=(6,5))
#     disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
#     plt.title(f"Confusion Matrix @ {snr} dB (acc={acc:.3f})")
#     plt.tight_layout()
#     plt.show()


