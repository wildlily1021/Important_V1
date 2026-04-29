import scipy.signal as signal
import numpy as np
from matplotlib import pyplot as plt


def signal_ideal(Fs, Fc, Rs, rec_wave, Modulation, SNR):
    global data_modu

    def calculate_evm(rec_wave, ideal_wave):
        # 提取符号（可以使用峰值检测或其他方法）
        rec_symbols = rec_wave  # 假设rec_wave已经是符号形式

        # 计算误差向量
        error_vector = rec_symbols - ideal_wave

        # 计算误差向量的幅度
        evm_vector = np.abs(error_vector)

        # 计算EVM（以dB表示）
        evm = np.sqrt(np.mean(evm_vector ** 2)) / np.sqrt(np.mean(np.abs(ideal_wave) ** 2))
        evm_dB = 20 * np.log10(evm)
        return [evm, evm_dB]

    def calculate_papr(signal):
        """
        计算接收信号的峰值与平均功率比（PAPR）

        参数:
        signal (numpy array): 接收信号（复数形式）

        返回:
        PAPR (float): 峰值与平均功率比
        """
        # 计算信号瞬时功率
        power = np.abs(signal) ** 2

        # 计算峰值功率
        peak_power = np.max(power)

        # 计算平均功率
        avg_power = np.mean(power)

        # 计算PAPR
        papr = peak_power / avg_power
        return papr

    import numpy as np

    def demodulate_signal(rx_symbols, M):
        if M == 1:  # QPSK
            # 解调过程：根据接收到的信号角度找到最接近的符号
            angles = np.angle(rx_symbols)
            # 确保角度范围在 [0, 2*pi) 内
            normalized_angles = (angles + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi) 区间
            demodulated_symbols = np.round(normalized_angles / (2 * np.pi / 4)) % 4  # 四舍五入并映射到 [0, 3]
        elif M == 2:  # 8PSK
            # 解调过程：根据接收到的信号角度找到最接近的符号
            angles = np.angle(rx_symbols)
            # 确保角度范围在 [0, 2*pi) 内
            normalized_angles = (angles + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi) 区间
            demodulated_symbols = np.round(normalized_angles / (2 * np.pi / 8)) % 8  # 四舍五入并映射到 [0, 7]
        elif M == 3:  # 16QAM
            demodulated_symbols = demodulate_16qam(rx_symbols)  # 16QAM解调
        elif M == 4:  # 64QAM
            demodulated_symbols = demodulate_64qam(rx_symbols)  # 64QAM解调
        else:
            demodulated_symbols = 0
        return demodulated_symbols.astype(int)

    def demodulate_16qam(rx_symbols):
        print(get_16qam_constellation())
        constellation_index = get_16qam_constellation()
        distances = np.abs(rx_symbols[:, np.newaxis] - constellation_index)  # 计算距离
        closest_symbol_index = np.argmin(distances, axis=1)  # 找到最接近的符号
        return closest_symbol_index  # 返回符号的索引

    def demodulate_64qam(rx_symbols):
        constellation_index = get_64qam_constellation()
        distances = np.abs(rx_symbols[:, np.newaxis] - constellation_index)  # 计算距离
        closest_symbol_index = np.argmin(distances, axis=1)  # 找到最接近的符号
        return closest_symbol_index  # 返回符号的索引

    # 16QAM 星座图生成
    def get_16qam_constellation():
        # 添加归一化因子，确保星座点功率为1
        normalization_factor = np.sqrt(10)  # 16QAM的归一化系数（平均符号能量为10）
        real_part = np.array([-3, -1, 1, 3]) / normalization_factor
        imag_part = np.array([-3, -1, 1, 3]) / normalization_factor
        constellation = np.array([r + 1j * i for r in real_part for i in imag_part])
        return constellation

    # 64QAM 星座图生成
    def get_64qam_constellation():
        normalization_factor = np.sqrt(42)  # 16QAM的归一化系数（平均符号能量为10）
        real_part = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / normalization_factor  # 实部取值
        imag_part = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / normalization_factor  # 虚部取值
        constellation = np.array([r + 1j * i for r in real_part for i in imag_part])
        return constellation

    def AWGN(x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x * np.conj(x)) / len(x)
        npower = xpower / snr
        if isinstance(x[0], complex):
            noise = (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * np.sqrt(0.5 * npower)  # Complex Number
        else:
            noise = np.random.randn(len(x)) * np.sqrt(npower)  # Real Number
        return x + noise

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

    def qammod(data, M):
        constellation = np.array([0])
        if M == 16:
            constellation = np.array([
                (-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
                (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
                (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
                (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j),
            ]) # / np.sqrt(10)
        elif M == 64:
            # QAM-64 星座图
            I = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            Q = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = np.array([(i + 1j * q) for i in I for q in Q])
            # 正则化，使星座图的功率为1
            # constellation /= np.sqrt((np.abs(constellation) ** 2).mean())

        # Map the data to the constellation points
        mapped_data = constellation[(data % M).astype(int)]
        return mapped_data

    # 信号调制
    # Modulation = 1  # QPSK = 1; 8PSK = 2; 16QAM = 3; 64QAM = 4
    def Modulation_select(data, Modulation, rate):
        data_modu = 0
        if Modulation == 1:
            #     data_rand = np.random.randint(0, 4, size=symbol_num)
            data_upsample = np.repeat(data, rate)
            data_modu = np.exp(1j * np.pi / 2 * data_upsample + 1j * np.pi / 4)

        elif Modulation == 2:
            data_upsample = np.repeat(data, rate)
            data_modu = np.exp(1j * np.pi / 4 * data_upsample + 1j * np.pi / 8)

        elif Modulation == 3:
            graycode16 = np.array([0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10])
            data_upsample = np.repeat(data, rate)
            msg16 = graycode16[data_upsample]
            data_modu = qammod(msg16, 16)

        elif Modulation == 4:
            graycode64 = np.array([
                0, 1, 3, 2, 6, 7, 5, 4, 8, 9, 11, 10, 14, 15, 13, 12,
                24, 25, 27, 26, 30, 31, 29, 28, 16, 17, 19, 18, 22, 23, 21, 20,
                48, 49, 51, 50, 54, 55, 53, 52, 56, 57, 59, 58, 62, 63, 61, 60,
                40, 41, 43, 42, 46, 47, 45, 44, 32, 33, 35, 34, 38, 39, 37, 36
            ])
            data_upsample = np.repeat(data, rate)
            msg64 = graycode64[data_upsample]
            data_modu = qammod(msg64, 64)

        return data_modu

    # 1. 信号处理
    N = len(rec_wave)
    rate = int(Fs // Rs)  # 每符号采样点数
    t = np.arange(0, N) / Fs  # 采样时间矢量
    DC_out = np.exp(1j * 2 * np.pi * Fc * t) * rec_wave

    # 根升余弦滤波器
    rolloff = 0.35  # 滚降系数
    span = 6  # 冲激响应截断符号数
    filter1 = rcosdesign(rolloff, span, rate)
    data_tran = signal.convolve(DC_out, filter1, mode='same')

    # # 检查滤波器输出
    # plt.plot(np.abs(data_tran))
    # plt.title("Filtered Signal Amplitude")
    # plt.show()

    # 2. 符号同步
    samplingOffset = 4
    rxSymbols = data_tran[samplingOffset::rate]  # 按符号速率采样

    # 3. 信号功率归一化
    avg_power = np.mean(np.abs(rxSymbols) ** 2)
    rxSymbols = rxSymbols / np.sqrt(avg_power)

    # 4. 信号解调
    demodulatedSymbols = demodulate_signal(rxSymbols, Modulation)

    # # 5. 可视化星座图
    # constellation = get_16qam_constellation()
    # plt.scatter(np.real(rxSymbols), np.imag(rxSymbols), label="Received Symbols")
    # plt.scatter(np.real(constellation), np.imag(constellation), color='red', label="Constellation")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # 4.信号反调制
    data_modu = Modulation_select(demodulatedSymbols, Modulation, rate)
    print("data_modu:", data_modu)
    # 5.加噪绘制星座图
    signal_noise = AWGN(data_modu, SNR)

    [evm_percentage, evm_db] = calculate_evm(signal_noise[1:len(signal_noise)], data_modu[1:len(signal_noise)])
    PARP = calculate_papr(rec_wave)

    return [evm_percentage, evm_db, PARP, signal_noise]



