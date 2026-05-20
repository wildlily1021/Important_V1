import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import welch

from photo_save import photo_save, photo_save_test, photo_save_final, photo_save_scipy
from scipy.signal import hilbert, correlate, welch
from scipy.fft import fftshift, fft
from scipy.ndimage import uniform_filter1d
import pywt


def estimate_symbol_rate_from_waveform(rec_wave, Fs):
    signal_array = np.asarray(rec_wave)
    if np.iscomplexobj(signal_array) and np.max(np.abs(np.imag(signal_array))) > 1e-12:
        envelope = np.abs(signal_array)
    else:
        envelope = np.abs(hilbert(np.real(signal_array)))

    if envelope.size < 4:
        return 0.0

    diff_envelope = np.abs(envelope[1:]) - np.abs(envelope[:-1])
    fft_diff = np.abs(fft(np.concatenate(([0], diff_envelope))))
    fft_diff_fast = uniform_filter1d(np.abs(fft_diff), size=1)
    fft_diff_slow = uniform_filter1d(np.abs(fft_diff), size=100)
    fft_ratio = fft_diff_fast / np.maximum(fft_diff_slow, np.finfo(float).eps)

    sample_count = len(signal_array)
    delta_f = Fs / sample_count
    search_slice = fft_ratio[: int(sample_count // 2)]
    peak_index = int(np.argmax(search_slice)) if search_slice.size else 0
    return float(peak_index * delta_f)


def signal_create(Fs, Fc, Rs, SNR, Modulation):
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

    def smooth(data, window_size=80, method='simple'):
        """
        对数据进行平滑处理

        :param data: 原始数据
        :param window_size: 窗口大小，默认为3
        :param method: 平滑处理方法，可选'simple'或'weighted'，默认为'simple'
        :return: 平滑处理后的数据
        """
        if method == 'simple':
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode='valid')
        elif method == 'weighted':
            weights = np.arange(1, window_size + 1)
            weights = weights / np.sum(weights)
            return np.convolve(data, weights, mode='valid')
        else:
            raise ValueError("Unsupported smoothing method")

    # def qammod(data, M):
    #     constellation = np.array([0])
    #     if M == 4:
    #         # 4QAM 星座图
    #         I = np.array([-1, 1])
    #         Q = np.array([-1, 1])
    #         constellation = np.array([i + 1j * q for i in I for q in Q])
    #         # constellation /= np.sqrt((np.abs(constellation) ** 2).mean())
    #     elif M == 16:
    #         # 16QAM 星座图
    #         constellation = np.array([
    #             (-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
    #             (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
    #             (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
    #             (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j),
    #         ]) # / np.sqrt(10)
    #     elif M == 64:
    #         # 64QAM 星座图
    #         I = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    #         Q = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    #         constellation = np.array([i + 1j * q for i in I for q in Q])
    #         # constellation /= np.sqrt((np.abs(constellation) ** 2).mean())
    #     elif M == 256:
    #         # 256QAM 星座图
    #         I = np.arange(-15, 16, 2)
    #         Q = np.arange(-15, 16, 2)
    #         constellation = np.array([i + 1j * q for i in I for q in Q])
    #         # constellation /= np.sqrt((np.abs(constellation) ** 2).mean())
    #
    #     # Map the data to the constellation points
    #     mapped_data = constellation[(data % M).astype(int)]
    #     return mapped_data

    def AWGN(x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x * np.conj(x)) / len(x)
        npower = xpower / snr
        if isinstance(x[0], complex):
            noise = (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * np.sqrt(0.5 * npower)  # Complex Number
        else:
            noise = np.random.randn(len(x)) * np.sqrt(npower)  # Real Number
        return x + noise

    # 信号调制
    # Modulation = 1  # QPSK = 1; 8PSK = 2; 16QAM = 3; 64QAM = 4; 256QAM = 5; 16PSK = 6; 4QAM = 7; BPSK = 8
    def Modulation_select(Modulation, symbol_num, rate):
        """
        通用调制选择函数，使用冲激+零插值上采样方式。
        """
        mod_order_map = {
            1: 4,       # QPSK
            2: 8,       # 8PSK
            3: 16,      # 16QAM
            4: 64,      # 64QAM
            5: 256,     # 256QAM
            6: 16,      # 16PSK
            7: 2,       # BPSK
            # 9: 2,  # OOK
            # 10: 4,  # 4ASK
            # 11: 8,  # 8ASK
            # 12: 16,  # 16PSK
            # 13: 32,  # 32PSK
            # 14: 16,  # 16APSK
            # 15: 32,  # 32APSK
            # 16: 64,  # 64APSK
            # 17: 128,  # 128APSK
            # 18: 32,  # 32QAM
            # 19: 128,  # 128QAM
            # 20: 2,  # GMSK
            # 21: 4,  # OQPSK
            # 22: 256  # 256QAM (alias, optional)
        }

        if Modulation not in mod_order_map:
            raise ValueError("Unsupported Modulation")

        M = mod_order_map[Modulation]
        data_rand = np.random.randint(0, M, size=symbol_num)
        up_len = symbol_num * rate
        data_modu = np.zeros(up_len, dtype=complex)

        if Modulation in [7]:  # BPSK
            mapped = 2 * data_rand - 1
            data_modu[::rate] = mapped.astype(complex)

        elif Modulation in [1, 21]:  # QPSK, 4QAM, OQPSK
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

        elif Modulation == 20:
            raise NotImplementedError("GMSK not implemented")

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
        idx = index % total
        symbol = []
        base = 0
        for r_idx, n in enumerate(ring):
            if idx < base + n:
                radius = 1 + 0.3 * r_idx
                angle = 2 * np.pi * (idx - base) / n
                symbol = radius * np.exp(1j * angle)
                break
            base += n
        return symbol / np.sqrt(np.mean(np.abs(symbol) ** 2))

    def SNR_Analysis(rec_wave, Fs):
        global i_up, i_down
        N_freq = 1024  # FFT点数
        # 计算功率谱密度（PSD）
        noverlap = 40  # 窗口重叠
        nfft = 1024  # FFT 点数
        # 设置图像大小和DPI

        # 创建一个新的图形
        # plt.figure()
        # 计算STFT
        fre_psdx, psdx = welch(rec_wave, fs=Fs, nperseg=N_freq, noverlap=noverlap, nfft=N_freq, return_onesided=False)
        positive_mask = fre_psdx >= 0
        fre_psdx = fre_psdx[positive_mask]
        psdx = psdx[positive_mask]
        plt.semilogy(fre_psdx, psdx)
        # 分段计算
        count = 200  # 分段数目
        split_length = N_freq // count  # 每一段的长度
        data_count = np.zeros(count)

        for k in range(count):  # 分段求和
            data_count[k] = np.sum(psdx[split_length * k:split_length * (k + 1)])

        # 计算差分
        diff_count = data_count[2: count] - data_count[1: count - 1] # np.diff(data_count)  # 求分段和差分

        # 查找峰值和谷值
        diff_max = np.max(diff_count)  # 差分峰值
        i_max = np.argmax(diff_count)  # 差分峰值位置
        diff_min = np.min(diff_count)  # 差分谷值
        i_min = np.argmin(diff_count)  # 差分谷值位置

        # 寻找开始点
        for k in range(1, i_max + 1):
            if diff_count[i_max - k] < 0:
                i_up = (i_max - k) * split_length
                break
            else:
                i_up = 0

        # 寻找结束点
        for k in range(i_min + 1, len(diff_count)):
            if diff_count[k] > 0:
                i_down = k * split_length
                break
            else:
                i_down = len(psdx)

        # 计算频率
        fre_up = i_up * Fs / N_freq  # 开始频率
        fre_down = i_down * Fs / N_freq  # 结束频率
        valid_bandwidth = fre_down - fre_up  # 信号有效带宽
        # 计算噪声功率谱和信噪比估计值
        N_0 = np.mean(psdx[:i_up]) # 噪声功率谱密度估计
        delta_f = Fs / N_freq  # 频率间隔
        SIGNAL = np.sum(psdx[i_up:i_down])
        S_N_estimation = 10 * np.log10(((SIGNAL * delta_f) -N_0 * valid_bandwidth) / (N_0*Fs))

        print(f"信噪比估计值: {S_N_estimation} dB")
        # plt.close("all")
        return S_N_estimation

    def Rs_Analysis(rec_wave, Fs):
        rec_wave_RS = np.real(rec_wave)

        # Hilbert变换
        Hx_rec_wave = hilbert(rec_wave_RS)
        bl = np.abs(Hx_rec_wave)

        # 差分计算
        diff_Hx = np.abs(bl[1:]) - np.abs(bl[:-1])

        # FFT计算
        fft_diff_Hx = np.abs(fft(np.concatenate(([0], diff_Hx))))

        # 平滑处理
        fft_diff_Hx1 = uniform_filter1d(np.abs(fft_diff_Hx), size=1)
        fft_diff_Hx2 = uniform_filter1d(np.abs(fft_diff_Hx), size=100)
        fft_diff_Hx = fft_diff_Hx1 / fft_diff_Hx2

        # 频率计算
        delta_f = Fs / N
        f = np.arange(0, Fs / 2, delta_f)
        D_max, idex = np.max(fft_diff_Hx[:int(N // 2)]), np.argmax(fft_diff_Hx[:int(N // 2)])
        f_idex = idex * delta_f
        symbol_cen_record = f_idex
        return symbol_cen_record

    # 参数配置
    N = 1e6
    # plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 增加图像大小和DPI
    rate = round(Fs / Rs)  # 每符号采样点数
    symbol_num = int(N // rate)  # 符号数
    N_index = symbol_num * rate
    t = np.arange(0, N_index) / Fs  # 采样时间矢量
    rolloff = 0.35  # 滚降系数
    span = 6  # 冲激响应截断符号数
    # 根升余弦滤波器
    filter1 = rcosdesign(rolloff, span, rate)
    # print(data_modu[:6])
    # 信号成型——不加噪声
    data_modu = Modulation_select(Modulation, symbol_num, rate)
    data_tran = signal.convolve(data_modu, filter1, mode='same')
    signal_amp = 0.2
    signal_wave = signal_amp * data_tran * np.exp(1j * 2 * np.pi * Fc * t)
    print("signal_wave:", signal_wave[0:3])
    np.savetxt('signal_wave.txt', signal_wave, fmt='%.8f')

    # 信号加噪
    signal_noise = AWGN(signal_wave, SNR)
    #         signal_noise = signal_wave
    # print("after_add_noise:", signal_noise[:3])
    # 图像生成与原始数据载入
    rec_wave = signal_noise
    Fc_text = (Fc / 1e9)
    Rs_text = (Rs / 1e9)
    SNR_text = round(SNR)
    Modulation_text = round(Modulation)
    np.savetxt(f'signal_{Fc_text:.2f}_{Rs_text:.2f}_{SNR_text}_{Modulation_text}.txt', rec_wave, fmt='%.8f')
    print("rec_wave:", rec_wave[0:3])
    num = 1
    Band = (1 - rolloff / 2) * Rs
    Fs_index = Fs
    Rs_index = Rs / 1e9
    [min_val, max_val] = photo_save(rec_wave, Fs_index, num, Rs_index)
    plt.clf()
    plt.close("all")
    # 计算每个采样点的幅度
    magnitude = np.abs(rec_wave)
    # 计算平均幅度
    magnitude_GUJI = np.mean(magnitude)
    SNR_GUJI = SNR_Analysis(rec_wave, Fs)
    plt.clf()
    plt.close("all")
    RS_GUJI = estimate_symbol_rate_from_waveform(rec_wave, Fs)
    plt.clf()
    plt.close("all")
    # return Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI
    return [Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI, min_val, max_val]

def signal_read(rec_wave, Fs, keep_negative_frequencies=False):
    #rec_wave接收到的信号，Fs采样频率

    def SNR_Analysis(rec_wave, Fs):
        global i_up, i_down
        N_freq = 1024  # FFT点数
        # 计算功率谱密度（PSD）
        noverlap = 40  # 窗口重叠
        nfft = 1024  # FFT 点数
        # 设置图像大小和DPI

        # 创建一个新的图形
        # plt.figure()
        # 计算STFT
        fre_psdx, psdx = welch(rec_wave, fs=Fs, nperseg=N_freq, noverlap=noverlap, nfft=N_freq, return_onesided=False)
        positive_mask = fre_psdx >= 0
        fre_psdx = fre_psdx[positive_mask]
        psdx = psdx[positive_mask]
        plt.semilogy(fre_psdx, psdx)
        # 分段计算
        count = 200  # 分段数目
        split_length = N_freq // count  # 每一段的长度
        data_count = np.zeros(count)

        for k in range(count):  # 分段求和
            data_count[k] = np.sum(psdx[split_length * k:split_length * (k + 1)])

        # 计算差分
        diff_count = data_count[2: count] - data_count[1: count - 1] # np.diff(data_count)  # 求分段和差分

        # 查找峰值和谷值
        diff_max = np.max(diff_count)  # 差分峰值
        i_max = np.argmax(diff_count)  # 差分峰值位置
        diff_min = np.min(diff_count)  # 差分谷值
        i_min = np.argmin(diff_count)  # 差分谷值位置

        # 寻找开始点
        for k in range(1, i_max + 1):
            if diff_count[i_max - k] < 0:
                i_up = (i_max - k) * split_length
                break
            else:
                i_up = 0

        # 寻找结束点
        for k in range(i_min + 1, len(diff_count)):
            if diff_count[k] > 0:
                i_down = k * split_length
                break
            else:
                i_down = len(psdx)

        # 计算频率
        fre_up = i_up * Fs / N_freq  # 开始频率
        fre_down = i_down * Fs / N_freq  # 结束频率
        valid_bandwidth = fre_down - fre_up  # 信号有效带宽
        # 计算噪声功率谱和信噪比估计值
        N_0 = np.mean(psdx[:i_up]) # 噪声功率谱密度估计
        delta_f = Fs / N_freq  # 频率间隔
        SIGNAL = np.sum(psdx[i_up:i_down])
        S_N_estimation = 10 * np.log10(((SIGNAL * delta_f) -N_0 * valid_bandwidth) / (N_0*Fs))

        print(f"信噪比估计值: {S_N_estimation} dB")
        # plt.close("all")
        return S_N_estimation

    def Rs_Analysis(rec_wave, Fs):
        rec_wave_RS = np.real(rec_wave)

        # Hilbert变换
        Hx_rec_wave = hilbert(rec_wave_RS)
        bl = np.abs(Hx_rec_wave)

        # 差分计算
        diff_Hx = np.abs(bl[1:]) - np.abs(bl[:-1])

        # FFT计算
        fft_diff_Hx = np.abs(fft(np.concatenate(([0], diff_Hx))))

        # 平滑处理
        fft_diff_Hx1 = uniform_filter1d(np.abs(fft_diff_Hx), size=1)
        fft_diff_Hx2 = uniform_filter1d(np.abs(fft_diff_Hx), size=100)
        fft_diff_Hx = fft_diff_Hx1 / fft_diff_Hx2

        # 频率计算
        N = len(rec_wave)
        delta_f = Fs / N
        f = np.arange(0, Fs / 2, delta_f)
        D_max, idex = np.max(fft_diff_Hx[:int(N // 2)]), np.argmax(fft_diff_Hx[:int(N // 2)])
        f_idex = idex * delta_f
        symbol_cen_record = f_idex
        return symbol_cen_record
    # 参数配置
    plt.clf()
    plt.close("all")
    # plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 增加图像大小和DPI
    np.savetxt('rec_wave_read.txt', rec_wave, fmt='%.8f')
    print("rec_wave:", rec_wave[0:3])
    num = 2
    Fs_index = Fs
    Rs_index = 0
    photo_save_scipy(
        rec_wave,
        Fs_index,
        num,
        Rs_index,
        keep_negative_frequencies=False,
        enable_local_recognition=False,
    )
    plt.clf()
    plt.close("all")
    # 计算每个采样点的幅度
    magnitude = np.abs(rec_wave)
    # 计算平均幅度
    magnitude_GUJI = np.mean(magnitude)
    SNR_GUJI = SNR_Analysis(rec_wave, Fs)
    plt.clf()
    plt.close("all")
    RS_GUJI = estimate_symbol_rate_from_waveform(rec_wave, Fs)
    plt.clf()
    plt.close("all")
    # return Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI
    return [Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI]

def signal_create_test(Fs, Fc, Rs, SNR, Modulation, image_dir):
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

    def smooth(data, window_size=80, method='simple'):
        """
        对数据进行平滑处理

        :param data: 原始数据
        :param window_size: 窗口大小，默认为3
        :param method: 平滑处理方法，可选'simple'或'weighted'，默认为'simple'
        :return: 平滑处理后的数据
        """
        if method == 'simple':
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode='valid')
        elif method == 'weighted':
            weights = np.arange(1, window_size + 1)
            weights = weights / np.sum(weights)
            return np.convolve(data, weights, mode='valid')
        else:
            raise ValueError("Unsupported smoothing method")

    def qammod(data, M):
        constellation = np.array([0])
        if M == 16:
            constellation = np.array([
                (-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
                (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
                (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
                (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j),
            ]) / np.sqrt(10)
        elif M == 64:
            # QAM-64 星座图
            I = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            Q = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = np.array([(i + 1j * q) for i in I for q in Q])
            # 正则化，使星座图的功率为1
            constellation /= np.sqrt((np.abs(constellation) ** 2).mean())

        # Map the data to the constellation points
        mapped_data = constellation[(data % M).astype(int)]
        print("Mapped data:", mapped_data)
        return mapped_data

    def AWGN(x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x * np.conj(x)) / len(x)
        npower = xpower / snr
        if isinstance(x[0], complex):
            noise = (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * np.sqrt(0.5 * npower)  # Complex Number
        else:
            noise = np.random.randn(len(x)) * np.sqrt(npower)  # Real Number
        return x + noise

    # 信号调制
    # Modulation = 1  # QPSK = 1; 8PSK = 2; 16QAM = 3; 64QAM = 4
    def Modulation_select(Modulation):
        data_modu = 0
        if Modulation == 1:
            #     data_rand = np.random.randint(0, 4, size=symbol_num)
            data_rand = np.random.randint(0, 4, size=symbol_num)
            data_upsample = np.repeat(data_rand, rate)
            data_modu = np.exp(1j * np.pi / 2 * data_upsample + 1j * np.pi / 4)

        elif Modulation == 2:
            data_rand = np.random.randint(0, 8, size=symbol_num)
            data_upsample = np.repeat(data_rand, rate)
            data_modu = np.exp(1j * np.pi / 4 * data_upsample + 1j * np.pi / 8)

        elif Modulation == 3:
            data_rand = np.random.randint(0, 16, size=symbol_num)
            graycode16 = np.array([0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10])
            data_upsample = np.repeat(data_rand, rate)
            msg16 = graycode16[data_upsample]
            data_modu = qammod(msg16, 16)

        elif Modulation == 4:
            data_rand = np.random.randint(0, 64, size=symbol_num)
            graycode64 = np.array([
                0, 1, 3, 2, 6, 7, 5, 4, 8, 9, 11, 10, 14, 15, 13, 12,
                24, 25, 27, 26, 30, 31, 29, 28, 16, 17, 19, 18, 22, 23, 21, 20,
                48, 49, 51, 50, 54, 55, 53, 52, 56, 57, 59, 58, 62, 63, 61, 60,
                40, 41, 43, 42, 46, 47, 45, 44, 32, 33, 35, 34, 38, 39, 37, 36
            ])
            data_upsample = np.repeat(data_rand, rate)
            msg64 = graycode64[data_upsample]
            data_modu = qammod(msg64, 64)

        return data_modu

    def SNR_Analysis(rec_wave, Fs):
        global i_up, i_down
        N_freq = 1024  # FFT点数
        # 计算功率谱密度（PSD）
        noverlap = 40  # 窗口重叠
        nfft = 1024  # FFT 点数
        # 设置图像大小和DPI

        # 创建一个新的图形
        # plt.figure()
        # 计算STFT
        fre_psdx, psdx = welch(rec_wave, fs=Fs, nperseg=N_freq, noverlap=noverlap, nfft=N_freq, return_onesided=False)
        positive_mask = fre_psdx >= 0
        fre_psdx = fre_psdx[positive_mask]
        psdx = psdx[positive_mask]
        plt.semilogy(fre_psdx, psdx)
        # 分段计算
        count = 200  # 分段数目
        split_length = N_freq // count  # 每一段的长度
        data_count = np.zeros(count)

        for k in range(count):  # 分段求和
            data_count[k] = np.sum(psdx[split_length * k:split_length * (k + 1)])

        # 计算差分
        diff_count = data_count[2: count] - data_count[1: count - 1] # np.diff(data_count)  # 求分段和差分

        # 查找峰值和谷值
        diff_max = np.max(diff_count)  # 差分峰值
        i_max = np.argmax(diff_count)  # 差分峰值位置
        diff_min = np.min(diff_count)  # 差分谷值
        i_min = np.argmin(diff_count)  # 差分谷值位置

        # 寻找开始点
        for k in range(1, i_max + 1):
            if diff_count[i_max - k] < 0:
                i_up = (i_max - k) * split_length
                break
            else:
                i_up = 0

        # 寻找结束点
        for k in range(i_min + 1, len(diff_count)):
            if diff_count[k] > 0:
                i_down = k * split_length
                break
            else:
                i_down = len(psdx)

        # 计算频率
        fre_up = i_up * Fs / N_freq  # 开始频率
        fre_down = i_down * Fs / N_freq  # 结束频率
        valid_bandwidth = fre_down - fre_up  # 信号有效带宽
        # 计算噪声功率谱和信噪比估计值
        N_0 = np.mean(psdx[:i_up]) # 噪声功率谱密度估计
        delta_f = Fs / N_freq  # 频率间隔
        SIGNAL = np.sum(psdx[i_up:i_down])
        S_N_estimation = 10 * np.log10(((SIGNAL * delta_f) -N_0 * valid_bandwidth) / (N_0*Fs))

        print(f"信噪比估计值: {S_N_estimation} dB")
        # plt.close("all")
        return S_N_estimation

    def Rs_Analysis(rec_wave, Fs):
        rec_wave_RS = np.real(rec_wave)

        # Hilbert变换
        Hx_rec_wave = hilbert(rec_wave_RS)
        bl = np.abs(Hx_rec_wave)

        # 差分计算
        diff_Hx = np.abs(bl[1:]) - np.abs(bl[:-1])

        # FFT计算
        fft_diff_Hx = np.abs(fft(np.concatenate(([0], diff_Hx))))

        # 平滑处理
        fft_diff_Hx1 = uniform_filter1d(np.abs(fft_diff_Hx), size=1)
        fft_diff_Hx2 = uniform_filter1d(np.abs(fft_diff_Hx), size=100)
        fft_diff_Hx = fft_diff_Hx1 / fft_diff_Hx2

        # 频率计算
        delta_f = Fs / N
        f = np.arange(0, Fs / 2, delta_f)
        D_max, idex = np.max(fft_diff_Hx[:int(N // 2)]), np.argmax(fft_diff_Hx[:int(N // 2)])
        f_idex = idex * delta_f
        symbol_cen_record = f_idex
        return symbol_cen_record

    # 参数配置
    N = 4e6
    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 增加图像大小和DPI
    rate = round(Fs / Rs)  # 每符号采样点数
    symbol_num = int(N // rate)  # 符号数
    N_index = symbol_num * rate
    t = np.arange(0, N_index) / Fs  # 采样时间矢量
    rolloff = 0.35  # 滚降系数
    span = 6  # 冲激响应截断符号数
    # 根升余弦滤波器
    filter1 = rcosdesign(rolloff, span, rate)
    # print(data_modu[:6])
    # 信号成型——不加噪声
    data_modu = Modulation_select(Modulation)
    data_tran = signal.convolve(data_modu, filter1, mode='same')
    signal_amp = 0.2
    signal_wave = signal_amp * data_tran * np.exp(1j * 2 * np.pi * Fc * t)
    signal_noise = AWGN(signal_wave, SNR)
    # 图像生成与原始数据载入
    rec_wave = signal_noise
    Fs_index = Fs
    Rs_index = Rs / 1e9
    Fc_index = Fc / 1e9
    photo_save_final(rec_wave, Fs_index, Rs_index, Fc_index, SNR, Modulation, image_dir)
    plt.clf()
    plt.close("all")
    SNR_GUJI = SNR_Analysis(rec_wave, Fs)
    plt.clf()
    plt.close("all")
    RS_GUJI = Rs_Analysis(rec_wave, Fs)
    plt.clf()
    plt.close("all")

    # return Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI
    return [Fs, rec_wave, SNR_GUJI, RS_GUJI]

    # return RS_GUJI

def signal_create_test_inter(Fs, Fc, Rs, SNR, Modulation, image_dir):
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

    def smooth(data, window_size=80, method='simple'):
        """
        对数据进行平滑处理

        :param data: 原始数据
        :param window_size: 窗口大小，默认为3
        :param method: 平滑处理方法，可选'simple'或'weighted'，默认为'simple'
        :return: 平滑处理后的数据
        """
        if method == 'simple':
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode='valid')
        elif method == 'weighted':
            weights = np.arange(1, window_size + 1)
            weights = weights / np.sum(weights)
            return np.convolve(data, weights, mode='valid')
        else:
            raise ValueError("Unsupported smoothing method")

    def qammod(data, M):
        constellation = np.array([0])
        if M == 16:
            constellation = np.array([
                (-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
                (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
                (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
                (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j),
            ]) / np.sqrt(10)
        elif M == 64:
            # QAM-64 星座图
            I = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            Q = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = np.array([(i + 1j * q) for i in I for q in Q])
            # 正则化，使星座图的功率为1
            constellation /= np.sqrt((np.abs(constellation) ** 2).mean())

        # Map the data to the constellation points
        mapped_data = constellation[(data % M).astype(int)]
        print("Mapped data:", mapped_data)
        return mapped_data

    def add_interference(signal, Fs, JSR_dB, interference_freq):
        """
        向信号中添加单频干扰
        :param signal: 原始信号（复数或实数）
        :param Fs: 采样率 (Hz)
        :param JSR_dB: 干信比 (dB)
        :param interference_freq: 干扰频率 (Hz)
        :return: 含干扰的信号
        """
        t = np.arange(len(signal)) / Fs  # 时间向量

        # 计算信号功率
        signal_power = np.mean(np.abs(signal) ** 2)

        # 根据JSR计算干扰幅度
        JSR_linear = 10 ** (JSR_dB / 10)
        interference_amplitude = np.sqrt(JSR_linear * signal_power)

        # 生成复数单频干扰（与信号同类型）
        if np.iscomplexobj(signal):
            interference = interference_amplitude * np.exp(1j * 2 * np.pi * interference_freq * t)
        else:
            interference = interference_amplitude * np.sin(2 * np.pi * interference_freq * t)

        return signal + interference
    def AWGN(x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x * np.conj(x)) / len(x)
        npower = xpower / snr
        if isinstance(x[0], complex):
            noise = (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * np.sqrt(0.5 * npower)  # Complex Number
        else:
            noise = np.random.randn(len(x)) * np.sqrt(npower)  # Real Number
        return x + noise

    # 信号调制
    # Modulation = 1  # QPSK = 1; 8PSK = 2; 16QAM = 3; 64QAM = 4
    def Modulation_select(Modulation):
        data_modu = 0
        if Modulation == 1:
            #     data_rand = np.random.randint(0, 4, size=symbol_num)
            data_rand = np.random.randint(0, 4, size=symbol_num)
            data_upsample = np.repeat(data_rand, rate)
            data_modu = np.exp(1j * np.pi / 2 * data_upsample + 1j * np.pi / 4)

        elif Modulation == 2:
            data_rand = np.random.randint(0, 8, size=symbol_num)
            data_upsample = np.repeat(data_rand, rate)
            data_modu = np.exp(1j * np.pi / 4 * data_upsample + 1j * np.pi / 8)

        elif Modulation == 3:
            data_rand = np.random.randint(0, 16, size=symbol_num)
            graycode16 = np.array([0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10])
            data_upsample = np.repeat(data_rand, rate)
            msg16 = graycode16[data_upsample]
            data_modu = qammod(msg16, 16)

        elif Modulation == 4:
            data_rand = np.random.randint(0, 64, size=symbol_num)
            graycode64 = np.array([
                0, 1, 3, 2, 6, 7, 5, 4, 8, 9, 11, 10, 14, 15, 13, 12,
                24, 25, 27, 26, 30, 31, 29, 28, 16, 17, 19, 18, 22, 23, 21, 20,
                48, 49, 51, 50, 54, 55, 53, 52, 56, 57, 59, 58, 62, 63, 61, 60,
                40, 41, 43, 42, 46, 47, 45, 44, 32, 33, 35, 34, 38, 39, 37, 36
            ])
            data_upsample = np.repeat(data_rand, rate)
            msg64 = graycode64[data_upsample]
            data_modu = qammod(msg64, 64)

        return data_modu

    def SNR_Analysis(rec_wave, Fs):
        global i_up, i_down
        N_freq = 1024  # FFT点数
        # 计算功率谱密度（PSD）
        noverlap = 40  # 窗口重叠
        nfft = 1024  # FFT 点数
        # 设置图像大小和DPI

        # 创建一个新的图形
        # plt.figure()
        # 计算STFT
        fre_psdx, psdx = welch(rec_wave, fs=Fs, nperseg=N_freq, noverlap=noverlap, nfft=N_freq, return_onesided=False)
        positive_mask = fre_psdx >= 0
        fre_psdx = fre_psdx[positive_mask]
        psdx = psdx[positive_mask]
        plt.semilogy(fre_psdx, psdx)
        # 分段计算
        count = 200  # 分段数目
        split_length = N_freq // count  # 每一段的长度
        data_count = np.zeros(count)

        for k in range(count):  # 分段求和
            data_count[k] = np.sum(psdx[split_length * k:split_length * (k + 1)])

        # 计算差分
        diff_count = data_count[2: count] - data_count[1: count - 1] # np.diff(data_count)  # 求分段和差分

        # 查找峰值和谷值
        diff_max = np.max(diff_count)  # 差分峰值
        i_max = np.argmax(diff_count)  # 差分峰值位置
        diff_min = np.min(diff_count)  # 差分谷值
        i_min = np.argmin(diff_count)  # 差分谷值位置

        # 寻找开始点
        for k in range(1, i_max + 1):
            if diff_count[i_max - k] < 0:
                i_up = (i_max - k) * split_length
                break
            else:
                i_up = 0

        # 寻找结束点
        for k in range(i_min + 1, len(diff_count)):
            if diff_count[k] > 0:
                i_down = k * split_length
                break
            else:
                i_down = len(psdx)

        # 计算频率
        fre_up = i_up * Fs / N_freq  # 开始频率
        fre_down = i_down * Fs / N_freq  # 结束频率
        valid_bandwidth = fre_down - fre_up  # 信号有效带宽
        # 计算噪声功率谱和信噪比估计值
        N_0 = np.mean(psdx[:i_up]) # 噪声功率谱密度估计
        delta_f = Fs / N_freq  # 频率间隔
        SIGNAL = np.sum(psdx[i_up:i_down])
        S_N_estimation = 10 * np.log10(((SIGNAL * delta_f) -N_0 * valid_bandwidth) / (N_0*Fs))

        print(f"信噪比估计值: {S_N_estimation} dB")
        # plt.close("all")
        return S_N_estimation

    def Rs_Analysis(rec_wave, Fs):
        rec_wave_RS = np.real(rec_wave)

        # Hilbert变换
        Hx_rec_wave = hilbert(rec_wave_RS)
        bl = np.abs(Hx_rec_wave)

        # 差分计算
        diff_Hx = np.abs(bl[1:]) - np.abs(bl[:-1])

        # FFT计算
        fft_diff_Hx = np.abs(fft(np.concatenate(([0], diff_Hx))))

        # 平滑处理
        fft_diff_Hx1 = uniform_filter1d(np.abs(fft_diff_Hx), size=1)
        fft_diff_Hx2 = uniform_filter1d(np.abs(fft_diff_Hx), size=100)
        fft_diff_Hx = fft_diff_Hx1 / fft_diff_Hx2

        # 频率计算
        delta_f = Fs / N
        f = np.arange(0, Fs / 2, delta_f)
        D_max, idex = np.max(fft_diff_Hx[:int(N // 2)]), np.argmax(fft_diff_Hx[:int(N // 2)])
        f_idex = idex * delta_f
        symbol_cen_record = f_idex
        return symbol_cen_record

    # 参数配置
    N = 4e6
    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 增加图像大小和DPI
    rate = round(Fs / Rs)  # 每符号采样点数
    symbol_num = int(N // rate)  # 符号数
    N_index = symbol_num * rate
    t = np.arange(0, N_index) / Fs  # 采样时间矢量
    rolloff = 0.35  # 滚降系数
    span = 6  # 冲激响应截断符号数
    # 根升余弦滤波器
    filter1 = rcosdesign(rolloff, span, rate)
    # print(data_modu[:6])
    # 信号成型——不加噪声
    data_modu = Modulation_select(Modulation)
    data_tran = signal.convolve(data_modu, filter1, mode='same')
    signal_amp = 0.2
    signal_wave = signal_amp * data_tran * np.exp(1j * 2 * np.pi * Fc * t)
    # 添加单频干扰（参数示例）
    JSR_dB = -5  # 干信比设置为10dB（干扰明显）
    interference_freq = Fc + 0.2 * 1e9  # 干扰频率比载波高1kHz（确保在信号带宽内）
    signal_with_interference = add_interference(signal_wave, Fs, JSR_dB, interference_freq)

    signal_noise = AWGN(signal_with_interference, SNR)
    # 图像生成与原始数据载入
    rec_wave = signal_noise
    Fs_index = Fs
    Rs_index = Rs / 1e9
    Fc_index = Fc / 1e9
    photo_save_final(rec_wave, Fs_index, Rs_index, Fc_index, SNR, Modulation, image_dir)
    plt.clf()
    plt.close("all")
    SNR_GUJI = SNR_Analysis(rec_wave, Fs)
    plt.clf()
    plt.close("all")
    RS_GUJI = Rs_Analysis(rec_wave, Fs)
    plt.clf()
    plt.close("all")

    # return Fs, rec_wave, magnitude_GUJI, SNR_GUJI, RS_GUJI
    return [Fs, rec_wave, SNR_GUJI, RS_GUJI]

    # return RS_GUJI
