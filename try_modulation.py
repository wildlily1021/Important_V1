import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

# 你的 AWGN 加噪函数
def AWGN(x, snr_dB):
    snr = 10 ** (snr_dB / 10.0)
    xpower = np.sum(x * np.conj(x)) / len(x)
    npower = xpower / snr
    if isinstance(x[0], complex):
        noise = (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * np.sqrt(0.5 * npower)
    else:
        noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise


# 1) 参数
Fs = 10e6     # 采样率
Rs = 1e6      # 符号率
Fc = 2e6      # 载波
SNR_dB = 30   # 信噪比
beta = 0.35   # 滚降
span = 6      # 脉冲长度（符号数）
sps = int(Fs/Rs)

# 2) 发送端：QPSK调制
num_sym = 10000
bits = np.random.randint(0, 4, num_sym)
tx_sym = np.exp(1j*(np.pi/4 + bits*np.pi/2))
print("能量：", np.mean(np.abs(tx_sym) ** 2))

# 3) 根升余弦成型滤波
h_rrc = rcosdesign(beta, span, sps)
print("RRC 滤波器能量：", np.sum(np.abs(h_rrc)**2))

upsampled = np.zeros(num_sym*sps, dtype=complex)
upsampled[::sps] = tx_sym
tx_shaped = signal.lfilter(h_rrc, 1, upsampled)
print("成型后符号能量：", np.mean(np.abs(tx_shaped)**2))

# 3) 上变频 + AWGN
t       = np.arange(len(tx_shaped)) / Fs
tx_rf   = tx_shaped * np.exp(1j*2*np.pi*Fc*t)
rx_rf   = AWGN(tx_rf, SNR_dB)
print("加噪后信号能量：", np.mean(np.abs(rx_rf)**2))

# 5) 下变频
rx_bb = rx_rf * np.exp(-1j*2*np.pi*Fc*t)

# 6) 匹配滤波
rx_filt = signal.lfilter(h_rrc, 1, rx_bb)
print("匹配滤波后符号能量：", np.mean(np.abs(rx_filt)**2))

# 7) 延迟补偿 & 抽取
delay = (len(h_rrc)-1)//2
rx_samps = rx_filt[delay::sps]
print("抽取后前10个幅度：", np.abs(rx_samps[:10]))

# 8) 绘制星座
plt.figure(figsize=(5,5))
plt.scatter(rx_samps.real, rx_samps.imag, s=2, alpha=0.5)
# 理想点
ideal = np.exp(1j*(np.pi/4 + np.arange(4)*np.pi/2))
plt.scatter(ideal.real, ideal.imag, c='r', marker='x')
plt.title("QPSK Constellation")
plt.axis('equal'); plt.grid(True)
plt.show()
