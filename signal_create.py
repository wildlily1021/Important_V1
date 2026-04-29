import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import matplotlib.pyplot as plt
from excelwrite import writexlsx
from Save_photo import Photo_save
import os


# Define constants:这些部分未来会通过Qt的上位机进行输入
Fs = 50e9                # Sampling rate：采样频率
Fc = 16e9                # Carrier frequency：载波频率
N = int(3e6)             # Number of samples：采样点数
A = 2                    # Signal amplitude：信号幅度
Rs = 3.5e9               # Symbol rate：符号速率
rate = int(np.floor(Fs / Rs))  # Samples per symbol：每个符号的采样点数
symbol_num = int(np.floor(N / rate))  # Number of symbols：符号数               
N = symbol_num * rate # 采样点数等于符号数乘以每个符号的采样点数
t = np.arange(0, N) / Fs  # Time vector：时间向量
rolloff = 0.35            # Rolloff factor：滚降因子
span = 6                  # Span of the filter in symbols：滤波器符号数
Modulation = 1            # 1 for QPSK, 2 for 8PSK, 3 for 16QAM, 4 for 64QAM：调制方式

# Generate root raised cosine filter
# 生成根升余弦滤波器
filter_taps = signal.firwin2(span * rate + 1,
                             [0, rolloff, rolloff, 1.0],
                             [1, 1, 0, 0])

# 信号生成函数 接收6个参数：（输入）调制方式，采样频率，载波频率，采样点数，信噪比，符号速率
def signal_create(Modulation, Fs, Fc, N, SNR, Rs):
    # Define constants:这些部分未来会通过Qt的上位机进行输入
    # Fs = 50e9  # Sampling rate
    # Fc = 16e9  # Carrier frequency
    # N = int(3e6)  # Number of samples
    # A = 2  # Signal amplitude
    # Rs = 3.5e9  # Symbol rate

    #np.floor() 向下取整
    rate = int(np.floor(Fs / Rs))  # Samples per symbol：每个符号的采样点数
    symbol_num = int(np.floor(N / rate))  # Number of symbols：符号数 
    N = symbol_num * rate # 采样点数等于符号数乘以每个符号的采样点数
    t = np.arange(0, N) / Fs  # Time vector：时间向量
    rolloff = 0.35  # Rolloff factor：滚降因子
    span = 6  # Span of the filter in symbols：滤波器符号数
    Modulation = 1  # 1 for QPSK, 2 for 8PSK, 3 for 16QAM, 4 for 64QAM：调制方式
    # Modulation
    if Modulation == 1:
        data_rand = np.random.randint(0, 3, symbol_num)
        data_upsample = np.repeat(data_rand, rate)
        data_modu = np.exp(1j * np.pi / 2 * data_upsample + 1j * np.pi / 4)
    elif Modulation == 2:
        data_rand = np.random.randint(0, 8, symbol_num)
        data_upsample = np.repeat(data_rand, rate)
        data_modu = np.exp(1j * np.pi / 4 * data_upsample + 1j * np.pi / 8)
    elif Modulation == 3:
        data_rand = np.random.randint(0, 16, symbol_num)
        graycode16 = np.array([0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10])
        data_upsample = np.repeat(data_rand, rate)
        msg16 = graycode16[data_upsample]
        data_modu = signal.qammod(msg16, 16)
    elif Modulation == 4:
        data_rand = np.random.randint(0, 64, symbol_num)
        graycode64 = np.array([0, 1, 3, 2, 6, 7, 5, 4, 8, 9, 11, 10, 14, 15, 13, 12,
                               24, 25, 27, 26, 30, 31, 29, 28, 16, 17, 19, 18, 22, 23,
                               21, 20, 48, 49, 51, 50, 54, 55, 53, 52, 56, 57, 59, 58,
                               62, 63, 61, 60, 40, 41, 43, 42, 46, 47, 45, 44, 32, 33,
                               35, 34, 38, 39, 37, 36])
        data_upsample = np.repeat(data_rand, rate)
        msg64 = graycode64[data_upsample]
        data_modu = signal.qammod(msg64, 64)



    # Generate root raised cosine filter
    filter_taps = signal.firwin2(span * rate + 1,
                                 [0, rolloff, rolloff, 1.0],
                                 [1, 1, 0, 0])

    # Loop to generate images
    for i in range(1, 2):
        Fc = (16 + 0.5 * (i - 0.5)) * 1e9
        Low_Rs = 3.5
        Hig_Rs = 3.6
        step_Rs = 0.1
        for Rs in np.arange(Low_Rs, Hig_Rs + step_Rs, step_Rs):
            data_tran = np.convolve(data_modu, filter_taps)
            data_tran = data_tran[(len(filter_taps) - 1) // 2:-(len(filter_taps) - 1) // 2]
            signal_amp = 0.2
            signal_wave = signal_amp * data_tran * np.exp(1j * 2 * np.pi * Fc * t)

            # Add noise
            SNR = -10
            signal_noise = signal_wave + np.random.normal(0, np.sqrt(10 ** (-SNR / 10)), signal_wave.shape)

            rec_wave = signal_noise
            num = int((i - 1) * ((Hig_Rs - Low_Rs) / step_Rs + 1) + (Rs - Low_Rs) / step_Rs + 1)
            # ES_Center, ES_3dB, ES_snr = Matlab_estimate(rec_wave, Fc, Fs, Rs)
            bandwidth_the = (1 - rolloff / 2) * Rs
            Center_the = Fc
            SNR_the = SNR

            def save_wave_to_txt(wave, filename):
                np.savetxt(filename, wave.view(float))  # Convert complex to float view for saving
            # Save rec_wave to text file
            file_name_txt = f'rec_wave_{num}_{SNR}.txt'
            file_path_txt = os.path.join(file_name_txt)
            save_wave_to_txt(data_modu, file_path_txt)

            # Save photos
            Photo_save(signal_noise, Fs, num, SNR)
            # writexlsx(num, Fs, Fc, Rs, bandwidth_the, SNR_the, ES_Center, ES_3dB, ES_snr)

signal_create(Modulation)