import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from PIL import Image
from scipy.signal import hilbert
def photo_save(rec_wave, Fs, num, Rs):
    # 使用 spectrogram 函数绘制频谱图
    window = np.hamming(256)  # 窗函数
    noverlap = 220  # 窗口重叠
    nfft = 256  # FFT 点数
    # 设置图像大小和DPI
    # 计算STFT
    p, f, t, _ = plt.specgram(rec_wave, NFFT=nfft, Fs=Fs, Fc=0, window=window, noverlap=noverlap, cmap='jet',
                              xextent=None, pad_to=None, sides='twosided', scale_by_freq=None,
                              mode='psd', scale='default')
    # plt.show()
    # 将功率谱转换为 dB
    p = 10 * np.log10(p)
    print("p_picture:", p[0:10])
    max_value = np.max(p)
    min_value = np.min(p)
    # 对dB值进行规划
    # p_db = np.clip(p, -40, 40)
    # 仅在显示层面保留非负频率（不改变原始p的计算）
    try:
        if np.any(f < 0):
            mask_pos = f >= 0
            f_disp = f[mask_pos] / 1000.0
            p_disp = p[mask_pos, :]
        else:
            f_disp = f / 1000.0
            p_disp = p
    except Exception:
        f_disp = f / 1000.0
        p_disp = p
    # 绘制图像（使用非负频率部分）
    plt.pcolormesh(t, f_disp, p_disp, cmap='jet', vmin=-60, vmax=60)
    # 去除坐标轴
    plt.axis('off')
    # plt.colorbar()
    # 保存图像
    if Rs != 0:
        plt.savefig(f'./signal_ana/STFT_Org.jpg', bbox_inches='tight',
                    pad_inches=0)
        # plt.savefig(f'./STFT_Org.jpg', bbox_inches='tight', pad_inches=0)
        # 清除当前图形
        plt.clf()
        # plt.close('all')
    else:
        plt.savefig(f'./signal_ana/STFT_Org_txt.jpg', bbox_inches='tight',
                    pad_inches=0)
        # plt.savefig(f'./STFT_Org.jpg', bbox_inches='tight', pad_inches=0)
        # 清除当前图形
        plt.clf()
        # plt.close('all')

    return [min_value, max_value]

    # # 绘制眼图的函数部分
    # # 分离实部和虚部
    # real_part = np.real(rec_wave)
    # imag_part = np.imag(rec_wave)
    # # 计算幅度
    # magnitude = np.abs(rec_wave)
    # num_symbols = len(rec_wave)
    # # 每符号的采样点数
    # samples_per_symbol = int(Fs // Rs)
    # # 计算总的采样符号数
    # total_symbols = len(real_part) // samples_per_symbol
    # downsample_factor = 10
    # # 对信号进行下采样以提高绘图效率
    # if downsample_factor > 1:
    #     signal = real_part[::downsample_factor]
    #     samples_per_symbol //= downsample_factor  # 更新每符号的采样点数
    # # 计算眼图的步长和截取数据
    # step_size = samples_per_symbol
    # truncated_signal = signal[:step_size * total_symbols]  # 截取整数个符号的信号
    # N = len(rec_wave)
    # y = truncated_signal  # [:(samples_per_symbol * 1000)]
    # eyediagram(y, 4, offset=samples_per_symbol, cmap=plt.cm.coolwarm)
    # plt.clf()
    # plt.close('all')


def photo_save_test(rec_wave, Fs, Rs, Fc, SNR, Modulation, image_dir):
    # 使用 spectrogram 函数绘制频谱图
    window = np.hamming(256)  # 窗函数
    noverlap = 220  # 窗口重叠
    nfft = 256  # FFT 点数
    # 设置图像大小和DPI
    # 计算STFT
    p, f, t, _ = plt.specgram(rec_wave, NFFT=nfft, Fs=Fs, Fc=0, window=window, noverlap=noverlap, cmap='jet',
                              xextent=None, pad_to=None, sides='twosided', scale_by_freq=None,
                              mode='psd', scale='default')
    # plt.show()
    # 将功率谱转换为 dB
    p = 10 * np.log10(p)
    # 仅在显示层面保留非负频率
    try:
        if np.any(f < 0):
            mask_pos = f >= 0
            f_disp = f[mask_pos] / 1000.0
            p_disp = p[mask_pos, :]
        else:
            f_disp = f / 1000.0
            p_disp = p
    except Exception:
        f_disp = f / 1000.0
        p_disp = p
    # 绘制图像（使用非负频率部分）
    plt.pcolormesh(t, f_disp, p_disp, cmap='jet', vmin=-60, vmax=60)
    # 去除坐标轴
    plt.axis('off')
    # plt.colorbar()
    # 保存图像
    #
    # if Rs == 0.5:
    #     fileflag = 32.3416
    # elif Rs == 0.75:
    #     fileflag = 48.5125
    # elif Rs == 1:
    #     fileflag = 64.6833
    # elif Rs == 1.25:
    #     fileflag = 80.8541
    # elif Rs == 1.5:
    #     fileflag = 97.0250
    # elif Rs == 1.75:
    #     fileflag = 113.1358
    # elif Rs == 2:
    #     fileflag = 129.3666
    # elif Rs == 2.25:
    #     fileflag = 145.5374
    # elif Rs == 2.5:
    #     fileflag = 161.7082
    # elif Rs == 2.75:
    #     fileflag = 177.8791
    # elif Rs == 3:
    #     fileflag = 194.0499
    # else:
    #     fileflag = 0
    #
    # Fc_index = round(Fc, 2)
    # SNR_index = round(SNR, 1)

    plt.savefig(
        f'./dataset_YOLO/image_dataset.jpg',
        bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'./STFT_Org.jpg', bbox_inches='tight', pad_inches=0)
    # 清除当前图形
    plt.clf()
    # plt.close('all')
def photo_save_final(rec_wave, Fs, Rs, Fc, SNR, Modulation, image_dir):
    # 使用 spectrogram 函数绘制频谱图
    window = np.hamming(256)  # 窗函数
    noverlap = 220  # 窗口重叠
    nfft = 256  # FFT 点数
    # 设置图像大小和DPI
    # 计算STFT
    p, f, t, _ = plt.specgram(rec_wave, NFFT=nfft, Fs=Fs, Fc=0, window=window, noverlap=noverlap, cmap='jet',
                              xextent=None, pad_to=None, sides='twosided', scale_by_freq=None,
                              mode='psd', scale='default')
    # plt.show()
    # 将功率谱转换为 dB
    p = 10 * np.log10(p)
    # 仅在显示层面保留非负频率
    try:
        if np.any(f < 0):
            mask_pos = f >= 0
            f_disp = f[mask_pos] / 1000.0
            p_disp = p[mask_pos, :]
        else:
            f_disp = f / 1000.0
            p_disp = p
    except Exception:
        f_disp = f / 1000.0
        p_disp = p
    # 绘制图像（使用非负频率部分）
    plt.pcolormesh(t, f_disp, p_disp, cmap='jet', vmin=-60, vmax=60)
    # 去除坐标轴
    plt.axis('off')

    plt.savefig(image_dir, bbox_inches='tight', pad_inches=0)
    # 清除当前图形
    plt.clf()
    # plt.close('all')

def get_amplitude_from_rgb(rgb_value, min_val, max_val):
    # 生成一个 jet 色型的颜色轴，假设dB范围为[-100, 0]
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap('jet')

    dB_range = np.linspace(min_val, max_val, 2048)
    colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=norm).to_rgba(np.linspace(min_val, max_val, 2048))
    colorbar_rgb = colorbar[:, :3]  # 只取前3个通道（RGB）
    differences = np.sqrt(np.sum((colorbar_rgb - rgb_value) ** 2, axis=1))
    idx = np.argmin(differences)
    db_value = dB_range[idx]

    return db_value
    # # 创建一个jet色图
    # colormap = plt.get_cmap('jet')
    #
    # # 生成一个包含所有可能值的列表
    # num_colors = 2048
    # gradient = np.linspace(0, 1, num_colors)
    # colors = colormap(gradient)[:, :3]  # 获取RGB值
    #
    # # 确定幅度范围
    # clim_min = min_val  # 默认最小值
    # clim_max = max_val  # 默认最大值
    # amplitudes = np.linspace(clim_min, clim_max, num_colors)
    #
    # # 创建KD树用于快速查找最近邻
    # tree = KDTree(colors)
    #
    # # 查找与输入RGB值最接近的颜色
    # distance, index = tree.query(rgb_value)
    #
    # # 返回对应的幅度值
    # return amplitudes[index]

def photo_save_scipy(rec_wave, Fs, num, Rs):
    # 使用 spectrogram 函数绘制频谱图
    window = np.hamming(256)  # 窗函数
    noverlap = 220  # 窗口重叠
    nfft = 256  # FFT 点数
    # 设置图像大小和DPI
    # 计算STFT
    p, f, t, _ = plt.specgram(rec_wave, NFFT=nfft, Fs=Fs, Fc=0, window=window, noverlap=noverlap, cmap='jet',
                              xextent=None, pad_to=None, sides='twosided', scale_by_freq=None,
                              mode='psd', scale='default')
    # plt.show()
    # 将功率谱转换为 dB
    p = 10 * np.log10(p)
    print("p_picture:", p[0:10])
    max_value = np.max(p)
    min_value = np.min(p)
    # 对dB值进行规划
    # p_db = np.clip(p, -40, 40)
    # 绘制图像
    plt.pcolormesh(t, f / 1000, p, cmap='jet', vmin=-60, vmax=60)
    # 去除坐标轴
    plt.axis('off')
    # plt.colorbar()
    # 保存图像
    if Rs != 0:
        plt.savefig(f'./signal_ana/STFT_Org_txt.jpg', bbox_inches='tight',
                    pad_inches=0)
        # plt.savefig(f'./STFT_Org.jpg', bbox_inches='tight', pad_inches=0)
        # 清除当前图形
        plt.clf()
        # plt.close('all')
    else:
        plt.savefig(f'./signal_ana/STFT_Org_txt.jpg', bbox_inches='tight',
                    pad_inches=0)
        # plt.savefig(f'./STFT_Org.jpg', bbox_inches='tight', pad_inches=0)
        # 清除当前图形
        plt.clf()
    # 读入图像
    img = Image.open('./signal_ana/STFT_Org_txt.jpg')
    img_np = np.array(img)

    h, w, c = img_np.shape
    print("图像尺寸:", h, w, c)

    # 选取要复制的区域：上方 35%~45%
    top_start = int(h * 0.35)
    top_end = int(h * 0.45)
    patch = img_np[top_start:top_end, :, :]

    # 目标区域：下方 60%~90%
    target_start = int(h * 0.45)
    target_end = int(h * 0.90)

    # 计算要覆盖的区域高度
    target_height = target_end - target_start
    patch_resized = np.tile(patch, (int(np.ceil(target_height / patch.shape[0])), 1, 1))[:target_height, :, :]

    # 覆盖
    img_np[target_start:target_end, :, :] = patch_resized

    # 转回图像保存
    out_img = Image.fromarray(img_np)
    out_img.save('./signal_ana/STFT_Org_cropped.jpg')
    print("覆盖完成，已保存为 STFT_Org_cropped.jpg")

    return [min_value, max_value]