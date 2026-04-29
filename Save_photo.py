import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import matplotlib.colors as mcolors


def Photo_save(rec_wave, Fs, num, SNR):
    # STFT parameters
    nperseg = 256
    noverlap = 220
    nfft = 512

    # Perform STFT
    f, t, Zxx = stft(rec_wave, Fs, window=('kaiser', 5), nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Convert magnitude to dB
    sdb = 20 * np.log10(np.abs(Zxx) + 1e-10)

    # 仅在显示层面保留非负频率（不修改原始STFT计算）
    try:
        if np.any(f < 0):
            mask_pos = f >= 0
            f_disp = f[mask_pos] / 1000.0
            sdb_disp = sdb[mask_pos, :]
        else:
            f_disp = f / 1000.0
            sdb_disp = sdb
    except Exception:
        f_disp = f / 1000.0
        sdb_disp = sdb

    # Plotting (use non-negative frequency portion for display)
    plt.figure(figsize=(10, 5), frameon=False)
    plt.pcolormesh(t, f_disp, sdb_disp, shading='gouraud', cmap='viridis',
                   norm=mcolors.Normalize(vmin=np.max(sdb_disp) - 60, vmax=np.max(sdb_disp)))

    # Remove axes and border
    plt.axis('off')

    # File name
    file_name = f'photo_{num}_{SNR}.jpg'

    # Save the figure
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

    return 'ok'


# Example usage
rec_wave = np.random.randn(1024)  # Example data
Fs = 50e9  # Example sampling rate
num = 1  # Example number
SNR = -10  # Example SNR value

Photo_save(rec_wave, Fs, num, SNR)
