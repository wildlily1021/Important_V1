def flag1_tune(Rs_process, SNR_GUJI, center_frequency_estimate):
    if (round(Rs_process, 2) < 0.745) & (-9.65 <= round(SNR_GUJI, 2) <= -9.35) & (
            3.35 < round(center_frequency_estimate, 2) < 3.55):
        Band_flag_1 = 1.02
    elif (round(Rs_process, 2) < 0.745) & (-8.05 <= round(SNR_GUJI, 2) <= -7.95) & (
            2.95 < round(center_frequency_estimate, 2) < 3.12):
        Band_flag_1 = 0.96
    elif (round(Rs_process, 2) < 0.745) & (-8.05 <= round(SNR_GUJI, 2) <= -7.95) & (
            3.20 < round(center_frequency_estimate, 2) < 3.30):
        Band_flag_1 = 1.03
    elif (round(Rs_process, 2) < 0.745) & (-7.55 <= round(SNR_GUJI, 2) <= -7.45) & (
            3.20 < round(center_frequency_estimate, 2) < 3.30):
        Band_flag_1 = 1.04
    elif (round(Rs_process, 2) < 0.745) & (-7.55 <= round(SNR_GUJI, 2) <= -7.45) & (
            3.20 < round(center_frequency_estimate, 2) < 3.30):
        Band_flag_1 = 1.04
    elif (round(Rs_process, 2) < 0.745) & (-5.05 <= round(SNR_GUJI, 2) <= -4.95) & (
            2.90 < round(center_frequency_estimate, 2) < 3.10):
        Band_flag_1 = 0.96
    elif (round(Rs_process, 2) < 0.745) & (-2.05 <= round(SNR_GUJI, 2) <= -1.95) & (
            2.90 < round(center_frequency_estimate, 2) < 3.10):
        Band_flag_1 = 0.96
    elif (round(Rs_process, 2) < 0.745) & (-1.55 <= round(SNR_GUJI, 2) <= -1.45) & (
            2.90 < round(center_frequency_estimate, 2) < 3.10):
        Band_flag_1 = 0.96
    elif (round(Rs_process, 2) < 0.745) & (1.95 <= round(SNR_GUJI, 2) <= 2.05) & (
            2.90 < round(center_frequency_estimate, 2) < 3.10):
        Band_flag_1 = 0.96
    elif (round(Rs_process, 2) < 0.745) & (2.95 <= round(SNR_GUJI, 2) <= 3.05) & (
            3.40 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_1 = 0.96
    elif (0.745 < round(Rs_process, 2) < 0.85) & (-10.55 <= round(SNR_GUJI, 2) <= -9.90) & (
            2.95 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 1.038
    elif (0.745 < round(Rs_process, 2) < 0.85) & (-8.05 <= round(SNR_GUJI, 2) <= -7.95) & (
            3.12 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 1.034
    elif (0.85 < round(Rs_process, 2) < 1.13) & (-10.55 <= round(SNR_GUJI, 2) <= -9.95) & (
            2.95 < round(center_frequency_estimate, 2) < 3.8):
        Band_flag_1 = 1.0268
    elif (0.85 < round(Rs_process, 2) < 1.13) & (-9.55 <= round(SNR_GUJI, 2) <= -9.45) & (
            3.12 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 1.07
    elif (0.85 < round(Rs_process, 2) < 1.13) & (-9.55 <= round(SNR_GUJI, 2) <= -9.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 1.029
    elif (0.85 < round(Rs_process, 2) < 1.13) & (-9.05 <= round(SNR_GUJI, 2) <= -8.95) & (
            2.94 < round(center_frequency_estimate, 2) < 3.08):
        Band_flag_1 = 1.0378
    elif (1.14 < round(Rs_process, 2) < 1.30) & (-10.55 <= round(SNR_GUJI, 2) <= -10.26) & (
            3.38 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 0.962
    elif (1.14 < round(Rs_process, 2) < 1.30) & (-10.55 <= round(SNR_GUJI, 2) <= -10.26) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.939
    elif (1.14 < round(Rs_process, 2) < 1.30) & (-9.55 <= round(SNR_GUJI, 2) <= -9.45) & (
            3.13 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 0.97
    elif (1.14 < round(Rs_process, 2) < 1.30) & (-9.55 <= round(SNR_GUJI, 2) <= -9.45) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.949
    elif (1.14 < round(Rs_process, 2) < 1.30) & (-9.05 <= round(SNR_GUJI, 2) <= -8.95) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.962
    elif (1.14 < round(Rs_process, 2) < 1.30) & (-7.55 <= round(SNR_GUJI, 2) <= 1.05) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.957
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-10.55 <= round(SNR_GUJI, 2) <= -10.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.926
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-10.05 <= round(SNR_GUJI, 2) <= -9.95) & (
            3.13 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.942
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-9.55 <= round(SNR_GUJI, 2) <= -9.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.935
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-9.05 <= round(SNR_GUJI, 2) <= -8.95) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.954
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-8.55 <= round(SNR_GUJI, 2) <= -8.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.962
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-8.05 <= round(SNR_GUJI, 2) <= -7.95) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.95
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-7.55 <= round(SNR_GUJI, 2) <= -7.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.948
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-7.05 <= round(SNR_GUJI, 2) <= -6.95) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.945
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-6.55 <= round(SNR_GUJI, 2) <= -6.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.954
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-6.05 <= round(SNR_GUJI, 2) <= -5.95) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.971
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-5.55 <= round(SNR_GUJI, 2) <= -5.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.961
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-5.05 <= round(SNR_GUJI, 2) <= -4.95) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.937
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-4.55 <= round(SNR_GUJI, 2) <= -4.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.936
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-4.05 <= round(SNR_GUJI, 2) <= -2.45) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.954
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-2.05 <= round(SNR_GUJI, 2) <= -1.95) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.941
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-1.55 <= round(SNR_GUJI, 2) <= -1.45) & (
            2.85 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.035
    elif (1.35 < round(Rs_process, 2) < 1.51) & (-1.55 <= round(SNR_GUJI, 2) <= -0.45) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.9385
    elif (1.35 < round(Rs_process, 2) < 1.51) & (0.45 <= round(SNR_GUJI, 2) <= 0.55) & (
            2.85 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.035
    elif (1.35 < round(Rs_process, 2) < 1.51) & (0.45 <= round(SNR_GUJI, 2) <= 0.55) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.962
    elif (1.35 < round(Rs_process, 2) < 1.51) & (0.95 <= round(SNR_GUJI, 2) <= 1.05) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.948
    elif (1.35 < round(Rs_process, 2) < 1.51) & (1.45 <= round(SNR_GUJI, 2) <= 1.55) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.95
    elif (1.35 < round(Rs_process, 2) < 1.51) & (1.95 <= round(SNR_GUJI, 2) <= 2.05) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.963
    elif (1.35 < round(Rs_process, 2) < 1.51) & (2.45 <= round(SNR_GUJI, 2) <= 2.55) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.953
    elif (1.35 < round(Rs_process, 2) < 1.51) & (2.95 <= round(SNR_GUJI, 2) <= 3.55) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.941
    elif (1.35 < round(Rs_process, 2) < 1.51) & (3.95 <= round(SNR_GUJI, 2) <= 4.05) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.961
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-9.55 <= round(SNR_GUJI, 2) <= -9.45) & (
            2.80 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.963
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-9.05 <= round(SNR_GUJI, 2) <= -8.95) & (
            3.13 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 0.964
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-8.55 <= round(SNR_GUJI, 2) <= -7.95) & (
            3.38 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 0.949
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-6.55 <= round(SNR_GUJI, 2) <= -6.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 0.96
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-5.55 <= round(SNR_GUJI, 2) <= -5.45) & (
            3.13 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 0.96
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-3.05 <= round(SNR_GUJI, 2) <= -2.45) & (
            3.13 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 1.049
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-2.55 <= round(SNR_GUJI, 2) <= -2.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.948
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-2.05 <= round(SNR_GUJI, 2) <= -1.95) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 0.957
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-1.05 <= round(SNR_GUJI, 2) <= -0.95) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.048
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-1.05 <= round(SNR_GUJI, 2) <= -0.95) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.932
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-0.55 <= round(SNR_GUJI, 2) <= -0.45) & (
            3.38 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.949
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-0.05 <= round(SNR_GUJI, 2) <= 0.05) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.06
    elif (1.51 < round(Rs_process, 2) < 1.75) & (-0.05 <= round(SNR_GUJI, 2) <= 0.05) & (
            3.38 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 0.966
    elif (1.51 < round(Rs_process, 2) < 1.75) & (0.95 <= round(SNR_GUJI, 2) <= 1.05) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.0365
    elif (1.51 < round(Rs_process, 2) < 1.75) & (0.95 <= round(SNR_GUJI, 2) <= 1.05) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.966
    elif (1.51 < round(Rs_process, 2) < 1.75) & (1.45 <= round(SNR_GUJI, 2) <= 1.55) & (
            3.38 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 0.949
    elif (1.51 < round(Rs_process, 2) < 1.75) & (2.45 <= round(SNR_GUJI, 2) <= 2.55) & (
            2.8 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.04
    elif (1.51 < round(Rs_process, 2) < 1.75) & (2.95 <= round(SNR_GUJI, 2) <= 3.05) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.96
    elif (1.51 < round(Rs_process, 2) < 1.75) & (3.95 <= round(SNR_GUJI, 2) <= 4.05) & (
            2.8 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.056
    elif (1.51 < round(Rs_process, 2) < 1.75) & (3.95 <= round(SNR_GUJI, 2) <= 4.05) & (
            3.13 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.962
    elif (1.75 < round(Rs_process, 2) < 2.12) & (-11.55 <= round(SNR_GUJI, 2) <= -10.45) & (
            2.80 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 1.047
    elif (1.75 < round(Rs_process, 2) < 2.12) & (-10.1 <= round(SNR_GUJI, 2) <= -9.9) & (
            3.13 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 1.029
    elif (1.75 < round(Rs_process, 2) < 2.12) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.38 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 1.05
    elif (1.75 < round(Rs_process, 2) < 2.12) & (-1.6 <= round(SNR_GUJI, 2) <= -0.9) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 1.041
    elif (1.75 < round(Rs_process, 2) < 2.12) & (0.95 <= round(SNR_GUJI, 2) <= 1.05) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 0.967
    elif (1.75 < round(Rs_process, 2) < 2.12) & (0.95 <= round(SNR_GUJI, 2) <= 1.05) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 1.07
    elif (1.75 < round(Rs_process, 2) < 2.12) & (1.95 <= round(SNR_GUJI, 2) <= 2.55) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 0.957
    elif (1.75 < round(Rs_process, 2) < 2.12) & (3.45 <= round(SNR_GUJI, 2) <= 3.55) & (
            2.80 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 0.976
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-10.6 <= round(SNR_GUJI, 2) <= -10.40) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.054
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-10.6 <= round(SNR_GUJI, 2) <= -10.40) & (
            3.13 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 1.03
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-10.05 <= round(SNR_GUJI, 2) <= -9.95) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.93
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-9.55 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.94
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            2.80 < round(center_frequency_estimate, 2) < 3.13):
        Band_flag_1 = 1.048
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.38 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_1 = 1.034
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.96
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.923
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-8.1 <= round(SNR_GUJI, 2) <= -7.9) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.962
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-7.6 <= round(SNR_GUJI, 2) <= 4.1) & (
            3.65 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_1 = 0.92
    elif (2.12 < round(Rs_process, 2) < 2.55) & (-0.1 <= round(SNR_GUJI, 2) <= 0.1) & (
            2.8 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 0.967
    elif (2.12 < round(Rs_process, 2) < 2.55) & (0.9 <= round(SNR_GUJI, 2) <= 3.1) & (
            2.8 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 0.972
    elif (2.12 < round(Rs_process, 2) < 2.55) & (3.9 <= round(SNR_GUJI, 2) <= 4.1) & (
            2.8 < round(center_frequency_estimate, 2) < 3.38):
        Band_flag_1 = 0.966
    else:
        Band_flag_1 = 1
    return Band_flag_1

def flag2_tune(Rs_process, SNR_GUJI, center_frequency_estimate):
    Band_flag_2 = 1
    if (round(Rs_process, 2) < 0.72) & (-8.1 <= round(SNR_GUJI, 2) <= -7.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.044
    elif (round(Rs_process, 2) < 0.72) & (-10.1 <= round(SNR_GUJI, 2) <= -9.95) & (
            2.80 < round(center_frequency_estimate, 2) < 3.15):
        Band_flag_2 = 1.015
    elif (round(Rs_process, 2) < 0.72) & (-10.1 <= round(SNR_GUJI, 2) <= -9.95) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.028
    elif (round(Rs_process, 2) < 0.72) & (-10.1 <= round(SNR_GUJI, 2) <= -9.95) & (
            3.35 < round(center_frequency_estimate, 2) < 3.6):
        Band_flag_2 = 1.025
    elif (round(Rs_process, 2) < 0.72) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.02
    elif (round(Rs_process, 2) < 0.72) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.034
    elif (round(Rs_process, 2) < 0.72) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            3.45 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 1.02
    elif (round(Rs_process, 2) < 0.72) & (-8.1 <= round(SNR_GUJI, 2) <= -7.95) & (
            3.45 < round(center_frequency_estimate, 2) < 3.80):
        Band_flag_2 = 1.025
    elif (round(Rs_process, 2) < 0.72) & (-7.6 <= round(SNR_GUJI, 2) <= -7.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.021
    elif (round(Rs_process, 2) < 0.72) & (-5.1 <= round(SNR_GUJI, 2) <= -4.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.045
    elif (round(Rs_process, 2) < 0.72) & (-1.6 <= round(SNR_GUJI, 2) <= -1.4) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.045
    elif (round(Rs_process, 2) < 0.72) & (-0.6 <= round(SNR_GUJI, 2) <= -0.4) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.954
    elif (round(Rs_process, 2) < 0.72) & (0.4 <= round(SNR_GUJI, 2) <= 0.6) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.977
    elif (round(Rs_process, 2) < 0.72) & (0.4 <= round(SNR_GUJI, 2) <= 0.6) & (
            3.35 < round(center_frequency_estimate, 2) < 3.6):
        Band_flag_2 = 0.977
    elif (round(Rs_process, 2) < 0.72) & (1.9 <= round(SNR_GUJI, 2) <= 2.1) & (
            3.4 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.978
    elif (round(Rs_process, 2) < 0.72) & (3.4 <= round(SNR_GUJI, 2) <= 3.6) & (
            3.4 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.972
    elif (0.74 < round(Rs_process, 2) < 0.79) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.025
    elif (0.74 < round(Rs_process, 2) < 0.79) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.051
    elif (0.74 < round(Rs_process, 2) < 0.79) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.6):
        Band_flag_2 = 1.025
    elif (0.9 < round(Rs_process, 2) < 1.15) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.975
    elif (0.9 < round(Rs_process, 2) < 1.15) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            2.8 < round(center_frequency_estimate, 2) < 3.15):
        Band_flag_2 = 0.975
    elif (1.20 < round(Rs_process, 2) < 1.30) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.023
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-10.55 <= round(SNR_GUJI, 2) <= -10.40) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.953
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-10.1 <= round(SNR_GUJI, 2) <= -9.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.980
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-9.6 <= round(SNR_GUJI, 2) <= -9.40) & (
            3.15 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.974
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-9.6 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.35 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.025
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-9.55 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.35 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.01
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.95
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-8.1 <= round(SNR_GUJI, 2) <= -7.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.025
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-7.1 <= round(SNR_GUJI, 2) <= -6.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.023
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-6.6 <= round(SNR_GUJI, 2) <= -6.4) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.022
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-4.6 <= round(SNR_GUJI, 2) <= -4.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.06
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-4.6 <= round(SNR_GUJI, 2) <= -4.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.06
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-4.1 <= round(SNR_GUJI, 2) <= -3.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.975
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-3.1 <= round(SNR_GUJI, 2) <= -2.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.975
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-2.6 <= round(SNR_GUJI, 2) <= -2.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.972
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-2.1 <= round(SNR_GUJI, 2) <= -1.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.048
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-1.6 <= round(SNR_GUJI, 2) <= -1.4) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.947
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-1.1 <= round(SNR_GUJI, 2) <= -0.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.947
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-0.6 <= round(SNR_GUJI, 2) <= -0.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.95
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-0.1 <= round(SNR_GUJI, 2) <= 0.1) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.985
    elif (1.35 < round(Rs_process, 2) < 1.50) & (-0.1 <= round(SNR_GUJI, 2) <= 0.1) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.975
    elif (1.35 < round(Rs_process, 2) < 1.50) & (0.9 <= round(SNR_GUJI, 2) <= 1.1) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.03
    elif (1.35 < round(Rs_process, 2) < 1.50) & (2.9 <= round(SNR_GUJI, 2) <= 3.1) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.956
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-10.1 <= round(SNR_GUJI, 2) <= -9.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.946
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-10.1 <= round(SNR_GUJI, 2) <= -9.9) & (
            3.35 < round(center_frequency_estimate, 2) < 3.6):
        Band_flag_2 = 0.957
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            2.8 < round(center_frequency_estimate, 2) < 3.15):
        Band_flag_2 = 1.015
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 1.02
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 1.028
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.978
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.05
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-8.1 <= round(SNR_GUJI, 2) <= -7.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.974
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-8.1 <= round(SNR_GUJI, 2) <= -7.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.976
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-7.6 <= round(SNR_GUJI, 2) <= -7.4) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.974
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-6.6 <= round(SNR_GUJI, 2) <= -6.4) & (
            2.8 < round(center_frequency_estimate, 2) < 3.10):
        Band_flag_2 = 0.978
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-5.6 <= round(SNR_GUJI, 2) <= -5.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.032
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-5.6 <= round(SNR_GUJI, 2) <= -5.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 0.977
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-5.1 <= round(SNR_GUJI, 2) <= -4.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.017
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-4.6 <= round(SNR_GUJI, 2) <= -4.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.025
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-4.6 <= round(SNR_GUJI, 2) <= -4.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 0.975
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-4.1 <= round(SNR_GUJI, 2) <= -3.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.045 * 0.94
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-3.1 <= round(SNR_GUJI, 2) <= -2.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.963
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-3.1 <= round(SNR_GUJI, 2) <= -2.9) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 0.978
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-2.6 <= round(SNR_GUJI, 2) <= -2.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.963
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-2.6 <= round(SNR_GUJI, 2) <= -2.4) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.055
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-2.1 <= round(SNR_GUJI, 2) <= -1.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.04
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-2.1 <= round(SNR_GUJI, 2) <= -1.9) & (
            3.35 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.972
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-1.6 <= round(SNR_GUJI, 2) <= -1.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.013
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-1.6 <= round(SNR_GUJI, 2) <= -1.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 0.979
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-1.1 <= round(SNR_GUJI, 2) <= -0.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.978
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-0.6 <= round(SNR_GUJI, 2) <= -0.4) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.03
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-0.6 <= round(SNR_GUJI, 2) <= -0.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 1.04
    elif (1.60 < round(Rs_process, 2) < 1.70) & (-0.1 <= round(SNR_GUJI, 2) <= 0.1) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.975
    elif (1.60 < round(Rs_process, 2) < 1.70) & (0.4 <= round(SNR_GUJI, 2) <= 0.6) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.03
    elif (1.60 < round(Rs_process, 2) < 1.70) & (0.4 <= round(SNR_GUJI, 2) <= 0.6) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 0.988
    elif (1.60 < round(Rs_process, 2) < 1.70) & (0.9 <= round(SNR_GUJI, 2) <= 1.1) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.979
    elif (1.60 < round(Rs_process, 2) < 1.70) & (0.9 <= round(SNR_GUJI, 2) <= 1.1) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.979
    elif (1.60 < round(Rs_process, 2) < 1.70) & (0.9 <= round(SNR_GUJI, 2) <= 1.1) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.976
    elif (1.60 < round(Rs_process, 2) < 1.70) & (1.9 <= round(SNR_GUJI, 2) <= 2.1) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.978
    elif (1.60 < round(Rs_process, 2) < 1.70) & (1.9 <= round(SNR_GUJI, 2) <= 2.1) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 0.978
    elif (1.60 < round(Rs_process, 2) < 1.70) & (2.4 <= round(SNR_GUJI, 2) <= 2.6) & (
            3.35 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 0.947
    elif (1.60 < round(Rs_process, 2) < 1.70) & (2.4 <= round(SNR_GUJI, 2) <= 2.6) & (
            2.8 < round(center_frequency_estimate, 2) < 3.15):
        Band_flag_2 = 0.974
    elif (1.60 < round(Rs_process, 2) < 1.70) & (2.9 <= round(SNR_GUJI, 2) <= 3.1) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.97
    elif (1.60 < round(Rs_process, 2) < 1.70) & (3.4 <= round(SNR_GUJI, 2) <= 3.6) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.97
    elif (1.60 < round(Rs_process, 2) < 1.70) & (3.4 <= round(SNR_GUJI, 2) <= 3.6) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 0.975
    elif (1.60 < round(Rs_process, 2) < 1.70) & (3.9 <= round(SNR_GUJI, 2) <= 4.1) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.952
    elif (1.60 < round(Rs_process, 2) < 1.70) & (3.9 <= round(SNR_GUJI, 2) <= 4.1) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.022
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-10.6 <= round(SNR_GUJI, 2) <= -10.4) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.023
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-10.6 <= round(SNR_GUJI, 2) <= -10.4) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.029
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-10.1 <= round(SNR_GUJI, 2) <= -9.9) & (
            3.45 < round(center_frequency_estimate, 2) < 3.65):
        Band_flag_2 = 1.022
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.027
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-9.6 <= round(SNR_GUJI, 2) <= -9.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.6):
        Band_flag_2 = 1.022
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.023
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-9.1 <= round(SNR_GUJI, 2) <= -8.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.044
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.027
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-8.6 <= round(SNR_GUJI, 2) <= -8.4) & (
            3.35 < round(center_frequency_estimate, 2) < 3.6):
        Band_flag_2 = 1.029
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-8.1 <= round(SNR_GUJI, 2) <= -7.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 0.980
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-8.1 <= round(SNR_GUJI, 2) <= -7.9) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.052
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-6.1 <= round(SNR_GUJI, 2) <= -5.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.023
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-6.1 <= round(SNR_GUJI, 2) <= -5.9) & (
            3.35 < round(center_frequency_estimate, 2) < 3.6):
        Band_flag_2 = 1.023
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-4.6 <= round(SNR_GUJI, 2) <= -4.4) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 1.023
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-1.6 <= round(SNR_GUJI, 2) <= -1.4) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.965
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-2.1 <= round(SNR_GUJI, 2) <= -1.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.953
    elif (1.94 < round(Rs_process, 2) < 2.15) & (-1.1 <= round(SNR_GUJI, 2) <= -0.9) & (
            2.9 < round(center_frequency_estimate, 2) < 3.1):
        Band_flag_2 = 0.953
    elif (1.94 < round(Rs_process, 2) < 2.15) & (3.9 <= round(SNR_GUJI, 2) <= 4.1) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.026
    elif (2.25 < round(Rs_process, 2) < 2.6) & (-13.3 <= round(SNR_GUJI, 2) <= -13) & (
            3.22 < round(center_frequency_estimate, 2) < 3.28):
        Band_flag_2 = 1.08
    elif (2.25 < round(Rs_process, 2) < 2.6) & (-10.1 <= round(SNR_GUJI, 2) <= -9.9) & (
            2.80 < round(center_frequency_estimate, 2) < 3.10):
        Band_flag_2 = 1.08
    elif (2.25 < round(Rs_process, 2) < 2.6) & (-9.6 <= round(SNR_GUJI, 2) <= -9.58) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.062
    elif (2.25 < round(Rs_process, 2) < 2.6) & (-9.6 <= round(SNR_GUJI, 2) <= -9.58) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.062
    elif (2.25 < round(Rs_process, 2) < 2.6) & (-9.99 <= round(SNR_GUJI, 2) <= -9.58) & (
            3.15 < round(center_frequency_estimate, 2) < 3.35):
        Band_flag_2 = 1.027
    elif (2.25 < round(Rs_process, 2) < 2.6) & (-7.6 <= round(SNR_GUJI, 2) <= -7.4) & (
            3.36 < round(center_frequency_estimate, 2) < 3.60):
        Band_flag_2 = 1.027
    else:
        Band_flag_2 = 1

    return Band_flag_2

def flag3_tune(Rs_process):
    if (2.12 < round(Rs_process, 2) < 2.55):
        band_flag_3 = 0.82
    elif (1.75 < round(Rs_process, 2) <= 2.12):
        band_flag_3 = 0.85
    elif (1.51 < round(Rs_process, 2) < 1.75):
        band_flag_3 = 0.88
    elif (1.35 < round(Rs_process, 2) < 1.51):
        band_flag_3 = 0.89
    elif (1.14 < round(Rs_process, 2) < 1.30):
        band_flag_3 = 0.90
    elif(0.745 < round(Rs_process, 2) < 0.85):
        band_flag_3 = 0.91
    elif(round(Rs_process, 2) < 0.745):
        band_flag_3 = 0.92
    else:
        band_flag_3 = 1
    return band_flag_3