# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import welch

from photo_save import photo_save, photo_save_test, photo_save_final, photo_save_scipy
from scipy.signal import hilbert, correlate, welch
from scipy.fft import fftshift, fft
from scipy.ndimage import uniform_filter1d
import pywt
import pandas as pd


def load_excel_signal(file_path, normalize=True):
    """
    Load signal data from Excel/CSV file and convert to complex form

    Parameters:
        file_path: Excel or CSV file path (.xlsx, .xls, .csv)
        normalize: Whether to normalize the signal, default True

    Returns:
        Complex signal array (a + nj form)
    """
    try:
        # Determine file type by extension
        if file_path.lower().endswith('.csv'):
            # CSV file format: each line contains "real,imaginary" or just "value"
            df = pd.read_csv(file_path, header=None)
        else:
            # Excel file format
            df = pd.read_excel(file_path, header=None)

        # Get number of columns
        num_cols = df.shape[1]

        if num_cols >= 2:
            # Two or more columns: first column as real part (a), second column as imaginary part (n)
            a = df.iloc[:, 0].values  # First column
            n = df.iloc[:, 1].values  # Second column
        elif num_cols == 1:
            # Only one column: use the same column for both real and imaginary parts (a + aj)
            a = df.iloc[:, 0].values  # First column
            n = a.copy()  # Duplicate as imaginary part
            print(f"Single column detected: using a + aj form")
        else:
            raise ValueError("File contains no data columns")

        # Convert to complex form (a + nj)
        complex_signal = a + 1j * n

        # Normalization processing
        if normalize:
            # Calculate peak power of signal
            peak_power = np.max(np.abs(complex_signal) ** 2)
            if peak_power > 0:
                # Normalize so peak amplitude is 1
                complex_signal = complex_signal / np.sqrt(peak_power)

            # Additional power normalization (optional)
            signal_power = np.mean(np.abs(complex_signal) ** 2)
            if signal_power > 0:
                complex_signal = complex_signal / np.sqrt(signal_power)

        print(f"File data loaded successfully: {len(complex_signal)} sample points")
        print(f"Real part range: [{np.min(a):.4f}, {np.max(a):.4f}]")
        print(f"Imaginary part range: [{np.min(n):.4f}, {np.max(n):.4f}]")
        if normalize:
            print(f"Normalized signal power: {np.mean(np.abs(complex_signal) ** 2):.4f}")

        return complex_signal

    except Exception as e:
        print(f"Error loading file: {str(e)}")
        raise

def normalize_signal(signal_data, method='peak'):
    """
    Normalize the signal

    Parameters:
        signal_data: Input signal (complex or real)
        method: Normalization method
            - 'peak': Peak amplitude normalization (normalize to 1)
            - 'power': Power normalization (average power is 1)
            - 'both': Both peak and power normalization

    Returns:
        Normalized signal
    """
    signal_normalized = signal_data.copy()

    if method in ['peak', 'both']:
        # Peak amplitude normalization
        peak_amplitude = np.max(np.abs(signal_normalized))
        if peak_amplitude > 0:
            signal_normalized = signal_normalized / peak_amplitude

    if method in ['power', 'both']:
        # Power normalization
        signal_power = np.mean(np.abs(signal_normalized) ** 2)
        if signal_power > 0:
            signal_normalized = signal_normalized / np.sqrt(signal_power)

    return signal_normalized
