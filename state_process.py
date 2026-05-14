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


def _normalize_complex_signal(complex_signal):
    complex_signal = np.asarray(complex_signal, dtype=np.complex128)

    peak_power = np.max(np.abs(complex_signal) ** 2)
    if peak_power > 0:
        complex_signal = complex_signal / np.sqrt(peak_power)

    signal_power = np.mean(np.abs(complex_signal) ** 2)
    if signal_power > 0:
        complex_signal = complex_signal / np.sqrt(signal_power)

    return complex_signal


def _to_analytic_signal(real_signal):
    real_signal = np.asarray(real_signal, dtype=float)
    if real_signal.size == 0:
        raise ValueError("File contains no valid signal samples")
    return hilbert(real_signal)


def _clean_numeric_frame(df):
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.dropna(axis=1, how='all')
    numeric_df = numeric_df.dropna(axis=0, how='all')
    return numeric_df


def _estimate_sample_rate(timestamps):
    timestamps = np.asarray(timestamps, dtype=float)
    timestamps = timestamps[np.isfinite(timestamps)]
    if timestamps.size < 2:
        return None

    deltas = np.diff(timestamps)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0 or np.any(deltas <= 0):
        return None

    sample_interval = np.median(deltas)
    if sample_interval <= 0:
        return None

    return 1.0 / sample_interval


def load_excel_signal(file_path, normalize=True, return_fs=False):
    """
    Load signal data from Excel/CSV file and convert to complex form.

    Parameters:
        file_path: Excel or CSV file path (.xlsx, .xls, .csv)
        normalize: Whether to normalize the signal, default True
        return_fs: Return (signal, sample_rate) when True

    Returns:
        Complex signal array, or (complex signal array, sample_rate).
        Single-column real signals are converted to analytic signals by Hilbert
        transform. Two-column CSV files are treated as timestamp + real signal
        when the first column is strictly increasing.
    """
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
        else:
            df = pd.read_excel(file_path, header=None)

        numeric_df = _clean_numeric_frame(df)
        num_cols = numeric_df.shape[1]
        sample_rate = None

        if num_cols >= 2:
            first_two_cols = numeric_df.iloc[:, :2].dropna()
            if first_two_cols.empty:
                raise ValueError("File contains no valid numeric signal samples")

            first_col = first_two_cols.iloc[:, 0].to_numpy(dtype=float)
            second_col = first_two_cols.iloc[:, 1].to_numpy(dtype=float)
            sample_rate = _estimate_sample_rate(first_col)

            if file_path.lower().endswith('.csv') and sample_rate is not None:
                real_signal = second_col
                complex_signal = _to_analytic_signal(real_signal)
                print(f"Timestamp column detected: Fs = {sample_rate:.6e} Hz")
            else:
                real_signal = first_col
                imag_signal = second_col
                complex_signal = real_signal + 1j * imag_signal
                print("I/Q columns detected: using first column as I and second column as Q")
        elif num_cols == 1:
            real_signal = numeric_df.iloc[:, 0].dropna().to_numpy(dtype=float)
            complex_signal = _to_analytic_signal(real_signal)
            print("Single real-signal column detected: using Hilbert analytic signal")
        else:
            raise ValueError("File contains no data columns")

        if normalize:
            complex_signal = _normalize_complex_signal(complex_signal)

        print(f"File data loaded successfully: {len(complex_signal)} sample points")
        print(f"Real part range: [{np.min(np.real(complex_signal)):.4f}, {np.max(np.real(complex_signal)):.4f}]")
        print(f"Imaginary part range: [{np.min(np.imag(complex_signal)):.4f}, {np.max(np.imag(complex_signal)):.4f}]")
        if normalize:
            print(f"Normalized signal power: {np.mean(np.abs(complex_signal) ** 2):.4f}")

        if return_fs:
            return complex_signal, sample_rate
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
