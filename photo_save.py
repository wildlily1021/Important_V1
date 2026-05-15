from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stft_pipeline import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_MODEL_VMAX_DB,
    DEFAULT_MODEL_VMIN_DB,
    compute_spectrogram_matrix,
    create_stft_run_artifacts,
    render_spectrogram_image,
)


SIGNAL_ANA_DIR = Path(__file__).resolve().parent / "signal_ana"


def photo_save(rec_wave, Fs, num, Rs, *, keep_negative_frequencies=False, enable_local_recognition=False):
    create_stft_run_artifacts(
        rec_wave,
        Fs,
        output_dir=SIGNAL_ANA_DIR,
        source_tag="generated",
        image_width_px=DEFAULT_IMAGE_WIDTH,
        image_height_px=DEFAULT_IMAGE_HEIGHT,
        model_vmin_db=DEFAULT_MODEL_VMIN_DB,
        model_vmax_db=DEFAULT_MODEL_VMAX_DB,
        enable_local_recognition=enable_local_recognition,
        keep_negative_frequencies=keep_negative_frequencies,
    )
    return [DEFAULT_MODEL_VMIN_DB, DEFAULT_MODEL_VMAX_DB]


def photo_save_test(rec_wave, Fs, Rs, Fc, SNR, Modulation, image_dir, *, keep_negative_frequencies=False):
    _save_single_stft_image(rec_wave, Fs, image_dir, keep_negative_frequencies=keep_negative_frequencies)


def photo_save_final(rec_wave, Fs, Rs, Fc, SNR, Modulation, image_dir, *, keep_negative_frequencies=False):
    _save_single_stft_image(rec_wave, Fs, image_dir, keep_negative_frequencies=keep_negative_frequencies)


def get_amplitude_from_rgb(rgb_value, min_val, max_val):
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap("jet")

    dB_range = np.linspace(min_val, max_val, 2048)
    colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=norm).to_rgba(np.linspace(min_val, max_val, 2048))
    colorbar_rgb = colorbar[:, :3]
    differences = np.sqrt(np.sum((colorbar_rgb - rgb_value) ** 2, axis=1))
    idx = np.argmin(differences)
    return dB_range[idx]


def photo_save_scipy(rec_wave, Fs, num, Rs, *, keep_negative_frequencies=False, enable_local_recognition=False):
    create_stft_run_artifacts(
        rec_wave,
        Fs,
        output_dir=SIGNAL_ANA_DIR,
        source_tag="file",
        image_width_px=DEFAULT_IMAGE_WIDTH,
        image_height_px=DEFAULT_IMAGE_HEIGHT,
        model_vmin_db=DEFAULT_MODEL_VMIN_DB,
        model_vmax_db=DEFAULT_MODEL_VMAX_DB,
        enable_local_recognition=enable_local_recognition,
        keep_negative_frequencies=keep_negative_frequencies,
    )
    return [DEFAULT_MODEL_VMIN_DB, DEFAULT_MODEL_VMAX_DB]


def _save_single_stft_image(rec_wave, Fs, image_dir, *, keep_negative_frequencies=False):
    _, _, matrix_db, _ = compute_spectrogram_matrix(
        rec_wave,
        Fs,
        keep_negative_frequencies=keep_negative_frequencies,
    )
    render_spectrogram_image(
        matrix_db,
        image_dir,
        image_width_px=DEFAULT_IMAGE_WIDTH,
        image_height_px=DEFAULT_IMAGE_HEIGHT,
        vmin=DEFAULT_MODEL_VMIN_DB,
        vmax=DEFAULT_MODEL_VMAX_DB,
    )
