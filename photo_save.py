from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stft_pipeline import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    build_stft_products,
    compute_spectrogram_matrix,
    render_spectrogram_image,
)


SIGNAL_ANA_DIR = Path("./signal_ana")


def photo_save(rec_wave, Fs, num, Rs):
    result = build_stft_products(
        rec_wave,
        Fs,
        panorama_image_path=SIGNAL_ANA_DIR / "STFT_Org.jpg",
        panorama_metadata_path=SIGNAL_ANA_DIR / "STFT_Org.json",
        local_image_path=SIGNAL_ANA_DIR / "STFT_Org_local.jpg",
        local_metadata_path=SIGNAL_ANA_DIR / "STFT_Org_local.json",
        mask_image_path=SIGNAL_ANA_DIR / "STFT_Org_mask.jpg",
        image_width_px=DEFAULT_IMAGE_WIDTH,
        image_height_px=DEFAULT_IMAGE_HEIGHT,
    )
    return [result["vmin"], result["vmax"]]


def photo_save_test(rec_wave, Fs, Rs, Fc, SNR, Modulation, image_dir):
    _save_single_stft_image(rec_wave, Fs, image_dir)


def photo_save_final(rec_wave, Fs, Rs, Fc, SNR, Modulation, image_dir):
    _save_single_stft_image(rec_wave, Fs, image_dir)


def get_amplitude_from_rgb(rgb_value, min_val, max_val):
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap("jet")

    dB_range = np.linspace(min_val, max_val, 2048)
    colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=norm).to_rgba(np.linspace(min_val, max_val, 2048))
    colorbar_rgb = colorbar[:, :3]
    differences = np.sqrt(np.sum((colorbar_rgb - rgb_value) ** 2, axis=1))
    idx = np.argmin(differences)
    return dB_range[idx]


def photo_save_scipy(rec_wave, Fs, num, Rs):
    result = build_stft_products(
        rec_wave,
        Fs,
        panorama_image_path=SIGNAL_ANA_DIR / "STFT_Org_txt.jpg",
        panorama_metadata_path=SIGNAL_ANA_DIR / "STFT_Org_txt.json",
        local_image_path=SIGNAL_ANA_DIR / "STFT_Org_cropped.jpg",
        local_metadata_path=SIGNAL_ANA_DIR / "STFT_Org_cropped.json",
        mask_image_path=SIGNAL_ANA_DIR / "STFT_Org_mask.jpg",
        image_width_px=DEFAULT_IMAGE_WIDTH,
        image_height_px=DEFAULT_IMAGE_HEIGHT,
    )
    return [result["vmin"], result["vmax"]]


def _save_single_stft_image(rec_wave, Fs, image_dir):
    freqs_hz, _, matrix_db, _ = compute_spectrogram_matrix(rec_wave, Fs)
    if freqs_hz.size:
        half_span = float(Fs) / 2.0
        mask = (freqs_hz >= -half_span) & (freqs_hz <= half_span)
        if np.any(mask):
            matrix_db = matrix_db[mask, :]

    render_spectrogram_image(
        matrix_db,
        image_dir,
        image_width_px=DEFAULT_IMAGE_WIDTH,
        image_height_px=DEFAULT_IMAGE_HEIGHT,
    )
