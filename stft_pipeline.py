from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any
import uuid

import cv2
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter1d
from scipy.signal import spectrogram, welch


EPSILON = 1e-20
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SIGNAL_ANA_DIR = PROJECT_ROOT / "signal_ana"
DEFAULT_IMAGE_WIDTH = 3720
DEFAULT_IMAGE_HEIGHT = 2772
DEFAULT_TARGET_DF_HZ = 5e6
DEFAULT_LOCAL_TRIGGER_RATIO = 0.15
DEFAULT_ENABLE_LOCAL_RECOGNITION = False
DEFAULT_MODEL_VMIN_DB = -70.0
DEFAULT_MODEL_VMAX_DB = 0.0
CURRENT_STFT_RUN_PATH = DEFAULT_SIGNAL_ANA_DIR / "current_stft_run.json"


def select_stft_params(
    fs_hz: float,
    signal_length: int | None = None,
    *,
    target_df_hz: float = DEFAULT_TARGET_DF_HZ,
    min_nfft: int = 512,
    max_nfft: int = 4096,
    overlap_ratio: float = 0.85,
) -> dict[str, int]:
    fs_hz = float(max(fs_hz, 1.0))
    target_bins = max(min_nfft, int(math.ceil(fs_hz / max(target_df_hz, 1.0))))
    nfft = _next_power_of_two(target_bins)
    nfft = max(min_nfft, min(max_nfft, nfft))

    if signal_length is not None and signal_length > 0:
        nperseg = min(int(signal_length), nfft)
    else:
        nperseg = nfft

    nperseg = max(64, nperseg)
    noverlap = min(nperseg - 1, int(round(nperseg * overlap_ratio)))

    return {
        "nfft": int(nfft),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
    }


def compute_spectrogram_matrix(
    samples: np.ndarray,
    fs_hz: float,
    *,
    params: dict[str, int] | None = None,
    scaling: str = "spectrum",
    relative_to_peak: bool = True,
    keep_negative_frequencies: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    signal = _as_complex_array(samples)
    params = params or select_stft_params(fs_hz, signal.size)
    scaling = scaling.lower().strip()
    if scaling not in {"density", "spectrum"}:
        raise ValueError(f"Unsupported spectrogram scaling: {scaling}")

    freqs_hz, times_s, spectrum = spectrogram(
        signal,
        fs=float(fs_hz),
        window=np.hamming(params["nperseg"]),
        nperseg=params["nperseg"],
        noverlap=params["noverlap"],
        nfft=params["nfft"],
        return_onesided=False,
        scaling=scaling,
        mode="psd",
    )

    freqs_hz = np.fft.fftshift(freqs_hz)
    spectrum = np.fft.fftshift(spectrum, axes=0)
    if not keep_negative_frequencies:
        positive_mask = freqs_hz >= 0
        freqs_hz = freqs_hz[positive_mask]
        spectrum = spectrum[positive_mask, :]

    matrix_db = 10.0 * np.log10(np.maximum(spectrum, EPSILON))
    if relative_to_peak:
        peak_db = float(np.max(matrix_db)) if matrix_db.size else 0.0
        if not np.isfinite(peak_db):
            peak_db = 0.0
        matrix_db = matrix_db - peak_db

    return freqs_hz, times_s, matrix_db, params


def estimate_psd_band(
    samples: np.ndarray,
    fs_hz: float,
    *,
    nfft: int | None = None,
    keep_negative_frequencies: bool = False,
) -> dict[str, Any]:
    signal = _as_complex_array(samples)
    nfft = int(nfft or max(4096, select_stft_params(fs_hz, signal.size)["nfft"]))
    nperseg = min(nfft, max(64, signal.size))
    noverlap = min(nperseg - 1, int(round(nperseg * 0.75)))

    freqs_hz, psd = welch(
        signal,
        fs=float(fs_hz),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        scaling="density",
    )

    freqs_hz = np.fft.fftshift(freqs_hz)
    psd = np.fft.fftshift(psd)
    if not keep_negative_frequencies:
        positive_mask = freqs_hz >= 0
        freqs_hz = freqs_hz[positive_mask]
        psd = psd[positive_mask]
    freqs_hz, psd = _sort_frequency_and_values(freqs_hz, psd)
    psd = np.maximum(psd, EPSILON)
    psd_db = 10.0 * np.log10(np.maximum(psd, EPSILON))
    smooth_size = max(5, int(round(nfft / 512)))
    psd_db_smooth = uniform_filter1d(psd_db, size=smooth_size, mode="nearest")

    peak_idx = int(np.argmax(psd_db_smooth))
    peak_db = float(psd_db_smooth[peak_idx])
    noise_floor_db = float(np.percentile(psd_db_smooth, 35))
    threshold_db = float(max(noise_floor_db + 6.0, peak_db - 20.0))

    band_mask = psd_db_smooth >= threshold_db
    if not np.any(band_mask):
        band_mask[peak_idx] = True

    start_idx, end_idx = _segment_containing_index(band_mask, peak_idx)
    df_hz = float(abs(freqs_hz[1] - freqs_hz[0])) if freqs_hz.size > 1 else float(fs_hz)

    f_min_hz = float(freqs_hz[start_idx] - 0.5 * df_hz)
    f_max_hz = float(freqs_hz[end_idx] + 0.5 * df_hz)
    bandwidth_hz = max(df_hz, f_max_hz - f_min_hz)
    coarse_band = {
        "f_min_hz": f_min_hz,
        "f_max_hz": f_max_hz,
        "center_hz": 0.5 * (f_min_hz + f_max_hz),
        "bandwidth_hz": float(bandwidth_hz),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "threshold_db": threshold_db,
    }
    final_measurement = _measure_final_psd_band(
        freqs_hz=freqs_hz,
        psd=psd,
        psd_db_smooth=psd_db_smooth,
        coarse_start_idx=start_idx,
        coarse_end_idx=end_idx,
        peak_idx=peak_idx,
        noise_floor_db=noise_floor_db,
        smooth_size=smooth_size,
        df_hz=df_hz,
    )
    f_min_3db_hz = float(final_measurement["f_left_3db_hz"])
    f_max_3db_hz = float(final_measurement["f_right_3db_hz"])
    bandwidth_3db_hz = max(df_hz, f_max_3db_hz - f_min_3db_hz)

    signal_power = float(np.sum(psd[start_idx : end_idx + 1]) * df_hz)
    noise_psd = float(10.0 ** (noise_floor_db / 10.0))
    noise_power = max(noise_psd * bandwidth_hz, EPSILON)
    carrier_power = max(signal_power - noise_power, EPSILON)
    cnr_db = 10.0 * math.log10(carrier_power / noise_power)
    frequency_span_hz = float(freqs_hz[-1] - freqs_hz[0] + df_hz) if freqs_hz.size > 1 else float(fs_hz)

    result = {
        "f_min_hz": f_min_hz,
        "f_max_hz": f_max_hz,
        "center_hz": coarse_band["center_hz"],
        "bandwidth_hz": float(bandwidth_hz),
        "bandwidth_3db_hz": float(bandwidth_3db_hz),
        "f_min_3db_hz": f_min_3db_hz,
        "f_max_3db_hz": f_max_3db_hz,
        "center_3db_hz": float(final_measurement["center_hz"]),
        "eta": float(bandwidth_hz / max(frequency_span_hz, 1.0)),
        "cnr_db": float(cnr_db),
        "threshold_db": threshold_db,
        "noise_floor_db": noise_floor_db,
        "peak_db": peak_db,
        "peak_hz": float(freqs_hz[peak_idx]),
        "nfft": int(nfft),
        "df_hz": float(df_hz),
        "psd_band_coarse": coarse_band,
        "psd_measure_final": final_measurement,
        "debug": {
            "psd_linear_min": float(np.min(psd)),
            "psd_linear_max": float(np.max(psd)),
            "psd_db_min": float(np.min(psd_db)),
            "psd_db_max": float(np.max(psd_db)),
            "psd_db_smooth_min": float(np.min(psd_db_smooth)),
            "psd_db_smooth_max": float(np.max(psd_db_smooth)),
            "freq_min_hz": float(freqs_hz[0]) if freqs_hz.size else 0.0,
            "freq_max_hz": float(freqs_hz[-1]) if freqs_hz.size else 0.0,
        },
    }
    _log_psd_measurement_debug(result)
    return result


def _measure_final_psd_band(
    *,
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    psd_db_smooth: np.ndarray,
    coarse_start_idx: int,
    coarse_end_idx: int,
    peak_idx: int,
    noise_floor_db: float,
    smooth_size: int,
    df_hz: float,
) -> dict[str, float | str | bool]:
    local_slice = slice(coarse_start_idx, coarse_end_idx + 1)
    local_freqs = freqs_hz[local_slice]
    local_psd = np.maximum(psd[local_slice], EPSILON)
    local_psd_db = 10.0 * np.log10(local_psd)
    local_smooth_size = _bounded_odd_window(local_psd_db.size, max(smooth_size * 4, int(round(local_psd_db.size * 0.08))))
    local_psd_db_smooth = uniform_filter1d(local_psd_db, size=local_smooth_size, mode="nearest")
    local_refine_size = _bounded_odd_window(local_psd_db.size, max(5, smooth_size))
    local_psd_db_refine = uniform_filter1d(local_psd_db, size=local_refine_size, mode="nearest")

    local_peak_idx = int(np.argmax(local_psd_db_smooth))
    peak_db_local = float(local_psd_db_smooth[local_peak_idx])
    peak_hz = float(local_freqs[local_peak_idx])
    plateau_gate_db = max(noise_floor_db + 6.0, peak_db_local - 10.0)
    plateau_mask = local_psd_db_smooth >= plateau_gate_db
    if np.any(plateau_mask):
        plateau_ref_db = float(np.median(local_psd_db_smooth[plateau_mask]))
    else:
        plateau_ref_db = peak_db_local
    threshold_3db_db = plateau_ref_db - 3.0

    above_threshold_mask = local_psd_db_smooth >= threshold_3db_db
    above_threshold_mask = _bridge_small_false_gaps(
        above_threshold_mask,
        max_gap_bins=max(1, local_smooth_size // 2),
    )
    mask_indices = np.flatnonzero(above_threshold_mask)
    segment_count = 0
    if mask_indices.size:
        segment_count = int(np.count_nonzero(np.diff(mask_indices) > 1) + 1)

    left_cross = None
    right_cross = None
    if mask_indices.size:
        left_idx = int(mask_indices[0])
        right_idx = int(mask_indices[-1])

        for idx in range(min(left_idx + 1, local_psd_db_refine.size - 1), 0, -1):
            if local_psd_db_refine[idx - 1] < threshold_3db_db <= local_psd_db_refine[idx]:
                left_cross = _interpolate_threshold_crossing(
                    local_freqs[idx - 1],
                    local_psd_db_refine[idx - 1],
                    local_freqs[idx],
                    local_psd_db_refine[idx],
                    threshold_3db_db,
                )
                break
        if left_cross is None and left_idx > 0:
            left_cross = _interpolate_threshold_crossing(
                local_freqs[left_idx - 1],
                local_psd_db_refine[left_idx - 1],
                local_freqs[left_idx],
                local_psd_db_refine[left_idx],
                threshold_3db_db,
            )

        for idx in range(max(right_idx, 0), local_psd_db_refine.size - 1):
            if local_psd_db_refine[idx] >= threshold_3db_db > local_psd_db_refine[idx + 1]:
                right_cross = _interpolate_threshold_crossing(
                    local_freqs[idx],
                    local_psd_db_refine[idx],
                    local_freqs[idx + 1],
                    local_psd_db_refine[idx + 1],
                    threshold_3db_db,
                )
                break
        if right_cross is None and right_idx < local_psd_db_refine.size - 1:
            right_cross = _interpolate_threshold_crossing(
                local_freqs[right_idx],
                local_psd_db_refine[right_idx],
                local_freqs[right_idx + 1],
                local_psd_db_refine[right_idx + 1],
                threshold_3db_db,
            )

    final_left = float(left_cross) if left_cross is not None else float(local_freqs[0] - 0.5 * df_hz)
    final_right = float(right_cross) if right_cross is not None else float(local_freqs[-1] + 0.5 * df_hz)
    if final_right < final_left:
        final_left, final_right = final_right, final_left

    centroid_threshold_db = max(noise_floor_db + 3.0, peak_db_local - 10.0)
    centroid_mask = local_psd_db_smooth >= centroid_threshold_db
    if np.any(centroid_mask):
        centroid_weights = np.maximum(local_psd[centroid_mask], EPSILON)
        centroid_hz = float(np.average(local_freqs[centroid_mask], weights=centroid_weights))
    else:
        centroid_hz = peak_hz

    crossings_found = mask_indices.size > 0
    center_hz = 0.5 * (final_left + final_right) if crossings_found else centroid_hz
    bandwidth_hz = max(df_hz, final_right - final_left)

    return {
        "method": "3db_outer_edges" if crossings_found else "centroid_fallback",
        "success": bool(crossings_found),
        "peak_hz": peak_hz,
        "peak_db": peak_db_local,
        "reference_db": plateau_ref_db,
        "centroid_hz": centroid_hz,
        "threshold_3db_db": threshold_3db_db,
        "f_left_3db_hz": final_left,
        "f_right_3db_hz": final_right,
        "center_hz": center_hz,
        "bandwidth_hz": bandwidth_hz,
        "local_smooth_size": int(local_smooth_size),
        "local_refine_size": int(local_refine_size),
        "left_cross_found": bool(left_cross is not None),
        "right_cross_found": bool(right_cross is not None),
        "segment_count": segment_count,
    }


def build_stft_products(
    samples: np.ndarray,
    fs_hz: float,
    *,
    panorama_image_path: str | Path,
    panorama_metadata_path: str | Path | None = None,
    local_image_path: str | Path | None = None,
    local_metadata_path: str | Path | None = None,
    mask_image_path: str | Path | None = None,
    image_width_px: int = DEFAULT_IMAGE_WIDTH,
    image_height_px: int = DEFAULT_IMAGE_HEIGHT,
    local_trigger_ratio: float = DEFAULT_LOCAL_TRIGGER_RATIO,
    model_vmin_db: float = DEFAULT_MODEL_VMIN_DB,
    model_vmax_db: float = DEFAULT_MODEL_VMAX_DB,
    enable_local_recognition: bool = DEFAULT_ENABLE_LOCAL_RECOGNITION,
    keep_negative_frequencies: bool = False,
    run_id: str | None = None,
    source_tag: str = "stft",
) -> dict[str, Any]:
    panorama_image_path = Path(panorama_image_path)
    panorama_metadata_path = Path(panorama_metadata_path or panorama_image_path.with_suffix(".json"))
    local_image_path = Path(local_image_path) if local_image_path else panorama_image_path.with_name(
        f"{panorama_image_path.stem}_local{panorama_image_path.suffix}"
    )
    local_metadata_path = Path(local_metadata_path or local_image_path.with_suffix(".json"))
    mask_image_path = Path(mask_image_path) if mask_image_path else panorama_image_path.with_name(
        f"{panorama_image_path.stem}_mask{panorama_image_path.suffix}"
    )

    signal = _as_complex_array(samples)
    params = select_stft_params(fs_hz, signal.size)
    psd_band = estimate_psd_band(
        signal,
        fs_hz,
        nfft=max(4096, params["nfft"]),
        keep_negative_frequencies=keep_negative_frequencies,
    )
    psd_measure_final = psd_band.get("psd_measure_final") or {}
    freqs_hz, _, matrix_db, params = compute_spectrogram_matrix(
        signal,
        fs_hz,
        params=params,
        scaling="spectrum",
        relative_to_peak=True,
        keep_negative_frequencies=keep_negative_frequencies,
    )

    panorama_limits = render_spectrogram_image(
        matrix_db,
        panorama_image_path,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        vmin=model_vmin_db,
        vmax=model_vmax_db,
    )

    if freqs_hz.size:
        full_freq_min_hz = float(freqs_hz[0])
        full_freq_max_hz = float(freqs_hz[-1])
    else:
        full_freq_min_hz = -float(fs_hz) / 2.0 if keep_negative_frequencies else 0.0
        full_freq_max_hz = float(fs_hz) / 2.0
    recognition_image_path = panorama_image_path
    recognition_metadata_path = panorama_metadata_path
    run_id = run_id or create_run_id(source_tag)

    panorama_meta: dict[str, Any] = {
        "run_id": run_id,
        "source_tag": source_tag,
        "role": "panorama",
        "freq_min_hz": full_freq_min_hz,
        "freq_max_hz": full_freq_max_hz,
        "frequency_span_hz": float(full_freq_max_hz - full_freq_min_hz),
        "Fs_hz": float(fs_hz),
        "image_width_px": int(image_width_px),
        "image_height_px": int(image_height_px),
        "nfft": int(params["nfft"]),
        "nperseg": int(params["nperseg"]),
        "noverlap": int(params["noverlap"]),
        "psd_band": psd_band,
        "psd_measure_final": psd_measure_final,
        "bandwidth_3db_hz": float(psd_measure_final.get("bandwidth_hz", psd_band["bandwidth_3db_hz"])),
        "cnr_db": float(psd_band["cnr_db"]),
        "bbox_inches": None,
        "display_min_db": float(panorama_limits["vmin"]),
        "display_max_db": float(panorama_limits["vmax"]),
        "stft_scaling": "spectrum",
        "relative_db": True,
        "keep_negative_frequencies": bool(keep_negative_frequencies),
        "mask_image_path": _portable_path(mask_image_path),
        "selection_mask_image_path": _portable_path(mask_image_path),
        "local_image_path": None,
        "use_local_recognition": False,
    }

    local_window = None
    if enable_local_recognition and local_trigger_ratio > 0:
        local_window = determine_local_frequency_window(
            psd_band,
            full_freq_min_hz,
            full_freq_max_hz,
            trigger_ratio=local_trigger_ratio,
        )

    if local_window is not None:
        mask_limits = {
            "f_min_hz": float(local_window["f_min_hz"]),
            "f_max_hz": float(local_window["f_max_hz"]),
        }
        render_frequency_mask(
            mask_image_path,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            full_freq_min_hz=full_freq_min_hz,
            full_freq_max_hz=full_freq_max_hz,
            selected_freq_min_hz=mask_limits["f_min_hz"],
            selected_freq_max_hz=mask_limits["f_max_hz"],
        )

        local_freqs_hz, local_matrix_db = slice_frequency_window(
            freqs_hz,
            matrix_db,
            local_window["f_min_hz"],
            local_window["f_max_hz"],
        )
        local_limits = render_spectrogram_image(
            local_matrix_db,
            local_image_path,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            vmin=model_vmin_db,
            vmax=model_vmax_db,
        )

        local_meta: dict[str, Any] = {
            "run_id": run_id,
            "source_tag": source_tag,
            "role": "local",
            "freq_min_hz": float(local_window["f_min_hz"]),
            "freq_max_hz": float(local_window["f_max_hz"]),
            "frequency_span_hz": float(local_window["f_max_hz"] - local_window["f_min_hz"]),
            "Fs_hz": float(fs_hz),
            "image_width_px": int(image_width_px),
            "image_height_px": int(image_height_px),
            "nfft": int(params["nfft"]),
            "nperseg": int(params["nperseg"]),
            "noverlap": int(params["noverlap"]),
            "psd_band": psd_band,
            "psd_measure_final": psd_measure_final,
            "bandwidth_3db_hz": float(psd_measure_final.get("bandwidth_hz", psd_band["bandwidth_3db_hz"])),
            "cnr_db": float(psd_band["cnr_db"]),
            "bbox_inches": None,
            "display_min_db": float(local_limits["vmin"]),
            "display_max_db": float(local_limits["vmax"]),
            "stft_scaling": "spectrum",
            "relative_db": True,
            "keep_negative_frequencies": bool(keep_negative_frequencies),
            "use_local_recognition": False,
            "source_full_image_path": _portable_path(panorama_image_path),
            "selection_mask_image_path": _portable_path(mask_image_path),
            "matrix_freq_min_hz": float(local_freqs_hz[0]) if local_freqs_hz.size else float(local_window["f_min_hz"]),
            "matrix_freq_max_hz": float(local_freqs_hz[-1]) if local_freqs_hz.size else float(local_window["f_max_hz"]),
        }
        write_metadata(local_metadata_path, local_meta)

        panorama_meta["local_image_path"] = _portable_path(local_image_path)
    else:
        render_frequency_mask(
            mask_image_path,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            full_freq_min_hz=full_freq_min_hz,
            full_freq_max_hz=full_freq_max_hz,
            selected_freq_min_hz=psd_band["f_min_hz"],
            selected_freq_max_hz=psd_band["f_max_hz"],
        )

    write_metadata(panorama_metadata_path, panorama_meta)

    return {
        "run_id": run_id,
        "source_tag": source_tag,
        "params": params,
        "psd_band": psd_band,
        "recognition_image_path": str(recognition_image_path),
        "recognition_metadata_path": str(recognition_metadata_path),
        "panorama_image_path": str(panorama_image_path),
        "panorama_metadata_path": str(panorama_metadata_path),
        "local_image_path": str(local_image_path) if panorama_meta["local_image_path"] else None,
        "local_metadata_path": str(local_metadata_path) if panorama_meta["local_image_path"] else None,
        "selection_mask_image_path": str(mask_image_path),
        "vmin": float(panorama_limits["vmin"]),
        "vmax": float(panorama_limits["vmax"]),
        "use_local_recognition": bool(panorama_meta["use_local_recognition"]),
    }


def create_run_id(source_tag: str = "stft") -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    token = uuid.uuid4().hex[:8]
    return f"{source_tag}_{timestamp}_{token}"


def create_stft_run_artifacts(
    samples: np.ndarray,
    fs_hz: float,
    *,
    output_dir: str | Path = DEFAULT_SIGNAL_ANA_DIR,
    source_tag: str = "stft",
    run_id: str | None = None,
    image_width_px: int = DEFAULT_IMAGE_WIDTH,
    image_height_px: int = DEFAULT_IMAGE_HEIGHT,
    local_trigger_ratio: float = DEFAULT_LOCAL_TRIGGER_RATIO,
    model_vmin_db: float = DEFAULT_MODEL_VMIN_DB,
    model_vmax_db: float = DEFAULT_MODEL_VMAX_DB,
    enable_local_recognition: bool = DEFAULT_ENABLE_LOCAL_RECOGNITION,
    keep_negative_frequencies: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    run_id = run_id or create_run_id(source_tag)
    prefix = output_dir / run_id

    result = build_stft_products(
        samples,
        fs_hz,
        panorama_image_path=prefix.with_name(f"{run_id}_model_full.jpg"),
        panorama_metadata_path=prefix.with_name(f"{run_id}_model_full.json"),
        local_image_path=prefix.with_name(f"{run_id}_model_local.jpg"),
        local_metadata_path=prefix.with_name(f"{run_id}_model_local.json"),
        mask_image_path=prefix.with_name(f"{run_id}_model_mask.jpg"),
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        local_trigger_ratio=local_trigger_ratio,
        model_vmin_db=model_vmin_db,
        model_vmax_db=model_vmax_db,
        enable_local_recognition=enable_local_recognition,
        keep_negative_frequencies=keep_negative_frequencies,
        run_id=run_id,
        source_tag=source_tag,
    )

    manifest = {
        "run_id": run_id,
        "source_tag": source_tag,
        "model_full_image_path": str(Path(result["panorama_image_path"]).resolve()),
        "model_full_metadata_path": str(Path(result["panorama_metadata_path"]).resolve()),
        "model_local_image_path": str(Path(result["local_image_path"]).resolve()) if result["local_image_path"] else None,
        "model_local_metadata_path": str(Path(result["local_metadata_path"]).resolve()) if result["local_metadata_path"] else None,
        "recognition_image_path": str(Path(result["recognition_image_path"]).resolve()),
        "recognition_metadata_path": str(Path(result["recognition_metadata_path"]).resolve()),
        "selection_mask_image_path": str(Path(result["selection_mask_image_path"]).resolve()),
        "mask_image_path": str((output_dir / f"{run_id}_unet_mask.jpg").resolve()),
        "annotated_image_path": str((output_dir / f"{run_id}_annotated.jpg").resolve()),
        "display_image_path": str((output_dir / f"{run_id}_display.jpg").resolve()),
        "use_local_recognition": result["use_local_recognition"],
        "enable_local_recognition": bool(enable_local_recognition),
        "keep_negative_frequencies": bool(keep_negative_frequencies),
        "model_vmin_db": float(model_vmin_db),
        "model_vmax_db": float(model_vmax_db),
    }
    write_current_stft_run(manifest)
    return manifest


def determine_local_frequency_window(
    psd_band: dict[str, float],
    full_freq_min_hz: float,
    full_freq_max_hz: float,
    *,
    trigger_ratio: float = DEFAULT_LOCAL_TRIGGER_RATIO,
) -> dict[str, float] | None:
    full_span_hz = float(full_freq_max_hz - full_freq_min_hz)
    band_span_hz = float(max(psd_band["bandwidth_hz"], psd_band["bandwidth_3db_hz"]))
    if full_span_hz <= 0:
        return None

    eta = band_span_hz / full_span_hz
    if eta >= trigger_ratio:
        return None

    target_span_hz = max(full_span_hz * trigger_ratio, band_span_hz * 1.8, psd_band["bandwidth_3db_hz"] * 3.0)
    target_span_hz = min(full_span_hz, target_span_hz)
    center_hz = float(psd_band["center_hz"])
    half_span_hz = 0.5 * target_span_hz
    freq_min_hz = center_hz - half_span_hz
    freq_max_hz = center_hz + half_span_hz

    if freq_min_hz < full_freq_min_hz:
        shift = full_freq_min_hz - freq_min_hz
        freq_min_hz += shift
        freq_max_hz += shift
    if freq_max_hz > full_freq_max_hz:
        shift = freq_max_hz - full_freq_max_hz
        freq_min_hz -= shift
        freq_max_hz -= shift

    freq_min_hz = max(full_freq_min_hz, freq_min_hz)
    freq_max_hz = min(full_freq_max_hz, freq_max_hz)

    return {
        "f_min_hz": float(freq_min_hz),
        "f_max_hz": float(freq_max_hz),
        "eta": float(eta),
    }


def slice_frequency_window(
    freqs_hz: np.ndarray,
    matrix_db: np.ndarray,
    freq_min_hz: float,
    freq_max_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (freqs_hz >= freq_min_hz) & (freqs_hz <= freq_max_hz)
    if not np.any(mask):
        center_idx = int(np.argmin(np.abs(freqs_hz - 0.5 * (freq_min_hz + freq_max_hz))))
        left = max(0, center_idx - 8)
        right = min(freqs_hz.size, center_idx + 9)
        return freqs_hz[left:right], matrix_db[left:right, :]
    return freqs_hz[mask], matrix_db[mask, :]


def render_spectrogram_image(
    matrix_db: np.ndarray,
    output_path: str | Path,
    *,
    image_width_px: int = DEFAULT_IMAGE_WIDTH,
    image_height_px: int = DEFAULT_IMAGE_HEIGHT,
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_db = np.asarray(matrix_db, dtype=np.float32)
    if matrix_db.ndim != 2 or matrix_db.size == 0:
        raise ValueError("STFT matrix must be a non-empty 2D array")

    vmax = float(np.max(matrix_db)) if vmax is None else float(vmax)
    vmin = float(vmax - 70.0) if vmin is None else float(vmin)
    if not np.isfinite(vmax):
        vmax = 0.0
    if not np.isfinite(vmin) or vmin >= vmax:
        vmin = vmax - 1.0

    flipped = np.flipud(matrix_db)
    resized = cv2.resize(flipped, (int(image_width_px), int(image_height_px)), interpolation=cv2.INTER_LINEAR)
    normalized = Normalize(vmin=vmin, vmax=vmax, clip=True)(resized)
    rgb = (cm.get_cmap("jet")(normalized)[..., :3] * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return {"vmin": float(vmin), "vmax": float(vmax)}


def render_frequency_mask(
    output_path: str | Path,
    *,
    image_width_px: int,
    image_height_px: int,
    full_freq_min_hz: float,
    full_freq_max_hz: float,
    selected_freq_min_hz: float,
    selected_freq_max_hz: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((int(image_height_px), int(image_width_px)), dtype=np.uint8)
    top_row = frequency_to_pixel_y(selected_freq_max_hz, full_freq_min_hz, full_freq_max_hz, image_height_px)
    bottom_row = frequency_to_pixel_y(selected_freq_min_hz, full_freq_min_hz, full_freq_max_hz, image_height_px)
    row_start = max(0, min(top_row, bottom_row))
    row_end = min(int(image_height_px), max(top_row, bottom_row))
    mask[row_start:row_end, :] = 255
    cv2.imwrite(str(output_path), mask)


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def write_current_stft_run(manifest: dict[str, Any], path: str | Path = CURRENT_STFT_RUN_PATH) -> None:
    write_metadata(path, manifest)


def load_current_stft_run(path: str | Path = CURRENT_STFT_RUN_PATH) -> dict[str, Any] | None:
    return load_stft_metadata(path)


def load_stft_metadata(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    metadata_path = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_recognition_image_path(path: str | Path) -> str:
    image_path = Path(path)
    metadata = load_stft_metadata(image_path)
    if not metadata:
        return str(image_path)

    local_image_path = metadata.get("local_image_path")
    if metadata.get("use_local_recognition") and local_image_path:
        candidate = Path(local_image_path)
        candidate_paths = [candidate]
        if not candidate.is_absolute():
            candidate_paths.append((Path.cwd() / candidate).resolve())
            candidate_paths.append(image_path.parent / candidate.name)
        for item in candidate_paths:
            if item.exists():
                return str(item)

    return str(image_path)


def get_current_recognition_image_path(path: str | Path = CURRENT_STFT_RUN_PATH) -> str | None:
    manifest = load_current_stft_run(path)
    if not manifest:
        return None
    recognition_path = manifest.get("recognition_image_path")
    return str(recognition_path) if recognition_path else None


def pixel_to_frequency_hz(pixel_y: float, metadata: dict[str, Any], *, image_height_px: int | None = None) -> float:
    freq_min_hz = float(metadata.get("freq_min_hz", 0.0))
    freq_max_hz = float(metadata.get("freq_max_hz", 0.0))
    height = int(image_height_px or metadata.get("image_height_px") or DEFAULT_IMAGE_HEIGHT)
    if height <= 0:
        return 0.5 * (freq_min_hz + freq_max_hz)

    ratio = float(np.clip(pixel_y / float(height), 0.0, 1.0))
    return float(freq_max_hz - ratio * (freq_max_hz - freq_min_hz))


def pixel_height_to_bandwidth_hz(
    pixel_height: float,
    metadata: dict[str, Any],
    *,
    image_height_px: int | None = None,
) -> float:
    freq_min_hz = float(metadata.get("freq_min_hz", 0.0))
    freq_max_hz = float(metadata.get("freq_max_hz", 0.0))
    height = int(image_height_px or metadata.get("image_height_px") or DEFAULT_IMAGE_HEIGHT)
    if height <= 0:
        return 0.0
    return float(max(pixel_height, 0.0) / float(height) * (freq_max_hz - freq_min_hz))


def frequency_to_pixel_y(
    frequency_hz: float,
    freq_min_hz: float,
    freq_max_hz: float,
    image_height_px: int,
) -> int:
    span = max(freq_max_hz - freq_min_hz, 1.0)
    ratio = float(np.clip((freq_max_hz - frequency_hz) / span, 0.0, 1.0))
    return int(round(ratio * image_height_px))


def _sort_frequency_and_values(freqs_hz: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if freqs_hz.size <= 1:
        return freqs_hz, values
    order = np.argsort(freqs_hz)
    return freqs_hz[order], values[order]


def _bounded_odd_window(length: int, proposal: int) -> int:
    if length <= 1:
        return 1
    size = max(3, min(int(proposal), int(length)))
    if size % 2 == 0:
        size -= 1
    if size < 3:
        size = 3 if length >= 3 else length
    return max(1, size)


def _bridge_small_false_gaps(mask: np.ndarray, *, max_gap_bins: int) -> np.ndarray:
    filled = np.asarray(mask, dtype=bool).copy()
    if max_gap_bins <= 0 or filled.size == 0:
        return filled

    false_start = None
    for idx, value in enumerate(filled):
        if not value and false_start is None:
            false_start = idx
        elif value and false_start is not None:
            gap_size = idx - false_start
            if false_start > 0 and gap_size <= max_gap_bins:
                filled[false_start:idx] = True
            false_start = None

    return filled


def _interpolate_threshold_crossing(
    freq_a_hz: float,
    value_a_db: float,
    freq_b_hz: float,
    value_b_db: float,
    threshold_db: float,
) -> float:
    if value_b_db == value_a_db:
        return 0.5 * (freq_a_hz + freq_b_hz)
    ratio = (threshold_db - value_a_db) / (value_b_db - value_a_db)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(freq_a_hz + ratio * (freq_b_hz - freq_a_hz))


def _log_psd_measurement_debug(result: dict[str, Any]) -> None:
    coarse = result.get("psd_band_coarse") or {}
    final = result.get("psd_measure_final") or {}
    debug = result.get("debug") or {}
    print(
        "[PSD] "
        f"freq=({debug.get('freq_min_hz', 0.0):.3e},{debug.get('freq_max_hz', 0.0):.3e}) "
        f"lin=({debug.get('psd_linear_min', 0.0):.3e},{debug.get('psd_linear_max', 0.0):.3e}) "
        f"dB=({debug.get('psd_db_min', 0.0):.2f},{debug.get('psd_db_max', 0.0):.2f})"
    )
    print(
        "[PSD] "
        f"coarse=({coarse.get('f_min_hz', 0.0):.3e},{coarse.get('f_max_hz', 0.0):.3e}) "
        f"peak={result.get('peak_hz', 0.0):.3e}Hz/{result.get('peak_db', 0.0):.2f}dB "
        f"noise={result.get('noise_floor_db', 0.0):.2f}dB "
        f"final=({final.get('f_left_3db_hz', 0.0):.3e},{final.get('f_right_3db_hz', 0.0):.3e}) "
        f"center={final.get('center_hz', 0.0):.3e}Hz "
        f"bw={final.get('bandwidth_hz', 0.0):.3e}Hz "
        f"method={final.get('method', 'unknown')} "
        f"segments={final.get('segment_count', 0)}"
    )


def _segment_containing_index(mask: np.ndarray, index: int) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return index, index

    splits = np.where(np.diff(indices) > 1)[0] + 1
    segments = np.split(indices, splits)
    for segment in segments:
        if segment[0] <= index <= segment[-1]:
            return int(segment[0]), int(segment[-1])

    strongest = max(segments, key=len)
    return int(strongest[0]), int(strongest[-1])


def _as_complex_array(samples: np.ndarray) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if signal.size < 64:
        raise ValueError("Signal is too short to build a stable STFT image")
    return signal


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _portable_path(path: Path) -> str:
    return str(path).replace("\\", "/")
