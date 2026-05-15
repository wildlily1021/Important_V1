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
) -> dict[str, float]:
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

    peak_minus_3db = peak_db - 3.0
    left_3db = peak_idx
    right_3db = peak_idx
    while left_3db > 0 and psd_db_smooth[left_3db - 1] >= peak_minus_3db:
        left_3db -= 1
    while right_3db < psd_db_smooth.size - 1 and psd_db_smooth[right_3db + 1] >= peak_minus_3db:
        right_3db += 1

    f_min_hz = float(freqs_hz[start_idx] - 0.5 * df_hz)
    f_max_hz = float(freqs_hz[end_idx] + 0.5 * df_hz)
    f_min_3db_hz = float(freqs_hz[left_3db] - 0.5 * df_hz)
    f_max_3db_hz = float(freqs_hz[right_3db] + 0.5 * df_hz)
    bandwidth_hz = max(df_hz, f_max_hz - f_min_hz)
    bandwidth_3db_hz = max(df_hz, f_max_3db_hz - f_min_3db_hz)

    signal_power = float(np.sum(psd[start_idx : end_idx + 1]) * df_hz)
    noise_psd = float(10.0 ** (noise_floor_db / 10.0))
    noise_power = max(noise_psd * bandwidth_hz, EPSILON)
    carrier_power = max(signal_power - noise_power, EPSILON)
    cnr_db = 10.0 * math.log10(carrier_power / noise_power)
    frequency_span_hz = float(freqs_hz[-1] - freqs_hz[0] + df_hz) if freqs_hz.size > 1 else float(fs_hz)

    return {
        "f_min_hz": f_min_hz,
        "f_max_hz": f_max_hz,
        "center_hz": 0.5 * (f_min_hz + f_max_hz),
        "bandwidth_hz": float(bandwidth_hz),
        "bandwidth_3db_hz": float(bandwidth_3db_hz),
        "f_min_3db_hz": f_min_3db_hz,
        "f_max_3db_hz": f_max_3db_hz,
        "eta": float(bandwidth_hz / max(frequency_span_hz, 1.0)),
        "cnr_db": float(cnr_db),
        "threshold_db": threshold_db,
        "noise_floor_db": noise_floor_db,
        "peak_db": peak_db,
        "nfft": int(nfft),
        "df_hz": float(df_hz),
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
        "bandwidth_3db_hz": float(psd_band["bandwidth_3db_hz"]),
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
            "bandwidth_3db_hz": float(psd_band["bandwidth_3db_hz"]),
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
