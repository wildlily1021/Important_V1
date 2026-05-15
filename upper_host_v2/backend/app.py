from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from legacy_pipeline import analyze_file, analyze_generated, detect_sample_rate, preview_file

app = FastAPI(title="Signal Analysis Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_WAVEFORM_POINTS = 1_200
MAX_CONSTELLATION_POINTS = 2_200
MAX_FFT_POINTS = 2_048


class SignalFileRequest(BaseModel):
    file_path: str


class SignalAnalysisRequest(BaseModel):
    mode: Literal["file", "generated"] = "file"
    file_path: str | None = None
    fs_hz: float | None = Field(default=None, gt=0)
    fc_hz: float = Field(default=3.25e9)
    rs_hz: float = Field(default=1.75e9, gt=0)
    snr_db: float = Field(default=-10.0)
    modulation: int = Field(default=1, ge=1, le=4)


@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Backend is running. Open /health to check the API status.",
        "health_url": "http://127.0.0.1:8000/health",
    }


@app.get("/health")
def health():
    return {"ok": True, "message": "Python backend is running"}


@app.post("/signals/inspect")
def inspect_signal_file(request: SignalFileRequest):
    path = validate_signal_file(request.file_path)
    preview_lines = preview_file(path)
    detected_sample_rate_hz = None
    try:
        detected_sample_rate_hz = detect_sample_rate(path)
    except Exception:
        detected_sample_rate_hz = None
    return {
        "ok": True,
        "name": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "detected_format": path.suffix.lower().lstrip(".") or "unknown",
        "detected_sample_rate_hz": detected_sample_rate_hz,
        "preview_lines": preview_lines,
    }


@app.post("/signals/analyze")
def analyze_signal(request: SignalAnalysisRequest):
    try:
        if request.mode == "file":
            if not request.file_path:
                raise HTTPException(status_code=400, detail="file_path is required for file mode")
            path = validate_signal_file(request.file_path)
            result = analyze_file(
                file_path=path,
                fs_hz=request.fs_hz,
                fc_hz=request.fc_hz,
                rs_hz=request.rs_hz,
                snr_db=request.snr_db,
                modulation=request.modulation,
            )
        else:
            if request.fs_hz is None:
                raise HTTPException(status_code=400, detail="fs_hz is required for generated mode")
            result = analyze_generated(
                fs_hz=request.fs_hz,
                fc_hz=request.fc_hz,
                rs_hz=request.rs_hz,
                snr_db=request.snr_db,
                modulation=request.modulation,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    signal_samples = coerce_complex_list(result.signal_samples)
    constellation_samples = coerce_complex_list(result.constellation_samples)
    waveform = build_waveform(signal_samples)
    constellation = build_constellation(constellation_samples)
    fft_data = build_fft(signal_samples, result.sample_rate_hz)

    return {
        "ok": True,
        "mode": result.mode,
        "name": result.name,
        "path": result.path,
        "detected_sample_rate_hz": result.detected_sample_rate_hz,
        "sample_rate_hz": result.sample_rate_hz,
        "input": result.input_values,
        "signal": {
            "total_points": len(signal_samples),
            "truncated": False,
        },
        "preview_lines": result.preview_lines,
        "stats": build_stats(signal_samples),
        "estimates": result.estimates,
        "display_values": result.display_values,
        "quality": result.quality,
        "errors": result.errors,
        "bandwidth_true_hz": result.estimates["bandwidth_true_hz"],
        "waveform": waveform,
        "fft": fft_data,
        "constellation": constellation,
        "stft": {
            "frequency_hz": [],
            "time_s": [],
            "matrix": [],
        },
        "images": result.images,
    }


def validate_signal_file(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Signal file does not exist")
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Selected path is not a file")
    return path


def coerce_complex_list(values) -> list[complex]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, complex):
        return [values]
    if not isinstance(values, list):
        try:
            values = list(values)
        except Exception:
            return []
    result: list[complex] = []
    for value in values:
        try:
            result.append(complex(value))
        except Exception:
            continue
    return result


def build_waveform(samples: list[complex]) -> list[dict]:
    points = downsample_complex(samples, MAX_WAVEFORM_POINTS)
    return [
        {
            "index": index,
            "real": value.real,
            "imag": value.imag,
            "magnitude": abs(value),
        }
        for index, value in enumerate(points)
    ]


def build_constellation(samples: list[complex]) -> list[dict]:
    points = downsample_complex(samples, MAX_CONSTELLATION_POINTS)
    return [{"real": value.real, "imag": value.imag} for value in points]


def build_fft(samples: list[complex], fs_hz: float) -> dict:
    values = downsample_complex(samples, MAX_FFT_POINTS)
    if len(values) < 2:
        return {"frequency_hz": [], "magnitude": []}

    try:
        import numpy as np  # type: ignore

        array = np.asarray(values, dtype=np.complex128)
        spectrum = np.fft.fftshift(np.fft.fft(array))
        frequencies = np.fft.fftshift(np.fft.fftfreq(array.size, d=1.0 / fs_hz))
        magnitudes = np.abs(spectrum)
        return {
            "frequency_hz": frequencies.tolist(),
            "magnitude": magnitudes.tolist(),
        }
    except Exception:
        size = len(values)
        spectrum: list[complex] = []
        for k in range(size):
            total = 0j
            for index, value in enumerate(values):
                angle = -2 * math.pi * k * index / size
                total += value * complex(math.cos(angle), math.sin(angle))
            spectrum.append(total)
        midpoint = size // 2
        shifted = spectrum[midpoint:] + spectrum[:midpoint]
        freqs = [((index - midpoint) / size) * fs_hz for index in range(size)]
        return {
            "frequency_hz": freqs,
            "magnitude": [abs(value) for value in shifted],
        }


def build_stats(samples: list[complex]) -> dict:
    if not samples:
        return {
            "real_min": 0,
            "real_max": 0,
            "real_mean": 0,
            "imag_min": 0,
            "imag_max": 0,
            "imag_mean": 0,
            "magnitude_min": 0,
            "magnitude_max": 0,
            "magnitude_mean": 0,
        }

    real_values = [value.real for value in samples]
    imag_values = [value.imag for value in samples]
    magnitudes = [abs(value) for value in samples]
    return {
        "real_min": min(real_values),
        "real_max": max(real_values),
        "real_mean": fmean(real_values),
        "imag_min": min(imag_values),
        "imag_max": max(imag_values),
        "imag_mean": fmean(imag_values),
        "magnitude_min": min(magnitudes),
        "magnitude_max": max(magnitudes),
        "magnitude_mean": fmean(magnitudes),
    }


def downsample_complex(values: list[complex], target: int) -> list[complex]:
    if len(values) <= target:
        return list(values)
    if target <= 1:
        return [values[0]]
    step = (len(values) - 1) / (target - 1)
    return [values[round(index * step)] for index in range(target)]
