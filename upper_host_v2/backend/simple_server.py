from __future__ import annotations

import argparse
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import fmean
from typing import Any

from legacy_pipeline import analyze_file, analyze_generated, detect_sample_rate, preview_file

MAX_WAVEFORM_POINTS = 1_200
MAX_CONSTELLATION_POINTS = 2_200
MAX_FFT_POINTS = 2_048
MAX_FFT_FRAMES = 14


class SignalBackendHandler(BaseHTTPRequestHandler):
    server_version = "SignalBackend/2.0"

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path in {"/", "/health"}:
            self.send_json({"ok": True, "message": "Python backend is running"})
            return

        self.send_error_json(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self):  # noqa: N802
        try:
            body = self.read_json_body()
            if self.path == "/signals/inspect":
                self.send_json(inspect_signal_file(body))
                return
            if self.path == "/signals/analyze":
                self.send_json(analyze_signal(body))
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "Endpoint not found")
        except HttpError as exc:
            self.send_error_json(exc.status, exc.message)
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, "Invalid JSON body") from exc
        if not isinstance(data, dict):
            raise HttpError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return data

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_error_json(self, status: HTTPStatus, message: str):
        self.send_json({"detail": message}, status)

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):  # noqa: A002
        print(f"{self.address_string()} - {format % args}")


class HttpError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def inspect_signal_file(request: dict[str, Any]) -> dict[str, Any]:
    path = validate_signal_file(str(request.get("file_path", "")))
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
        "preview_lines": preview_file(path),
    }


def analyze_signal(request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode", "file")
    if mode == "file":
        file_path = request.get("file_path")
        if not file_path:
            raise HttpError(HTTPStatus.BAD_REQUEST, "file_path is required for file mode")
        result = analyze_file(
            file_path=validate_signal_file(str(file_path)),
            fs_hz=optional_positive_float(request.get("fs_hz")),
            fc_hz=required_positive_float(request.get("fc_hz"), "fc_hz"),
            rs_hz=required_positive_float(request.get("rs_hz"), "rs_hz"),
            snr_db=required_float(request.get("snr_db"), "snr_db"),
            modulation=required_int(request.get("modulation"), "modulation"),
        )
    elif mode == "generated":
        fs_hz = optional_positive_float(request.get("fs_hz"))
        if fs_hz is None:
            raise HttpError(HTTPStatus.BAD_REQUEST, "fs_hz is required for generated mode")
        result = analyze_generated(
            fs_hz=fs_hz,
            fc_hz=required_positive_float(request.get("fc_hz"), "fc_hz"),
            rs_hz=required_positive_float(request.get("rs_hz"), "rs_hz"),
            snr_db=required_float(request.get("snr_db"), "snr_db"),
            modulation=required_int(request.get("modulation"), "modulation"),
        )
    else:
        raise HttpError(HTTPStatus.BAD_REQUEST, "mode must be file or generated")

    signal_samples = coerce_complex_list(result.signal_samples)
    constellation_samples = coerce_complex_list(result.constellation_samples)

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
        "waveform": build_waveform(signal_samples),
        "fft": build_fft(signal_samples, result.sample_rate_hz),
        "constellation": build_constellation(constellation_samples),
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
        raise HttpError(HTTPStatus.NOT_FOUND, "Signal file does not exist")
    if not path.is_file():
        raise HttpError(HTTPStatus.BAD_REQUEST, "Selected path is not a file")
    return path


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HttpError(HTTPStatus.BAD_REQUEST, f"Expected a number, got {value!r}") from exc
    return numeric


def required_float(value: Any, field_name: str) -> float:
    numeric = optional_float(value)
    if numeric is None:
        raise HttpError(HTTPStatus.BAD_REQUEST, f"{field_name} is required")
    return numeric


def optional_positive_float(value: Any) -> float | None:
    numeric = optional_float(value)
    if numeric is not None and numeric <= 0:
        raise HttpError(HTTPStatus.BAD_REQUEST, "Numeric values must be greater than 0")
    return numeric


def required_positive_float(value: Any, field_name: str) -> float:
    numeric = optional_positive_float(value)
    if numeric is None:
        raise HttpError(HTTPStatus.BAD_REQUEST, f"{field_name} is required")
    return numeric


def required_int(value: Any, field_name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise HttpError(HTTPStatus.BAD_REQUEST, f"{field_name} must be an integer") from exc
    return numeric


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


def build_waveform(samples: list[complex]) -> list[dict[str, float]]:
    points = downsample_complex(samples, MAX_WAVEFORM_POINTS)
    return [
        {
            "index": float(index),
            "real": value.real,
            "imag": value.imag,
            "magnitude": abs(value),
        }
        for index, value in enumerate(points)
    ]


def build_constellation(samples: list[complex]) -> list[dict[str, float]]:
    points = downsample_complex(samples, MAX_CONSTELLATION_POINTS)
    return [{"real": value.real, "imag": value.imag} for value in points]


def build_fft(samples: list[complex], fs_hz: float) -> dict[str, Any]:
    frames = build_fft_frames(samples, fs_hz)
    if not frames:
        return {"frequency_hz": [], "magnitude": [], "frames": []}

    return {
        "frequency_hz": frames[0]["frequency_hz"],
        "magnitude": frames[0]["magnitude"],
        "frames": frames,
    }


def build_fft_frames(samples: list[complex], fs_hz: float) -> list[dict[str, list[float]]]:
    if len(samples) < 2:
        return []

    chunk_size = min(MAX_FFT_POINTS, len(samples))
    if chunk_size < 2:
        return []

    if len(samples) <= chunk_size:
        starts = [0]
    else:
        max_start = len(samples) - chunk_size
        if MAX_FFT_FRAMES <= 1:
            starts = [0]
        else:
            starts = [round(index * max_start / (MAX_FFT_FRAMES - 1)) for index in range(MAX_FFT_FRAMES)]

    frames: list[dict[str, list[float]]] = []
    for start in starts:
        frame = compute_positive_fft(samples[start : start + chunk_size], fs_hz)
        if frame["frequency_hz"]:
            frames.append(frame)
    return frames


def compute_positive_fft(values: list[complex], fs_hz: float) -> dict[str, list[float]]:
    if len(values) < 2:
        return {"frequency_hz": [], "magnitude": []}

    try:
        import numpy as np  # type: ignore

        array = np.asarray(values, dtype=np.complex128)
        spectrum = np.fft.fftshift(np.fft.fft(array))
        frequencies = np.fft.fftshift(np.fft.fftfreq(array.size, d=1.0 / fs_hz))
        positive_mask = frequencies >= 0
        magnitudes = 2.0 / array.size * np.abs(spectrum)
        return {
            "frequency_hz": frequencies[positive_mask].tolist(),
            "magnitude": magnitudes[positive_mask].tolist(),
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
        positive_pairs = [(frequency, value) for frequency, value in zip(freqs, shifted) if frequency >= 0]
        return {
            "frequency_hz": [frequency for frequency, _ in positive_pairs],
            "magnitude": [2.0 / size * abs(value) for _, value in positive_pairs],
        }


def build_stats(samples: list[complex]) -> dict[str, float]:
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


def main():
    parser = argparse.ArgumentParser(description="Run the local signal-analysis backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SignalBackendHandler)
    print(f"Python backend is running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
