from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

from legacy_pipeline import analyze_file


SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xlsx", ".xls"}
MODULATION_NAMES = {
    1: "QPSK",
    2: "8PSK",
    3: "16QAM",
    4: "64QAM",
}


@dataclass
class SignalParams:
    fs_hz: float | None
    fc_hz: float
    rs_hz: float
    snr_db: float
    modulation: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch signal analysis on one file or a whole dataset folder, then export per-file "
            "errors plus dataset-level statistics."
        ),
        epilog=(
            "Simple example:\n"
            "  conda run -n million python batch_evaluate.py D:\\dataset "
            "--fc-hz 3.25e9 --rs-hz 1.75e9 --snr-db -10 --modulation 1\n\n"
            "Manifest example (optional CSV columns): file, fs_hz, fc_hz, rs_hz, snr_db, modulation"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        help="Path to one signal file or a directory containing many signal files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for exported CSV/JSON files. Defaults to backend/batch_reports/<timestamp>.",
    )
    parser.add_argument(
        "--manifest",
        help=(
            "Optional CSV manifest for per-file truth values. Supported columns: "
            "file, fs_hz, fc_hz, rs_hz, snr_db, modulation."
        ),
    )
    parser.add_argument(
        "--fs-hz",
        type=float,
        default=None,
        help="Sampling rate in Hz. Leave empty to auto-detect from CSV/XLSX/XLS when possible.",
    )
    parser.add_argument("--fc-hz", type=float, default=3.25e9, help="Reference carrier center frequency in Hz.")
    parser.add_argument("--rs-hz", type=float, default=1.75e9, help="Reference symbol rate in Hz.")
    parser.add_argument("--snr-db", type=float, default=-10.0, help="Reference carrier-to-noise ratio in dB.")
    parser.add_argument(
        "--modulation",
        type=int,
        choices=sorted(MODULATION_NAMES),
        default=1,
        help="Modulation type: 1=QPSK, 2=8PSK, 3=16QAM, 4=64QAM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only analyze the first N files after sorting. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subfolders when input_path is a directory.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one file fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return 1

    default_params = SignalParams(
        fs_hz=args.fs_hz,
        fc_hz=args.fc_hz,
        rs_hz=args.rs_hz,
        snr_db=args.snr_db,
        modulation=args.modulation,
    )

    manifest_rows = load_manifest(Path(args.manifest).expanduser().resolve()) if args.manifest else {}
    signal_files = collect_signal_files(input_path, recursive=not args.no_recursive)
    if args.limit is not None:
        signal_files = signal_files[: max(args.limit, 0)]

    if not signal_files:
        print(f"No supported signal files were found under: {input_path}")
        return 1

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(signal_files)} signal file(s).")
    print(f"Results will be written to: {output_dir}")

    records: list[dict[str, Any]] = []
    for index, signal_file in enumerate(signal_files, start=1):
        try:
            params = resolve_signal_params(
                signal_file=signal_file,
                input_root=input_path if input_path.is_dir() else input_path.parent,
                defaults=default_params,
                manifest_rows=manifest_rows,
            )
            result = analyze_file(
                file_path=signal_file,
                fs_hz=params.fs_hz,
                fc_hz=params.fc_hz,
                rs_hz=params.rs_hz,
                snr_db=params.snr_db,
                modulation=params.modulation,
            )
            record = build_success_record(signal_file, params, result)
            records.append(record)
            print(
                f"[{index}/{len(signal_files)}] OK    {signal_file.name}  "
                f"cf={format_number(record['center_frequency_error_percent'])}%  "
                f"bw={format_number(record['bandwidth_error_percent'])}%  "
                f"snr={format_number(record['snr_error_db'])} dB"
            )
        except Exception as exc:
            record = build_failure_record(signal_file, exc)
            records.append(record)
            print(f"[{index}/{len(signal_files)}] FAIL  {signal_file.name}  {exc}")
            if args.stop_on_error:
                break

    csv_path = output_dir / "per_file_results.csv"
    json_path = output_dir / "summary.json"
    txt_path = output_dir / "summary.txt"

    write_csv(csv_path, records)
    summary = build_summary(records, input_path, output_dir, default_params, manifest_rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(build_summary_text(summary), encoding="utf-8")

    print("")
    print(f"Per-file results: {csv_path}")
    print(f"Summary JSON:     {json_path}")
    print(f"Summary TXT:      {txt_path}")
    print_recommended_thresholds(summary)

    return 0 if summary["success_count"] > 0 else 1


def resolve_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        return Path(raw_output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "batch_reports" / timestamp


def collect_signal_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    if recursive:
        candidates = input_path.rglob("*")
    else:
        candidates = input_path.glob("*")

    files = [path.resolve() for path in candidates if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files, key=lambda path: str(path).lower())


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    rows_by_key: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Manifest CSV is missing a header row")
        if "file" not in reader.fieldnames:
            raise ValueError("Manifest CSV must include a 'file' column")

        for row in reader:
            file_value = (row.get("file") or "").strip()
            if not file_value:
                continue
            rows_by_key[normalize_key(file_value)] = row
    return rows_by_key


def resolve_signal_params(
    *,
    signal_file: Path,
    input_root: Path,
    defaults: SignalParams,
    manifest_rows: dict[str, dict[str, str]],
) -> SignalParams:
    row = lookup_manifest_row(signal_file, input_root, manifest_rows)
    if not row:
        return defaults

    return SignalParams(
        fs_hz=parse_optional_float(row.get("fs_hz"), defaults.fs_hz, "fs_hz", signal_file),
        fc_hz=parse_required_float(row.get("fc_hz"), defaults.fc_hz, "fc_hz", signal_file),
        rs_hz=parse_required_float(row.get("rs_hz"), defaults.rs_hz, "rs_hz", signal_file),
        snr_db=parse_required_float(row.get("snr_db"), defaults.snr_db, "snr_db", signal_file),
        modulation=parse_required_int(row.get("modulation"), defaults.modulation, "modulation", signal_file),
    )


def lookup_manifest_row(
    signal_file: Path,
    input_root: Path,
    manifest_rows: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    if not manifest_rows:
        return None

    candidates = [normalize_key(signal_file.name), normalize_key(str(signal_file))]
    try:
        candidates.append(normalize_key(str(signal_file.relative_to(input_root))))
    except ValueError:
        pass

    for key in candidates:
        row = manifest_rows.get(key)
        if row:
            return row
    return None


def build_success_record(signal_file: Path, params: SignalParams, result) -> dict[str, Any]:
    estimates = result.estimates
    errors = result.errors
    center_frequency_error_hz = abs(float(estimates["center_frequency_hz"]) - float(params.fc_hz))
    bandwidth_error_hz = abs(float(estimates["bandwidth_hz"]) - float(estimates["bandwidth_true_hz"]))
    snr_error_db = abs(float(estimates["snr_db"]) - float(params.snr_db))

    return {
        "status": "ok",
        "file_name": signal_file.name,
        "file_path": str(signal_file),
        "message": "",
        "detected_sample_rate_hz": result.detected_sample_rate_hz,
        "sample_rate_hz": result.sample_rate_hz,
        "reference_fs_hz": params.fs_hz,
        "reference_fc_hz": params.fc_hz,
        "reference_rs_hz": params.rs_hz,
        "reference_snr_db": params.snr_db,
        "reference_modulation": params.modulation,
        "reference_modulation_name": MODULATION_NAMES.get(params.modulation, str(params.modulation)),
        "estimated_center_frequency_hz": estimates["center_frequency_hz"],
        "estimated_bandwidth_hz": estimates["bandwidth_hz"],
        "estimated_symbol_rate_hz": estimates["rs_hz"],
        "estimated_snr_db": estimates["snr_db"],
        "estimated_evm_percent": estimates["evm_percent"],
        "estimated_papr": estimates["papr"],
        "bandwidth_true_hz": estimates["bandwidth_true_hz"],
        "center_frequency_error_hz": center_frequency_error_hz,
        "center_frequency_error_percent": errors["center_frequency_percent"],
        "bandwidth_error_hz": bandwidth_error_hz,
        "bandwidth_error_percent": errors["bandwidth_percent"],
        "snr_error_db": snr_error_db,
    }


def build_failure_record(signal_file: Path, exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "file_name": signal_file.name,
        "file_path": str(signal_file),
        "message": str(exc),
        "detected_sample_rate_hz": None,
        "sample_rate_hz": None,
        "reference_fs_hz": None,
        "reference_fc_hz": None,
        "reference_rs_hz": None,
        "reference_snr_db": None,
        "reference_modulation": None,
        "reference_modulation_name": None,
        "estimated_center_frequency_hz": None,
        "estimated_bandwidth_hz": None,
        "estimated_symbol_rate_hz": None,
        "estimated_snr_db": None,
        "estimated_evm_percent": None,
        "estimated_papr": None,
        "bandwidth_true_hz": None,
        "center_frequency_error_hz": None,
        "center_frequency_error_percent": None,
        "bandwidth_error_hz": None,
        "bandwidth_error_percent": None,
        "snr_error_db": None,
    }


def write_csv(csv_path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "file_name",
        "file_path",
        "message",
        "detected_sample_rate_hz",
        "sample_rate_hz",
        "reference_fs_hz",
        "reference_fc_hz",
        "reference_rs_hz",
        "reference_snr_db",
        "reference_modulation",
        "reference_modulation_name",
        "estimated_center_frequency_hz",
        "estimated_bandwidth_hz",
        "estimated_symbol_rate_hz",
        "estimated_snr_db",
        "estimated_evm_percent",
        "estimated_papr",
        "bandwidth_true_hz",
        "center_frequency_error_hz",
        "center_frequency_error_percent",
        "bandwidth_error_hz",
        "bandwidth_error_percent",
        "snr_error_db",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def build_summary(
    records: list[dict[str, Any]],
    input_path: Path,
    output_dir: Path,
    defaults: SignalParams,
    manifest_rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    successes = [record for record in records if record["status"] == "ok"]
    failures = [record for record in records if record["status"] != "ok"]

    center_frequency_errors = collect_metric(successes, "center_frequency_error_percent")
    bandwidth_errors = collect_metric(successes, "bandwidth_error_percent")
    snr_errors = collect_metric(successes, "snr_error_db")

    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_files": len(records),
        "success_count": len(successes),
        "failure_count": len(failures),
        "used_manifest": bool(manifest_rows),
        "default_parameters": {
            "fs_hz": defaults.fs_hz,
            "fc_hz": defaults.fc_hz,
            "rs_hz": defaults.rs_hz,
            "snr_db": defaults.snr_db,
            "modulation": defaults.modulation,
            "modulation_name": MODULATION_NAMES.get(defaults.modulation, str(defaults.modulation)),
        },
        "metrics": {
            "center_frequency_error_percent": summarize_metric(center_frequency_errors, unit="% of symbol rate"),
            "bandwidth_error_percent": summarize_metric(bandwidth_errors, unit="% of symbol rate"),
            "snr_error_db": summarize_metric(snr_errors, unit="dB"),
        },
        "recommended_thresholds": {
            "p90": {
                "center_frequency_percent": percentile(center_frequency_errors, 0.90),
                "bandwidth_percent": percentile(bandwidth_errors, 0.90),
                "snr_db": percentile(snr_errors, 0.90),
            },
            "p95": {
                "center_frequency_percent": percentile(center_frequency_errors, 0.95),
                "bandwidth_percent": percentile(bandwidth_errors, 0.95),
                "snr_db": percentile(snr_errors, 0.95),
            },
            "p99": {
                "center_frequency_percent": percentile(center_frequency_errors, 0.99),
                "bandwidth_percent": percentile(bandwidth_errors, 0.99),
                "snr_db": percentile(snr_errors, 0.99),
            },
            "max": {
                "center_frequency_percent": max(center_frequency_errors) if center_frequency_errors else None,
                "bandwidth_percent": max(bandwidth_errors) if bandwidth_errors else None,
                "snr_db": max(snr_errors) if snr_errors else None,
            },
        },
        "failures": [
            {
                "file_name": record["file_name"],
                "file_path": record["file_path"],
                "message": record["message"],
            }
            for record in failures
        ],
    }


def build_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Batch Evaluation Summary",
        "========================",
        f"Input path:      {summary['input_path']}",
        f"Output dir:      {summary['output_dir']}",
        f"Generated at:    {summary['generated_at']}",
        f"Total files:     {summary['total_files']}",
        f"Success count:   {summary['success_count']}",
        f"Failure count:   {summary['failure_count']}",
        f"Used manifest:   {summary['used_manifest']}",
        "",
        "Recommended thresholds",
        "----------------------",
    ]

    for label in ("p90", "p95", "p99", "max"):
        section = summary["recommended_thresholds"][label]
        lines.append(
            f"{label}: center_frequency={format_number(section['center_frequency_percent'])}%  "
            f"bandwidth={format_number(section['bandwidth_percent'])}%  "
            f"snr={format_number(section['snr_db'])} dB"
        )

    lines.extend(
        [
            "",
            "Metric details",
            "--------------",
        ]
    )

    for metric_name, metric_summary in summary["metrics"].items():
        lines.append(f"{metric_name} ({metric_summary['unit']}):")
        lines.append(
            f"  count={metric_summary['count']}  mean={format_number(metric_summary['mean'])}  "
            f"median={format_number(metric_summary['median'])}  p95={format_number(metric_summary['p95'])}  "
            f"max={format_number(metric_summary['max'])}"
        )

    if summary["failures"]:
        lines.extend(["", "Failures", "--------"])
        for failure in summary["failures"]:
            lines.append(f"{failure['file_name']}: {failure['message']}")

    lines.append("")
    return "\n".join(lines)


def print_recommended_thresholds(summary: dict[str, Any]) -> None:
    thresholds = summary["recommended_thresholds"]
    if summary["success_count"] == 0:
        print("No successful files were analyzed, so no thresholds can be recommended.")
        return

    print("")
    print("Suggested thresholds from this dataset:")
    for label in ("p90", "p95", "p99"):
        section = thresholds[label]
        print(
            f"  {label}: center_frequency={format_number(section['center_frequency_percent'])}%  "
            f"bandwidth={format_number(section['bandwidth_percent'])}%  "
            f"snr={format_number(section['snr_db'])} dB"
        )


def collect_metric(records: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field_name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def summarize_metric(values: list[float], *, unit: str) -> dict[str, Any]:
    return {
        "unit": unit,
        "count": len(values),
        "min": min(values) if values else None,
        "mean": fmean(values) if values else None,
        "median": median(values) if values else None,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    if ratio <= 0:
        return min(values)
    if ratio >= 1:
        return max(values)

    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value

    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def parse_optional_float(raw_value: Any, default: float | None, field_name: str, signal_file: Path) -> float | None:
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} for {signal_file.name}: {raw_value}") from exc


def parse_required_float(raw_value: Any, default: float, field_name: str, signal_file: Path) -> float:
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} for {signal_file.name}: {raw_value}") from exc


def parse_required_int(raw_value: Any, default: int, field_name: str, signal_file: Path) -> int:
    if raw_value in (None, ""):
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} for {signal_file.name}: {raw_value}") from exc
    if value not in MODULATION_NAMES:
        raise ValueError(f"Unsupported {field_name} for {signal_file.name}: {value}")
    return value


def normalize_key(value: str) -> str:
    return str(Path(value)).replace("\\", "/").lower()


def format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
