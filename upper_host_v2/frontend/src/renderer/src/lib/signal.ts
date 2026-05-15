export const API_BASES = ['http://127.0.0.1:8000', 'http://localhost:8000']

export const UNIT_FACTORS = {
  Hz: 1,
  kHz: 1e3,
  MHz: 1e6,
  GHz: 1e9
} as const

export const MODULATION_OPTIONS = [
  { value: 1, label: 'QPSK' },
  { value: 2, label: '8PSK' },
  { value: 3, label: '16QAM' },
  { value: 4, label: '64QAM' }
] as const

export const QUALITY_LABELS = ['载噪比误差', '载波中心频率误差', '载波-3dB带宽误差', 'EVM']

export const QUALITY_COLORS = ['#f97316', '#38bdf8', '#4ade80', '#f472b6']

export type FrequencyUnit = keyof typeof UNIT_FACTORS

export type WorkflowMode = 'file' | 'generated'

export type BackendState = {
  ok: boolean
  message: string
}

export type SelectedSignalFile = {
  path: string
  name: string
  size: number
}

export type InspectResult = {
  ok: boolean
  name: string
  path: string
  size_bytes: number
  size_mb: number
  detected_format: string
  detected_sample_rate_hz: number | null
  preview_lines: string[]
}

export type WaveformPoint = {
  index: number
  real: number
  imag: number
  magnitude: number
}

export type SpectrumData = {
  frequency_hz: number[]
  magnitude: number[]
  frames?: Array<{
    frequency_hz: number[]
    magnitude: number[]
  }>
}

export type ConstellationPoint = {
  real: number
  imag: number
}

export type StftData = {
  frequency_hz: number[]
  time_s: number[]
  matrix: number[][]
}

export type StatsResult = {
  real_min: number
  real_max: number
  real_mean: number
  imag_min: number
  imag_max: number
  imag_mean: number
  magnitude_min: number
  magnitude_max: number
  magnitude_mean: number
}

export type EstimateResult = {
  center_frequency_hz: number
  bandwidth_hz: number
  rs_hz: number
  snr_db: number
  evm_percent: number
  papr: number
}

export type QualityScores = {
  snr: number
  center_frequency: number
  bandwidth: number
  evm: number
}

export type ErrorMetrics = {
  center_frequency_percent: number | null
  bandwidth_percent: number | null
  snr_db: number | null
}

export type AnalysisResult = {
  ok: boolean
  mode: WorkflowMode
  name: string
  path: string | null
  detected_sample_rate_hz: number | null
  sample_rate_hz: number
  input: {
    fs_hz: number | null
    fc_hz: number
    rs_hz: number
    snr_db: number
    modulation: number
    modulation_name: string
  }
  signal: {
    total_points: number
    truncated: boolean
  }
  preview_lines: string[]
  stats: StatsResult
  estimates: EstimateResult
  display_values: {
    center_frequency: string
    symbol_rate: string
    bandwidth: string
    snr: string
    magnitude: string
    evm: string
    papr: string
  }
  bandwidth_true_hz: number
  quality: QualityScores
  errors: ErrorMetrics
  waveform: WaveformPoint[]
  fft: SpectrumData
  constellation: ConstellationPoint[]
  stft: StftData
  images: {
    stft_source: string | null
    stft_annotated: string | null
    stft_display: string | null
    mask: string | null
    grad_cam: string | null
  }
}

export type SignalFormState = {
  fsValue: string
  fsUnit: FrequencyUnit
  fcValue: string
  fcUnit: FrequencyUnit
  rsValue: string
  rsUnit: FrequencyUnit
  snrValue: string
  modulation: number
}

export const DEFAULT_SIGNAL_FORM: SignalFormState = {
  fsValue: '10',
  fsUnit: 'GHz',
  fcValue: '3.25',
  fcUnit: 'GHz',
  rsValue: '1.75',
  rsUnit: 'GHz',
  snrValue: '-10',
  modulation: 1
}

export function toHertz(value: string, unit: FrequencyUnit): number {
  const numeric = Number.parseFloat(value)
  if (!Number.isFinite(numeric)) {
    return 0
  }
  return numeric * UNIT_FACTORS[unit]
}

export function parseSNR(value: string): number {
  const numeric = Number.parseFloat(value)
  return Number.isFinite(numeric) ? numeric : 0
}

export function fromHertz(value: number): { value: string; unit: FrequencyUnit } {
  if (!Number.isFinite(value) || value === 0) {
    return { value: '0', unit: 'Hz' }
  }

  const units: FrequencyUnit[] = ['GHz', 'MHz', 'kHz', 'Hz']
  for (const unit of units) {
    const scaled = value / UNIT_FACTORS[unit]
    if (Math.abs(scaled) >= 1) {
      return {
        value: trimNumber(scaled),
        unit
      }
    }
  }

  return { value: trimNumber(value), unit: 'Hz' }
}

export function trimNumber(value: number, digits = 4): string {
  if (!Number.isFinite(value)) {
    return '-'
  }
  if (value === 0) {
    return '0'
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(2)
  }
  return value.toFixed(digits).replace(/\.?0+$/, '')
}

export function formatFrequency(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? Number.NaN)) {
    return '-'
  }
  const normalized = fromHertz(value ?? 0)
  return `${normalized.value} ${normalized.unit}`
}

export function formatMetric(value: number | null | undefined, suffix = ''): string {
  if (!Number.isFinite(value ?? Number.NaN)) {
    return '-'
  }
  return `${trimNumber(value ?? 0)}${suffix}`
}

export function buildRequestPayload(mode: WorkflowMode, form: SignalFormState, filePath?: string | null) {
  return {
    mode,
    file_path: mode === 'file' ? filePath ?? null : null,
    fs_hz: toHertz(form.fsValue, form.fsUnit),
    fc_hz: toHertz(form.fcValue, form.fcUnit),
    rs_hz: toHertz(form.rsValue, form.rsUnit),
    snr_db: parseSNR(form.snrValue),
    modulation: form.modulation
  }
}

export function buildQualityScores(result: AnalysisResult, form: SignalFormState): QualityScores {
  const fcInput = toHertz(form.fcValue, form.fcUnit)
  const rsInput = toHertz(form.rsValue, form.rsUnit)
  const snrInput = parseSNR(form.snrValue)
  const evmPercent = result.estimates.evm_percent

  const centerFrequency =
    rsInput > 0
      ? clamp(25 - ((Math.abs(result.estimates.center_frequency_hz - fcInput) / rsInput) * 20 - 1), 0, 25)
      : 0

  const bandwidth =
    rsInput > 0
      ? clamp(25 - ((Math.abs(result.estimates.bandwidth_hz - result.bandwidth_true_hz) / rsInput) * 10 - 1), 0, 25)
      : 0

  const snr = clamp(25 - Math.abs(result.estimates.snr_db - snrInput) * 2.5, 0, 25)

  let evm = 15
  if (form.modulation === 1) {
    evm = 25 - ((evmPercent / 17.5) * 10 - 1) * 5
  } else if (form.modulation === 2) {
    evm = 25 - ((evmPercent / 12) * 10 - 1) * 5
  } else if (form.modulation === 3) {
    evm = 25 - ((evmPercent / 12.5) * 10 - 1) * 5
  } else if (form.modulation === 4) {
    evm = 25 - ((evmPercent / 8) * 10 - 1) * 5
  }

  return {
    snr,
    center_frequency: centerFrequency,
    bandwidth,
    evm: clamp(evm, 0, 25)
  }
}

export function buildErrorMetrics(result: AnalysisResult, form: SignalFormState): ErrorMetrics {
  const fcInput = toHertz(form.fcValue, form.fcUnit)
  const rsInput = toHertz(form.rsValue, form.rsUnit)
  const snrInput = parseSNR(form.snrValue)

  return {
    center_frequency_percent:
      rsInput > 0 ? Math.abs(((result.estimates.center_frequency_hz - fcInput) / rsInput) * 100) : null,
    bandwidth_percent:
      rsInput > 0
        ? Math.abs(((result.estimates.bandwidth_hz - result.bandwidth_true_hz) / rsInput) * 100)
        : null,
    snr_db: Math.abs(result.estimates.snr_db - snrInput)
  }
}

export function qualityToArray(quality: QualityScores): number[] {
  return [quality.snr, quality.center_frequency, quality.bandwidth, quality.evm]
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}
