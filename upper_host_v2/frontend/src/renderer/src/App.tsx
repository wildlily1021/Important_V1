import { useEffect, useMemo, useState } from 'react'

import { LineChart, RadarChart, ScatterChart } from './components/Plots'
import {
  API_BASES,
  DEFAULT_SIGNAL_FORM,
  MODULATION_OPTIONS,
  type AnalysisResult,
  type BackendState,
  type ErrorMetrics,
  type InspectResult,
  type QualityScores,
  type SelectedSignalFile,
  type SignalFormState,
  type WorkflowMode,
  buildErrorMetrics,
  buildQualityScores,
  buildRequestPayload,
  formatFrequency,
  formatMetric,
  fromHertz
} from './lib/signal'

const BACKEND_LAUNCH_HINT =
  'conda run -n million python simple_server.py --host 127.0.0.1 --port 8000'

function App(): React.JSX.Element {
  const [apiBase, setApiBase] = useState(API_BASES[0])
  const [backend, setBackend] = useState<BackendState>({
    ok: false,
    message: '正在检查 Python 后端...'
  })
  const [form, setForm] = useState<SignalFormState>(DEFAULT_SIGNAL_FORM)
  const [selectedFile, setSelectedFile] = useState<SelectedSignalFile | null>(null)
  const [inspectResult, setInspectResult] = useState<InspectResult | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [qualitySeries, setQualitySeries] = useState<Array<{ label: string; values: QualityScores }>>([])
  const [currentErrors, setCurrentErrors] = useState<ErrorMetrics | null>(null)
  const [isWorking, setIsWorking] = useState(false)
  const [assessmentCount, setAssessmentCount] = useState(0)

  const spectrumPoints = useMemo(() => {
    if (!analysisResult) {
      return []
    }

    return toSpectrumPoints(analysisResult.fft.frequency_hz, analysisResult.fft.magnitude)
  }, [analysisResult])

  const spectrumFrames = useMemo(() => {
    if (!analysisResult) {
      return []
    }

    const frames = analysisResult.fft.frames?.length ? analysisResult.fft.frames : [analysisResult.fft]
    return frames
      .map((frame) => toSpectrumPoints(frame.frequency_hz, frame.magnitude))
      .filter((frame) => frame.length >= 2)
  }, [analysisResult])

  function toSpectrumPoints(frequencies: number[], magnitudes: number[]): Array<{ x: number; y: number }> {
    return frequencies
      .map((frequency, index) => ({
        x: frequency,
        y: Math.max(magnitudes[index] ?? 0, 0)
      }))
      .filter((point) => point.x >= 0 && Number.isFinite(point.x) && Number.isFinite(point.y))
  }

  const resultRows = useMemo(() => {
    if (!analysisResult) {
      return []
    }

    return [
      { label: '信号EVM', unit: '%', value: analysisResult.display_values.evm },
      { label: '信号PAPR', unit: '', value: analysisResult.display_values.papr },
      { label: '载波符号速率', unit: '(Hz)', value: analysisResult.display_values.symbol_rate },
      { label: '载波-3dB带宽', unit: '(Hz)', value: analysisResult.display_values.bandwidth },
      { label: '载噪比', unit: '(dB)', value: analysisResult.display_values.snr },
      { label: '载波中心频率', unit: '(Hz)', value: analysisResult.display_values.center_frequency }
    ]
  }, [analysisResult])

  const previewLines = inspectResult?.preview_lines ?? analysisResult?.preview_lines ?? []

  const galleryItems = useMemo(
    () => [
      {
        key: 'grad_cam',
        title: 'Grad-CAM',
        src: analysisResult?.images.grad_cam ?? null
      },
      {
        key: 'mask',
        title: '掩膜结果',
        src: analysisResult?.images.mask ?? null
      },
      {
        key: 'stft_source',
        title: '原始 STFT',
        src: analysisResult?.images.stft_source ?? null
      },
      {
        key: 'stft_annotated',
        title: '标注 STFT',
        src: analysisResult?.images.stft_annotated ?? null
      }
    ],
    [analysisResult]
  )

  useEffect(() => {
    void checkBackend()
  }, [])

  async function checkBackend(): Promise<void> {
    setBackend({
      ok: false,
      message: '正在检查 Python 后端...'
    })

    const failures: string[] = []

    for (const base of API_BASES) {
      try {
        const response = await fetch(`${base}/health`)
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = (await response.json()) as BackendState
        setApiBase(base)
        setBackend({
          ok: data.ok,
          message: `Python 后端已连接：${base}`
        })
        return
      } catch (error) {
        failures.push(`${base}: ${normalizeErrorMessage(error)}`)
      }
    }

    setBackend({
      ok: false,
      message: `后端未连接，请先启动 Python 服务。${failures.join(' | ')}`
    })
  }

  async function chooseSignalFile(): Promise<void> {
    const file = await window.api.selectSignalFile()
    if (!file) {
      return
    }

    setSelectedFile(file)
    setInspectResult(null)
    setAnalysisResult(null)
    setCurrentErrors(null)
    setQualitySeries([])
    setAssessmentCount(0)
    setBackend((current) => ({
      ...current,
      message: `已选择文件：${file.name}`
    }))

    try {
      const inspect = await postJson<InspectResult>(`${apiBase}/signals/inspect`, {
        file_path: file.path
      })
      setInspectResult(inspect)

      if (inspect.detected_sample_rate_hz) {
        const nextFs = fromHertz(inspect.detected_sample_rate_hz)
        setForm((current) => ({
          ...current,
          fsValue: nextFs.value,
          fsUnit: nextFs.unit
        }))
      }
    } catch (error) {
      setInspectResult(null)
      setBackend({
        ok: !isConnectivityError(error),
        message: `文件读取失败：${normalizeErrorMessage(error)}`
      })
    }
  }

  function clearSelectedFile(): void {
    setSelectedFile(null)
    setInspectResult(null)
    setAnalysisResult(null)
    setCurrentErrors(null)
    setQualitySeries([])
    setAssessmentCount(0)
    setBackend((current) => ({
      ...current,
      message: '已切换到仿真信号模式，可直接点击“信号测试”使用当前参数生成并分析。'
    }))
  }

  async function runAnalysis(targetMode: WorkflowMode = selectedFile ? 'file' : 'generated'): Promise<void> {
    if (!backend.ok) {
      setBackend({
        ok: false,
        message: `后端未连接。请在 upper_host_v2/backend 目录运行：${BACKEND_LAUNCH_HINT}`
      })
      return
    }

    if (targetMode === 'file' && !selectedFile) {
      setBackend((current) => ({
        ...current,
        ok: false,
        message: '请先选择要分析的数据文件，或者清空文件后使用当前参数生成信号。'
      }))
      return
    }

    setIsWorking(true)
    setAnalysisResult(null)
    setCurrentErrors(null)
    setQualitySeries([])
    setAssessmentCount(0)
    setBackend((current) => ({
      ...current,
      message:
        targetMode === 'file'
          ? '正在调用旧版 PyQt 分析链路处理文件，请稍候...'
          : '正在按当前参数生成信号，并调用旧版 PyQt 分析链路，请稍候...'
    }))

    try {
      const result = await postJson<AnalysisResult>(
        `${apiBase}/signals/analyze`,
        buildRequestPayload(targetMode, form, selectedFile?.path)
      )

      setAnalysisResult(result)

      if (result.detected_sample_rate_hz) {
        const nextFs = fromHertz(result.detected_sample_rate_hz)
        setForm((current) => ({
          ...current,
          fsValue: nextFs.value,
          fsUnit: nextFs.unit
        }))
      }

      if (targetMode === 'generated') {
        setInspectResult({
          ok: true,
          name: result.name,
          path: '',
          size_bytes: 0,
          size_mb: 0,
          detected_format: 'generated',
          detected_sample_rate_hz: result.sample_rate_hz,
          preview_lines: result.preview_lines
        })
      }

      setBackend({
        ok: true,
        message: `分析完成：${result.name}，共 ${result.signal.total_points.toLocaleString()} 个采样点。`
      })
    } catch (error) {
      setBackend({
        ok: !isConnectivityError(error),
        message: `信号分析失败：${normalizeErrorMessage(error)}`
      })
    } finally {
      setIsWorking(false)
    }
  }

  async function runSignalTest(): Promise<void> {
    const targetMode: WorkflowMode = selectedFile ? 'file' : 'generated'
    await runAnalysis(targetMode)
  }

  function confirmInputs(): void {
    setBackend((current) => ({
      ...current,
      ok: current.ok,
      message: `参数已确认：Fs ${form.fsValue}${form.fsUnit}，Fc ${form.fcValue}${form.fcUnit}，Rs ${form.rsValue}${form.rsUnit}。`
    }))
  }

  function startAssessment(): void {
    if (!analysisResult) {
      setBackend((current) => ({
        ...current,
        ok: false,
        message: '请先完成一次信号分析，再点击“开始评估”。'
      }))
      return
    }

    const nextCount = assessmentCount + 1
    const quality = buildQualityScores(analysisResult, form)
    const label = nextCount === 1 ? '当前信号' : `对比 ${nextCount}`
    const nextSeries = [...qualitySeries, { label, values: quality }].slice(-4)

    setQualitySeries(nextSeries)
    setAssessmentCount(nextCount)
    setBackend((current) => ({
      ...current,
      ok: true,
      message: `质量评估已更新，当前显示 ${nextSeries.length} 组结果。`
    }))
  }

  function calculateErrors(): void {
    if (!analysisResult) {
      setBackend((current) => ({
        ...current,
        ok: false,
        message: '请先完成一次信号分析，再点击“计算误差”。'
      }))
      return
    }

    const errors = buildErrorMetrics(analysisResult, form)
    setCurrentErrors(errors)
    setBackend((current) => ({
      ...current,
      ok: true,
      message: '误差计算完成。'
    }))
  }

  function updateForm<K extends keyof SignalFormState>(key: K, value: SignalFormState[K]): void {
    setForm((current) => ({
      ...current,
      [key]: value
    }))
  }

  return (
    <main className="legacy-shell">
      <header className="legacy-header">
        <div className="legacy-header__title">
          <h1>新一代移动通信信号测试技术软件</h1>
          <p>Electron 前端 + Python 旧版算法链路</p>
        </div>
        <button className="legacy-status" type="button" onClick={() => void checkBackend()}>
          <span className={`legacy-status__dot ${backend.ok ? 'online' : 'offline'}`} />
          <span>{backend.message}</span>
        </button>
      </header>

      <section className="legacy-grid">
        <div className="legacy-left-stack">
          <LegacyPanel title="测试配置" accent="red">
            <div className="legacy-button-stack">
              <button className="legacy-big-button" type="button" onClick={() => void chooseSignalFile()}>
                选择文件
              </button>
              <button className="legacy-big-button" type="button" disabled={isWorking} onClick={() => void runSignalTest()}>
                {isWorking ? '分析中...' : '信号测试'}
              </button>
            </div>

            <div className="legacy-file-box">
              <div className="legacy-file-box__label">当前数据源</div>
              <div className="legacy-file-box__value">
                {selectedFile?.name ?? '未选择文件，将使用右下角参数生成仿真信号。'}
              </div>
              <div className="legacy-file-box__hint">
                {inspectResult?.detected_sample_rate_hz
                  ? `自动识别采样率：${formatFrequency(inspectResult.detected_sample_rate_hz)}`
                  : '若文件中未识别到采样率，请在右下角手动填写 Fs。'}
              </div>
            </div>

            {!backend.ok && (
              <div className="legacy-warning-box">
                <strong>当前后端未连接</strong>
                <span>请在 `upper_host_v2/backend` 目录使用原来的 `million` 环境启动：</span>
                <code>{BACKEND_LAUNCH_HINT}</code>
              </div>
            )}

            <div className="legacy-preview-box">
              <div className="legacy-preview-box__title">文件预览</div>
              {previewLines.length ? (
                previewLines.slice(0, 4).map((line, index) => (
                  <div key={`${index}-${line}`} className="legacy-preview-line">
                    {line}
                  </div>
                ))
              ) : (
                <div className="legacy-preview-line legacy-preview-line--muted">选择文件后在这里显示前几行预览。</div>
              )}
            </div>

            <button className="legacy-text-action" type="button" onClick={clearSelectedFile} disabled={!selectedFile}>
              清空文件并切换到仿真模式
            </button>
          </LegacyPanel>
        </div>

        <LegacyPanel title="分析图像结果" accent="green" className="legacy-analysis-panel">
          <div className="legacy-analysis-grid legacy-analysis-grid--panel">
            {galleryItems.map((item) => (
              <ImageCard key={item.key} title={item.title} src={item.src} emptyLabel="分析完成后显示对应图像" />
            ))}
          </div>
        </LegacyPanel>

        <LegacyPanel title="参数分析结果" accent="cyan" className="legacy-result-panel">
          <div className="legacy-result-list">
            {resultRows.length ? (
              resultRows.map((row) => <ResultRow key={row.label} label={row.label} unit={row.unit} value={row.value} />)
            ) : (
              <div className="legacy-empty-box">完成一次信号分析后，这里会显示和旧版上位机一致的参数结果。</div>
            )}
          </div>
        </LegacyPanel>

        <LegacyPanel title="星座图" accent="orange" className="legacy-constellation-panel">
          <ScatterChart points={analysisResult?.constellation ?? []} emptyLabel="分析完成后在这里显示星座图。" />
        </LegacyPanel>

        <LegacyPanel title="频谱分析" accent="yellow" className="legacy-fft-panel">
          <LineChart points={spectrumPoints} frames={spectrumFrames} color="#fff200" emptyLabel="分析完成后在这里显示频谱曲线。" />
        </LegacyPanel>

        <LegacyPanel title="信号质量评估" accent="magenta" className="legacy-quality-panel">
          <div className="legacy-quality-layout">
            <div className="legacy-quality-config-grid">
              <CompactInput
                label="采样频率 (Hz)"
                value={form.fsValue}
                unit={form.fsUnit}
                onValueChange={(value) => updateForm('fsValue', value)}
                onUnitChange={(value) => updateForm('fsUnit', value)}
              />
              <CompactSelect
                label="调制方式"
                value={String(form.modulation)}
                options={MODULATION_OPTIONS.map((option) => ({
                  label: option.label,
                  value: String(option.value)
                }))}
                onChange={(value) => updateForm('modulation', Number(value))}
              />
              <button className="legacy-mini-button legacy-mini-button--accent" type="button" onClick={confirmInputs}>
                确定
              </button>

              <CompactInput
                label="载波频率 (Hz)"
                value={form.fcValue}
                unit={form.fcUnit}
                onValueChange={(value) => updateForm('fcValue', value)}
                onUnitChange={(value) => updateForm('fcUnit', value)}
              />
              <CompactPlainInput
                label="载噪比 (dB)"
                value={form.snrValue}
                onChange={(value) => updateForm('snrValue', value)}
              />
              <div className="legacy-quality-actions">
                <button className="legacy-mini-button" type="button" onClick={startAssessment} disabled={!analysisResult}>
                  开始评估
                </button>
                <button
                  className="legacy-mini-button legacy-mini-button--ghost"
                  type="button"
                  onClick={calculateErrors}
                  disabled={!analysisResult}
                >
                  计算误差
                </button>
              </div>

              <div className="legacy-quality-span-2">
                <CompactInput
                  label="符号速率 (sy/s)"
                  value={form.rsValue}
                  unit={form.rsUnit}
                  onValueChange={(value) => updateForm('rsValue', value)}
                  onUnitChange={(value) => updateForm('rsUnit', value)}
                />
              </div>
            </div>

            <div className="legacy-quality-chart">
              <RadarChart series={qualitySeries} emptyLabel='点击“开始评估”后在这里显示质量雷达图。' />
            </div>

            <div className="legacy-error-strip">
              <ErrorInline label="载波中心频率误差" value={currentErrors?.center_frequency_percent ?? null} suffix="%" />
              <ErrorInline label="载波-3dB带宽误差" value={currentErrors?.bandwidth_percent ?? null} suffix="%" />
              <ErrorInline label="载噪比误差" value={currentErrors?.snr_db ?? null} suffix="dB" />
            </div>
          </div>
        </LegacyPanel>

        <LegacyPanel title="STFT 对照" accent="cyan" className="legacy-stft-panel">
          <StftReferencePanel
            src={
              analysisResult?.images.stft_display ??
              analysisResult?.images.stft_source ??
              analysisResult?.images.stft_annotated ??
              null
            }
            rotateFallback={!analysisResult?.images.stft_display}
            alt="STFT 对照"
            emptyLabel="分析完成后在这里显示 STFT 对照图。"
          />
        </LegacyPanel>
      </section>
    </main>
  )
}

function LegacyPanel({
  title,
  accent,
  className,
  children
}: {
  title: string
  accent: 'red' | 'cyan' | 'green' | 'orange' | 'yellow' | 'magenta'
  className?: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <section className={`legacy-panel legacy-panel--${accent} ${className ?? ''}`}>
      <div className="legacy-panel__title">{title}</div>
      <div className="legacy-panel__body">{children}</div>
    </section>
  )
}

function ResultRow({
  label,
  unit,
  value
}: {
  label: string
  unit: string
  value: string
}): React.JSX.Element {
  return (
    <div className="legacy-result-row">
      <div className="legacy-result-row__label">{label}</div>
      <div className="legacy-result-row__unit">{unit}</div>
      <div className="legacy-result-row__value">{value}</div>
    </div>
  )
}

function CompactInput({
  label,
  value,
  unit,
  onValueChange,
  onUnitChange
}: {
  label: string
  value: string
  unit: SignalFormState['fsUnit']
  onValueChange: (value: string) => void
  onUnitChange: (value: SignalFormState['fsUnit']) => void
}): React.JSX.Element {
  return (
    <label className="legacy-compact-field">
      <span>{label}</span>
      <div className="legacy-compact-field__input">
        <input value={value} onChange={(event) => onValueChange(event.target.value)} />
        <select value={unit} onChange={(event) => onUnitChange(event.target.value as SignalFormState['fsUnit'])}>
          <option value="GHz">G</option>
          <option value="MHz">M</option>
          <option value="kHz">K</option>
          <option value="Hz">Hz</option>
        </select>
      </div>
    </label>
  )
}

function CompactPlainInput({
  label,
  value,
  onChange
}: {
  label: string
  value: string
  onChange: (value: string) => void
}): React.JSX.Element {
  return (
    <label className="legacy-compact-field">
      <span>{label}</span>
      <div className="legacy-compact-field__input legacy-compact-field__input--single">
        <input value={value} onChange={(event) => onChange(event.target.value)} />
      </div>
    </label>
  )
}

function CompactSelect({
  label,
  value,
  options,
  onChange
}: {
  label: string
  value: string
  options: Array<{ label: string; value: string }>
  onChange: (value: string) => void
}): React.JSX.Element {
  return (
    <label className="legacy-compact-field">
      <span>{label}</span>
      <div className="legacy-compact-field__input legacy-compact-field__input--single">
        <select value={value} onChange={(event) => onChange(event.target.value)}>
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </label>
  )
}

function ImageCard({
  title,
  src,
  emptyLabel
}: {
  title: string
  src: string | null
  emptyLabel: string
}): React.JSX.Element {
  return (
    <div className="legacy-image-card">
      <div className="legacy-image-card__title">{title}</div>
      <ImagePanel src={src} alt={title} emptyLabel={emptyLabel} />
    </div>
  )
}

function ErrorInline({
  label,
  value,
  suffix
}: {
  label: string
  value: number | null
  suffix: string
}): React.JSX.Element {
  return (
    <div className="legacy-error-inline">
      <span>{label}</span>
      <strong>{value === null ? '-' : `${formatMetric(value)} ${suffix}`}</strong>
    </div>
  )
}

function ImagePanel({
  src,
  alt,
  emptyLabel
}: {
  src: string | null
  alt: string
  emptyLabel: string
}): React.JSX.Element {
  if (!src) {
    return <div className="legacy-empty-box">{emptyLabel}</div>
  }

  return (
    <div className="image-frame">
      <img className="analysis-image" src={src} alt={alt} />
    </div>
  )
}

function StftReferencePanel({
  src,
  alt,
  emptyLabel,
  rotateFallback
}: {
  src: string | null
  alt: string
  emptyLabel: string
  rotateFallback: boolean
}): React.JSX.Element {
  if (!src) {
    return <div className="legacy-empty-box">{emptyLabel}</div>
  }

  return (
    <div className="stft-reference stft-reference--plot">
      <img
        className={`stft-reference__image ${rotateFallback ? 'stft-reference__image--fallback-rotate' : ''}`}
        src={src}
        alt={alt}
      />
    </div>
  )
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  })

  if (!response.ok) {
    const errorText = await response.text()
    let detail = errorText

    try {
      const parsed = JSON.parse(errorText) as { detail?: string; message?: string }
      detail = parsed.detail ?? parsed.message ?? errorText
    } catch {
      detail = errorText
    }

    throw new Error(detail)
  }

  return (await response.json()) as T
}

function isConnectivityError(error: unknown): boolean {
  return error instanceof Error && error.message.includes('Failed to fetch')
}

function normalizeErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    if (isConnectivityError(error)) {
      return `无法连接后端，请确认已运行：${BACKEND_LAUNCH_HINT}`
    }

    return error.message
  }

  return String(error)
}

export default App
