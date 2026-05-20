import { useEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'

import type { ConstellationPoint, QualityScores, StftData } from '../lib/signal'
import { QUALITY_COLORS, QUALITY_LABELS, qualityToArray } from '../lib/signal'

type LinePoint = {
  x: number
  y: number
}

type RadarSeries = {
  label: string
  values: QualityScores
}

const CONSTELLATION_BOX = {
  width: 1000,
  height: 1000,
  left: 112,
  right: 940,
  top: 58,
  bottom: 910
}

const DEFAULT_SPECTRUM_Y_LIMIT = 0.03
const RADAR_LEVELS = [5, 10, 15, 20, 25]
const RADAR_VIEWBOX = {
  width: 360,
  height: 360,
  centerX: 180,
  centerY: 180,
  radius: 128,
  labelOffset: 38
}

export function LineChart({
  points,
  frames,
  color = '#ffff00',
  emptyLabel
}: {
  points: LinePoint[]
  frames?: LinePoint[][]
  color?: string
  emptyLabel: string
}): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const plotRef = useRef<HTMLDivElement | null>(null)
  const plotSize = useElementSize(plotRef)
  const frameList = useMemo(() => {
    return (frames?.length ? frames : [points])
      .map((frame) =>
        frame
          .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y) && point.x >= 0)
          .map((point) => ({ x: point.x, y: Math.max(point.y, 0) }))
          .sort((left, right) => left.x - right.x)
      )
      .filter((frame) => frame.length >= 2)
  }, [frames, points])
  const [frameIndex, setFrameIndex] = useState(0)
  const [yLimit, setYLimit] = useState(DEFAULT_SPECTRUM_Y_LIMIT)
  const frameSignature = useMemo(
    () => frameList.map((frame) => `${frame.length}:${frame[0]?.x ?? 0}:${frame[frame.length - 1]?.x ?? 0}`).join('|'),
    [frameList]
  )

  useEffect(() => {
    setFrameIndex(findStrongestFrameIndex(frameList))
    setYLimit(chooseSpectrumYLimit(frameList))
  }, [frameList, frameSignature])

  useEffect(() => {
    if (frameList.length <= 1) {
      return
    }

    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % frameList.length)
    }, 300)

    return () => window.clearInterval(timer)
  }, [frameList.length, frameSignature])

  const activePoints = frameList[frameIndex % Math.max(frameList.length, 1)] ?? []

  useEffect(() => {
    const canvas = canvasRef.current
    const rect = canvas?.parentElement?.getBoundingClientRect()
    const width = plotSize.width || rect?.width || canvas?.clientWidth || 0
    const height = plotSize.height || rect?.height || canvas?.clientHeight || 0
    if (!canvas || activePoints.length < 2 || width <= 0 || height <= 0) {
      return
    }

    drawSpectrumCanvas(canvas, activePoints, {
      width,
      height,
      yLimit,
      color
    })
  }, [activePoints, color, plotSize.height, plotSize.width, yLimit])

  if (!frameList.length) {
    return <div className="empty-state">{emptyLabel}</div>
  }

  return (
    <div className="instrument-plot instrument-plot--spectrum">
      <div ref={plotRef} className="spectrum-canvas-shell">
        <canvas ref={canvasRef} className="spectrum-canvas" />
      </div>
      <div className="plot-controls">
        <button
          className="plot-control-dot"
          type="button"
          onClick={() =>
            setYLimit((current) => {
              const step = current <= 0.01 ? 0.001 : 0.01
              return Math.max(0.001, Number((current - step).toFixed(4)))
            })
          }
        >
          -
        </button>
        <span className="plot-control-readout">{formatLimitReadout(yLimit)}</span>
        <button
          className="plot-control-dot"
          type="button"
          onClick={() =>
            setYLimit((current) => {
              const step = current < 0.01 ? 0.001 : 0.01
              return Number((current + step).toFixed(4))
            })
          }
        >
          +
        </button>
        <span className="plot-live-badge">{frameList.length > 1 ? `动态窗口 ${frameIndex + 1}/${frameList.length}` : '单帧频谱'}</span>
      </div>
    </div>
  )
}

export function ScatterChart({
  points,
  emptyLabel
}: {
  points: ConstellationPoint[]
  emptyLabel: string
}): React.JSX.Element {
  const usablePoints = useMemo(
    () => points.filter((point) => Number.isFinite(point.real) && Number.isFinite(point.imag)).slice(0, 2200),
    [points]
  )
  const pointsPerFrame = Math.max(1, Math.min(1000, Math.floor(usablePoints.length / 10) || usablePoints.length))
  const [startIndex, setStartIndex] = useState(0)
  const pointSignature = useMemo(
    () => `${usablePoints.length}:${usablePoints[0]?.real ?? 0}:${usablePoints[usablePoints.length - 1]?.imag ?? 0}`,
    [usablePoints]
  )

  useEffect(() => {
    setStartIndex(0)
  }, [pointSignature])

  useEffect(() => {
    if (usablePoints.length <= pointsPerFrame) {
      return
    }

    const timer = window.setInterval(() => {
      setStartIndex((current) => (current + pointsPerFrame) % usablePoints.length)
    }, 1000)

    return () => window.clearInterval(timer)
  }, [pointSignature, pointsPerFrame, usablePoints.length])

  const chart = useMemo(() => {
    if (!usablePoints.length) {
      return null
    }

    const activePoints =
      usablePoints.length <= pointsPerFrame
        ? usablePoints
        : wrapSlice(usablePoints, startIndex, pointsPerFrame)

    return buildConstellationChart(usablePoints, activePoints)
  }, [pointsPerFrame, startIndex, usablePoints])

  if (!chart) {
    return <div className="empty-state">{emptyLabel}</div>
  }

  return (
    <div className="instrument-plot instrument-plot--constellation">
      <svg className="instrument-plot__svg" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id="constellation-clip">
            <rect
              x={chart.box.left}
              y={chart.box.top}
              width={chart.box.right - chart.box.left}
              height={chart.box.bottom - chart.box.top}
            />
          </clipPath>
        </defs>
        <rect className="plot-bg" x="0" y="0" width="1000" height="1000" />
        {chart.ticks.map((tick) => (
          <g key={`grid-${tick.value}`}>
            <line className="axis-line" x1={tick.x} y1={chart.box.top} x2={tick.x} y2={chart.box.bottom} />
            <line className="axis-line" x1={chart.box.left} y1={tick.y} x2={chart.box.right} y2={tick.y} />
            <text className="constellation-tick-label" x={tick.x} y={chart.box.bottom + 42} textAnchor="middle">
              {tick.value}
            </text>
            <text className="constellation-tick-label" x={chart.box.left - 34} y={tick.y + 12} textAnchor="end">
              {tick.value}
            </text>
          </g>
        ))}
        <text className="constellation-axis-title" x={(chart.box.left + chart.box.right) / 2} y="980" textAnchor="middle">
          I_Phase
        </text>
        <text
          className="constellation-axis-title"
          x="40"
          y={(chart.box.top + chart.box.bottom) / 2}
          textAnchor="middle"
          transform={`rotate(-90 40 ${(chart.box.top + chart.box.bottom) / 2})`}
        >
          Q_Phase
        </text>
        <g key={startIndex} clipPath="url(#constellation-clip)">
          {chart.points.map((point, index) => (
            <circle
              key={`${index}-${point.x.toFixed(2)}-${point.y.toFixed(2)}`}
              className="scatter-point"
              cx={point.x}
              cy={point.y}
              r="3.6"
            />
          ))}
        </g>
      </svg>
      <div className="plot-live-badge plot-live-badge--constellation">
        {usablePoints.length > pointsPerFrame ? `动态刷新 ${pointsPerFrame} 点/帧` : '单帧星座图'}
      </div>
    </div>
  )
}

export function HeatmapChart({
  stft,
  emptyLabel
}: {
  stft: StftData | null
  emptyLabel: string
}): React.JSX.Element {
  if (!stft || !stft.matrix.length) {
    return <div className="empty-state">{emptyLabel}</div>
  }

  const rows = stft.matrix.length
  const cols = stft.matrix[0]?.length ?? 0
  if (rows === 0 || cols === 0) {
    return <div className="empty-state">{emptyLabel}</div>
  }

  const cellWidth = 84 / cols
  const cellHeight = 74 / rows

  return (
    <svg className="plot-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
      <rect className="plot-bg" x="0" y="0" width="100" height="100" rx="6" />
      {stft.matrix.map((row, rowIndex) =>
        row.map((value, colIndex) => (
          <rect
            key={`${rowIndex}-${colIndex}`}
            x={8 + colIndex * cellWidth}
            y={12 + rowIndex * cellHeight}
            width={cellWidth + 0.2}
            height={cellHeight + 0.2}
            fill={heatmapColor(value)}
          />
        ))
      )}
    </svg>
  )
}

export function RadarChart({
  series,
  emptyLabel
}: {
  series: RadarSeries[]
  emptyLabel: string
}): React.JSX.Element {
  if (!series.length) {
    return <div className="empty-state">{emptyLabel}</div>
  }

  const axes = QUALITY_LABELS.length
  const angles = Array.from({ length: axes }, (_, index) => (Math.PI * 2 * index) / axes - Math.PI / 2)

  return (
    <div className="radar-shell">
      <div className="radar-stage">
        <svg
          className="plot-svg radar-svg"
          viewBox={`0 0 ${RADAR_VIEWBOX.width} ${RADAR_VIEWBOX.height}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <rect className="plot-bg" x="0" y="0" width={RADAR_VIEWBOX.width} height={RADAR_VIEWBOX.height} rx="6" />
          {RADAR_LEVELS.map((level) => (
            <circle
              key={level}
              className={`radar-grid ${level === RADAR_LEVELS[RADAR_LEVELS.length - 1] ? 'radar-grid--outer' : ''}`.trim()}
              cx={RADAR_VIEWBOX.centerX}
              cy={RADAR_VIEWBOX.centerY}
              r={radarRadius(level)}
            />
          ))}
          {RADAR_LEVELS.map((level) => {
            const tick = radarTickPosition(level)
            return (
              <text key={`tick-${level}`} className="radar-tick-label" x={tick.x} y={tick.y} textAnchor="start">
                {level}
              </text>
            )
          })}
          {angles.map((angle, index) => {
            const endpoint = radarPointAtValue(angle, RADAR_LEVELS[RADAR_LEVELS.length - 1])
            return (
              <line
                key={`${angle}-${index}`}
                className="radar-axis"
                x1={RADAR_VIEWBOX.centerX}
                y1={RADAR_VIEWBOX.centerY}
                x2={endpoint.x}
                y2={endpoint.y}
              />
            )
          })}
          {QUALITY_LABELS.map((label, index) => {
            const { x, y } = radarLabelPosition(angles[index])
            const lines = radarLabelLines(label)

            return (
              <text key={label} className="radar-label" x={x} y={y} textAnchor={radarTextAnchor(angles[index])}>
                {lines.map((line, lineIndex) => (
                  <tspan key={`${label}-${line}`} x={x} y={y - ((lines.length - 1) * 8) / 2 + lineIndex * 15}>
                    {line}
                  </tspan>
                ))}
              </text>
            )
          })}
          {series.map((item, index) => {
            const color = QUALITY_COLORS[index % QUALITY_COLORS.length]
            const values = qualityToArray(item.values)
            const polygon = radarPolygon(values, angles)

            return (
              <g key={`${item.label}-${index}`}>
                <polygon className="radar-fill" style={{ fill: `${color}24`, stroke: color }} points={polygon} />
                {values.map((value, valueIndex) => {
                  const point = radarPointAtValue(angles[valueIndex], value)
                  return (
                    <circle
                      key={`${item.label}-${QUALITY_LABELS[valueIndex]}`}
                      className="radar-point"
                      cx={point.x}
                      cy={point.y}
                      r="3.6"
                      style={{ fill: color, stroke: '#0d0f11' }}
                    />
                  )
                })}
              </g>
            )
          })}
        </svg>
      </div>
      <div className="radar-legend">
        {series.map((item, index) => (
          <div key={item.label} className="legend-row">
            <span className="legend-swatch" style={{ backgroundColor: QUALITY_COLORS[index % QUALITY_COLORS.length] }} />
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function radarRadius(value: number): number {
  const maxLevel = RADAR_LEVELS[RADAR_LEVELS.length - 1]
  return (clamp(value, 0, maxLevel) / maxLevel) * RADAR_VIEWBOX.radius
}

function radarPointAtValue(angle: number, value: number): { x: number; y: number } {
  const radius = radarRadius(value)
  return {
    x: RADAR_VIEWBOX.centerX + Math.cos(angle) * radius,
    y: RADAR_VIEWBOX.centerY + Math.sin(angle) * radius
  }
}

function radarTickPosition(level: number): { x: number; y: number } {
  return {
    x: RADAR_VIEWBOX.centerX + 8,
    y: RADAR_VIEWBOX.centerY - radarRadius(level) + 4
  }
}

function radarLabelPosition(angle: number): { x: number; y: number } {
  const distance = RADAR_VIEWBOX.radius + RADAR_VIEWBOX.labelOffset
  return {
    x: RADAR_VIEWBOX.centerX + Math.cos(angle) * distance,
    y: RADAR_VIEWBOX.centerY + Math.sin(angle) * distance
  }
}

function radarLabelLines(label: string): string[] {
  switch (label) {
    case '载波中心频率误差':
      return ['载波中心', '频率误差']
    case '载噪比误差':
      return ['载噪比', '误差']
    case '载波-3dB带宽误差':
      return ['载波-3dB', '带宽误差']
    default:
      return [label]
  }
}

function radarTextAnchor(angle: number): 'start' | 'middle' | 'end' {
  const horizontal = Math.cos(angle)
  if (Math.abs(horizontal) < 0.25) {
    return 'middle'
  }
  return horizontal > 0 ? 'start' : 'end'
}

function radarPolygon(values: number[], angles: number[]): string {
  return angles
    .map((angle, index) => {
      const point = radarPointAtValue(angle, values[index] ?? 0)
      return `${point.x},${point.y}`
    })
    .join(' ')
}

function drawSpectrumCanvas(
  canvas: HTMLCanvasElement,
  points: LinePoint[],
  options: { width: number; height: number; yLimit: number; color: string }
): void {
  const dpr = window.devicePixelRatio || 1
  const width = Math.max(1, Math.floor(options.width))
  const height = Math.max(1, Math.floor(options.height))
  canvas.width = Math.floor(width * dpr)
  canvas.height = Math.floor(height * dpr)
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`

  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = '#000000'
  ctx.fillRect(0, 0, width, height)

  const plot = {
    left: clamp(width * 0.04, 76, 104),
    right: width - clamp(width * 0.012, 18, 28),
    top: clamp(height * 0.06, 18, 28),
    bottom: height - clamp(height * 0.16, 48, 64)
  }
  const plotWidth = plot.right - plot.left
  const plotHeight = plot.bottom - plot.top
  const xLimit = normalizeSpectrumLimit(Math.max(...points.map((point) => point.x), 1))
  const yLimit = Math.max(options.yLimit, Number.EPSILON)
  const xExponent = axisExponent(xLimit)
  const yExponent = axisExponent(yLimit)
  const xScale = 10 ** xExponent
  const yScale = 10 ** yExponent
  const tickFont = `${clamp(width / 120, 15, 22)}px Segoe UI, Microsoft YaHei, sans-serif`
  const labelFont = `${clamp(width / 110, 16, 22)}px Segoe UI, Microsoft YaHei, sans-serif`
  const scaleFont = `${clamp(width / 130, 13, 18)}px Segoe UI, Microsoft YaHei, sans-serif`

  ctx.save()
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.42)'
  ctx.lineWidth = 1
  ctx.setLineDash([3, 3])

  for (let index = 0; index <= 5; index += 1) {
    const x = plot.left + (index / 5) * plotWidth
    ctx.beginPath()
    ctx.moveTo(x, plot.top)
    ctx.lineTo(x, plot.bottom)
    ctx.stroke()
  }

  for (let index = 0; index <= 6; index += 1) {
    const y = plot.bottom - (index / 6) * plotHeight
    ctx.beginPath()
    ctx.moveTo(plot.left, y)
    ctx.lineTo(plot.right, y)
    ctx.stroke()
  }
  ctx.restore()

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.72)'
  ctx.lineWidth = 1
  ctx.setLineDash([])
  ctx.beginPath()
  ctx.moveTo(plot.left, plot.bottom)
  ctx.lineTo(plot.right, plot.bottom)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(plot.left, plot.top)
  ctx.lineTo(plot.left, plot.bottom)
  ctx.stroke()

  ctx.fillStyle = '#f5f5f5'
  ctx.font = tickFont
  ctx.textBaseline = 'middle'
  ctx.textAlign = 'center'
  for (let index = 0; index <= 5; index += 1) {
    const x = plot.left + (index / 5) * plotWidth
    ctx.fillText(formatTick((xLimit * index) / 5 / xScale), x, plot.bottom + 20)
  }

  ctx.textAlign = 'right'
  for (let index = 0; index <= 6; index += 1) {
    const y = plot.bottom - (index / 6) * plotHeight
    ctx.fillText(formatTick((yLimit * index) / 6 / yScale), plot.left - 10, y)
  }

  ctx.font = labelFont
  ctx.textAlign = 'center'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText('Frequency (Hz)', (plot.left + plot.right) / 2, height - 8)
  ctx.save()
  ctx.translate(22, (plot.top + plot.bottom) / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.fillText('Amplitude', 0, 0)
  ctx.restore()

  ctx.font = scaleFont
  ctx.textAlign = 'left'
  ctx.textBaseline = 'top'
  drawExponentText(ctx, `×10`, yExponent, plot.left, 4)
  ctx.textAlign = 'right'
  drawExponentText(ctx, `×10`, xExponent, plot.right - 2, plot.bottom + 28, true)

  ctx.save()
  ctx.beginPath()
  ctx.rect(plot.left, plot.top, plotWidth, plotHeight)
  ctx.clip()
  ctx.strokeStyle = options.color
  ctx.lineWidth = clamp(width / 1050, 1.6, 2.4)
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.beginPath()
  points.forEach((point, index) => {
    const x = plot.left + (point.x / xLimit) * plotWidth
    const y = plot.bottom - (Math.min(point.y, yLimit) / yLimit) * plotHeight
    if (index === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })
  ctx.stroke()
  ctx.restore()
}

function useElementSize(ref: RefObject<HTMLElement | null>): { width: number; height: number } {
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const element = ref.current
    if (!element) {
      return
    }

    let animationFrame = 0
    const update = (): void => {
      const rect = element.getBoundingClientRect()
      setSize((current) => {
        const next = {
          width: rect.width,
          height: rect.height
        }
        if (Math.abs(current.width - next.width) < 0.5 && Math.abs(current.height - next.height) < 0.5) {
          return current
        }
        return next
      })
    }

    update()
    animationFrame = window.requestAnimationFrame(update)

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', update)
      return () => {
        window.cancelAnimationFrame(animationFrame)
        window.removeEventListener('resize', update)
      }
    }

    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => {
      window.cancelAnimationFrame(animationFrame)
      observer.disconnect()
    }
  }, [ref])

  return size
}

function buildConstellationChart(allPoints: ConstellationPoint[], activePoints: ConstellationPoint[]) {
  const box = CONSTELLATION_BOX
  const plotWidth = box.right - box.left
  const plotHeight = box.bottom - box.top
  const domain = Math.max(2, percentile(allPoints.flatMap((point) => [Math.abs(point.real), Math.abs(point.imag)]), 0.995))
  const scale = Math.min(Math.max(domain * 1.08, 2), 3)
  const toX = (value: number): number => box.left + ((value + scale) / (scale * 2)) * plotWidth
  const toY = (value: number): number => box.bottom - ((value + scale) / (scale * 2)) * plotHeight

  return {
    box,
    points: activePoints.map((point) => ({
      x: toX(point.real),
      y: toY(point.imag)
    })),
    ticks: [-2, -1, 0, 1, 2].map((tick) => ({
      value: tick,
      x: toX(tick),
      y: toY(tick)
    }))
  }
}

function wrapSlice<T>(values: T[], startIndex: number, size: number): T[] {
  if (values.length <= size) {
    return values
  }

  const result: T[] = []
  for (let index = 0; index < size; index += 1) {
    result.push(values[(startIndex + index) % values.length])
  }
  return result
}

function percentile(values: number[], ratio: number): number {
  const finite = values.filter((value) => Number.isFinite(value)).sort((left, right) => left - right)
  if (!finite.length) {
    return 0
  }
  const index = Math.min(finite.length - 1, Math.max(0, Math.floor((finite.length - 1) * ratio)))
  return finite[index]
}

function findStrongestFrameIndex(frames: LinePoint[][]): number {
  if (!frames.length) {
    return 0
  }

  let strongestIndex = 0
  let strongestPeak = -Infinity
  frames.forEach((frame, index) => {
    const peak = percentile(
      frame.map((point) => point.y),
      0.995
    )
    if (peak > strongestPeak) {
      strongestPeak = peak
      strongestIndex = index
    }
  })
  return strongestIndex
}

function chooseSpectrumYLimit(frames: LinePoint[][]): number {
  const values = frames.flatMap((frame) => frame.map((point) => point.y))
  const peak = percentile(values, 0.995)
  if (!Number.isFinite(peak) || peak <= 0) {
    return DEFAULT_SPECTRUM_Y_LIMIT
  }

  return roundUpNiceLimit(peak * 1.18)
}

function roundUpNiceLimit(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return DEFAULT_SPECTRUM_Y_LIMIT
  }

  const exponent = Math.floor(Math.log10(value))
  const scale = 10 ** exponent
  const normalized = value / scale
  const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 3 ? 3 : normalized <= 5 ? 5 : 10
  return Math.max(0.001, Number((nice * scale).toPrecision(4)))
}

function formatLimitReadout(value: number): string {
  return value >= 0.01 ? value.toFixed(3) : value.toFixed(4)
}

function axisExponent(value: number): number {
  if (!Number.isFinite(value) || value === 0) {
    return 0
  }
  const exponent = Math.floor(Math.log10(Math.abs(value)))
  return exponent >= 3 || exponent <= -2 ? exponent : 0
}

function normalizeSpectrumLimit(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 1
  }

  const exponent = Math.floor(Math.log10(value))
  const scale = 10 ** exponent
  const normalized = value / scale
  const rounded = Math.ceil(normalized * 10) / 10
  return rounded * scale
}

function formatTick(value: number): string {
  if (!Number.isFinite(value)) {
    return '0'
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(0)
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1).replace(/\.0$/, '')
  }
  return value.toFixed(1).replace(/\.0$/, '')
}

function drawExponentText(
  ctx: CanvasRenderingContext2D,
  base: string,
  exponent: number,
  x: number,
  y: number,
  alignRight = false
): void {
  const baseWidth = ctx.measureText(base).width
  if (alignRight) {
    const exponentWidth = ctx.measureText(String(exponent)).width * 0.72
    ctx.fillText(base, x - exponentWidth, y)
    const previousFont = ctx.font
    ctx.font = previousFont.replace(/(\d+(?:\.\d+)?)px/, (_, size) => `${Number(size) * 0.72}px`)
    ctx.fillText(String(exponent), x, y - 7)
    ctx.font = previousFont
    return
  }

  ctx.fillText(base, x, y)
  const previousFont = ctx.font
  ctx.font = previousFont.replace(/(\d+(?:\.\d+)?)px/, (_, size) => `${Number(size) * 0.72}px`)
  ctx.fillText(String(exponent), x + baseWidth + 1, y - 7)
  ctx.font = previousFont
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function heatmapColor(value: number): string {
  const clamped = Math.max(0, Math.min(1, value))
  const red = Math.round(12 + clamped * 234)
  const green = Math.round(40 + clamped * 150)
  const blue = Math.round(88 + (1 - clamped) * 130)
  return `rgb(${red}, ${green}, ${blue})`
}
