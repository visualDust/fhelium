import type { TopLevelSpec } from 'vega-lite'

type ChartTheme = 'light' | 'dark'

interface BackendMeasurement {
  backend: string
  display: string
  nttOnlyPercentSlower: number
  ckksPercentSlower: number
  recommended: boolean
}

interface BackendSelectionData {
  thresholdPercent: number
  rows: BackendMeasurement[]
}

interface Palette {
  foreground: string
  secondary: string
  grid: string
  nearTie: string
  threshold: string
  nttOnly: string
  ckks: string
  recommendation: string
}

function parseData(value: unknown): BackendSelectionData {
  if (typeof value !== 'object' || value === null) {
    throw new TypeError('NTT backend chart data must be an object')
  }
  const candidate = value as Partial<BackendSelectionData>
  if (
    typeof candidate.thresholdPercent !== 'number'
    || !Array.isArray(candidate.rows)
    || candidate.rows.length === 0
  ) {
    throw new TypeError('NTT backend chart data is missing its threshold or rows')
  }
  for (const row of candidate.rows) {
    if (
      typeof row.backend !== 'string'
      || typeof row.display !== 'string'
      || typeof row.nttOnlyPercentSlower !== 'number'
      || typeof row.ckksPercentSlower !== 'number'
      || typeof row.recommended !== 'boolean'
    ) {
      throw new TypeError('NTT backend chart contains an invalid measurement row')
    }
  }
  return candidate as BackendSelectionData
}

function measurementPanel({
  rows,
  thresholdPercent,
  valueField,
  labelField,
  title,
  color,
  palette,
  markRecommendation,
}: {
  rows: Array<BackendMeasurement & Record<string, unknown>>
  thresholdPercent: number
  valueField: string
  labelField: string
  title: string
  color: string
  palette: Palette
  markRecommendation: boolean
}) {
  const order = rows.map((row) => row.display)
  const xScale = { domain: [-0.35, 8], nice: false }
  const xAxis = {
    title: 'Relative latency above the fastest result (%)',
    titleColor: palette.foreground,
    titleFontSize: 12,
    titlePadding: 10,
    labelColor: palette.secondary,
    labelFontSize: 11,
    values: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    grid: true,
    gridColor: palette.grid,
    gridOpacity: 0.85,
    tickColor: palette.grid,
    domainColor: palette.grid,
  }
  const yEncoding = {
    field: 'display',
    type: 'nominal',
    sort: order,
    axis: {
      title: null,
      labelColor: palette.foreground,
      labelFontSize: 13,
      labelFontWeight: 500,
      labelLimit: 260,
      labelPadding: 10,
      ticks: false,
      domain: false,
    },
  }
  const layers: Record<string, unknown>[] = [
    {
      data: { values: [{ start: 0, end: thresholdPercent }] },
      mark: { type: 'rect', color: palette.nearTie },
      encoding: {
        x: { field: 'start', type: 'quantitative', scale: xScale, axis: xAxis },
        x2: { field: 'end' },
      },
    },
    {
      data: { values: [{ threshold: thresholdPercent }] },
      mark: {
        type: 'rule',
        color: palette.threshold,
        strokeDash: [5, 4],
        strokeWidth: 1.5,
      },
      encoding: {
        x: {
          field: 'threshold',
          type: 'quantitative',
          scale: xScale,
          axis: xAxis,
        },
      },
    },
    {
      data: { values: rows },
      mark: {
        type: 'bar',
        color,
        opacity: 0.35,
        size: 8,
        cornerRadiusEnd: 4,
      },
      encoding: {
        x: {
          field: valueField,
          type: 'quantitative',
          scale: xScale,
          axis: xAxis,
        },
        x2: { datum: 0 },
        y: yEncoding,
      },
    },
    {
      data: { values: rows },
      mark: { type: 'point', filled: true, color, size: 105 },
      encoding: {
        x: {
          field: valueField,
          type: 'quantitative',
          scale: xScale,
          axis: xAxis,
        },
        y: yEncoding,
        tooltip: [
          { field: 'backend', type: 'nominal', title: 'Backend' },
          {
            field: valueField,
            type: 'quantitative',
            title: 'Relative latency above the fastest result (%)',
            format: '.3f',
          },
        ],
      },
    },
    {
      data: { values: rows },
      mark: {
        type: 'text',
        align: 'left',
        baseline: 'middle',
        dx: 8,
        color,
        fontSize: 11,
        fontWeight: 650,
      },
      encoding: {
        x: {
          field: valueField,
          type: 'quantitative',
          scale: xScale,
          axis: xAxis,
        },
        y: yEncoding,
        text: { field: labelField },
      },
    },
  ]
  if (markRecommendation) {
    layers.push({
      data: { values: rows },
      transform: [{ filter: 'datum.recommended === true' }],
      mark: {
        type: 'point',
        filled: false,
        size: 330,
        stroke: palette.recommendation,
        strokeWidth: 2.5,
      },
      encoding: {
        x: {
          field: valueField,
          type: 'quantitative',
          scale: xScale,
          axis: xAxis,
        },
        y: yEncoding,
      },
    })
  }
  return {
    width: 'container',
    height: 180,
    title: {
      text: title,
      anchor: 'start',
      color: palette.foreground,
      fontSize: 15,
      fontWeight: 650,
      offset: 10,
    },
    layer: layers,
  }
}

export function nttBackendSelectionSpec(
  value: unknown,
  theme: ChartTheme,
): TopLevelSpec {
  const data = parseData(value)
  const dark = theme === 'dark'
  const palette: Palette = {
    foreground: dark ? '#e5e7eb' : '#243230',
    secondary: dark ? '#aeb8b5' : '#60706d',
    grid: dark ? '#343a3d' : '#dfe7e5',
    nearTie: dark ? '#173a35' : '#e8f4f1',
    threshold: dark ? '#7bc0b2' : '#448c7e',
    nttOnly: dark ? '#5ed4c2' : '#0f766e',
    ckks: dark ? '#8fb4ff' : '#315b9c',
    recommendation: dark ? '#ffbd75' : '#c65d09',
  }
  const rows = data.rows.map((row) => ({
    ...row,
    nttOnlyLabel: `${row.nttOnlyPercentSlower.toFixed(2)}%`,
    ckksLabel: `${row.ckksPercentSlower.toFixed(2)}%`,
  }))
  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v6.json',
    background: 'transparent',
    padding: { top: 8, right: 30, bottom: 6, left: 8 },
    spacing: 30,
    vconcat: [
      measurementPanel({
        rows,
        thresholdPercent: data.thresholdPercent,
        valueField: 'nttOnlyPercentSlower',
        labelField: 'nttOnlyLabel',
        title: 'NTT operations only',
        color: palette.nttOnly,
        palette,
        markRecommendation: false,
      }),
      measurementPanel({
        rows,
        thresholdPercent: data.thresholdPercent,
        valueField: 'ckksPercentSlower',
        labelField: 'ckksLabel',
        title: 'Complete CKKS operations',
        color: palette.ckks,
        palette,
        markRecommendation: true,
      }),
    ],
    resolve: { scale: { x: 'shared', y: 'shared' } },
    config: {
      font: 'Inter, ui-sans-serif, system-ui, sans-serif',
      view: { stroke: null },
    },
  } as unknown as TopLevelSpec
}
