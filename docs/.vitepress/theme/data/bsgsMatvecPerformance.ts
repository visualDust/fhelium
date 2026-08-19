export type WorkloadKind = 'ptCt' | 'ctCt'

export interface DepthConfiguration {
  depth: 7 | 16 | 34
  ringDimension: number
  qBits: readonly [reference: number, fhelium: number]
  qpBits: readonly [reference: number, fhelium: number]
}

export interface WorkloadMeasurement {
  referenceMs: number
  fheliumMs: number
}

export interface PlatformMeasurement {
  key: 'gpu1' | 'gpu2' | 'cpu'
  label: string
  hardware: string
  workload: string
  reference: string
  measured: string
  results: Record<DepthConfiguration['depth'], Record<WorkloadKind, WorkloadMeasurement>>
}

export const depthConfigurations = [
  {
    depth: 7,
    ringDimension: 16_384,
    qBits: [340, 340],
    qpBits: [400, 400],
  },
  {
    depth: 16,
    ringDimension: 32_768,
    qBits: [701, 701],
    qpBits: [821, 821],
  },
  {
    depth: 34,
    ringDimension: 65_536,
    qBits: [1_420, 1_420],
    qpBits: [1_660, 1_660],
  },
] as const satisfies readonly DepthConfiguration[]

export const platformMeasurements = [
  {
    key: 'gpu1',
    label: '1 GPU',
    hardware: 'RTX PRO 6000 Blackwell',
    workload: '256×256 dense matrix–vector',
    reference: 'Liberate 0.9.0',
    measured: 'FHElium 0.10',
    results: {
      7: {
        ptCt: { referenceMs: 74.198835, fheliumMs: 6.592828 },
        ctCt: { referenceMs: 108.054670, fheliumMs: 9.791920 },
      },
      16: {
        ptCt: { referenceMs: 126.610228, fheliumMs: 23.089189 },
        ctCt: { referenceMs: 192.366163, fheliumMs: 39.446611 },
      },
      34: {
        ptCt: { referenceMs: 454.819441, fheliumMs: 113.753381 },
        ctCt: { referenceMs: 621.227444, fheliumMs: 191.891703 },
      },
    },
  },
  {
    key: 'gpu2',
    label: '2 GPUs',
    hardware: 'RTX PRO 6000 Blackwell',
    workload: '256×256 dense matrix–vector',
    reference: 'Liberate 0.9.0',
    measured: 'FHElium 0.10',
    results: {
      7: {
        ptCt: { referenceMs: 116.622137, fheliumMs: 4.564264 },
        ctCt: { referenceMs: 210.891075, fheliumMs: 6.104738 },
      },
      16: {
        ptCt: { referenceMs: 135.267199, fheliumMs: 14.548636 },
        ctCt: { referenceMs: 249.927941, fheliumMs: 22.421380 },
      },
      34: {
        ptCt: { referenceMs: 276.499770, fheliumMs: 69.056179 },
        ctCt: { referenceMs: 395.654165, fheliumMs: 108.641759 },
      },
    },
  },
  {
    key: 'cpu',
    label: 'CPU',
    hardware: 'Threadripper PRO 9965WX',
    workload: '16×16 dense matrix–vector',
    reference: 'OpenFHE 1.4.2',
    measured: 'FHElium 0.10',
    results: {
      7: {
        ptCt: { referenceMs: 91.135531, fheliumMs: 40.879805 },
        ctCt: { referenceMs: 162.043363, fheliumMs: 69.235583 },
      },
      16: {
        ptCt: { referenceMs: 431.167449, fheliumMs: 151.945039 },
        ctCt: { referenceMs: 754.649231, fheliumMs: 276.640468 },
      },
      34: {
        ptCt: { referenceMs: 1_538.924870, fheliumMs: 700.088974 },
        ctCt: { referenceMs: 2_424.366936, fheliumMs: 1_138.100891 },
      },
    },
  },
] as const satisfies readonly PlatformMeasurement[]
