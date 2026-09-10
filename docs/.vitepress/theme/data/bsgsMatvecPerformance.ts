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
    measured: 'FHElium 0.20',
    results: {
      7: {
        ptCt: { referenceMs: 74.198835, fheliumMs: 6.319321 },
        ctCt: { referenceMs: 108.054670, fheliumMs: 7.761033 },
      },
      16: {
        ptCt: { referenceMs: 126.610228, fheliumMs: 22.992083 },
        ctCt: { referenceMs: 192.366163, fheliumMs: 29.634773 },
      },
      34: {
        ptCt: { referenceMs: 454.819441, fheliumMs: 104.901555 },
        ctCt: { referenceMs: 621.227444, fheliumMs: 126.321445 },
      },
    },
  },
  {
    key: 'gpu2',
    label: '2 GPUs',
    hardware: 'RTX PRO 6000 Blackwell',
    workload: '256×256 dense matrix–vector',
    reference: 'Liberate 0.9.0',
    measured: 'FHElium 0.20',
    results: {
      7: {
        ptCt: { referenceMs: 116.622137, fheliumMs: 4.395659 },
        ctCt: { referenceMs: 210.891075, fheliumMs: 5.830780 },
      },
      16: {
        ptCt: { referenceMs: 135.267199, fheliumMs: 13.830115 },
        ctCt: { referenceMs: 249.927941, fheliumMs: 17.922600 },
      },
      34: {
        ptCt: { referenceMs: 276.499770, fheliumMs: 60.912082 },
        ctCt: { referenceMs: 395.654165, fheliumMs: 79.159949 },
      },
    },
  },
  {
    key: 'cpu',
    label: 'CPU',
    hardware: 'Threadripper PRO 9965WX',
    workload: '16×16 dense matrix–vector',
    reference: 'OpenFHE 1.4.2',
    measured: 'FHElium 0.20',
    results: {
      7: {
        ptCt: { referenceMs: 91.315008, fheliumMs: 38.383031 },
        ctCt: { referenceMs: 147.203014, fheliumMs: 44.701132 },
      },
      16: {
        ptCt: { referenceMs: 401.927965, fheliumMs: 160.119467 },
        ctCt: { referenceMs: 707.881538, fheliumMs: 175.684921 },
      },
      34: {
        ptCt: { referenceMs: 1_435.338548, fheliumMs: 628.465334 },
        ctCt: { referenceMs: 2_454.963876, fheliumMs: 809.544331 },
      },
    },
  },
] as const satisfies readonly PlatformMeasurement[]
