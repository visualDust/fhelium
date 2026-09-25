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
  fheliumJitMs: number
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
    measured: 'FHElium',
    results: {
      7: {
        ptCt: { referenceMs: 74.198835, fheliumMs: 8.964460, fheliumJitMs: 4.999339 },
        ctCt: { referenceMs: 108.054670, fheliumMs: 10.046077, fheliumJitMs: 5.965834 },
      },
      16: {
        ptCt: { referenceMs: 126.610228, fheliumMs: 19.904646, fheliumJitMs: 17.816226 },
        ctCt: { referenceMs: 192.366163, fheliumMs: 23.570510, fheliumJitMs: 20.812153 },
      },
      34: {
        ptCt: { referenceMs: 454.819441, fheliumMs: 72.091918, fheliumJitMs: 70.671163 },
        ctCt: { referenceMs: 621.227444, fheliumMs: 86.249296, fheliumJitMs: 84.517840 },
      },
    },
  },
  {
    key: 'gpu2',
    label: '2 GPUs',
    hardware: 'RTX PRO 6000 Blackwell',
    workload: '256×256 dense matrix–vector',
    reference: 'Liberate 0.9.0',
    measured: 'FHElium',
    results: {
      7: {
        ptCt: { referenceMs: 116.622137, fheliumMs: 6.226670, fheliumJitMs: 4.023083 },
        ctCt: { referenceMs: 210.891075, fheliumMs: 8.155194, fheliumJitMs: 5.213323 },
      },
      16: {
        ptCt: { referenceMs: 135.267199, fheliumMs: 12.774520, fheliumJitMs: 11.359318 },
        ctCt: { referenceMs: 249.927941, fheliumMs: 16.733571, fheliumJitMs: 14.367518 },
      },
      34: {
        ptCt: { referenceMs: 276.499770, fheliumMs: 46.397627, fheliumJitMs: 45.317555 },
        ctCt: { referenceMs: 395.654165, fheliumMs: 60.771623, fheliumJitMs: 59.373772 },
      },
    },
  },
  {
    key: 'cpu',
    label: 'CPU',
    hardware: 'Threadripper PRO 9965WX',
    workload: '16×16 dense matrix–vector',
    reference: 'OpenFHE 1.4.2',
    measured: 'FHElium',
    results: {
      7: {
        ptCt: { referenceMs: 91.315008, fheliumMs: 29.765600, fheliumJitMs: 27.969600 },
        ctCt: { referenceMs: 147.203014, fheliumMs: 47.861400, fheliumJitMs: 43.619000 },
      },
      16: {
        ptCt: { referenceMs: 401.927965, fheliumMs: 103.084900, fheliumJitMs: 100.686700 },
        ctCt: { referenceMs: 707.881538, fheliumMs: 161.159100, fheliumJitMs: 158.581800 },
      },
      34: {
        ptCt: { referenceMs: 1_435.338548, fheliumMs: 523.535200, fheliumJitMs: 523.493100 },
        ctCt: { referenceMs: 2_454.963876, fheliumMs: 736.290100, fheliumJitMs: 729.616000 },
      },
    },
  },
] as const satisfies readonly PlatformMeasurement[]
