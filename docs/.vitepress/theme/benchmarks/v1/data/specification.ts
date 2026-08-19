export const benchmarkV1CaseIds = [
  'ckks-depth-aware-single-operations',
  'indexed-ntt-operations',
  'dense-matmul-ptct',
  'dense-matmul-ctct',
  'polynomial-method-matrix',
] as const

export type BenchmarkV1CaseId = (typeof benchmarkV1CaseIds)[number]
export type BenchmarkV1ExecutionBackend = 'cpu' | 'cuda'

export const benchmarkV1ManifestSha256 =
  '5b9ce22abf59cb5b37dcc59062e2f856df40584470082fd0a4c6f08ee9b81c4b'

export const benchmarkV1CaseDefinitions: Record<
  BenchmarkV1CaseId,
  { benchmark: string; profile: string; workloadId: string }
> = {
  'ckks-depth-aware-single-operations': {
    benchmark: 'ckks-depth-aware-single-operations',
    profile: 'core',
    workloadId: 'ckks-depth-aware-single-operations',
  },
  'indexed-ntt-operations': {
    benchmark: 'indexed-ntt-operations',
    profile: 'core',
    workloadId: 'indexed-ntt-operations',
  },
  'dense-matmul-ptct': {
    benchmark: 'dense-matrix-multiplication-ptct',
    profile: 'core',
    workloadId: 'dense-matrix-multiplication-ptct',
  },
  'dense-matmul-ctct': {
    benchmark: 'dense-matrix-multiplication-ctct',
    profile: 'core',
    workloadId: 'dense-matrix-multiplication-ctct',
  },
  'polynomial-method-matrix': {
    benchmark: 'polynomial-evaluation',
    profile: 'core',
    workloadId: 'polynomial-evaluation',
  },
}

const benchmarkV1CaseIdSet = new Set<string>(benchmarkV1CaseIds)

export function isBenchmarkV1CaseId(value: string): value is BenchmarkV1CaseId {
  return benchmarkV1CaseIdSet.has(value)
}

export const benchmarkV1CaseLabels: Record<BenchmarkV1CaseId, string> = {
  'ckks-depth-aware-single-operations': 'Operations',
  'indexed-ntt-operations': 'Indexed NTT',
  'dense-matmul-ptct': 'PT × CT · 16×16',
  'dense-matmul-ctct': 'CT × CT · 16×16',
  'polynomial-method-matrix': 'Polynomials',
}
