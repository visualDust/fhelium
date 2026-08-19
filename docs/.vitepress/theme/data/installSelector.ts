import catalogJson from './install-catalog.json'

export type OsId = 'linux-x86_64' | 'macos-arm64'
export type MethodId = 'prebuilt-pip' | 'source-pip' | 'source-github'
export type ComputeId = 'cpu' | `cuda-${number}`

export type InstallSelection = {
  os: OsId
  method: MethodId
  torch: string
  compute: ComputeId
}

export type TorchDistribution = {
  os: OsId
  torch: string
  compute: ComputeId
  requirement: string
  index_url: string
}

export type BinaryRecipe = InstallSelection & {
  configuration: string
  fhelium_version: string
  simple_index_url: string
  published: boolean
}

export type SourceProfile = {
  os: OsId
  method: 'source-pip' | 'source-github'
  native_backend_by_compute: {
    cpu: 'CPU'
    cuda?: 'CPU+CUDA'
  }
}

export type InstallCatalog = {
  schema_version: 1
  fhelium_version: string
  torch_distributions: TorchDistribution[]
  binary_recipes: BinaryRecipe[]
  source_profiles: SourceProfile[]
}

export const installCatalog = catalogJson as InstallCatalog

export const osLabels: Record<OsId, string> = {
  'linux-x86_64': 'Linux x86-64',
  'macos-arm64': 'macOS Apple Silicon',
}

export const methodLabels: Record<MethodId, string> = {
  'prebuilt-pip': 'Pip',
  'source-pip': 'Build from source (pip)',
  'source-github': 'Build from source (GitHub)',
}

export function computeLabel(compute: ComputeId): string {
  if (compute === 'cpu') return 'CPU'
  const version = compute.slice(5)
  return `CUDA ${version.slice(0, -1)}.${version.slice(-1)}`
}
