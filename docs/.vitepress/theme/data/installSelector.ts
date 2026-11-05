import catalogJson from './install-catalog.json'

export type OsId = 'linux-x86_64' | 'windows-x86_64' | 'macos-arm64'
export type MethodId = 'prebuilt-pip' | 'source-pip' | 'source-github' | 'source-local'
export type ComputeId = 'cpu' | 'rocm' | `cuda-${number}`

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
  requirement?: string
  index_url?: string
  preinstalled?: boolean
}

export type BinaryRecipe = InstallSelection & {
  configuration: string
  fhelium_version: string
  simple_index_url: string
  published: boolean
}

export type SourceProfile = {
  os: OsId
  method: 'source-pip' | 'source-github' | 'source-local'
  native_backend_by_compute: {
    cpu?: 'CPU'
    cuda?: 'CPU+CUDA'
    rocm?: 'CPU+HIP'
  }
}

export type InstallCatalog = {
  schema_version: 1
  fhelium_version: string
  torch_distributions: TorchDistribution[]
  binary_recipes: BinaryRecipe[]
  source_profiles: SourceProfile[]
}

const publishedCatalog = catalogJson as InstallCatalog

export const installCatalog: InstallCatalog = {
  ...publishedCatalog,
  torch_distributions: [
    ...publishedCatalog.torch_distributions,
    { os: 'linux-x86_64', torch: 'installed-rocm', compute: 'rocm', preinstalled: true },
  ],
  source_profiles: [
    ...publishedCatalog.source_profiles,
    { os: 'linux-x86_64', method: 'source-local', native_backend_by_compute: { rocm: 'CPU+HIP' } },
  ],
}

export const osLabels: Record<OsId, string> = {
  'linux-x86_64': 'Linux x86-64',
  'windows-x86_64': 'Windows x86-64',
  'macos-arm64': 'macOS Apple Silicon',
}

export const methodLabels: Record<MethodId, string> = {
  'prebuilt-pip': 'Pip',
  'source-pip': 'Build from source (pip)',
  'source-github': 'Build from source (GitHub)',
  'source-local': 'Build from source',
}

export function computeLabel(compute: ComputeId): string {
  if (compute === 'cpu') return 'CPU'
  if (compute === 'rocm') return 'AMD ROCm'
  const version = compute.slice(5)
  return `CUDA ${version.slice(0, -1)}.${version.slice(-1)}`
}
