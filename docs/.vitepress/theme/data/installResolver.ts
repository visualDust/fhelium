import {
  installCatalog,
  type BinaryRecipe,
  type ComputeId,
  type InstallCatalog,
  type InstallSelection,
  type MethodId,
  type OsId,
  type SourceProfile,
  type TorchDistribution,
} from './installSelector'

export type InstallAxis = keyof InstallSelection
export type InstallOption = { id: string; label: string }
export type ResolvedInstall = {
  selection: InstallSelection
  torch: TorchDistribution
  binary?: BinaryRecipe
  source?: SourceProfile
  commands: readonly string[]
}

const axisOrder: readonly InstallAxis[] = ['os', 'method', 'torch', 'compute']
const osOrder: readonly OsId[] = ['linux-x86_64', 'macos-arm64']
const methodOrder: readonly MethodId[] = ['prebuilt-pip', 'source-pip', 'source-github']
const torchOrder = ['2.13', '2.12'] as const
const computeOrder: readonly ComputeId[] = ['cuda-130', 'cuda-129', 'cpu']

function availableBinaryRecipes(catalog: InstallCatalog): BinaryRecipe[] {
  return catalog.binary_recipes.filter(recipe => recipe.published)
}

function candidateSelections(catalog: InstallCatalog): InstallSelection[] {
  const binary = availableBinaryRecipes(catalog)
  const source = catalog.source_profiles.flatMap(profile =>
    catalog.torch_distributions
      .filter(distribution => distribution.os === profile.os)
      .filter(distribution => {
        if (profile.os === 'macos-arm64') return distribution.compute === 'cpu'
        if (distribution.compute === 'cpu') return true
        return profile.native_backend_by_compute.cuda !== undefined
      })
      .map(distribution => ({
        os: profile.os,
        method: profile.method,
        torch: distribution.torch,
        compute: distribution.compute,
      })),
  )
  return [
    ...binary.map(({ os, method, torch, compute }) => ({ os, method, torch, compute })),
    ...source,
  ]
}

function prefixMatches(
  candidate: InstallSelection,
  selection: Partial<InstallSelection>,
  axis: InstallAxis,
): boolean {
  const axisIndex = axisOrder.indexOf(axis)
  return axisOrder.slice(0, axisIndex).every(key =>
    selection[key] === undefined || selection[key] === candidate[key],
  )
}

export function optionsFor(
  axis: InstallAxis,
  selection: Partial<InstallSelection>,
  catalog: InstallCatalog = installCatalog,
): string[] {
  const values = new Set(
    candidateSelections(catalog)
      .filter(candidate => prefixMatches(candidate, selection, axis))
      .map(candidate => candidate[axis]),
  )
  const order: readonly string[] =
    axis === 'os' ? osOrder
      : axis === 'method' ? methodOrder
        : axis === 'torch' ? torchOrder
          : computeOrder
  return order.filter(value => values.has(value))
}

export function defaultSelection(
  catalog: InstallCatalog = installCatalog,
): InstallSelection {
  const selection: Partial<InstallSelection> = {}
  for (const axis of axisOrder) {
    const next = optionsFor(axis, selection, catalog)[0]
    if (next === undefined) throw new Error(`No legal installation option for ${axis}`)
    Object.assign(selection, { [axis]: next })
  }
  return selection as InstallSelection
}

export function changeSelection(
  selection: InstallSelection,
  axis: InstallAxis,
  value: string,
  catalog: InstallCatalog = installCatalog,
): InstallSelection {
  const changedIndex = axisOrder.indexOf(axis)
  const next: Partial<InstallSelection> = {}
  for (const key of axisOrder.slice(0, changedIndex)) {
    Object.assign(next, { [key]: selection[key] })
  }
  Object.assign(next, { [axis]: value })
  for (const key of axisOrder.slice(changedIndex + 1)) {
    const candidate = optionsFor(key, next, catalog)[0]
    if (candidate === undefined) throw new Error(`No legal installation option for ${key}`)
    Object.assign(next, { [key]: candidate })
  }
  return next as InstallSelection
}

function distributionFor(
  selection: InstallSelection,
  catalog: InstallCatalog,
): TorchDistribution {
  const candidates = catalog.torch_distributions.filter(item =>
    item.torch === selection.torch
    && item.compute === selection.compute
    && item.os === selection.os,
  )
  if (candidates.length !== 1) {
    throw new Error(`Expected one Torch distribution, found ${candidates.length}`)
  }
  return candidates[0]
}

function torchCommand(distribution: TorchDistribution): string {
  return `python -m pip install --index-url ${distribution.index_url} "${distribution.requirement}"`
}

export function resolveInstall(
  selection: InstallSelection,
  catalog: InstallCatalog = installCatalog,
): ResolvedInstall {
  const torch = distributionFor(selection, catalog)
  if (selection.method === 'prebuilt-pip') {
    const recipes = availableBinaryRecipes(catalog).filter(recipe =>
      axisOrder.every(axis => recipe[axis] === selection[axis]),
    )
    if (recipes.length !== 1) {
      throw new Error(`Expected one published binary recipe, found ${recipes.length}`)
    }
    const binary = recipes[0]
    return {
      selection,
      torch,
      binary,
      commands: [
        torchCommand(torch),
        `python -m pip install --only-binary=fhelium \
  --extra-index-url ${binary.simple_index_url} \
  "fhelium==${binary.fhelium_version}"`,
      ],
    }
  }

  const profiles = catalog.source_profiles.filter(profile =>
    profile.os === selection.os && profile.method === selection.method,
  )
  if (profiles.length !== 1) throw new Error(`Expected one source profile, found ${profiles.length}`)
  const source = profiles[0]
  const backend = selection.compute === 'cpu'
    ? source.native_backend_by_compute.cpu
    : source.native_backend_by_compute.cuda
  if (backend === undefined) throw new Error('Selected source profile has no CUDA backend')
  return {
    selection,
    torch,
    source,
    commands: [
      torchCommand(torch),
      `python -m pip install "scikit-build-core>=1.0.3" "cmake>=3.18" ninja`,
      selection.method === 'source-pip'
        ? `CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=${backend}" \
  python -m pip install --no-binary=fhelium \
    --no-build-isolation --no-cache-dir --verbose "fhelium==${catalog.fhelium_version}"`
        : `git clone --branch "v${catalog.fhelium_version}" --depth 1 https://github.com/VisualDust/fhelium.git
cd fhelium
CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=${backend}" \
  python -m pip install . --no-build-isolation --no-cache-dir --verbose`,
    ],
  }
}

export function validateCatalog(catalog: InstallCatalog = installCatalog): void {
  if (catalog.schema_version !== 1) throw new Error('Unsupported install catalog schema')
  const binaryKeys = new Set<string>()
  for (const recipe of catalog.binary_recipes) {
    const key = axisOrder.map(axis => recipe[axis]).join('|')
    if (binaryKeys.has(key)) throw new Error(`Duplicate binary recipe: ${key}`)
    binaryKeys.add(key)
  }
  for (const candidate of candidateSelections(catalog)) resolveInstall(candidate, catalog)
}

validateCatalog()
