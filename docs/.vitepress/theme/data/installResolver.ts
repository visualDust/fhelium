import {
  installCatalog,
  type BinaryRecipe,
  type ComputeId,
  type InstallCatalog,
  type InstallSelection,
  type LanguageId,
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

const axisOrder: readonly InstallAxis[] = ['os', 'compute', 'torch', 'language', 'method']
// Published binary recipes are keyed by these axes; every recipe provides both
// the Python package and the C++ library, but the installer offers C++ only for
// source builds.
const recipeAxes = ['os', 'compute', 'torch', 'method'] as const
const languageOrder: readonly LanguageId[] = ['python', 'cpp']
const osOrder: readonly OsId[] = [
  'linux-x86_64',
  'windows-x86_64',
  'macos-arm64',
]
const methodOrder: readonly MethodId[] = ['prebuilt-pip', 'source-pip', 'source-github', 'source-local']
const torchOrder = ['2.13', '2.12', 'installed-rocm'] as const
const computeOrder: readonly ComputeId[] = ['cuda-130', 'cuda-129', 'rocm', 'cpu']

function availableBinaryRecipes(catalog: InstallCatalog): BinaryRecipe[] {
  return catalog.binary_recipes.filter(recipe => recipe.published)
}

function candidateSelections(catalog: InstallCatalog): InstallSelection[] {
  const binary = availableBinaryRecipes(catalog)
  const source = catalog.source_profiles.flatMap(profile =>
    catalog.torch_distributions
      .filter(distribution => distribution.os === profile.os)
      .filter(distribution => {
        const family = distribution.compute.startsWith('cuda-') ? 'cuda' : distribution.compute
        return profile.native_backend_by_compute[family as 'cpu' | 'cuda' | 'rocm'] !== undefined
      })
      .flatMap(distribution => languageOrder.map(language => ({
        os: profile.os,
        method: profile.method,
        torch: distribution.torch,
        compute: distribution.compute,
        language,
      }))),
  )
  return [
    ...binary.map(({ os, method, torch, compute }) => ({ os, method, torch, compute, language: 'python' as const })),
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
          : axis === 'language' ? languageOrder
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
    const choices = optionsFor(key, next, catalog)
    const candidate = choices.includes(selection[key]) ? selection[key] : choices[0]
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

function binaryCommand(
  selection: InstallSelection,
  binary: BinaryRecipe,
): string {
  if (selection.os === 'windows-x86_64') {
    return `python -m pip install --only-binary=fhelium \`
  --extra-index-url ${binary.simple_index_url} \`
  "fhelium==${binary.fhelium_version}"`
  }
  return `python -m pip install --only-binary=fhelium \\
  --extra-index-url ${binary.simple_index_url} \\
  "fhelium==${binary.fhelium_version}"`
}

function sourcePipCommand(
  selection: InstallSelection,
  backend: 'CPU' | 'CPU+CUDA' | 'CPU+HIP',
  version: string,
): string {
  if (selection.os === 'windows-x86_64') {
    return `$env:CMAKE_ARGS = "-DFHELIUM_NATIVE_BACKENDS=${backend}"
python -m pip install --no-binary=fhelium \`
  --no-build-isolation --no-cache-dir --verbose \`
  "fhelium==${version}"`
  }
  return `CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=${backend}" \\
  python -m pip install --no-binary=fhelium \\
    --no-build-isolation --no-cache-dir --verbose "fhelium==${version}"`
}

function sourceGitHubCommand(
  selection: InstallSelection,
  backend: 'CPU' | 'CPU+CUDA' | 'CPU+HIP',
  version: string,
): string {
  if (selection.os === 'windows-x86_64') {
    return `git clone --branch "v${version}" --depth 1 https://github.com/VisualDust/fhelium.git
Set-Location fhelium
$env:CMAKE_ARGS = "-DFHELIUM_NATIVE_BACKENDS=${backend}"
python -m pip install . \`
  --no-build-isolation --no-cache-dir --verbose`
  }
  return `git clone --branch "v${version}" --depth 1 https://github.com/VisualDust/fhelium.git
cd fhelium
CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=${backend}" \\
  python -m pip install . --no-build-isolation --no-cache-dir --verbose`
}

export function resolveInstall(
  selection: InstallSelection,
  catalog: InstallCatalog = installCatalog,
): ResolvedInstall {
  const resolved = resolvePackageInstall(selection, catalog)
  if (selection.language === 'python') return resolved
  return { ...resolved, commands: [...resolved.commands, ...libraryCommands(selection)] }
}

function resolvePackageInstall(
  selection: InstallSelection,
  catalog: InstallCatalog,
): ResolvedInstall {
  const torch = distributionFor(selection, catalog)
  if (selection.method === 'prebuilt-pip') {
    const recipes = availableBinaryRecipes(catalog).filter(recipe =>
      recipeAxes.every(axis => recipe[axis] === selection[axis]),
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
        binaryCommand(selection, binary),
      ],
    }
  }

  const profiles = catalog.source_profiles.filter(profile =>
    profile.os === selection.os && profile.method === selection.method,
  )
  if (profiles.length !== 1) throw new Error(`Expected one source profile, found ${profiles.length}`)
  const source = profiles[0]
  const family = selection.compute.startsWith('cuda-') ? 'cuda' : selection.compute
  const backend = source.native_backend_by_compute[family as 'cpu' | 'cuda' | 'rocm']
  if (backend === undefined) throw new Error('Selected source profile has no matching backend')
  if (selection.method === 'source-local') {
    return {
      selection, torch, source,
      commands: [
        'python -m pip install "scikit-build-core==1.0.3" "cmake>=3.21" ninja',
        'CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU+HIP" python -m pip install . --no-build-isolation --no-cache-dir --verbose',
      ],
    }
  }
  return {
    selection,
    torch,
    source,
    commands: [
      torchCommand(torch),
      `python -m pip install "scikit-build-core>=1.0.3" "cmake>=3.18" ninja`,
      selection.method === 'source-pip'
        ? sourcePipCommand(selection, backend, catalog.fhelium_version)
        : sourceGitHubCommand(selection, backend, catalog.fhelium_version),
    ],
  }
}

const cmakePrefixScript =
  'import fhelium.native, torch.utils; print(fhelium.native.cmake_prefix_path + \';\' + torch.utils.cmake_prefix_path)'

// The installed fhelium package contains libfhelium, its headers, and the
// FHElium CMake package; these commands configure a CMake project against it.
export function libraryCommands(selection: InstallSelection): readonly string[] {
  const cuda = selection.compute.startsWith('cuda-')
  if (selection.os === 'windows-x86_64') {
    return [
      '# Configure a CMake project that calls find_package(FHElium CONFIG REQUIRED)',
      `$FHEliumPrefix = python -c "${cmakePrefixScript}"`,
      cuda
        ? 'cmake -S . -B build -DCMAKE_PREFIX_PATH="$FHEliumPrefix" `\n  -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\\bin\\nvcc.exe"'
        : 'cmake -S . -B build -DCMAKE_PREFIX_PATH="$FHEliumPrefix"',
      'cmake --build build --config Release',
    ]
  }
  const script = cmakePrefixScript.replace(/'/gu, '"')
  return [
    '# Configure a CMake project that calls find_package(FHElium CONFIG REQUIRED)',
    `FHELIUM_PREFIX="$(python -c '${script}')"`,
    cuda
      ? 'cmake -S . -B build -DCMAKE_PREFIX_PATH="$FHELIUM_PREFIX" \\\n  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"'
      : 'cmake -S . -B build -DCMAKE_PREFIX_PATH="$FHELIUM_PREFIX"',
    'cmake --build build',
  ]
}

export function validateCatalog(catalog: InstallCatalog = installCatalog): void {
  if (catalog.schema_version !== 1) throw new Error('Unsupported install catalog schema')
  const binaryKeys = new Set<string>()
  for (const recipe of catalog.binary_recipes) {
    const key = recipeAxes.map(axis => recipe[axis]).join('|')
    if (binaryKeys.has(key)) throw new Error(`Duplicate binary recipe: ${key}`)
    binaryKeys.add(key)
  }
  for (const candidate of candidateSelections(catalog)) resolveInstall(candidate, catalog)
}

validateCatalog()
