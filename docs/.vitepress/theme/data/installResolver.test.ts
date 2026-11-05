import assert from 'node:assert/strict'

import {
  installCatalog,
  type InstallCatalog,
  type InstallSelection,
} from './installSelector'
import {
  changeSelection,
  defaultSelection,
  optionsFor,
  resolveInstall,
  validateCatalog,
} from './installResolver'

const windowsCatalog: InstallCatalog = {
  schema_version: 1,
  fhelium_version: '0.20.0',
  torch_distributions: [
    {
      os: 'windows-x86_64',
      torch: '2.13',
      compute: 'cuda-130',
      requirement: 'torch==2.13.0+cu130',
      index_url: 'https://download.pytorch.org/whl/cu130',
    },
    {
      os: 'windows-x86_64',
      torch: '2.13',
      compute: 'cpu',
      requirement: 'torch==2.13.0+cpu',
      index_url: 'https://download.pytorch.org/whl/cpu',
    },
  ],
  binary_recipes: [
    {
      os: 'windows-x86_64',
      method: 'prebuilt-pip',
      torch: '2.13',
      compute: 'cuda-130',
      configuration: 'torch213-cu130',
      fhelium_version: '0.20.0',
      simple_index_url: 'https://download.example/torch213-cu130/simple/',
      published: false,
    },
    {
      os: 'windows-x86_64',
      method: 'prebuilt-pip',
      torch: '2.13',
      compute: 'cpu',
      configuration: 'torch213-cpu',
      fhelium_version: '0.20.0',
      simple_index_url: 'https://download.example/torch213-cpu/simple/',
      published: true,
    },
  ],
  source_profiles: [
    {
      os: 'windows-x86_64',
      method: 'source-pip',
      native_backend_by_compute: {
        cpu: 'CPU',
        cuda: 'CPU+CUDA',
      },
    },
    {
      os: 'windows-x86_64',
      method: 'source-github',
      native_backend_by_compute: {
        cpu: 'CPU',
        cuda: 'CPU+CUDA',
      },
    },
  ],
}

validateCatalog(windowsCatalog)
assert.deepEqual(optionsFor('os', {}, windowsCatalog), ['windows-x86_64'])
assert.deepEqual(
  optionsFor('method', { os: 'windows-x86_64' }, windowsCatalog),
  ['prebuilt-pip', 'source-pip', 'source-github'],
)
assert.deepEqual(
  optionsFor(
    'compute',
    {
      os: 'windows-x86_64',
      method: 'prebuilt-pip',
      torch: '2.13',
    },
    windowsCatalog,
  ),
  ['cuda-130', 'cpu'],
)

const sourceSelection: InstallSelection = {
  os: 'windows-x86_64',
  method: 'source-pip',
  torch: '2.13',
  compute: 'cuda-130',
}
const source = resolveInstall(sourceSelection, windowsCatalog)
assert.match(source.commands[2], /^\$env:CMAKE_ARGS = /u)
assert.match(source.commands[2], /`\n  --no-build-isolation/u)
assert.doesNotMatch(source.commands[2], /CMAKE_ARGS=.*\\\n/u)

const github = resolveInstall(
  { ...sourceSelection, method: 'source-github' },
  windowsCatalog,
)
assert.match(github.commands[2], /Set-Location fhelium/u)
assert.match(github.commands[2], /python -m pip install \. `\n/u)

const publishedWindowsCatalog: InstallCatalog = {
  ...windowsCatalog,
  binary_recipes: windowsCatalog.binary_recipes.map(recipe => ({
    ...recipe,
    published: true,
  })),
}
validateCatalog(publishedWindowsCatalog)
const binary = resolveInstall(
  { ...sourceSelection, method: 'prebuilt-pip' },
  publishedWindowsCatalog,
)
assert.match(binary.commands[1], /--only-binary=fhelium `\n/u)
assert.match(binary.commands[1], /--extra-index-url/u)

const linux = resolveInstall(
  {
    os: 'linux-x86_64',
    method: 'prebuilt-pip',
    torch: '2.13',
    compute: 'cuda-130',
  },
  installCatalog,
)
assert.match(linux.commands[1], /--only-binary=fhelium/u)
assert.match(linux.commands[1], /--extra-index-url/u)
assert.doesNotMatch(linux.commands[1], /\$env:|`/u)

// Compute selection determines the available package methods.
const amd = changeSelection(defaultSelection(), 'compute', 'rocm')
assert.equal(amd.method, 'source-local')
assert.deepEqual(optionsFor('method', amd), ['source-local'])
const amdCommands = resolveInstall(amd).commands.join('\n')
assert.match(amdCommands, /FHELIUM_NATIVE_BACKENDS=CPU\+HIP/u)
assert.match(amdCommands, /pip install \. --no-build-isolation --no-cache-dir/u)
assert.doesNotMatch(amdCommands, /--only-binary|--extra-index-url|fhelium==|git clone/u)
assert.ok(!optionsFor('compute', { os: 'windows-x86_64' }).includes('rocm'))
assert.ok(!optionsFor('compute', { os: 'macos-arm64' }).includes('rocm'))
const backToCuda = changeSelection(amd, 'compute', 'cuda-130')
assert.equal(backToCuda.method, 'prebuilt-pip')
assert.equal(backToCuda.torch, '2.13')
assert.match(resolveInstall(backToCuda).commands[1], /--only-binary=fhelium/u)
