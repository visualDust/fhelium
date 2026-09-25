import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import mathjax3 from 'markdown-it-mathjax3'
import { defineConfig } from 'vitepress'

const repository = 'https://github.com/VisualDust/fhelium'
const siteOrigin = process.env.DOCS_ORIGIN ?? 'https://fhelium.550w.host'
const base = process.env.DOCS_BASE ?? '/'

const apiReferencePath = fileURLToPath(
  new URL('./api-reference.json', import.meta.url),
)
const apiSidebarPath = fileURLToPath(
  new URL('./api-sidebar.json', import.meta.url),
)
const apiReferences = JSON.parse(
  readFileSync(apiReferencePath, 'utf8'),
) as Record<string, string>
const apiSidebar = JSON.parse(readFileSync(apiSidebarPath, 'utf8'))

function installApiReferences(md: any) {
  md.core.ruler.after('normalize', 'fhelium-api-reference', (state: any) => {
    const lines = state.src.split('\n')
    const expanded: string[] = []

    for (let index = 0; index < lines.length; index += 1) {
      const match = /^:::\s+(fhelium(?:\.[A-Za-z_]\w*)*)\s*$/.exec(
        lines[index],
      )
      if (!match) {
        expanded.push(lines[index])
        continue
      }

      const reference = match[1]
      const generated = apiReferences[reference]
      if (generated === undefined) {
        throw new Error(`No generated API reference found for ${reference}`)
      }

      expanded.push('', generated, '')
      while (
        index + 1 < lines.length &&
        lines[index + 1].trim() !== '' &&
        /^[ \t]/.test(lines[index + 1])
      ) {
        index += 1
      }
    }

    state.src = expanded.join('\n')
  })
}

function installMermaid(md: any) {
  const renderFence = md.renderer.rules.fence.bind(md.renderer.rules)

  md.renderer.rules.fence = (
    tokens: any[],
    index: number,
    options: any,
    env: any,
    self: any,
  ) => {
    const token = tokens[index]
    if (token.info.trim().split(/\s+/u)[0] === 'mermaid') {
      return `<MermaidDiagram code="${encodeURIComponent(token.content)}" />`
    }
    return renderFence(tokens, index, options, env, self)
  }
}

const learningSidebar = [
  {
    text: 'Start here',
    items: [
      { text: 'Overview', link: '/tutorial/' },
      { text: 'Installation', link: '/tutorial/installation' },
      { text: 'Tutorials', link: '/tutorial/tutorials' },
    ],
  },
  {
    text: 'Eager',
    collapsed: false,
    items: [
      {
        text: '01 - Basic CKKS workflow',
        link: '/tutorial/basic-ckks-workflow',
      },
      {
        text: '02 - Key creation and installation',
        link: '/tutorial/key-materials',
      },
      {
        text: '03 - Modulus-chain depth',
        link: '/tutorial/modulus-chain-depth',
      },
      {
        text: '04 - Actual scale management',
        link: '/tutorial/explicit-scale-management',
      },
      {
        text: '05 - NTT reuse and late relinearization',
        link: '/tutorial/late-relinearization-and-ntt-reuse',
      },
      {
        text: '06 - Rotation hoisting',
        link: '/tutorial/rotation-hoisting',
      },
      {
        text: '07 - Homogeneous batching',
        link: '/tutorial/homogeneous-batching',
      },
      {
        text: '08 - Compressed plaintexts',
        link: '/tutorial/compressed-plaintext',
      },
    ],
  },
  {
    text: 'Values, Serialization, and Artifacts',
    collapsed: false,
    items: [
      {
        text: '09 - Value movement and files',
        link: '/tutorial/value-memory-and-persistence',
      },
      {
        text: '10 - Named artifacts and generations',
        link: '/tutorial/artifact-store',
      },
    ],
  },
  {
    text: 'Compile',
    collapsed: false,
    items: [
      {
        text: '11 - JIT compilation',
        link: '/tutorial/compile-jit',
      },
      {
        text: '12 - Caller-composed Compile pipeline',
        link: '/tutorial/compose-and-execute-compile-pipeline',
      },
      {
        text: '13 - Textual Program IR',
        link: '/tutorial/ir-textual-program',
      },
      {
        text: '14 - Custom BSGS transformation',
        link: '/tutorial/customize-compile-pass-and-pipeline',
      },
      {
        text: '15 - Generated Python',
        link: '/tutorial/generate-python',
      },
      {
        text: '16 - Program materials and persistence',
        link: '/tutorial/compile-material-persistence',
      },
    ],
  },
  {
    text: 'Runtime',
    collapsed: false,
    items: [
      {
        text: '17 - Double-buffered execution',
        link: '/tutorial/reusable-value-buffer',
      },
      {
        text: '18 - CUDA Graph replay',
        link: '/tutorial/cuda-graph-matvec',
      },
    ],
  },
  {
    text: 'Residency',
    collapsed: false,
    items: [
      {
        text: '19 - Manual Residency',
        link: '/tutorial/explicit-residency',
      },
      {
        text: '20 - Automatic Residency admission',
        link: '/tutorial/automatic-residency',
      },
    ],
  },
  {
    text: 'Distributed',
    collapsed: false,
    items: [
      {
        text: '21 - Data-parallel encrypted batches',
        link: '/tutorial/spmd-independent-ciphertexts',
      },
      {
        text: '22 - Additive partial results',
        link: '/tutorial/spmd-rotation-parallel-matvec',
      },
      {
        text: '23 - RNS-sharded execution',
        link: '/tutorial/spmd-limb-parallel-pipeline',
      },
      {
        text: '24 - Rank-local collective IR',
        link: '/tutorial/rank-local-collective-ir',
      },
    ],
  },
  {
    text: 'Experimental',
    collapsed: false,
    items: [
      {
        text: '25 - Experimental bootstrapping',
        link: '/tutorial/composable-ckks-bootstrap',
      },
      {
        text: '26 - Experimental multiparty CKKS',
        link: '/tutorial/multiparty-ckks',
      },
    ],
  },
]

const howToSidebar = [
  {
    text: 'How-to guides',
    items: [
      { text: 'Overview', link: '/how-to/' },
    ],
  },
  {
    text: 'Describe and execute calculations',
    items: [
      { text: 'Evaluate CKKS data eagerly', link: '/how-to/evaluate-ckks-data' },
      { text: 'Build a Program and pipeline', link: '/how-to/build-program-pipeline' },
      { text: 'Compile a callable', link: '/how-to/compile-callable' },
      { text: 'Write a Compile pass', link: '/how-to/write-compilation-pass' },
      { text: 'Inspect Program transformations', link: '/how-to/visualize-mixed-level-ir' },
    ],
  },
  {
    text: 'Parameters and implementations',
    items: [
      { text: 'Choose parameters and depth', link: '/how-to/choose-preset-and-depth' },
      { text: 'Provision evaluation keys', link: '/how-to/provision-keyset' },
      { text: 'Select an operation implementation', link: '/how-to/select-operation-implementation' },
      { text: 'Choose an NTT implementation', link: '/how-to/choose-ntt-backend' },
    ],
  },
  {
    text: 'Materials and deployment',
    items: [
      { text: 'Bind and persist a compiled Program', link: '/how-to/persist-compiled-program' },
      { text: 'Manage named artifacts', link: '/how-to/manage-artifacts' },
      { text: 'Place computation on CPU or CUDA', link: '/how-to/switch-cpu-cuda' },
    ],
  },
  {
    text: 'Repeated execution and memory',
    items: [
      { text: 'Capture repeated execution', link: '/how-to/capture-repeated-evaluator' },
      { text: 'Choose Residency controls', link: '/how-to/choose-residency-control-level' },
      { text: 'Stream within a memory budget', link: '/how-to/stream-bounded-memory' },
      { text: 'Diagnose a Residency failure', link: '/how-to/diagnose-residency-failure' },
    ],
  },
  {
    text: 'Distributed execution',
    items: [
      { text: 'Partition work across GPUs', link: '/how-to/choose-multi-gpu-partition' },
      { text: 'Diagnose a distributed hang', link: '/how-to/diagnose-distributed-hang' },
    ],
  },
  {
    text: 'Diagnosis and performance',
    items: [
      { text: 'Diagnose value-state errors', link: '/how-to/diagnose-value-state-mismatch' },
      { text: 'Diagnose Compile preparation', link: '/how-to/diagnose-compile-preparation' },
      { text: 'Inspect runtime and topology', link: '/how-to/inspect-runtime-and-cuda' },
      { text: 'Profile and optimize a workload', link: '/how-to/optimize-workload' },
      { text: 'Choose a batch size', link: '/how-to/choose-homogeneous-batch-size' },
      { text: 'Measure NTT candidates', link: '/how-to/screen-ntt-backends' },
      { text: 'Run and submit a benchmark', link: '/benchmarks/run-and-submit' },
    ],
  },
  {
    text: 'Experimental',
    items: [
      { text: 'Compose a bootstrap circuit', link: '/how-to/compose-bootstrap-circuit' },
      { text: 'Implement a bootstrap component', link: '/how-to/implement-bootstrap-component' },
      { text: 'Use multiparty CKKS', link: '/how-to/use-multiparty-ckks' },
    ],
  },
]

const developerSidebar = [
  {
    text: 'Developer Guide',
    items: [
      { text: 'Overview', link: '/developer/' },
    ],
  },
  {
    text: 'Architecture overview',
    items: [
      { text: 'Execution stack', link: '/developer/engine-native-stack' },
      { text: 'Source tree', link: '/developer/source-tree' },
      { text: 'Security', link: '/developer/security' },
    ],
  },
  {
    text: 'Values and operations',
    items: [
      { text: 'Value state and Eager execution', link: '/developer/compiler-state-and-eager-execution' },
      { text: 'Operation definitions and registration', link: '/developer/operation-registration-and-selection' },
      { text: 'IR operations and implementations', link: '/developer/ir-operation-implementation-index' },
    ],
  },
  {
    text: 'Programs and Compile',
    items: [
      { text: 'IR and capture', link: '/developer/compiler-stack-internals' },
      { text: 'Compilation and passes', link: '/developer/compilation-and-passes' },
      { text: 'Materials and preparation', link: '/developer/materials-and-preparation' },
    ],
  },
  {
    text: 'Backend execution',
    items: [
      { text: 'Linking and prepared host execution', link: '/developer/prepared-host-execution' },
      { text: 'Generated kernels and fusion', link: '/developer/compiled-execution-and-kernels' },
    ],
  },
  {
    text: 'Arithmetic implementations',
    items: [
      { text: 'Encoding, randomness, and keys', link: '/developer/encoding-randomness-and-keys' },
      { text: 'RNS and NTT', link: '/developer/rns-and-ntt' },
      { text: 'Multiplication, key switching, and rescale', link: '/developer/multiplication-keyswitch-rescale' },
      { text: 'Compressed plaintext', link: '/developer/compressed-plaintext-internals' },
    ],
  },
  {
    text: 'Runtime, storage, and communication',
    items: [
      { text: 'Compilation persistence', link: '/developer/compilation-persistence' },
      { text: 'ArtifactStore', link: '/developer/artifact-store-v1' },
      { text: 'Buffers and CUDA Graphs', link: '/developer/execution-buffers-and-cuda-graphs' },
      { text: 'Residency state and ownership', link: '/developer/residency-state-and-ownership' },
      { text: 'Residency plans and execution', link: '/developer/residency-plans-and-execution' },
      { text: 'Distributed execution', link: '/developer/distributed-internals' },
    ],
  },
  {
    text: 'Contributing',
    items: [
      { text: 'Contributor guide', link: '/developer/contributing' },
      { text: 'Native operator workflow', link: '/developer/native-operator-workflow' },
      { text: 'Binary packaging', link: '/developer/binary-packaging-and-release' },
      { text: 'Documentation tooling', link: '/developer/documentation' },
    ],
  },
  {
    text: 'Experimental',
    items: [
      { text: 'CKKS bootstrapping', link: '/developer/composable-ckks-bootstrap' },
    ],
  },
]

const aboutSidebar = [
  {
    text: 'About FHElium',
    link: '/about/',
  },
  {
    text: 'Research',
    link: '/about/research',
  },
  {
    text: 'Branding',
    link: '/about/branding',
  },
]

export default defineConfig({
  lang: 'en-US',
  title: 'FHElium',
  description:
    'Cross-stack CKKS research with Python, PyTorch, CPU, and CUDA',
  base,
  cleanUrls: true,
  lastUpdated: true,
  vite: {
    server: {
      watch: {
        // Production output and Vite's dependency cache are generated trees;
        // watching them wastes inotify entries and can exhaust shared hosts.
        ignored: ['**/.vitepress/dist/**', '**/.vitepress/cache/**'],
      },
    },
    build: {
      // Mermaid diagram renderers are lazy-loaded as independent chunks.
      chunkSizeWarningLimit: 750,
    },
  },
  vue: {
    template: {
      compilerOptions: {
        // MathJax emits custom wrapper elements around its accessible SVG.
        isCustomElement: (tag) => tag.startsWith('mjx-'),
      },
    },
  },
  sitemap: {
    hostname: `${siteOrigin}${base}`,
  },
  markdown: {
    lineNumbers: true,
    config(md) {
      installApiReferences(md)
      installMermaid(md)
      md.use(mathjax3)
    },
  },
  head: [
    [
      'link',
      {
        rel: 'icon',
        type: 'image/svg+xml',
        href: `${base}brand/fhelium-mark.svg`,
      },
    ],
    [
      'meta',
      {
        name: 'theme-color',
        content: '#f7f7f2',
        media: '(prefers-color-scheme: light)',
      },
    ],
    [
      'meta',
      {
        name: 'theme-color',
        content: '#11131a',
        media: '(prefers-color-scheme: dark)',
      },
    ],
    ['meta', { name: 'color-scheme', content: 'light dark' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'FHElium' }],
    ['meta', { property: 'og:title', content: 'FHElium — FHE, built from the tensor up.' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'A tensor-native CKKS framework for fully homomorphic encryption on CPU and NVIDIA CUDA with Python and PyTorch.',
      },
    ],
    ['meta', { property: 'og:url', content: siteOrigin }],
    [
      'meta',
      {
        property: 'og:image',
        content: `${siteOrigin}${base}brand/fhelium-social-card.png`,
      },
    ],
    ['meta', { property: 'og:image:width', content: '1200' }],
    ['meta', { property: 'og:image:height', content: '630' }],
    [
      'meta',
      {
        property: 'og:image:alt',
        content:
          'FHElium — FHE, built from the tensor up. Python, PyTorch, CPU, and CUDA.',
      },
    ],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'FHElium — FHE, built from the tensor up.' }],
    [
      'meta',
      {
        name: 'twitter:description',
        content:
          'A tensor-native CKKS framework for fully homomorphic encryption on CPU and NVIDIA CUDA with Python and PyTorch.',
      },
    ],
    [
      'meta',
      {
        name: 'twitter:image',
        content: `${siteOrigin}${base}brand/fhelium-social-card.png`,
      },
    ],
  ],
  themeConfig: {
    logo: '/brand/fhelium-mark.svg',
    nav: [
      { text: 'Learn', link: '/tutorial/' },
      { text: 'Concepts', link: '/concepts/' },
      { text: 'How-to', link: '/how-to/' },
      { text: 'Benchmarks', link: '/benchmarks/' },
      { text: 'Developer', link: '/developer/' },
      { text: 'Blog', link: '/blog/' },
      { text: 'API', link: '/api/' },
      { text: 'About', link: '/about/' },
    ],
    sidebar: {
      '/benchmarks/': [{
        text: 'Benchmarks',
        items: [
          { text: 'Results', link: '/benchmarks/' },
          { text: 'Methodology', link: '/benchmarks/methodology' },
          { text: 'Run and submit', link: '/benchmarks/run-and-submit' },
        ],
      }],
      '/tutorial/': learningSidebar,
      '/concepts/': [
        {
          text: 'Foundations',
          items: [
            { text: 'Overview', link: '/concepts/' },
            {
              text: 'Terminology and mathematical model',
              link: '/concepts/terminology-and-mathematical-model',
            },
          ],
        },
        {
          text: 'Architecture',
          items: [
            {
              text: 'Architecture',
              link: '/concepts/architecture/system-overview',
            },
            {
              text: 'Ownership and responsibilities',
              link: '/concepts/architecture/ownership-and-responsibilities',
            },
          ],
        },
        {
          text: 'Programs and compilation',
          items: [
            {
              text: 'Neutral IR programs',
              link: '/concepts/neutral-ir-programs',
            },
            {
              text: 'Open compiler stack',
              link: '/concepts/open-compiler-stack',
            },
          ],
        },
        {
          text: 'CKKS semantics',
          items: [
            {
              text: 'Context and modulus chain',
              link: '/concepts/ckks/context-and-modulus-chain',
            },
            {
              text: 'Scale, depth, and RNS format',
              link: '/concepts/ckks/scale-depth-and-execution-format',
            },
            {
              text: 'Value model and identity',
              link: '/concepts/ckks/value-model-and-identity',
            },
            {
              text: 'State transitions and orthogonality',
              link: '/concepts/ckks/state-transitions-and-orthogonality',
            },
            {
              text: 'Scale and depth lifecycle',
              link: '/concepts/ckks/scale-and-depth-lifecycle',
            },
            {
              text: 'Evaluator operation transitions',
              link: '/concepts/ckks/evaluator-operation-transitions',
            },
            {
              text: 'Key lifecycle',
              link: '/concepts/ckks/key-lifecycle',
            },
          ],
        },
        {
          text: 'Distributed execution',
          items: [
            {
              text: 'Rank-local SPMD model',
              link: '/concepts/distributed/spmd-model',
            },
            {
              text: 'Communication semantics',
              link: '/concepts/distributed/communication-semantics',
            },
          ],
        },
        {
          text: 'Execution and lifecycle',
          items: [
            {
              text: 'Value signatures and buffers',
              link: '/concepts/execution/signatures-and-buffers',
            },
            {
              text: 'CUDA Graph model',
              link: '/concepts/execution/cuda-graph-model',
            },
            {
              text: 'Serialization and artifacts',
              link: '/concepts/execution/serialization-and-artifacts',
            },
            {
              text: 'Residency lifetimes',
              link: '/concepts/execution/residency-lifetimes',
            },
          ],
        },
        {
          text: 'Advanced CKKS mechanisms',
          items: [
            {
              text: 'Composable CKKS bootstrapping',
              link: '/concepts/ckks/composable-bootstrapping',
            },
          ],
        },
        {
          text: 'Performance',
          items: [
            {
              text: 'CKKS workload cost model',
              link: '/concepts/performance/cost-model',
            },
          ],
        },
      ],
      '/how-to/': howToSidebar,
      '/developer/': developerSidebar,
      '/api/': apiSidebar,
      '/about/': aboutSidebar,
    },
    search: {
      provider: 'local',
    },
    outline: {
      level: [2, 3],
      label: 'On this page',
    },
    editLink: {
      pattern: `${repository}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub',
    },
    socialLinks: [{ icon: 'github', link: repository }],
    externalLinkIcon: true,
    docFooter: {
      prev: 'Previous page',
      next: 'Next page',
    },
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'medium',
        timeStyle: 'short',
      },
    },
    returnToTopLabel: 'Back to top',
    sidebarMenuLabel: 'Menu',
    darkModeSwitchLabel: 'Appearance',
    footer: {
      message: 'Towards full-stack encrypted execution infrastructures',
      copyright: 'Released under the MIT License. Copyright <a href="https://github.com/VisualDust" target="_blank" rel="noopener noreferrer">VisualDust</a> and <a href="https://github.com/VisualDust/fhelium/graphs/contributors" target="_blank" rel="noopener noreferrer">FHElium contributors</a>',
    },
  },
})
