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
      {
        text: 'Support and security scope',
        link: '/tutorial/support-and-security',
      },
      { text: 'Tutorials', link: '/tutorial/tutorials' },
    ],
  },
  {
    text: 'CKKS',
    items: [
      {
        text: '01 - Basic CKKS workflow',
        link: '/tutorial/basic-ckks-workflow',
      },
      {
        text: '02 - Key material lifecycle',
        link: '/tutorial/key-materials',
      },
      {
        text: '04 - Modulus-chain depth',
        link: '/tutorial/modulus-chain-depth',
      },
      {
        text: '05 - Explicit scale management',
        link: '/tutorial/explicit-scale-management',
      },
      {
        text: '06 - Late relinearization and NTT reuse',
        link: '/tutorial/late-relinearization-and-ntt-reuse',
      },
    ],
  },
  {
    text: 'Performance',
    items: [
      {
        text: '07 - Rotation hoisting',
        link: '/tutorial/rotation-hoisting',
      },
    ],
  },
  {
    text: 'Distributed execution',
    collapsed: false,
    items: [
      {
        text: '08 - Independent ciphertexts',
        link: '/tutorial/spmd-independent-ciphertexts',
      },
      {
        text: '09 - Rotation-parallel matrix-vector',
        link: '/tutorial/spmd-rotation-parallel-matvec',
      },
      {
        text: '10 - Limb-parallel pipeline',
        link: '/tutorial/spmd-limb-parallel-pipeline',
      },
    ],
  },
  {
    text: 'Execution and lifecycle',
    collapsed: false,
    items: [
      {
        text: '03 - Values, memory, and persistence',
        link: '/tutorial/value-memory-and-persistence',
      },
      {
        text: '11 - CUDA Graph matrix-vector',
        link: '/tutorial/cuda-graph-matvec',
      },
      {
        text: '12 - Reusable value buffers',
        link: '/tutorial/reusable-value-buffer',
      },
      {
        text: '13 - Explicit residency plans and CUDA leases',
        link: '/tutorial/explicit-residency',
      },
      {
        text: '14 - Automatic residency admission',
        link: '/tutorial/automatic-residency',
      },
      {
        text: '15 - Homogeneous batching',
        link: '/tutorial/homogeneous-batching',
      },
      {
        text: '16 - Compressed plaintexts',
        link: '/tutorial/compressed-plaintext',
      },
    ],
  },
  {
    text: 'Features',
    collapsed: false,
    items: [
      {
        text: '17 - Refresh with composable bootstrapping',
        link: '/tutorial/composable-ckks-bootstrap',
      },
      {
        text: '18 - Synthetic multiparty CKKS',
        link: '/tutorial/multiparty-ckks',
      },
      {
        text: '19 - JIT programs',
        link: '/tutorial/unified-jit',
      },
      {
        text: '20 - JIT textual IR',
        link: '/tutorial/jit-textual-ir',
      },
      {
        text: '21 - JIT custom pipelines',
        link: '/tutorial/jit-custom-pipeline',
      },
    ],
  },
]

const developerSidebar = [
  {
    text: 'Developer Guide',
    items: [{ text: 'Overview', link: '/developer/' }],
  },
  {
    text: 'Architecture and native execution',
    items: [
      { text: 'Source tree', link: '/developer/source-tree' },
      {
        text: 'Python-to-native stack',
        link: '/developer/engine-native-stack',
      },
    ],
  },
  {
    text: 'Arithmetic internals',
    items: [
      {
        text: 'RNS and NTT',
        link: '/developer/rns-and-ntt',
      },
      {
        text: 'Multiplication, key switching, and rescale',
        link: '/developer/multiplication-keyswitch-rescale',
      },
      {
        text: 'CompressedPlaintext internals',
        link: '/developer/compressed-plaintext-internals',
      },
      {
        text: 'CKKS bootstrapping',
        link: '/developer/composable-ckks-bootstrap',
      },
    ],
  },
  {
    text: 'Distributed and execution',
    items: [
      {
        text: 'Distributed internals',
        link: '/developer/distributed-internals',
      },
      {
        text: 'Buffers and CUDA Graphs',
        link: '/developer/execution-buffers-and-cuda-graphs',
      },
    ],
  },
  {
    text: 'Just In Time',
    items: [
      {
        text: 'JIT internals',
        link: '/developer/unified-jit-internals',
      },
    ],
  },
  {
    text: 'Storage and residency',
    items: [
      {
        text: 'ArtifactStore internals',
        link: '/developer/artifact-store-v1',
      },
      {
        text: 'Residency state and ownership',
        link: '/developer/residency-state-and-ownership',
      },
      {
        text: 'Residency plans and execution',
        link: '/developer/residency-plans-and-execution',
      },
    ],
  },
  {
    text: 'Contributing',
    items: [
      { text: 'Contributor guide', link: '/developer/contributing' },
      {
        text: 'Mathematical and state invariants',
        link: '/developer/mathematical-notation-and-invariants',
      },
      {
        text: 'Native operator workflow',
        link: '/developer/native-operator-workflow',
      },
      {
        text: 'Documentation guide',
        link: '/developer/documentation',
      },
      {
        text: 'Binary packaging and release',
        link: '/developer/binary-packaging-and-release',
      },
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
      '/tutorial/': learningSidebar,
      '/concepts/': [
        {
          text: 'Concepts',
          items: [
            { text: 'Overview', link: '/concepts/' },
            {
              text: 'Programming model at a glance',
              link: '/concepts/programming-model',
            },
          ],
        },
        {
          text: 'Architecture',
          items: [
            {
              text: 'System overview',
              link: '/concepts/architecture/system-overview',
            },
            {
              text: 'Ownership and responsibilities',
              link: '/concepts/architecture/ownership-and-responsibilities',
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
              text: 'Value model and identity',
              link: '/concepts/ckks/value-model-and-identity',
            },
            {
              text: 'State transitions and orthogonality',
              link: '/concepts/ckks/state-transitions-and-orthogonality',
            },
            {
              text: 'Scale and level lifecycle',
              link: '/concepts/ckks/scale-and-level-lifecycle',
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
              text: 'Choose and switch CPU or CUDA',
              link: '/how-to/switch-cpu-cuda',
            },
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
          text: 'Features',
          items: [
            {
              text: 'Composable CKKS bootstrapping',
              link: '/concepts/ckks/composable-bootstrapping',
            },
            {
              text: 'JIT programs',
              link: '/concepts/unified-jit-programs',
            },
          ],
        },
        {
          text: 'Performance and terms',
          items: [
            {
              text: 'CKKS workload cost model',
              link: '/concepts/performance/cost-model',
            },
            { text: 'Glossary', link: '/concepts/glossary' },
          ],
        },
      ],
      '/how-to/': [
        {
          text: 'How-to guides',
          items: [{ text: 'Overview', link: '/how-to/' }],
        },
        {
          text: 'CKKS',
          items: [
            {
              text: 'Choose a preset and chain depth',
              link: '/how-to/choose-preset-and-depth',
            },
            {
              text: 'Provision the minimum keyset',
              link: '/how-to/provision-keyset',
            },
            {
              text: 'Diagnose a value-state mismatch',
              link: '/how-to/diagnose-value-state-mismatch',
            },
          ],
        },
        {
          text: 'Distributed execution',
          items: [
            {
              text: 'Choose a multi-GPU partition',
              link: '/how-to/choose-multi-gpu-partition',
            },
            {
              text: 'Diagnose a distributed hang',
              link: '/how-to/diagnose-distributed-hang',
            },
          ],
        },
        {
          text: 'Experimental public interfaces',
          items: [
            {
              text: 'Compose a bootstrap callable',
              link: '/how-to/compose-bootstrap-circuit',
            },
            {
              text: 'Implement a bootstrap component',
              link: '/how-to/implement-bootstrap-component',
            },
            {
              text: 'Use multiparty CKKS',
              link: '/how-to/use-multiparty-ckks',
            },
            {
              text: 'Visualize and inspect a JIT Program',
              link: '/how-to/visualize-jit-program',
            },
          ],
        },
        {
          text: 'Execution and lifecycle',
          items: [
            {
              text: 'Manage artifacts',
              link: '/how-to/manage-artifacts',
            },
            {
              text: 'Capture a repeated evaluator',
              link: '/how-to/capture-repeated-evaluator',
            },
            {
              text: 'Choose a Residency control level',
              link: '/how-to/choose-residency-control-level',
            },
            {
              text: 'Stream bounded CUDA memory',
              link: '/how-to/stream-bounded-memory',
            },
            {
              text: 'Diagnose a Residency failure',
              link: '/how-to/diagnose-residency-failure',
            },
          ],
        },
        {
          text: 'Performance',
          items: [
            {
              text: 'Inspect runtime and CUDA topology',
              link: '/how-to/inspect-runtime-and-cuda',
            },
            {
              text: 'Screen NTT backends',
              link: '/how-to/screen-ntt-backends',
            },
            {
              text: 'Analyze and choose an NTT backend',
              link: '/how-to/choose-ntt-backend',
            },
            {
              text: 'Benchmark a workload correctly',
              link: '/how-to/benchmark-a-workload',
            },
            {
              text: 'Choose a homogeneous batch size',
              link: '/how-to/choose-homogeneous-batch-size',
            },
            {
              text: 'Optimize a workload systematically',
              link: '/how-to/optimize-workload',
            },
          ],
        },
      ],
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
      copyright: 'Released under the MIT License. Copyright VisualDust and FHElium contributors',
    },
  },
})
