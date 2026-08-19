import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

import { parseBenchmarkCatalog } from '../../../.vitepress/theme/benchmarks/v1/data/catalog'

const catalogPath = fileURLToPath(
  new URL('../../../public/benchmarks/v1/catalog.json', import.meta.url),
)

export default {
  async paths() {
    const contents = await readFile(catalogPath, 'utf8')
    const catalog = parseBenchmarkCatalog(JSON.parse(contents) as unknown)
    const digests = [...new Set(catalog.runs.map((run) => run.raw_sha256))]
    return digests.map((digest) => ({
      params: { slug: `sha256-${digest}` },
      content: `<BenchmarkV1ResultDetail report-slug="sha256-${digest}" />`,
    }))
  },
}
