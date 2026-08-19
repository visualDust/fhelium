import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

import { defineLoader } from 'vitepress'

import { parseBenchmarkCatalog } from './catalog'

const catalogPath = fileURLToPath(
  new URL('../../../../../public/benchmarks/v1/catalog.json', import.meta.url),
)

/**
 * Load the compact, curated catalog during the VitePress build.
 *
 * This loader intentionally does not open raw run files, import FHElium or
 * CUDA modules, make network requests, or execute a benchmark.
 */
export default defineLoader({
  watch: ['../../../../../public/benchmarks/v1/catalog.json'],
  async load() {
    const contents = await readFile(catalogPath, 'utf8')
    let value: unknown
    try {
      value = JSON.parse(contents)
    } catch (error) {
      throw new Error(`Benchmark catalog is not valid JSON: ${String(error)}`)
    }
    return parseBenchmarkCatalog(value)
  },
})
