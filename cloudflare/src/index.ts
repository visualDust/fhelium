interface Env {
  FHELIUM_RELEASES: R2Bucket
}

const HTML_MEDIA_TYPE = 'text/html'
const SIMPLE_HTML_MEDIA_TYPE = 'application/vnd.pypi.simple.v1+html'
const JSON_MEDIA_TYPE = 'application/vnd.pypi.simple.v1+json'
const READ_METHODS = new Set(['GET', 'HEAD'])

function acceptsJson(request: Request): boolean {
  const accept = request.headers.get('Accept') ?? ''
  return accept
    .split(',')
    .map(item => item.split(';', 1)[0].trim().toLowerCase())
    .includes(JSON_MEDIA_TYPE)
}

function objectKey(request: Request): { key: string; negotiated: boolean } | null {
  const url = new URL(request.url)
  let path: string
  try {
    path = decodeURIComponent(url.pathname)
  }
  catch {
    return null
  }
  if (path.includes('\\') || path.split('/').includes('..')) return null

  const normalized = path.replace(/^\/+/, '')
  if (path.endsWith('/')) {
    return {
      key: `${normalized}${acceptsJson(request) ? 'index.json' : 'index.html'}`,
      negotiated: true,
    }
  }
  return { key: normalized, negotiated: false }
}

function contentType(
  key: string,
  object: R2ObjectBody,
  negotiated: boolean,
): string {
  if (negotiated && key.endsWith('index.json')) {
    return `${JSON_MEDIA_TYPE}; charset=utf-8`
  }
  if (negotiated && key.endsWith('index.html')) {
    return `${SIMPLE_HTML_MEDIA_TYPE}; charset=utf-8`
  }
  const stored = object.httpMetadata?.contentType
  if (stored) return stored
  if (key.endsWith('.html')) return `${HTML_MEDIA_TYPE}; charset=utf-8`
  if (key.endsWith('.json')) return `${JSON_MEDIA_TYPE}; charset=utf-8`
  if (key.endsWith('.whl') || key.endsWith('.tar.gz')) return 'application/octet-stream'
  return 'application/octet-stream'
}

function responseHeaders(
  key: string,
  object: R2ObjectBody,
  negotiated: boolean,
): Headers {
  const headers = new Headers()
  object.writeHttpMetadata(headers)
  headers.set('Content-Type', contentType(key, object, negotiated))
  headers.set('Content-Length', object.size.toString())
  headers.set('ETag', object.httpEtag)
  headers.set('X-Content-Type-Options', 'nosniff')
  if (negotiated) headers.set('Vary', 'Accept')
  if (!headers.has('Cache-Control')) {
    headers.set(
      'Cache-Control',
      key.includes('/artifacts/') || key.startsWith('artifacts/')
        ? 'public, max-age=31536000, immutable'
        : 'public, max-age=60, must-revalidate',
    )
  }
  return headers
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (!READ_METHODS.has(request.method)) {
      return new Response('Method Not Allowed\n', {
        status: 405,
        headers: { Allow: 'GET, HEAD' },
      })
    }

    const resolved = objectKey(request)
    if (resolved === null || resolved.key === '') {
      return new Response('Not Found\n', {
        status: 404,
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      })
    }

    const onlyIf = request.headers.has('If-None-Match')
      || request.headers.has('If-Modified-Since')
      ? request.headers
      : undefined
    const object = await env.FHELIUM_RELEASES.get(resolved.key, { onlyIf })
    if (object === null) {
      return new Response('Not Found\n', {
        status: 404,
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      })
    }
    if (!('body' in object)) return new Response(null, { status: 304 })

    const headers = responseHeaders(resolved.key, object, resolved.negotiated)
    return new Response(request.method === 'HEAD' ? null : object.body, {
      status: 200,
      headers,
    })
  },
} satisfies ExportedHandler<Env>
