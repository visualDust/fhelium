const PYTHON_TOKENS = /(#.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(import|from|as|def|return|for|in|if|else|True|False|None)\b|\b(\d+(?:\.\d+)?)\b|\b([A-Z][A-Za-z0-9_]*)(?=\()|\b([a-z_][A-Za-z0-9_]*)(?=\()/gm

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

/** Highlight a trusted, static Python example without changing its text. */
export function highlightPython(source: string): string {
  let cursor = 0
  let highlighted = ''

  for (const match of source.matchAll(PYTHON_TOKENS)) {
    const index = match.index ?? cursor
    highlighted += escapeHtml(source.slice(cursor, index))
    const tokenClass = match[1]
      ? 'tok-comment'
      : match[2]
        ? 'tok-string'
        : match[3]
          ? 'tok-keyword'
          : match[4]
            ? 'tok-number'
            : match[5]
              ? 'tok-type'
              : 'tok-call'
    highlighted += `<span class="${tokenClass}">${escapeHtml(match[0])}</span>`
    cursor = index + match[0].length
  }

  return highlighted + escapeHtml(source.slice(cursor))
}
