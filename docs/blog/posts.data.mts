import { createContentLoader } from 'vitepress'

export interface BlogPost {
  author: string
  category: string
  date: string
  description: string
  tags: string[]
  title: string
  url: string
}

function requiredString(
  value: unknown,
  field: string,
  source: string | undefined,
): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Blog post ${source ?? '<unknown>'} requires ${field}`)
  }
  return value.trim()
}

export default createContentLoader<BlogPost[]>('blog/posts/*.md', {
  transform(entries) {
    return entries
      .filter(({ frontmatter }) => frontmatter.draft !== true)
      .map(({ frontmatter, src, url }) => {
        const rawDate = frontmatter.date
        const date =
          rawDate instanceof Date
            ? rawDate.toISOString().slice(0, 10)
            : requiredString(rawDate, 'date', src)
        if (!/^\d{4}-\d{2}-\d{2}$/u.test(date)) {
          throw new Error(
            `Blog post ${src ?? '<unknown>'} date must use YYYY-MM-DD`,
          )
        }

        return {
          author: requiredString(frontmatter.author, 'author', src),
          category: requiredString(frontmatter.category, 'category', src),
          date,
          description: requiredString(
            frontmatter.description,
            'description',
            src,
          ),
          tags: Array.isArray(frontmatter.tags)
            ? frontmatter.tags.map(String)
            : [],
          title: requiredString(frontmatter.title, 'title', src),
          url,
        }
      })
      .sort(
        (left, right) =>
          right.date.localeCompare(left.date) ||
          left.title.localeCompare(right.title),
      )
  },
})
