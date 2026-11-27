import { onBeforeUnmount, onMounted, type Ref } from 'vue'

/**
 * Advance `active` through `ids` at a fixed interval after mount.
 * Rotation does not start when the reader prefers reduced motion; `stop` ends it permanently, for example on the first tab click.
 */
export function useAutoAdvance<T>(
  ids: readonly T[],
  active: Ref<T>,
  intervalMs: number,
  onAdvance?: (id: T) => void,
): { stop: () => void } {
  let timer: ReturnType<typeof setInterval> | undefined

  function stop(): void {
    if (timer === undefined) return
    clearInterval(timer)
    timer = undefined
  }

  onMounted(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    timer = setInterval(() => {
      const next = ids[(ids.indexOf(active.value) + 1) % ids.length]
      active.value = next
      onAdvance?.(next)
    }, intervalMs)
  })
  onBeforeUnmount(stop)

  return { stop }
}
