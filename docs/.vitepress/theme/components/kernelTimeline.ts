/** One kernel or copy: family index, start offset and duration in microseconds. */
export type KernelRow = [family: number, startUs: number, durationUs: number]

export interface KernelTimelineRow {
  label: string
  medianMs: number
  busyMs: number
  windowMs: number
  kernels: KernelRow[]
}

export interface KernelTimelineChart {
  ariaLabel: string
  context: string
  families: { label: string, light: string, dark: string }[]
  rows: KernelTimelineRow[]
}
