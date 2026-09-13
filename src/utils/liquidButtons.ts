/** Pointer-origin coordinates for the liquid button pour (--lx/--ly). */
export function attachLiquidButtons(): void {
  const onOver = (e: Event) => {
    const ev = e as PointerEvent
    if (!(ev.target instanceof Element)) return
    const btn = ev.target.closest('.btn')
    if (!(btn instanceof HTMLElement)) return
    const related = ev.relatedTarget
    if (related instanceof Element && btn.contains(related)) return
    const rect = btn.getBoundingClientRect()
    const x = Math.min(Math.max(ev.clientX - rect.left, 0), rect.width)
    const y = Math.min(Math.max(ev.clientY - rect.top, 0), rect.height)
    btn.style.setProperty('--lx', `${x}px`)
    btn.style.setProperty('--ly', `${y}px`)
  }
  document.addEventListener('pointerover', onOver)
}
