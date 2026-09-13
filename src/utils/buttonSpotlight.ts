/**
 * Cursor-follow color spotlight on button hover — the canonical mechanic.
 *
 * A small blurred coral dot sits under the button label and is repositioned
 * on pointermove by writing --x / --y (px, relative to the button). Only
 * opacity ever transitions; position never does. There is deliberately NO
 * pointerleave handler: the dot fades out wherever it last was and does not
 * jump back to the center.
 */

export function attachButtonSpotlight(): void {
  if (typeof document === 'undefined') return

  const onPointerMove = (e: PointerEvent) => {
    if (e.pointerType === 'touch') return
    const btn = (e.target as Element | null)?.closest?.('.btn') as HTMLElement | null
    if (!btn) return

    const rect = btn.getBoundingClientRect()
    btn.style.setProperty('--x', `${e.clientX - rect.left}px`)
    btn.style.setProperty('--y', `${e.clientY - rect.top}px`)
  }

  document.addEventListener('pointermove', onPointerMove, { passive: true })
}
