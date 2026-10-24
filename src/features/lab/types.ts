import type { ComponentType } from 'react'

/**
 * One instrument on the laboratory bench.
 *
 * The page renders whatever this describes; it never knows which model is
 * behind it. Adding an instrument means adding an entry to the registry, not
 * editing the page.
 */
export interface LabInstrument {
  readonly id: string
  readonly title: string
  readonly romanNumeral: string
  readonly description: string
  readonly component: ComponentType
}
