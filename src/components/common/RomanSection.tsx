import React from 'react'

const ROMANS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'] as const

interface RomanSectionProps {
  readonly index: number
  readonly of: number
  readonly title: string
  readonly className?: string
  readonly children: React.ReactNode
}

/**
 * Mandatory Atelier Zero section rule: hairline top border carrying
 * [Roman.] · [title] · [page-of-pages], opening every essay block.
 */
export const RomanSection: React.FC<RomanSectionProps> = ({
  index,
  of,
  title,
  className,
  children,
}) => (
  <section className={`sec-block ${className ?? ''}`}>
    <div className="sec-rule">
      <span className="roman">{ROMANS[index] ?? index + 1}.</span>
      <span className="sec-title">{title}</span>
      <span className="page-of">
        {String(index + 1).padStart(3, '0')} / {String(of).padStart(3, '0')}
      </span>
    </div>
    {children}
  </section>
)
