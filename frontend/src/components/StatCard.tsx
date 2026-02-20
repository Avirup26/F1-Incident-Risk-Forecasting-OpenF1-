'use client'
import { useEffect, useRef, useState } from 'react'
import { riskColor } from '@/lib/utils'

interface StatCardProps {
    label: string
    value: number | string
    unit?: string
    icon?: string
    color?: string
    decimals?: number
    animate?: boolean
    description?: string
}

export function StatCard({ label, value, unit = '', icon, color = 'var(--f1-red)', decimals = 0, animate = true, description }: StatCardProps) {
    const [displayed, setDisplayed] = useState(0)
    const ref = useRef<HTMLDivElement>(null)
    const numValue = typeof value === 'number' ? value : parseFloat(String(value)) || 0
    const isString = typeof value === 'string' && isNaN(parseFloat(value))

    useEffect(() => {
        if (!animate || isString) return
        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting) {
                let start = 0
                const duration = 1200
                const step = (timestamp: number) => {
                    if (!start) start = timestamp
                    const progress = Math.min((timestamp - start) / duration, 1)
                    const eased = 1 - Math.pow(1 - progress, 3)
                    setDisplayed(eased * numValue)
                    if (progress < 1) requestAnimationFrame(step)
                }
                requestAnimationFrame(step)
                observer.disconnect()
            }
        }, { threshold: 0.3 })
        if (ref.current) observer.observe(ref.current)
        return () => observer.disconnect()
    }, [numValue, animate, isString])

    const displayValue = isString ? value : displayed.toFixed(decimals)

    return (
        <div ref={ref} className="card-hover card-click" style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 12,
            padding: '20px 24px',
            position: 'relative',
            overflow: 'hidden',
            cursor: 'default',
        }}>
            {/* Background glow */}
            <div style={{
                position: 'absolute', top: -20, right: -20, width: 80, height: 80,
                borderRadius: '50%', background: `${color}10`, filter: 'blur(20px)',
                pointerEvents: 'none',
            }} />

            {/* Left accent bar */}
            <div style={{ position: 'absolute', left: 0, top: 16, bottom: 16, width: 3, background: color, borderRadius: '0 2px 2px 0' }} />

            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', paddingLeft: 8 }}>
                <div>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 500, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8 }}>
                        {label}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                        <span style={{ fontFamily: 'Barlow Condensed, sans-serif', fontSize: 36, fontWeight: 700, color, lineHeight: 1 }}>
                            {displayValue}
                        </span>
                        {unit && <span style={{ fontSize: 14, color: 'var(--text-secondary)', fontWeight: 500 }}>{unit}</span>}
                    </div>
                    {description && (
                        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>{description}</div>
                    )}
                </div>
                {icon && (
                    <div style={{ fontSize: 28, opacity: 0.7 }}>{icon}</div>
                )}
            </div>
        </div>
    )
}

export function RiskGauge({ score, size = 120 }: { score: number; size?: number }) {
    const color = riskColor(score)
    const pct = Math.min(score, 1)
    const circumference = 2 * Math.PI * 45
    const offset = circumference * (1 - pct * 0.75) // 270 degree arc

    return (
        <div style={{ position: 'relative', width: size, height: size }}>
            <svg width={size} height={size} viewBox="0 0 100 100" style={{ transform: 'rotate(-135deg)' }}>
                {/* Background track */}
                <circle cx="50" cy="50" r="45" fill="none" stroke="var(--border)" strokeWidth={8}
                    strokeDasharray={`${circumference * 0.75} ${circumference * 0.25}`} strokeLinecap="round" />
                {/* Fill */}
                <circle cx="50" cy="50" r="45" fill="none" stroke={color} strokeWidth={8}
                    strokeDasharray={`${circumference * 0.75 * pct} ${circumference * (1 - 0.75 * pct)}`}
                    strokeLinecap="round"
                    style={{ transition: 'stroke-dasharray 1s ease, stroke 0.5s ease', filter: `drop-shadow(0 0 4px ${color})` }} />
            </svg>
            <div style={{
                position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', paddingTop: 8,
            }}>
                <span style={{ fontFamily: 'Barlow Condensed, sans-serif', fontSize: size * 0.22, fontWeight: 700, color, lineHeight: 1 }}>
                    {(score * 100).toFixed(0)}%
                </span>
                <span style={{ fontSize: size * 0.09, color: 'var(--text-secondary)', letterSpacing: '0.05em' }}>RISK</span>
            </div>
        </div>
    )
}
