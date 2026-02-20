// Utility helpers
export function cn(...classes: (string | undefined | null | false)[]) {
    return classes.filter(Boolean).join(' ')
}

export function formatPct(v: number) {
    return `${(v * 100).toFixed(1)}%`
}

export function riskColor(score: number): string {
    if (score >= 0.6) return '#E8002D'
    if (score >= 0.35) return '#eab308'
    return '#22c55e'
}

export function flagColor(flag: string): string {
    const f = flag?.toLowerCase() ?? ''
    if (f.includes('safety car') || f.includes('sc')) return '#FFD700'
    if (f.includes('virtual')) return '#FFA500'
    if (f.includes('red')) return '#E8002D'
    if (f.includes('yellow') || f.includes('double yellow')) return '#eab308'
    if (f.includes('green')) return '#22c55e'
    if (f.includes('chequered') || f.includes('checkered')) return '#ffffff'
    return '#888888'
}

export const API_BASE = ''  // leverages Next.js rewrite proxy
