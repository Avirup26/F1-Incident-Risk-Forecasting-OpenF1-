'use client'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    LineChart, Line, Legend
} from 'recharts'

interface YearData {
    year: number
    sessions: number
    sc_windows: number
    total_windows: number
    sc_rate_pct: number
}

function CustomTooltip({ active, payload, label }: any) {
    if (!active || !payload?.length) return null
    return (
        <div className="custom-tooltip">
            <div style={{ color: 'var(--text-secondary)', fontWeight: 600, marginBottom: 6 }}>{label}</div>
            {payload.map((p: any, i: number) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <div style={{ width: 8, height: 8, borderRadius: 2, background: p.color }} />
                    <span style={{ color: 'var(--text-secondary)' }}>{p.name}:</span>
                    <span style={{ color: p.color, fontWeight: 600 }}>{typeof p.value === 'number' ? p.value.toFixed(p.name.includes('%') ? 1 : 0) : p.value}{p.name.includes('Rate') ? '%' : ''}</span>
                </div>
            ))}
        </div>
    )
}

export function YearComparisonChart({ data }: { data: YearData[] }) {
    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px 16px' }}>
            <div style={{ paddingLeft: 8, marginBottom: 16 }}>
                <h3 className="font-f1" style={{ margin: 0, fontSize: 14, color: 'var(--f1-red)' }}>SC/VSC EVENTS BY SEASON</h3>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>Safety car & virtual safety car windows per season (2018–2025)</div>
            </div>
            <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="year" tick={{ fill: 'var(--text-dim)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: 'var(--text-dim)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="sc_windows" name="SC/VSC Windows" fill="#E8002D" />
                </BarChart>
            </ResponsiveContainer>
        </div>
    )
}

export function SCRateLineChart({ data }: { data: YearData[] }) {
    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px 16px' }}>
            <div style={{ paddingLeft: 8, marginBottom: 16 }}>
                <h3 className="font-f1" style={{ margin: 0, fontSize: 14, color: 'var(--f1-red)' }}>SC/VSC RATE TREND</h3>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>Percentage of time-windows with active SC/VSC per season</div>
            </div>
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="year" tick={{ fill: 'var(--text-dim)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tickFormatter={v => `${v}%`} tick={{ fill: 'var(--text-dim)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Line type="monotone" dataKey="sc_rate_pct" name="SC/VSC Rate (%)" stroke="#FFD700" strokeWidth={2.5}
                        dot={{ fill: '#FFD700', r: 4, stroke: 'var(--bg-card)', strokeWidth: 2 }}
                        activeDot={{ r: 6, fill: '#FFD700', stroke: '#fff', strokeWidth: 2 }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    )
}

interface CircuitData {
    meeting_name: string
    sc_rate_pct: number
    sessions: number
}

export function CircuitRateChart({ data }: { data: CircuitData[] }) {
    const top = data.slice(0, 15)
    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px 16px' }}>
            <div style={{ paddingLeft: 8, marginBottom: 16 }}>
                <h3 className="font-f1" style={{ margin: 0, fontSize: 14, color: 'var(--f1-red)' }}>TOP CIRCUITS BY SC/VSC RATE</h3>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>Circuits with highest safety car deployment probability</div>
            </div>
            <ResponsiveContainer width="100%" height={380}>
                <BarChart data={top} layout="vertical" margin={{ top: 5, right: 40, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                    <XAxis type="number" tickFormatter={v => `${v}%`} tick={{ fill: 'var(--text-dim)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="meeting_name" width={150} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="sc_rate_pct" name="SC/VSC Rate (%)" fill="#E8002D" />
                </BarChart>
            </ResponsiveContainer>
        </div>
    )
}

interface FeatureData {
    feature: string
    importance: number
}

export function FeatureImportanceChart({ data }: { data: FeatureData[] }) {
    const top = data.slice(0, 15)
    const maxVal = Math.max(...top.map(d => d.importance))
    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px 16px' }}>
            <div style={{ paddingLeft: 8, marginBottom: 16 }}>
                <h3 className="font-f1" style={{ margin: 0, fontSize: 14, color: 'var(--f1-red)' }}>MODEL FEATURE IMPORTANCE</h3>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>Top features driving the LightGBM predictions</div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {top.map((f, i) => {
                    const pct = (f.importance / maxVal) * 100
                    const label = f.feature.replace(/_/g, ' ').replace('svd', 'Text SVD')
                    return (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <div style={{ minWidth: 160, fontSize: 12, color: 'var(--text-secondary)', textAlign: 'right' }}>{label}</div>
                            <div style={{ flex: 1, height: 12, background: 'var(--border)', borderRadius: 6, overflow: 'hidden' }}>
                                <div style={{
                                    width: `${pct}%`, height: '100%', borderRadius: 6,
                                    background: `linear-gradient(90deg, #E8002D ${100 - pct}%, #ff6b6b 100%)`,
                                    transition: 'width 1s ease',
                                }} />
                            </div>
                            <div style={{ minWidth: 50, fontSize: 11, color: 'var(--text-dim)', textAlign: 'right' }}>
                                {f.importance.toFixed(0)}
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
