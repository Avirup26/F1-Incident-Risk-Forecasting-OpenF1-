'use client'
import { useEffect, useRef, useState } from 'react'
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { riskColor } from '@/lib/utils'

interface TimelinePoint {
    timestamp: string
    risk_score: number
    y_sc_5m: number
}

interface Props {
    data: TimelinePoint[]
    scEventStarts: string[]
    threshold: number
    sessionName?: string
}

function CustomTooltip({ active, payload, label }: any) {
    if (!active || !payload?.length) return null
    const risk = payload[0]?.value ?? 0
    const color = riskColor(risk)
    const time = label ? new Date(label).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : ''
    return (
        <div className="custom-tooltip">
            <div style={{ color: 'var(--text-secondary)', fontSize: 11, marginBottom: 4 }}>{time}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                <span style={{ color }}>Risk: <strong>{(risk * 100).toFixed(1)}%</strong></span>
            </div>
            {payload[0]?.payload?.y_sc_5m === 1 && (
                <div style={{ color: '#FFD700', fontSize: 11, marginTop: 4 }}>⚡ SC/VSC Window</div>
            )}
        </div>
    )
}

export function RiskTimeline({ data, scEventStarts, threshold, sessionName }: Props) {
    const [hoveredTime, setHoveredTime] = useState<string | null>(null)

    // Format timestamps to short time for x-axis
    const formatted = data.map(d => ({
        ...d,
        time: d.timestamp,
        display_time: new Date(d.timestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
    }))

    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px 16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, paddingLeft: 8 }}>
                <div>
                    <h3 className="font-f1" style={{ margin: 0, fontSize: 14, color: 'var(--f1-red)' }}>SC/VSC RISK TIMELINE</h3>
                    {sessionName && <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>{sessionName}</div>}
                </div>
                <div style={{ display: 'flex', gap: 16 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 12, height: 2, background: 'var(--f1-red)' }} />
                        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Risk Score</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 2, height: 12, background: '#FFD700', borderRadius: 1 }} />
                        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>SC/VSC Event</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 12, height: 1, background: 'rgba(255,255,255,0.4)', borderTop: '1px dashed rgba(255,255,255,0.4)' }} />
                        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Threshold</span>
                    </div>
                </div>
            </div>

            <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={formatted} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                    <defs>
                        <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#E8002D" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#E8002D" stopOpacity={0.02} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="display_time" tick={{ fill: 'var(--text-dim)', fontSize: 11 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--text-dim)', fontSize: 11 }} axisLine={false} tickLine={false} width={45} />
                    <Tooltip content={<CustomTooltip />} />

                    {/* SC event vertical lines */}
                    {scEventStarts.map((ts, i) => (
                        <ReferenceLine key={i} x={new Date(ts).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                            stroke="#FFD700" strokeDasharray="4 4" strokeWidth={2}
                            label={{ value: 'SC/VSC', position: 'top', fill: '#FFD700', fontSize: 10 }} />
                    ))}

                    {/* Threshold line */}
                    <ReferenceLine y={threshold} stroke="rgba(255,255,255,0.4)" strokeDasharray="6 3" strokeWidth={1.5}>
                    </ReferenceLine>

                    <Area type="monotone" dataKey="risk_score" stroke="#E8002D" strokeWidth={2}
                        fill="url(#riskGrad)" dot={false} activeDot={{ r: 4, fill: '#E8002D', stroke: '#fff', strokeWidth: 2 }} />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    )
}
