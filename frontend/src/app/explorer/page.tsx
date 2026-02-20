'use client'
import { useState, useEffect, useCallback } from 'react'
import { SessionSelector } from '@/components/SessionSelector'
import { RiskTimeline } from '@/components/RiskTimeline'
import { SCEventFeed } from '@/components/SCEventFeed'
import { StatCard, RiskGauge } from '@/components/StatCard'

interface Session { session_key: number; meeting_name: string; year: number }

export default function ExplorerPage() {
    const [session, setSession] = useState<Session | null>(null)
    const [riskData, setRiskData] = useState<any>(null)
    const [messages, setMessages] = useState<any[]>([])
    const [loading, setLoading] = useState(false)
    const [threshold, setThreshold] = useState(0.3)

    useEffect(() => {
        if (!session) return
        setLoading(true)
        setRiskData(null)
        setMessages([])

        Promise.all([
            fetch(`/api/sessions/${session.session_key}/risk`).then(r => r.ok ? r.json() : null),
            fetch(`/api/sessions/${session.session_key}/race-control?year=${session.year}`).then(r => r.ok ? r.json() : []),
        ])
            .then(([risk, msgs]) => {
                setRiskData(risk)
                setMessages(msgs)
            })
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [session])

    const peakRisk = riskData?.peak_risk ?? 0

    return (
        <div style={{ maxWidth: 1280, margin: '0 auto', padding: '32px 24px 80px', display: 'grid', gridTemplateColumns: '300px 1fr', gap: 24, alignItems: 'start' }}>

            {/* Left sidebar */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <SessionSelector onSessionChange={setSession} />

                {/* Threshold slider */}
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: 20 }}>
                    <div className="font-f1" style={{ fontSize: 12, color: 'var(--f1-red)', marginBottom: 12 }}>⚙️ ALERT THRESHOLD</div>
                    <input type="range" min={0.1} max={0.9} step={0.05} value={threshold}
                        onChange={e => setThreshold(Number(e.target.value))}
                        style={{ width: '100%', accentColor: 'var(--f1-red)', cursor: 'pointer' }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
                        <span>10%</span>
                        <span style={{ color: 'var(--f1-red)', fontWeight: 600 }}>{(threshold * 100).toFixed(0)}%</span>
                        <span>90%</span>
                    </div>
                </div>

                {/* Risk gauge */}
                {riskData && (
                    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                        <div className="font-f1" style={{ fontSize: 12, color: 'var(--f1-red)', alignSelf: 'flex-start' }}>🎯 PEAK RISK SCORE</div>
                        <RiskGauge score={peakRisk} size={140} />
                    </div>
                )}

                {/* Session metrics */}
                {riskData && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        <StatCard label="Grid Points" value={riskData.grid_points} icon="⏱️" color="#3b82f6" animate={false} />
                        <StatCard label="SC/VSC Windows" value={riskData.sc_windows} icon="🚨" color="#FFD700" animate={false} />
                    </div>
                )}
            </div>

            {/* Main content */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                {/* Page header */}
                <div>
                    <h1 className="font-f1" style={{ fontSize: 28, color: '#fff', margin: '0 0 4px' }}>
                        {session ? session.meeting_name : 'Race Explorer'}
                    </h1>
                    {session && <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                        {session.year} · Session {session.session_key}
                    </div>}
                </div>

                {loading && (
                    <div style={{ display: 'flex', justifyContent: 'center', padding: 80 }}>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
                            <div style={{ width: 40, height: 40, border: '3px solid var(--border)', borderTopColor: 'var(--f1-red)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                            <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>Loading session data…</span>
                        </div>
                    </div>
                )}

                {!loading && !session && (
                    <div style={{ textAlign: 'center', padding: 80, color: 'var(--text-dim)' }}>
                        <div style={{ fontSize: 48, marginBottom: 16 }}>🏎️</div>
                        <div style={{ fontSize: 16 }}>Select a session to explore risk data</div>
                    </div>
                )}

                {!loading && riskData && (
                    <>
                        <RiskTimeline
                            data={riskData.timeline}
                            scEventStarts={riskData.sc_event_starts}
                            threshold={threshold}
                            sessionName={session?.meeting_name}
                        />
                        <SCEventFeed messages={messages} maxHeight={480} />
                    </>
                )}
            </div>

            <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
        </div>
    )
}
