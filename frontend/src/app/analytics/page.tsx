'use client'
import { useEffect, useState } from 'react'
import { YearComparisonChart, SCRateLineChart, CircuitRateChart, FeatureImportanceChart } from '@/components/AnalyticsCharts'
import { StatCard } from '@/components/StatCard'

export default function AnalyticsPage() {
    const [yearStats, setYearStats] = useState<any[]>([])
    const [circuitStats, setCircuitStats] = useState<any[]>([])
    const [features, setFeatures] = useState<any[]>([])
    const [overview, setOverview] = useState<any>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        Promise.all([
            fetch('/api/stats/sc-rate-by-year').then(r => r.json()),
            fetch('/api/stats/sc-rate-by-circuit').then(r => r.json()),
            fetch('/api/model/feature-importance?top_n=15').then(r => r.json()),
            fetch('/api/stats/overview').then(r => r.json()),
        ])
            .then(([y, c, f, o]) => {
                setYearStats(y)
                setCircuitStats(c)
                setFeatures(f)
                setOverview(o)
            })
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [])

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', padding: 120 }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
                    <div style={{ width: 40, height: 40, border: '3px solid var(--border)', borderTopColor: 'var(--f1-red)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                    <span style={{ color: 'var(--text-secondary)' }}>Loading analytics…</span>
                </div>
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
        )
    }

    return (
        <div style={{ maxWidth: 1280, margin: '0 auto', padding: '40px 24px 80px' }}>
            {/* Header */}
            <div style={{ marginBottom: 40 }}>
                <div className="font-f1" style={{ fontSize: 12, color: 'var(--f1-red)', letterSpacing: '0.2em', marginBottom: 8 }}>HISTORICAL ANALYSIS</div>
                <h1 className="font-f1" style={{ fontSize: 36, color: '#fff', margin: '0 0 8px' }}>F1 SAFETY CAR ANALYTICS</h1>
                <p style={{ fontSize: 14, color: 'var(--text-secondary)', margin: 0 }}>
                    Deep-dive into 7 seasons of Safety Car & VSC data across all circuits (2018–2025)
                </p>
            </div>

            {/* Overview stats */}
            {overview && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 40 }}>
                    <StatCard label="Total Sessions" value={overview.total_sessions} icon="📅" color="var(--f1-red)" />
                    <StatCard label="SC/VSC Rate" value={overview.positive_rate_pct} unit="%" decimals={2} icon="🚨" color="#FFD700" />
                    <StatCard label="SC/VSC Windows" value={overview.sc_event_windows} icon="🟡" color="#eab308" />
                    <StatCard label="Time Windows" value={overview.total_grid_points} icon="⏱️" color="#3b82f6" />
                </div>
            )}

            {/* Year charts */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
                {yearStats.length > 0 && <YearComparisonChart data={yearStats} />}
                {yearStats.length > 0 && <SCRateLineChart data={yearStats} />}
            </div>

            {/* Circuit + Features */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
                {circuitStats.length > 0 && <CircuitRateChart data={circuitStats} />}
                {features.length > 0 && <FeatureImportanceChart data={features} />}
            </div>

            {/* Season detail table */}
            {yearStats.length > 0 && (
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
                    <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--border)' }}>
                        <h3 className="font-f1" style={{ margin: 0, fontSize: 14, color: 'var(--f1-red)' }}>SEASON-BY-SEASON BREAKDOWN</h3>
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                            <thead>
                                <tr style={{ background: 'var(--bg-surface)' }}>
                                    {['Season', 'Races', 'SC/VSC Windows', 'Total Windows', 'SC/VSC Rate'].map(h => (
                                        <th key={h} style={{ padding: '12px 20px', textAlign: h === 'Season' ? 'left' : 'right', color: 'var(--text-dim)', fontWeight: 600, letterSpacing: '0.06em', fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {yearStats.map((y, i) => (
                                    <tr key={y.year} style={{ borderTop: '1px solid var(--border)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                                        <td style={{ padding: '14px 20px', fontFamily: 'Barlow Condensed, sans-serif', fontSize: 20, fontWeight: 700, color: '#fff' }}>{y.year}</td>
                                        <td style={{ padding: '14px 20px', textAlign: 'right', color: 'var(--text-secondary)' }}>{y.sessions}</td>
                                        <td style={{ padding: '14px 20px', textAlign: 'right', color: '#FFD700', fontWeight: 600 }}>{y.sc_windows?.toLocaleString()}</td>
                                        <td style={{ padding: '14px 20px', textAlign: 'right', color: 'var(--text-secondary)' }}>{y.total_windows?.toLocaleString()}</td>
                                        <td style={{ padding: '14px 20px', textAlign: 'right' }}>
                                            <span style={{
                                                fontFamily: 'Barlow Condensed, sans-serif', fontSize: 18, fontWeight: 700,
                                                color: y.sc_rate_pct > 5 ? 'var(--f1-red)' : y.sc_rate_pct > 3 ? '#eab308' : '#22c55e',
                                            }}>
                                                {y.sc_rate_pct?.toFixed(2)}%
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}
