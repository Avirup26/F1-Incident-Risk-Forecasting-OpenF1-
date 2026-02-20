'use client'
import { useEffect, useState } from 'react'

interface Session {
    session_key: number
    meeting_name: string
    year: number
}

interface Props {
    onSessionChange: (session: Session | null) => void
}

const YEARS = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]

const selectStyle: React.CSSProperties = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border-bright)',
    borderRadius: 8,
    color: 'var(--text-primary)',
    padding: '10px 14px',
    fontSize: 14,
    width: '100%',
    outline: 'none',
    appearance: 'none',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
}

export function SessionSelector({ onSessionChange }: Props) {
    const [year, setYear] = useState(2025)
    const [sessions, setSessions] = useState<Session[]>([])
    const [selectedKey, setSelectedKey] = useState<number | null>(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        setLoading(true)
        setSessions([])
        setSelectedKey(null)
        onSessionChange(null)
        fetch(`/api/sessions?year=${year}`)
            .then(r => r.json())
            .then(data => {
                setSessions(data)
                if (data.length > 0) {
                    setSelectedKey(data[0].session_key)
                    onSessionChange(data[0])
                }
            })
            .catch(() => setSessions([]))
            .finally(() => setLoading(false))
    }, [year])

    const handleSessionChange = (key: number) => {
        setSelectedKey(key)
        const s = sessions.find(s => s.session_key === key) ?? null
        onSessionChange(s)
    }

    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: 20 }}>
            <div className="font-f1" style={{ fontSize: 12, color: 'var(--f1-red)', marginBottom: 16 }}>📡 SESSION SELECTOR</div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div>
                    <label style={{ fontSize: 11, color: 'var(--text-dim)', display: 'block', marginBottom: 6, letterSpacing: '0.06em' }}>SEASON</label>
                    <div style={{ position: 'relative' }}>
                        <select style={selectStyle} value={year} onChange={e => setYear(Number(e.target.value))}>
                            {YEARS.map(y => <option key={y} value={y}>{y}</option>)}
                        </select>
                        <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-dim)' }}>▾</div>
                    </div>
                </div>

                <div>
                    <label style={{ fontSize: 11, color: 'var(--text-dim)', display: 'block', marginBottom: 6, letterSpacing: '0.06em' }}>RACE WEEKEND</label>
                    <div style={{ position: 'relative' }}>
                        <select style={{ ...selectStyle, opacity: loading ? 0.5 : 1 }} value={selectedKey ?? ''} onChange={e => handleSessionChange(Number(e.target.value))} disabled={loading || sessions.length === 0}>
                            {loading && <option>Loading…</option>}
                            {!loading && sessions.length === 0 && <option>No sessions</option>}
                            {sessions.map(s => (
                                <option key={s.session_key} value={s.session_key}>{s.meeting_name}</option>
                            ))}
                        </select>
                        <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-dim)' }}>▾</div>
                    </div>
                </div>
            </div>

            <div style={{ marginTop: 16, fontSize: 11, color: 'var(--text-dim)' }}>
                {sessions.length > 0 && `${sessions.length} race weekends available`}
            </div>
        </div>
    )
}
