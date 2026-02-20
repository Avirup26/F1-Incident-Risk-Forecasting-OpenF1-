'use client'
import { flagColor } from '@/lib/utils'

interface Message {
    date: string
    category?: string
    flag?: string
    message?: string
}

interface Props {
    messages: Message[]
    maxHeight?: number
}

export function SCEventFeed({ messages, maxHeight = 400 }: Props) {
    const scKeywords = ['safety car', 'vsc', 'virtual safety car']

    const isSCMessage = (m: Message) => {
        const text = `${m.category ?? ''} ${m.message ?? ''} ${m.flag ?? ''}`.toLowerCase()
        return scKeywords.some(k => text.includes(k))
    }

    const formatTime = (d: string) => {
        try { return new Date(d).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) }
        catch { return d }
    }

    const getCategoryColor = (m: Message) => {
        if (isSCMessage(m)) return '#FFD700'
        const cat = (m.category ?? '').toLowerCase()
        const flag = (m.flag ?? '').toLowerCase()
        if (flag.includes('red') || cat.includes('red')) return '#E8002D'
        if (flag.includes('yellow') || cat.includes('yellow')) return '#eab308'
        if (flag.includes('green') || cat.includes('green')) return '#22c55e'
        if (flag.includes('chequered')) return '#ffffff'
        return 'var(--text-secondary)'
    }

    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 16 }}>📻</span>
                <h3 className="font-f1" style={{ margin: 0, fontSize: 13, color: 'var(--f1-red)' }}>RACE CONTROL FEED</h3>
                <div style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-dim)' }}>{messages.length} messages</div>
            </div>
            <div style={{ maxHeight, overflowY: 'auto' }}>
                {messages.length === 0 ? (
                    <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>No messages available</div>
                ) : (
                    messages.map((m, i) => {
                        const color = getCategoryColor(m)
                        const isSC = isSCMessage(m)
                        return (
                            <div key={i} style={{
                                padding: '12px 20px',
                                borderBottom: '1px solid var(--border)',
                                background: isSC ? 'rgba(255,215,0,0.04)' : 'transparent',
                                display: 'flex', gap: 14, alignItems: 'flex-start',
                                transition: 'background 0.2s',
                            }} className="card-hover">
                                {/* Time */}
                                <div style={{ fontSize: 11, color: 'var(--text-dim)', minWidth: 56, fontVariantNumeric: 'tabular-nums', marginTop: 1 }}>
                                    {formatTime(m.date)}
                                </div>
                                {/* Color dot */}
                                <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, marginTop: 4, flexShrink: 0, boxShadow: isSC ? `0 0 6px ${color}` : 'none' }} className={isSC ? 'badge-sc' : ''} />
                                {/* Content */}
                                <div style={{ flex: 1 }}>
                                    {m.flag && m.flag !== 'None' && (
                                        <span style={{ fontSize: 10, fontWeight: 600, color, background: `${color}18`, borderRadius: 4, padding: '2px 6px', marginRight: 6, letterSpacing: '0.04em' }}>
                                            {m.flag}
                                        </span>
                                    )}
                                    {m.category && (
                                        <span style={{ fontSize: 10, color: 'var(--text-dim)', marginRight: 6 }}>{m.category}</span>
                                    )}
                                    <span style={{ fontSize: 13, color: isSC ? '#FFD700' : 'var(--text-primary)', fontWeight: isSC ? 600 : 400 }}>
                                        {m.message}
                                    </span>
                                </div>
                            </div>
                        )
                    })
                )}
            </div>
        </div>
    )
}
