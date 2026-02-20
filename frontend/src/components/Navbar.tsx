'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'

const links = [
    { href: '/', label: 'Home' },
    { href: '/explorer', label: 'Race Explorer' },
    { href: '/analytics', label: 'Analytics' },
    { href: '/model', label: 'Model' },
]

export function Navbar() {
    const pathname = usePathname()
    const [menuOpen, setMenuOpen] = useState(false)

    return (
        <nav style={{
            position: 'sticky', top: 0, zIndex: 50,
            background: 'rgba(3,3,3,0.92)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid var(--border)',
        }}>
            {/* Red racing stripe at very top */}
            <div style={{ height: 3, background: 'linear-gradient(90deg, var(--f1-red) 0%, #ff6b6b 50%, var(--f1-red) 100%)' }} />

            <div style={{ maxWidth: 1280, margin: '0 auto', padding: '0 24px', display: 'flex', alignItems: 'center', height: 60 }}>
                {/* Logo */}
                <Link href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 10, marginRight: 40 }}>
                    <div style={{
                        width: 32, height: 32, borderRadius: 6,
                        background: 'var(--f1-red)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontFamily: 'Barlow Condensed, sans-serif',
                        fontWeight: 800, fontSize: 16, color: '#fff',
                    }}>F1</div>
                    <div>
                        <div style={{ fontFamily: 'Barlow Condensed, sans-serif', fontWeight: 700, fontSize: 16, color: '#fff', letterSpacing: '0.05em', lineHeight: 1 }}>
                            RISK FORECASTER
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em' }}>2018 – 2025 · ML POWERED</div>
                    </div>
                </Link>

                {/* Desktop links */}
                <div style={{ display: 'flex', gap: 4, flex: 1 }}>
                    {links.map(l => {
                        const active = pathname === l.href
                        return (
                            <Link key={l.href} href={l.href} style={{
                                padding: '8px 16px',
                                borderRadius: 8,
                                textDecoration: 'none',
                                fontSize: 14,
                                fontWeight: 500,
                                color: active ? '#fff' : 'var(--text-secondary)',
                                background: active ? 'rgba(232,0,45,0.15)' : 'transparent',
                                border: `1px solid ${active ? 'var(--f1-red)' : 'transparent'}`,
                                transition: 'all 0.2s ease',
                            }}>
                                {l.label}
                            </Link>
                        )
                    })}
                </div>

                {/* Status pill */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderRadius: 20, background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)' }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', animation: 'scPulse 2s infinite' }} />
                    <span style={{ fontSize: 12, color: '#22c55e', fontWeight: 500 }}>LIVE</span>
                </div>
            </div>
        </nav>
    )
}
