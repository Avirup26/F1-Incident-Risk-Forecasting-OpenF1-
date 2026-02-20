import { StatCard } from '@/components/StatCard'
import Link from 'next/link'

async function getOverview() {
  try {
    const res = await fetch('http://localhost:8000/api/stats/overview', { next: { revalidate: 300 } })
    if (!res.ok) return null
    return res.json()
  } catch { return null }
}

async function getYearStats() {
  try {
    const res = await fetch('http://localhost:8000/api/stats/sc-rate-by-year', { next: { revalidate: 300 } })
    if (!res.ok) return []
    return res.json()
  } catch { return [] }
}

export default async function Home() {
  const overview = await getOverview()
  const yearStats = await getYearStats()

  return (
    <div style={{ maxWidth: 1280, margin: '0 auto', padding: '0 24px 80px' }}>

      {/* Hero */}
      <div style={{
        padding: '72px 0 56px',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Background radial glow */}
        <div style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          width: 600, height: 400, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(232,0,45,0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        <div className="font-f1" style={{ fontSize: 12, color: 'var(--f1-red)', letterSpacing: '0.2em', marginBottom: 16 }}>
          F1 DATA SCIENCE · 2018–2025
        </div>
        <h1 style={{
          fontFamily: 'Barlow Condensed, sans-serif',
          fontWeight: 800, fontSize: 'clamp(40px, 6vw, 80px)',
          margin: '0 0 16px',
          lineHeight: 1,
          background: 'linear-gradient(135deg, #ffffff 30%, #aaaaaa 100%)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
        }}>
          SAFETY CAR<br />
          <span style={{
            background: 'linear-gradient(135deg, #E8002D 0%, #ff6b6b 100%)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
          }}>RISK FORECASTER</span>
        </h1>
        <p style={{ fontSize: 18, color: 'var(--text-secondary)', maxWidth: 560, margin: '0 auto 40px', lineHeight: 1.6 }}>
          Predicting SC/VSC deployment probability every 30 seconds using LightGBM trained on 7 seasons of F1 telemetry, weather, and race control data.
        </p>

        <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
          <Link href="/explorer" style={{
            padding: '14px 32px', borderRadius: 8, textDecoration: 'none',
            background: 'var(--f1-red)', color: '#fff',
            fontWeight: 600, fontSize: 15, letterSpacing: '0.02em',
            transition: 'all 0.2s ease', display: 'inline-block',
          }}
            className="card-hover">
            🏎️ Explore Sessions
          </Link>
          <Link href="/analytics" style={{
            padding: '14px 32px', borderRadius: 8, textDecoration: 'none',
            background: 'transparent', color: 'var(--text-primary)',
            fontWeight: 600, fontSize: 15, letterSpacing: '0.02em',
            border: '1px solid var(--border-bright)',
            transition: 'all 0.2s ease', display: 'inline-block',
          }}
            className="card-hover">
            📊 View Analytics
          </Link>
        </div>
      </div>

      {/* Stats row */}
      {overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 48 }}>
          <StatCard label="Race Sessions" value={overview.total_sessions} icon="🏁" color="var(--f1-red)"
            description="Races analyzed across all seasons" />
          <StatCard label="Time Windows" value={overview.total_grid_points} icon="⏱️" color="#3b82f6"
            description="30-second prediction intervals" />
          <StatCard label="SC/VSC Rate" value={overview.positive_rate_pct} unit="%" decimals={2} icon="🚨" color="#FFD700"
            description="Of all windows have active SC/VSC" />
          <StatCard label="SC/VSC Windows" value={overview.sc_event_windows} icon="🟡" color="#eab308"
            description="Total safety car windows detected" />
          <StatCard label="Seasons Covered" value={overview.years_count} icon="📅" color="#22c55e"
            description={overview.years_covered?.join(', ')} animate={false} />
        </div>
      )}

      {/* Year timeline */}
      {yearStats.length > 0 && (
        <div style={{ marginBottom: 48 }}>
          <div className="font-f1" style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>SEASON BREAKDOWN</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
            {yearStats.map((y: any) => (
              <Link key={y.year} href={`/explorer?year=${y.year}`} style={{ textDecoration: 'none' }}>
                <div className="card-hover card-click" style={{
                  background: 'var(--bg-card)', border: '1px solid var(--border)',
                  borderRadius: 10, padding: '16px', textAlign: 'center',
                }}>
                  <div style={{ fontFamily: 'Barlow Condensed, sans-serif', fontSize: 24, fontWeight: 700, color: '#fff', marginBottom: 4 }}>{y.year}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>{y.sessions} races</div>
                  <div style={{
                    fontFamily: 'Barlow Condensed, sans-serif', fontSize: 20, fontWeight: 700,
                    color: y.sc_rate_pct > 5 ? 'var(--f1-red)' : y.sc_rate_pct > 3 ? '#eab308' : '#22c55e',
                  }}>
                    {y.sc_rate_pct?.toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>SC/VSC rate</div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Feature highlights */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginBottom: 48 }}>
        {[
          { icon: '🧠', title: 'LightGBM Model', desc: 'Trained on 137 race weekends with ensemble features: TF-IDF text, weather, race dynamics. ROC-AUC: 0.69', color: '#E8002D' },
          { icon: '⚡', title: 'No-Leakage Guarantee', desc: 'Strict as-of semantics — only data ≤ t used at prediction time. Temporal train/test splits by race weekend.', color: '#FFD700' },
          { icon: '🌦️', title: 'Real F1 Telemetry', desc: 'OpenF1 API data: car positions, race control messages, weather sensors, lap timing — every 30 seconds.', color: '#3b82f6' },
          { icon: '📡', title: '5-Minute Horizon', desc: 'Predicts SC/VSC deployment probability 5 minutes ahead giving teams critical strategy preparation time.', color: '#22c55e' },
        ].map((f, i) => (
          <div key={i} className="card-hover card-click racing-stripe" style={{
            background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: '24px 24px 24px 32px',
          }}>
            <div style={{ fontSize: 28, marginBottom: 12 }}>{f.icon}</div>
            <h3 style={{ fontFamily: 'Barlow Condensed, sans-serif', fontSize: 18, fontWeight: 700, color: f.color, margin: '0 0 8px' }}>{f.title}</h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.6 }}>{f.desc}</p>
          </div>
        ))}
      </div>

    </div>
  )
}
