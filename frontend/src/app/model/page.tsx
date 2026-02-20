'use client'
import { useEffect, useState } from 'react'
import { FeatureImportanceChart } from '@/components/AnalyticsCharts'

function MetricCard({ label, value, description, color = 'var(--f1-red)' }: any) {
    return (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 10, padding: '20px 24px', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: -10, right: -10, width: 60, height: 60, borderRadius: '50%', background: `${color}12`, filter: 'blur(15px)', pointerEvents: 'none' }} />
            <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8 }}>{label}</div>
            <div style={{ fontFamily: 'Barlow Condensed, sans-serif', fontSize: 32, fontWeight: 700, color, lineHeight: 1, marginBottom: 6 }}>{value}</div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{description}</div>
        </div>
    )
}

export default function ModelPage() {
    const [metrics, setMetrics] = useState<any>(null)
    const [alertPolicy, setAlertPolicy] = useState<any[]>([])
    const [features, setFeatures] = useState<any[]>([])

    useEffect(() => {
        Promise.all([
            fetch('/api/model/metrics').then(r => r.json()),
            fetch('/api/model/alert-policy').then(r => r.json()),
            fetch('/api/model/feature-importance?top_n=15').then(r => r.json()),
        ]).then(([m, a, f]) => {
            setMetrics(m)
            setAlertPolicy(a)
            setFeatures(f)
        }).catch(console.error)
    }, [])

    return (
        <div style={{ maxWidth: 1280, margin: '0 auto', padding: '40px 24px 80px' }}>
            {/* Header */}
            <div style={{ marginBottom: 40 }}>
                <div className="font-f1" style={{ fontSize: 12, color: 'var(--f1-red)', letterSpacing: '0.2em', marginBottom: 8 }}>MODEL EVALUATION</div>
                <h1 className="font-f1" style={{ fontSize: 36, color: '#fff', margin: '0 0 8px' }}>MODEL CARD</h1>
                <p style={{ fontSize: 14, color: 'var(--text-secondary)', margin: 0 }}>LightGBM trained on 137 race weekends · No temporal data leakage</p>
            </div>

            {/* Model comparison */}
            {metrics && (
                <div style={{ marginBottom: 40 }}>
                    <div className="font-f1" style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>MODEL PERFORMANCE COMPARISON</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                        {/* Baseline */}
                        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: 24 }}>
                            <div className="font-f1" style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>BASELINE (TF-IDF + LOGISTIC REGRESSION)</div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                                <MetricCard label="PR-AUC" value={metrics.baseline.pr_auc.toFixed(4)} description="Precision-Recall area" color="#888" />
                                <MetricCard label="ROC-AUC" value={metrics.baseline.roc_auc.toFixed(4)} description="Discrimination ability" color="#888" />
                                <MetricCard label="Brier" value={metrics.baseline.brier.toFixed(4)} description="Calibration error ↓" color="#888" />
                            </div>
                        </div>
                        {/* LightGBM */}
                        <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,0,45,0.3)', borderRadius: 12, padding: 24, boxShadow: '0 0 20px rgba(232,0,45,0.05)' }}>
                            <div className="font-f1" style={{ fontSize: 13, color: 'var(--f1-red)', marginBottom: 16 }}>🏆 LIGHTGBM (BEST MODEL)</div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                                <MetricCard label="PR-AUC" value={metrics.lgbm.pr_auc.toFixed(4)} description="Precision-Recall area ↑" color="var(--f1-red)" />
                                <MetricCard label="ROC-AUC" value={metrics.lgbm.roc_auc.toFixed(4)} description="Discrimination ability ↑" color="var(--f1-red)" />
                                <MetricCard label="Brier" value={metrics.lgbm.brier.toFixed(4)} description="Calibration error ↓" color="#22c55e" />
                            </div>
                        </div>
                    </div>

                    {/* Test set info */}
                    <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                        <MetricCard label="Test Set Rows" value={metrics.test_set.rows.toLocaleString()} description="Held-out grid points" color="#3b82f6" />
                        <MetricCard label="Test Meetings" value={metrics.test_set.meetings} description="Race weekends in test" color="#3b82f6" />
                        <MetricCard label="Positive Rate" value={`${metrics.test_set.positive_rate_pct}%`} description="SC/VSC in test set" color="#eab308" />
                    </div>
                </div>
            )}

            {/* Alert Policy + Feature Importance */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 32 }}>
                {/* Alert Policy Table */}
                {alertPolicy.length > 0 && (
                    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
                        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--border)' }}>
                            <h3 className="font-f1" style={{ margin: 0, fontSize: 14, color: 'var(--f1-red)' }}>ALERT POLICY ANALYSIS</h3>
                            <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>LightGBM at different thresholds</div>
                        </div>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                            <thead>
                                <tr style={{ background: 'var(--bg-surface)' }}>
                                    {['Threshold', 'Alerts/Race', 'Lead Time', 'TPR', 'FPR', 'Precision'].map(h => (
                                        <th key={h} style={{ padding: '10px 14px', textAlign: 'right', color: 'var(--text-dim)', fontWeight: 600, fontSize: 10, letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {alertPolicy.map((row, i) => (
                                    <tr key={i} style={{ borderTop: '1px solid var(--border)', background: row.threshold === 0.3 ? 'rgba(232,0,45,0.04)' : 'transparent' }}>
                                        <td style={{ padding: '10px 14px', textAlign: 'right', fontWeight: 700, color: row.threshold === 0.3 ? 'var(--f1-red)' : '#fff' }}>{(row.threshold * 100).toFixed(0)}%</td>
                                        <td style={{ padding: '10px 14px', textAlign: 'right', color: 'var(--text-secondary)' }}>{row.alerts_per_race}</td>
                                        <td style={{ padding: '10px 14px', textAlign: 'right', color: '#3b82f6' }}>{row.lead_time_s}s</td>
                                        <td style={{ padding: '10px 14px', textAlign: 'right', color: '#22c55e' }}>{(row.tpr * 100).toFixed(1)}%</td>
                                        <td style={{ padding: '10px 14px', textAlign: 'right', color: '#eab308' }}>{(row.fpr * 100).toFixed(1)}%</td>
                                        <td style={{ padding: '10px 14px', textAlign: 'right', color: 'var(--f1-red)' }}>{(row.precision * 100).toFixed(1)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {features.length > 0 && <FeatureImportanceChart data={features} />}
            </div>

            {/* No-leakage statement */}
            <div style={{ background: 'rgba(34,197,94,0.05)', border: '1px solid rgba(34,197,94,0.2)', borderRadius: 12, padding: 24 }}>
                <h3 className="font-f1" style={{ fontSize: 14, color: '#22c55e', margin: '0 0 12px' }}>✅ NO-LEAKAGE GUARANTEE</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
                    {[
                        { title: 'Strict As-Of Semantics', desc: 'Only data with timestamp ≤ t is used when computing features for grid point t.' },
                        { title: 'Weekend-Level Splits', desc: 'Train/test splits are by meeting_key (race weekend). No race weekend appears in both sets.' },
                        { title: 'No Random Splits', desc: 'Temporal ordering is preserved throughout. The model has never seen future races during training.' },
                    ].map((g, i) => (
                        <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                            <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', marginTop: 5, flexShrink: 0 }} />
                            <div>
                                <div style={{ fontWeight: 600, fontSize: 13, color: '#22c55e', marginBottom: 4 }}>{g.title}</div>
                                <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>{g.desc}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
