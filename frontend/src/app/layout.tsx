import type { Metadata } from 'next'
import './globals.css'
import { Navbar } from '@/components/Navbar'

export const metadata: Metadata = {
  title: 'F1 SC/VSC Risk Forecaster',
  description: 'Predict Safety Car & Virtual Safety Car incidents using ML on F1 telemetry data 2018–2025',
  keywords: ['F1', 'Formula 1', 'Safety Car', 'Machine Learning', 'Risk Forecasting'],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body>
        <Navbar />
        <main className="min-h-screen" style={{ background: 'var(--bg-base)' }}>
          {children}
        </main>
      </body>
    </html>
  )
}
