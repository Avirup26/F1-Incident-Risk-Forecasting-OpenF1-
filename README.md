# 🏎️ F1 Incident Risk Forecasting

> Predicts Safety Car (SC) and Virtual Safety Car (VSC) deployments in Formula 1 races using real-time telemetry from the [OpenF1 API](https://openf1.org). Generates a risk score every 30 seconds, forecasting whether a safety intervention will occur in the next 5 minutes.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![OpenF1 API](https://img.shields.io/badge/data-OpenF1-red.svg)](https://openf1.org)

---

## 🏗️ Architecture

```
OpenF1 API
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Data Ingestion (src/ingest_openf1/)                        │
│  • Resilient API client (retry, cache, rate limit)          │
│  • Fetchers: sessions, race_control, weather, position,     │
│    intervals, drivers                                        │
│  • Raw JSON → Bronze Parquet tables                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Timeline & Labeling (src/build_timeline/)                  │
│  • 30-second UTC time grid per session                      │
│  • SC/VSC event detection (category fields + text fallback) │
│  • y_sc_5m binary label + time_to_sc_seconds metric         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Feature Engineering (src/features/)                        │
│  • Text: rolling message counts, category entropy,          │
│    keyword flags (debris, crash, rain, yellow, red, ...)    │
│  • Weather: as-of join + rolling max_rainfall, temp_delta   │
│  • Dynamics: position changes, volatility, gap std,         │
│    pack density                                             │
│  • Silver (per-session) → Gold (master_timeline.parquet)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Modeling (src/models/)                                     │
│  • Baseline: TF-IDF + Logistic Regression                   │
│  • Strong: LightGBM + TruncatedSVD text features            │
│  • Time-series safe splits by meeting_key                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Evaluation (src/eval/)                                     │
│  • PR-AUC, ROC-AUC, Brier score                            │
│  • Alert policy analysis (alerts/race, lead time, FPR)      │
│  • Markdown evaluation report                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Dashboard (app/)                                 │
│  • Session selector (year → meeting → session)              │
│  • Interactive risk timeline with SC/VSC overlays           │
│  • Race control message drill-down                          │
│  • Feature importance visualization                         │
│  • Model card with no-leakage statement                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Setup

```bash
# Clone the repo
git clone https://github.com/Avirup26/F1-Incident-Risk-Forecasting-OpenF1-.git
cd F1-Incident-Risk-Forecasting-OpenF1-

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Full Pipeline

```bash
# Initialize directories
make setup
# or: python -m src.cli setup

# Fetch data from OpenF1 API (2024 season)
make ingest
# or: python -m src.cli ingest --year 2024

# Quick test with 2 sessions
make ingest-quick
# or: python -m src.cli ingest --year 2024 --limit 2

# Build features (timeline + labels + feature engineering)
make features
# or: python -m src.cli build_features

# Train models
make train
# or: python -m src.cli train

# Evaluate models
make evaluate
# or: python -m src.cli evaluate

# Launch Streamlit dashboard
make app
# or: streamlit run app/app.py
```

### Run Tests

```bash
make test
# or: pytest tests/ -v
```

---

## 📊 Data Sources

All data is fetched from the [OpenF1 API](https://openf1.org) (free, no auth required):

| Endpoint | Description | Used For |
|----------|-------------|----------|
| `/sessions` | Race session metadata | Session discovery |
| `/race_control` | SC/VSC messages, flags, incidents | Labels + text features |
| `/weather` | Track/air temperature, rainfall, wind | Weather features |
| `/position` | Driver positions over time | Dynamics features |
| `/intervals` | Gap to leader, interval to car ahead | Pack density features |
| `/drivers` | Driver metadata | UI display |

---

## 🤖 Model Approach

### Features
- **Text**: TF-IDF on race control messages (rolling 60s/180s/600s windows), keyword flags
- **Weather**: Rainfall, track temperature, wind speed (as-of joined)
- **Race Dynamics**: Position changes, pack density, gap standard deviation

### Models
| Model | Architecture | Primary Metric |
|-------|-------------|----------------|
| Baseline | TF-IDF (5k features) + Logistic Regression | PR-AUC |
| LightGBM | TF-IDF → SVD (100 dims) + numeric features | PR-AUC |

### Evaluation
- **Primary metric**: PR-AUC (handles class imbalance better than accuracy/ROC-AUC)
- **Secondary**: ROC-AUC, Brier score, calibration curves
- **Alert analysis**: Alerts per race, median lead time to actual events, false positive rate

---

## 🔒 No-Leakage Guarantee

This project is designed to be completely free of data leakage:

1. **As-of joins**: All features at time `t` use only data with timestamp `≤ t`
2. **Rolling windows**: Only past messages/readings are included (strictly `< t` or `≤ t`)
3. **Train/test splits**: Grouped by `meeting_key` (race weekend) — no weekend appears in both sets
4. **No random splits**: All splits are temporal/group-based

---

## 📁 Project Structure

```
F1-Incident-Risk-Forecasting-OpenF1-/
├── README.md
├── requirements.txt
├── .gitignore
├── Makefile
├── src/
│   ├── cli.py              # Click CLI entry point
│   ├── config.py           # Pydantic configuration
│   ├── ingest_openf1/      # Data fetching layer
│   ├── build_timeline/     # Timeline + SC/VSC labeling
│   ├── features/           # Feature engineering
│   ├── models/             # Baseline + LightGBM models
│   ├── eval/               # Metrics + report generation
│   └── utils/              # Logger + time utilities
├── app/
│   ├── app.py              # Streamlit dashboard
│   └── components/         # Reusable UI components
├── tests/                  # Unit tests
└── data/
    ├── raw/                # Raw JSON (gitignored)
    ├── bronze/             # Parquet tables (gitignored)
    ├── silver/             # Per-session features (gitignored)
    ├── gold/               # Master timeline (gitignored)
    ├── models/             # Trained models (gitignored)
    └── sample/             # Small demo dataset (tracked)
```

---

## ⚠️ Limitations

- SC/VSC events are rare (~5–15% of grid points) → imbalanced classification problem
- Model trained on historical data; performance may vary on new circuits or regulations
- Race control message latency may differ in real-time vs. historical API data
- OpenF1 API coverage starts from the 2023 season

---

## 🔮 Future Work

- Real-time inference mode (streaming from OpenF1 live endpoints)
- SHAP explanations for individual predictions
- Multi-class prediction (SC vs. VSC vs. Red Flag)
- Lap-level features (tire age, pit stop history)
- Ensemble of baseline + LightGBM models
