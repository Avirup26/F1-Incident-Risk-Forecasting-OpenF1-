"""
FastF1-based data ingestion package.

Replaces the OpenF1 API client with FastF1, which is:
- Completely FREE (no API key, no subscription)
- Well-maintained with rich F1 data
- Covers 2018+ seasons with full telemetry

Data fetched per session:
  - race_control_messages  → race_control.parquet
  - weather_data           → weather.parquet
  - laps                   → laps.parquet
  - car_data (telemetry)   → car_data.parquet
  - pos_data               → position.parquet
  - results                → results.parquet
"""
