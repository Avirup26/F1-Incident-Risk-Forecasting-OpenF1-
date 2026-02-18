.PHONY: setup ingest features train evaluate app test clean

PYTHON := python
SRC := src

setup:
	@echo "🏎️  Setting up F1 Risk Forecasting project..."
	$(PYTHON) -m src.cli setup
	@echo "✅ Setup complete."

ingest:
	@echo "📡 Fetching data from OpenF1 API..."
	$(PYTHON) -m src.cli ingest --year 2024

ingest-quick:
	@echo "📡 Quick ingest (2 sessions)..."
	$(PYTHON) -m src.cli ingest --year 2024 --limit 2

features:
	@echo "⚙️  Building features..."
	$(PYTHON) -m src.cli build_features

train:
	@echo "🤖 Training models..."
	$(PYTHON) -m src.cli train

evaluate:
	@echo "📊 Evaluating models..."
	$(PYTHON) -m src.cli evaluate

app:
	@echo "🚀 Launching Streamlit app..."
	streamlit run app/app.py

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --tb=short

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete."

pipeline: setup ingest features train evaluate
	@echo "🏁 Full pipeline complete!"
