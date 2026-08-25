#!/usr/bin/env bash
# Start the AI Market Predictor website.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

# Train anything that is missing or stale, so the first page load is instant.
./.venv/bin/python train_models.py

exec ./.venv/bin/streamlit run app.py
