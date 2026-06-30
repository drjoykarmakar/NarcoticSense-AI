#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Starting NarcoticSense AI from: $(pwd)"
echo "Installing only the app dependencies. This may take a minute the first time."
python3 -m pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "Opening app..."
python3 -m streamlit run "$(pwd)/app/streamlit_app.py"
