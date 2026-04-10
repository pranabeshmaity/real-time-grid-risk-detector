#!/bin/bash
echo "Starting UGIM Grid Oscillation System"
cd backend
source ../venv/bin/activate
python app/main_realtime.py
