#!/bin/bash
# Quick start script for NBA Foul Annotator

echo "🏀 Starting NBA Foul Frame Annotator..."
echo ""
echo "📂 Latest dataset: data/metadata/nba_fouls_multi-season_1213clips_20251015_094649.csv"
echo "   - 1,213 clips ready to annotate"
echo "   - 36,390 frames"
echo ""
echo "📍 App will open at: http://localhost:8501"
echo ""
echo "💡 Upload the CSV file from the sidebar to begin"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd annotation_tool
streamlit run app.py
