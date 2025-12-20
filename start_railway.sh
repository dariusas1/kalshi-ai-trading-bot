#!/bin/bash

# Railway startup script for Kalshi AI Trading Bot

echo "🚀 Starting Kalshi AI Trading Bot on Railway..."

# Set Railway's PORT for Streamlit if not already set
export PORT=${PORT:-8501}
echo "📡 Streamlit will run on port: $PORT"

# Initialize database
echo "🗄️ Initializing database..."
python init_database.py

# Start background processes
echo "🤖 Starting trading bot in background..."
python beast_mode_bot.py &
BOT_PID=$!

echo "📊 Starting analytics processor in background..."
python src/analytics/analytics_processor.py &
ANALYTICS_PID=$!

# Start Streamlit dashboard as main process
echo "🎛️ Starting Streamlit dashboard..."
exec streamlit run trading_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true