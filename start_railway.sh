#!/bin/bash

# Railway startup script for Kalshi AI Trading Bot

echo "🚀 Starting Kalshi AI Trading Bot on Railway..."

# Set Railway's PORT for Streamlit if not already set
export PORT=${PORT:-8501}
echo "📡 Streamlit will run on port: $PORT"

# Check if required environment variables are set
if [ -z "$KALSHI_API_KEY" ]; then
    echo "⚠️  Warning: KALSHI_API_KEY not set"
    echo "   The dashboard will start in demo mode without trading functionality"
    export DEMO_MODE=true
else
    echo "✅ KALSHI_API_KEY found - full functionality enabled"
    export DEMO_MODE=false
fi

# Initialize database (this should work without API keys)
echo "🗄️ Initializing database..."
python init_database.py

# Only start background processes if we have API keys
if [ "$DEMO_MODE" = false ]; then
    echo "🤖 Starting trading bot in background..."
    python beast_mode_bot.py &
    BOT_PID=$!

    echo "📊 Starting analytics processor in background..."
    python src/analytics/analytics_processor.py &
    ANALYTICS_PID=$!
else
    echo "🎭 Demo mode: Skipping background processes"
fi

# Start Streamlit dashboard as main process
echo "🎛️ Starting Streamlit dashboard..."
exec streamlit run trading_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true