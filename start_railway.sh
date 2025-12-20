#!/bin/bash
set -e

echo "🚀 Starting Kalshi AI Trading Bot Platform..."
export PYTHONPATH=$PYTHONPATH:.

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "📜 Loading environment variables from .env..."
    # Specifically extract live trading setting and export it
    export LIVE_TRADING_ENABLED=$(grep "^LIVE_TRADING_ENABLED=" .env | cut -d '=' -f2 | tr -d '"' | tr -d "'" || true)
    export ENABLE_PERFORMANCE_SYSTEM_MANAGER=$(grep "^ENABLE_PERFORMANCE_SYSTEM_MANAGER=" .env | cut -d '=' -f2 | tr -d '"' | tr -d "'" || true)
    if [ -z "$LIVE_TRADING_ENABLED" ]; then
        LIVE_TRADING_ENABLED="false"
    fi
fi

# 1. Initialize Database
echo "🗄️  Initializing database..."
python init_database.py

# 2. Start Analytics Processor in background
echo "📊 Starting Analytics Processor..."
python src/analytics/analytics_processor.py &

# 3. Start Beast Mode Bot in background
echo "🤖 Starting Beast Mode Bot..."
# Note: We use --live if the ENVIRONMENT variable LIVE_TRADING_ENABLED is 'true'
if [ "$LIVE_TRADING_ENABLED" = "true" ]; then
    echo "⚠️  LIVE TRADING ENABLED"
    python beast_mode_bot.py --live &
else
    echo "🧪 PAPER TRADING MODE"
    python beast_mode_bot.py &
fi

# 3b. Start Performance System Manager (optional)
if [ "$ENABLE_PERFORMANCE_SYSTEM_MANAGER" = "true" ]; then
    echo "📈 Starting Performance System Manager..."
    python performance_system_manager.py --start &
fi

# 4. Start Streamlit Dashboard in foreground
# Railway provides the PORT environment variable, default to 8501 for local
APP_PORT=${PORT:-8501}
echo "📈 Starting Dashboard on port $APP_PORT..."
streamlit run trading_dashboard.py --server.port $APP_PORT --server.address 0.0.0.0
