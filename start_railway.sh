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

    # Check if trading bot can start (has all dependencies)
    if python -c "import sys; sys.path.append('/app'); from beast_mode_dashboard import BeastModeDashboard; print('Dependencies OK')" 2>/dev/null; then
        python beast_mode_bot.py &
        BOT_PID=$!
        echo "✅ Trading bot started (PID: $BOT_PID)"
    else
        echo "⚠️  Trading bot has missing dependencies - skipping for now"
        echo "    Dashboard will still work with live API access"
        BOT_PID=""
    fi

    echo "📊 Starting analytics processor in background..."
    if [ -f "src/analytics/analytics_processor.py" ]; then
        python src/analytics/analytics_processor.py &
        ANALYTICS_PID=$!
        echo "✅ Analytics processor started (PID: $ANALYTICS_PID)"

        # Give analytics a moment to start and check if it's running
        sleep 2
        if ! kill -0 $ANALYTICS_PID 2>/dev/null; then
            echo "⚠️  Analytics processor failed to start - continuing without it"
            ANALYTICS_PID=""
        fi
    else
        echo "⚠️  Analytics processor not found - skipping"
        ANALYTICS_PID=""
    fi

    # Summary of started processes
    echo ""
    echo "📋 Background Processes Status:"
    echo "   Trading Bot: ${BOT_PID:+Running (PID: $BOT_PID)}"
    echo "   Analytics: ${ANALYTICS_PID:+Running (PID: $ANALYTICS_PID)}"
    echo ""
else
    echo "🎭 Demo mode: Skipping background processes"
fi

# Start Streamlit dashboard as main process
echo "🎛️ Starting Streamlit dashboard..."
exec streamlit run trading_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true