# 🤖 Kalshi AI Trading Bot

A sophisticated, multi-strategy AI-powered trading system for Kalshi prediction markets. This system uses advanced LLM reasoning (Grok-4), portfolio optimization with Kelly Criterion, and real-time market analysis to execute intelligent automated trading.

## ⚠️ **IMPORTANT DISCLAIMER**

**This is experimental software for educational and research purposes only.**

- **Trading involves substantial risk of loss**
- **Only trade with capital you can afford to lose**
- **Past performance does not guarantee future results**
- **This software is not financial advice**
- **Use at your own risk**

The authors are not responsible for any financial losses incurred through the use of this software.

---

## 🚀 Features

### Multi-Strategy Trading System
| Strategy | Allocation | Description |
|----------|------------|-------------|
| **Market Making** | 30% | Limit orders on both sides for spread profits |
| **Directional Trading** | 40% | AI-powered directional trades via portfolio optimization |
| **Quick Flip Scalping** | 30% | Rapid scalping of low-priced contracts (1¢-20¢) |

### Core Capabilities
- **🧠 Multi-Agent AI Analysis**: Forecaster, Critic, and Trader agents for comprehensive market evaluation
- **📊 Portfolio Optimization**: Kelly Criterion + Risk Parity allocation with dynamic rebalancing
- **⚡ Live Trading**: Direct Kalshi API integration for real-time order execution
- **📈 Real-time Dashboard**: Web-based monitoring with live P&L, positions, and performance metrics
- **🔄 Position Sync**: Automatic synchronization with Kalshi positions on startup

### Advanced Strategies
- **Volatility-Adjusted Sizing**: Dynamic position sizing based on market volatility
- **Theta Decay Exploitation**: Profit from time decay on high-probability events
- **ML Price Predictions**: Machine learning models for price movement forecasting
- **Arbitrage Detection**: Spread arbitrage and correlated market opportunities
- **Trailing Stop Losses**: Automatic profit protection with trailing stops

### AI Integration
- **Primary Model**: Grok-4 (xAI) for market analysis and decision making
- **Fallback Model**: Grok-3 for cost optimization
- **Multi-Model Ensemble**: Consensus-based decisions for high-stakes trades
- **Cost Controls**: Daily AI budget limits with automatic throttling

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Beast Mode Trading Bot 🚀                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Market    │    │     AI      │    │  Portfolio  │    │   Trade     │   │
│  │  Ingestion  │───▶│  Analysis   │───▶│Optimization │───▶│  Execution  │   │
│  │  (ingest)   │    │  (decide)   │    │   (Kelly)   │    │ (execute)   │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         SQLite Database                              │    │
│  │   Markets • Positions • Orders • Performance • AI Analysis Cache    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                      │            │
│         ▼                                                      ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Position   │    │ Performance │    │   P&L       │    │   Risk      │   │
│  │  Tracking   │    │  Scheduler  │    │  Tracker    │    │ Management  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                     External APIs
    ┌─────────────────────────────────────────────┐
    │  Kalshi API          │       xAI API        │
    │  • Order execution   │  • Grok-4 analysis   │
    │  • Market data       │  • Price predictions │
    │  • Position sync     │  • News sentiment    │
    └─────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Kalshi trading account with API access
- xAI API key (for Grok-4)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/kalshi-ai-trading-bot.git
   cd kalshi-ai-trading-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Copy the template and fill in your credentials:
   ```bash
   cp env.template .env
   ```
   
   Edit `.env` with your API keys:
   ```bash
   # Kalshi API Configuration
   KALSHI_API_KEY=your_kalshi_api_key_here
   
   # xAI API Configuration (Grok-4)
   XAI_API_KEY=your_xai_api_key_here
   
   # OpenAI API Configuration (fallback)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Trading Configuration
   LIVE_TRADING_ENABLED=false  # Set to 'true' for live trading
   ```

5. **Initialize the database**
   ```bash
   python init_database.py
   ```

---

## 🚀 Quick Start

### Beast Mode Trading (Recommended)
```bash
# Start the full trading system with all strategies
python beast_mode_bot.py --live

# Paper trading mode (default)
python beast_mode_bot.py

# With real-time dashboard
python beast_mode_bot.py --live --dashboard
```

### Dashboard Only
```bash
# Launch the web dashboard to monitor trading
python launch_dashboard.py
```

### Command Line Options
```bash
python beast_mode_bot.py [OPTIONS]

Options:
  --live           Enable live trading (real money)
  --dashboard      Enable real-time web dashboard
  --help           Show help message
```

---

## 📈 Trading Strategies

### 1. Market Making (30% allocation)
Provides liquidity by placing limit orders on both YES and NO sides:
- **Dynamic spread calculation** based on volatility and AI edge
- **Inventory management** to control net exposure
- **Skew-adjusted sizing** to reduce directional risk

### 2. Directional Trading (40% allocation)
AI-powered directional trades using portfolio optimization:
- **Multi-agent analysis**: Forecaster estimates probability, Critic validates, Trader decides
- **Kelly Criterion**: Optimal position sizing based on edge and confidence
- **Risk parity**: Balanced risk allocation across positions

### 3. Quick Flip Scalping (30% allocation)
Rapid scalping of low-priced contracts:
- **Entry range**: 1¢ - 20¢ contracts
- **Immediate sell orders**: Places exit order right after entry
- **Time-based adjustments**: Reduces target price if not filled
- **Loss cutting**: Market orders after 30 minutes if unprofitable

### 4. Advanced Strategies
- **Theta Decay**: Sell high-probability contracts near expiry
- **Volatility Sizing**: Adjust position size inversely to volatility
- **ML Predictions**: Use historical patterns for price forecasting
- **Arbitrage Detection**: Find spread and correlation opportunities

---

## ⚙️ Configuration

### Key Settings (`src/config/settings.py`)

```python
# Position Sizing
max_position_size_pct = 3.0    # Max 3% per position
max_positions = 6               # Max concurrent positions
kelly_fraction = 0.55           # 55% Kelly for balanced aggression

# Risk Management
max_daily_loss_pct = 8.0       # Stop trading at 8% daily loss
trailing_stop_distance = 0.05  # 5% trailing stop

# AI Configuration
primary_model = "grok-4"       # Primary AI model
daily_ai_budget = 12.0         # Daily AI spending limit ($)
min_confidence = 0.65          # Minimum confidence to trade

# Market Filtering
min_volume = 750               # Minimum volume threshold
max_time_to_expiry = 14 days   # Maximum expiry window
```

### Strategy Allocations
```python
market_making_allocation = 0.30   # 30% for market making
directional_allocation = 0.40    # 40% for directional trades
quick_flip_allocation = 0.30     # 30% for scalping
```

---

## 📊 Monitoring & Analytics

### Real-time Dashboard
Access the web dashboard at `http://localhost:8050`:
- Live P&L tracking
- Active positions overview
- Strategy performance breakdown
- AI decision logs
- Risk metrics visualization

### Performance Analysis
```bash
# Run comprehensive performance analysis
python performance_analysis.py

# Quick performance summary
python quick_performance_analysis.py

# View strategy-specific performance
python view_strategy_performance.py
```

### Position Management
```bash
# View current positions
python get_positions.py

# Sync positions with Kalshi
python sync_positions.py

# Portfolio health check
python portfolio_health_check.py
```

---

## 📁 Project Structure

```
kalshi-ai-trading-bot/
├── beast_mode_bot.py              # Main entry point
├── beast_mode_dashboard.py        # Real-time dashboard
├── trading_dashboard.py           # Detailed trading dashboard
├── src/
│   ├── clients/
│   │   ├── kalshi_client.py       # Kalshi API wrapper
│   │   ├── xai_client.py          # xAI/Grok client
│   │   └── openai_client.py       # OpenAI fallback
│   ├── config/
│   │   └── settings.py            # Configuration management
│   ├── jobs/
│   │   ├── ingest.py              # Market data ingestion
│   │   ├── decide.py              # AI decision making
│   │   ├── execute.py             # Order execution
│   │   ├── track.py               # Position tracking
│   │   └── evaluate.py            # Performance evaluation
│   ├── strategies/
│   │   ├── unified_trading_system.py  # Main strategy orchestrator
│   │   ├── market_making.py           # Market making strategy
│   │   ├── portfolio_optimization.py  # Kelly + risk parity
│   │   ├── quick_flip_scalping.py     # Scalping strategy
│   │   ├── theta_decay.py             # Time decay strategy
│   │   ├── volatility_sizing.py       # Vol-adjusted sizing
│   │   ├── ml_predictions.py          # ML price predictions
│   │   └── arbitrage_detector.py      # Arbitrage finder
│   └── utils/
│       ├── database.py            # SQLite database manager
│       ├── pnl_tracker.py         # P&L calculation
│       ├── stop_loss_calculator.py # Trailing stops
│       └── position_limits.py     # Risk limits
├── tests/                         # Test suite
├── logs/                          # Trading logs
├── requirements.txt               # Python dependencies
└── env.template                   # Environment template
```

---

## 🔧 Development

### Testing
```bash
# Run all tests
python run_tests.py

# Run specific test file
pytest tests/test_decide.py -v

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Type checking
mypy src/
```

---

## ⚠️ Important Notes

### Risk Warnings
- This is experimental software for educational purposes
- Trading involves substantial risk of loss
- Only trade with capital you can afford to lose
- Past performance does not guarantee future results

### API Rate Limits
- Kalshi: ~10 requests/second
- xAI: Monitor daily token usage
- Implement proper error handling and backoff

### Security Best Practices
- Never commit API keys or private keys
- Use environment variables for all secrets
- Regularly rotate API credentials
- Keep your Kalshi private key secure

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Kalshi](https://kalshi.com) for the prediction market platform
- [xAI](https://x.ai) for the Grok-4 AI model
- The open-source community for essential libraries

---

**Disclaimer**: This software is for educational and research purposes. Trading involves risk, and you should only trade with capital you can afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.
