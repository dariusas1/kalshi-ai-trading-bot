# Product Mission

## Purpose

A personal automated trading system that connects to my Kalshi account and handles everything end-to-end: researching markets, placing bets, reacting to live events, and growing the portfolio exponentially with minimal intervention. Set it, forget it, watch it compound.

## Core Philosophy

### Quality Over Quantity
- **NOT**: Analyzing 10,000+ markets, placing 50 x $3 bets spread thin
- **YES**: Focusing on 5-15 high-conviction opportunities, placing $30-100 bets with smart hedging

### Event-Driven, Time-Sensitive Trading
- **Primary Focus**: Events happening within 48 hours
- **Exception**: Long-term snipes only when odds are massively mispriced (>20% edge)
- **Live Adaptation**: React to in-game events, news, momentum shifts - like a human would

### Automated Intelligence
- Research markets automatically using AI + data APIs
- Place and adjust bets without manual intervention
- Hedge positions intelligently to lock in profits or minimize downside
- Compound gains systematically

## Trading Strategy

### Market Selection Criteria

**Time Filters:**
| Priority | Time to Event | Action |
|----------|---------------|--------|
| High | < 24 hours | Active trading, live adjustments |
| Medium | 24-48 hours | Pre-position based on edge |
| Low | 48+ hours | Only if edge > 20% (snipe opportunities) |
| Skip | > 7 days | Ignore unless exceptional |

**Quality Filters:**
- Focus on specific niches (sports, politics, economic data - pick 2-3)
- Minimum liquidity threshold (avoid illiquid markets)
- Minimum edge threshold (>5% estimated edge to enter)
- Maximum 10-20 markets analyzed per cycle (not 10,000)

### Position Sizing

**Concentrated, High-Conviction Bets:**
| Confidence | Position Size | Example |
|------------|---------------|---------|
| Very High (>80%) | $50-100 | Strong statistical edge + AI confirmation |
| High (70-80%) | $30-50 | Good edge, favorable odds |
| Medium (60-70%) | $15-30 | Moderate edge, worth small position |
| Below 60% | Skip | Not worth the risk |

### Hedging Strategy

**Smart Hedging for Consistent Wins:**
- **Pre-game**: Position on most likely outcome based on research
- **Live/In-game**: Hedge opposite side when odds shift favorably
- **Goal**: Lock in guaranteed profit or minimize loss regardless of outcome

**Example:**
```
Pre-game: Buy YES on Team A at $0.45 ($50)
In-game: Team A takes lead, YES now at $0.75
Action: Buy NO at $0.25 ($30) to lock in profit
Result: Guaranteed profit regardless of final outcome
```

### Live Event Adaptation

**React Like a Human Would:**
- Monitor live events (scores, news, momentum)
- Adjust positions when market hasn't caught up to reality
- Exit early if thesis is invalidated
- Double down if thesis is confirmed and odds still favorable

## Data & Research

### Primary Intelligence

**AI Analysis (Grok-4.1 + OpenAI fallback):**
- Market probability estimation
- Edge calculation vs. current odds
- Confidence scoring
- News/sentiment integration

**Statistical Data (Valyu.ai or similar):**
- Historical performance data
- Real-time stats and updates
- Event-specific metrics
- Odds movement tracking

### Research Flow
```
1. Filter: Only markets < 48hrs (or exceptional snipes)
2. Narrow: Apply niche/category filters (e.g., NBA, NFL, specific politics)
3. Analyze: AI + data analysis on 10-20 candidates max
4. Score: Rank by edge, confidence, liquidity
5. Select: Top 3-5 opportunities per cycle
6. Size: Position based on confidence tier
7. Hedge: Plan hedge points before entry
8. Monitor: Live adjustments during event
9. Exit: Take profits or cut losses systematically
```

## Technical Architecture

### Core Components

**Market Selection Engine:**
- Filters 60K+ markets down to 10-20 candidates
- Applies time, liquidity, category, and edge filters
- Runs every 15-30 minutes

**AI Analysis Pipeline:**
- Analyzes only filtered candidates (not entire market)
- Estimates true probability vs. market odds
- Calculates edge and confidence
- Cost: ~$0.005/analysis × 20 markets = $0.10/cycle

**Position Manager:**
- Tracks all open positions
- Monitors for hedge opportunities
- Executes live adjustments
- Enforces risk limits

**Live Event Monitor:**
- Watches active events for material changes
- Triggers re-analysis when significant news/score changes
- Executes hedges or exits automatically

### Data Sources

| Source | Purpose | Priority |
|--------|---------|----------|
| Kalshi API | Market data, order execution | Required |
| Grok-4.1 (xAI) | AI analysis, probability estimation | Primary |
| OpenAI | Fallback AI | Secondary |
| Valyu.ai | Statistical data, historical performance | Recommended |
| News APIs | Real-time news for live events | Optional |

## Risk Management

### Position Limits
- Max single position: 10-15% of portfolio
- Max correlated positions: 25% of portfolio
- Max total exposure: 60% of portfolio (keep cash for opportunities)

### Loss Limits
- Daily loss limit: 10% of portfolio
- Weekly loss limit: 20% of portfolio
- Single trade max loss: 5% of portfolio

### Hedging Rules
- Always have hedge plan before entry
- Execute hedge when profit lock-in available
- Never let winning position become big loser

## Success Metrics

### Primary Goal
Exponential portfolio growth through consistent, high-conviction trades with smart hedging.

### Key Metrics
| Metric | Target |
|--------|--------|
| Win Rate | >60% |
| Average Win/Loss Ratio | >1.5x |
| Daily AI Cost | <$1 |
| Markets Analyzed/Day | 20-50 (not 10,000) |
| Positions/Day | 3-10 |
| Monthly ROI | >15% |

## What This Is NOT

- ❌ Not a spray-and-pray system betting on everything
- ❌ Not a long-term holder (except rare snipes)
- ❌ Not analyzing 10,000 markets and burning AI credits
- ❌ Not placing $3 bets spread across 50 markets
- ❌ Not requiring constant manual monitoring

## What This IS

- ✅ A focused, high-conviction trading system
- ✅ Event-driven with <48hr time horizon
- ✅ Live-adaptive like a human trader
- ✅ Smart hedging for consistent profits
- ✅ Quality over quantity (fewer, bigger bets)
- ✅ Fully automated set-and-forget
- ✅ Optimized for capital growth and compounding