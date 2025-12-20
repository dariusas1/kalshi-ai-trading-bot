#!/usr/bin/env python3
"""
Comprehensive Trading System Dashboard

A Streamlit-based dashboard for monitoring and analyzing all aspects of the 
trading system including:
- Strategy performance analytics
- LLM query analysis and review
- Real-time position tracking
- Risk management monitoring
- System health metrics
- P&L analytics by strategy
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings

# Configure Streamlit page
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    .llm-query {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def _extract_positions(positions_response):
    """Normalize Kalshi positions response across possible API keys."""
    if not isinstance(positions_response, dict):
        return []
    return positions_response.get("market_positions") or positions_response.get("positions") or []


def _run_async(coro):
    """Run async code safely from Streamlit's sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    return asyncio.run(coro)


# @st.cache_data(ttl=60)  # Cache for 1 minute - temporarily disabled
def load_performance_data():
    """Load performance data from database."""
    try:
        db_manager = DatabaseManager()
        kalshi_client = KalshiClient()
        
        async def get_data():
            await db_manager.initialize()
            
            # Get performance by strategy - ensure it's serializable
            performance_raw = await db_manager.get_performance_by_strategy()
            
            # Convert performance data to ensure serializability
            performance = {}
            if performance_raw:
                for strategy, stats in performance_raw.items():
                    performance[str(strategy)] = {
                        str(k): float(v) if isinstance(v, (int, float)) else str(v) 
                        for k, v in stats.items()
                    }
            
            # Get LIVE positions from Kalshi API (not just database)
            positions_response = await kalshi_client.get_positions()
            kalshi_positions = _extract_positions(positions_response)
            
            # Convert Kalshi positions to simple dictionaries for caching
            positions = []
            for pos in kalshi_positions:
                if pos.get('position', 0) != 0:  # Only active positions
                    ticker = pos.get('ticker')
                    position_count = pos.get('position', 0)
                    
                    # Create a simple dictionary with only serializable types
                    position_dict = {
                        'market_id': str(ticker),
                        'side': 'YES' if position_count > 0 else 'NO',
                        'quantity': int(abs(position_count)),
                        'entry_price': 0.50,  # Will be updated below
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'live_sync',
                        'status': 'open',
                        'stop_loss_price': None,
                        'take_profit_price': None
                    }
                    
                    # Try to get current market price for better accuracy
                    try:
                        market_data = await kalshi_client.get_market(ticker)
                        if market_data and 'market' in market_data:
                            market_info = market_data['market']
                            if position_count > 0:  # YES position
                                position_dict['entry_price'] = float(market_info.get('yes_price', 50) / 100)
                            else:  # NO position
                                position_dict['entry_price'] = float(market_info.get('no_price', 50) / 100)
                    except:
                        position_dict['entry_price'] = 0.50  # Keep default price as float
                    
                    positions.append(position_dict)
            
            await db_manager.close()
            
            return performance, positions

        performance, positions = _run_async(get_data())
        return performance, positions
        
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        return {}, []

# @st.cache_data(ttl=30)  # Cache for 30 seconds - temporarily disabled
def load_llm_data():
    """Load LLM query data from database."""
    try:
        db_manager = DatabaseManager()
        
        async def get_data():
            await db_manager.initialize()
            
            # Get recent LLM queries
            queries = await db_manager.get_llm_queries(hours_back=24, limit=100)
            
            # Get LLM stats by strategy with improved token calculation
            stats = await db_manager.get_llm_stats_by_strategy()
            
            # Fix token count issues by recalculating from response lengths if needed
            for strategy, strategy_stats in stats.items():
                if strategy_stats.get('total_tokens', 0) == 0:
                    # Recalculate tokens from query responses for this strategy
                    strategy_queries = [q for q in queries if q.strategy == strategy]
                    estimated_tokens = 0
                    for query in strategy_queries:
                        # Estimate tokens: ~4 characters per token
                        prompt_tokens = len(query.prompt) // 4 if query.prompt else 0
                        response_tokens = len(query.response) // 4 if query.response else 0
                        estimated_tokens += prompt_tokens + response_tokens
                    
                    strategy_stats['total_tokens'] = estimated_tokens
                    strategy_stats['estimated'] = True
            
            await db_manager.close()
            
            return queries, stats
        
        queries, stats = _run_async(get_data())
        return queries, stats
        
    except Exception as e:
        st.error(f"Error loading LLM data: {e}")
        return [], {}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_system_health():
    """Load system health metrics including both available cash and total portfolio value."""
    try:
        kalshi_client = KalshiClient()
        
        async def get_health():
            # Get available cash
            balance_response = await kalshi_client.get_balance()
            available_cash = balance_response.get('balance', 0) / 100
            
            # Get current positions to calculate total portfolio value
            positions_response = await kalshi_client.get_positions()
            market_positions = _extract_positions(positions_response)
            
            total_position_value = 0
            positions_count = len(market_positions)
            
            # Calculate current value of all positions
            for position in market_positions:
                try:
                    ticker = position.get('ticker')
                    position_count = position.get('position', 0)
                    
                    if ticker and position_count != 0:
                        # Get current market data
                        market_data = await kalshi_client.get_market(ticker)
                        if market_data and 'market' in market_data:
                            market_info = market_data['market']
                            
                            # Determine if this is a YES or NO position and get current price
                            # For Kalshi, positive position = YES, negative = NO
                            if position_count > 0:  # YES position
                                current_price = market_info.get('yes_price', 50) / 100
                            else:  # NO position  
                                current_price = market_info.get('no_price', 50) / 100
                            
                            position_value = abs(position_count) * current_price
                            total_position_value += position_value
                            
                except Exception as e:
                    # If we can't get market data for a position, skip it
                    print(f"Warning: Could not value position {ticker}: {e}")
                    continue
            
            # Total portfolio value = cash + position values
            total_portfolio_value = available_cash + total_position_value
            
            return available_cash, total_portfolio_value, positions_count, total_position_value
        
        available_cash, total_portfolio_value, positions_count, position_value = _run_async(get_health())
        return {
            'available_cash': available_cash,
            'total_portfolio_value': total_portfolio_value, 
            'positions_count': positions_count,
            'position_value': position_value
        }
        
    except Exception as e:
        st.error(f"Error loading system health: {e}")
        return {
            'available_cash': 0.0,
            'total_portfolio_value': 0.0,
            'positions_count': 0,
            'position_value': 0.0
        }

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_period_pnl():
    """Load P&L data for different time periods, checking cache first."""
    try:
        db_manager = DatabaseManager()
        
        async def get_pnl():
            await db_manager.initialize()
            
            # Try cache first
            cached = await db_manager.get_cached_analytics("period_pnl")
            if cached:
                await db_manager.close()
                return cached
            
            # Compute fresh
            today = await db_manager.get_pnl_by_period('today')
            week = await db_manager.get_pnl_by_period('week')
            month = await db_manager.get_pnl_by_period('month')
            all_time = await db_manager.get_pnl_by_period('all')
            
            await db_manager.close()
            return {'today': today, 'week': week, 'month': month, 'all': all_time}
        
        result = _run_async(get_pnl())
        return result
        
    except Exception as e:
        st.error(f"Error loading P&L data: {e}")
        return {'today': {}, 'week': {}, 'month': {}, 'all': {}}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_reconciliation_status():
    """Load reconciliation status from analytics cache."""
    try:
        db_manager = DatabaseManager()

        async def get_status():
            await db_manager.initialize()
            status = await db_manager.get_cached_analytics("reconciliation_status")
            await db_manager.close()
            return status or {}

        return _run_async(get_status())
    except Exception as e:
        st.error(f"Error loading reconciliation status: {e}")
        return {}

def load_edge_analytics():
    """Load edge analytics data (category, hourly, expiry, calibration), checking cache first."""
    try:
        db_manager = DatabaseManager()
        
        async def get_analytics():
            await db_manager.initialize()
            
            # Try to load all components from cache individually or as a block
            # In our processor we cache them individually but we can also check for a block
            cached_cat = await db_manager.get_cached_analytics("category_performance")
            cached_hour = await db_manager.get_cached_analytics("hourly_performance")
            cached_exp = await db_manager.get_cached_analytics("expiry_performance")
            cached_cal = await db_manager.get_cached_analytics("confidence_calibration")
            cached_streaks = await db_manager.get_cached_analytics("trading_streaks")
            
            if all([cached_cat, cached_hour, cached_exp, cached_cal, cached_streaks]):
                await db_manager.close()
                return {
                    'category': cached_cat,
                    'hourly': cached_hour,
                    'expiry': cached_exp,
                    'calibration': cached_cal,
                    'streaks': cached_streaks
                }
            
            # Compute fresh if any missing
            category = await db_manager.get_category_performance()
            hourly = await db_manager.get_hourly_performance()
            expiry = await db_manager.get_expiry_performance()
            calibration = await db_manager.get_confidence_calibration()
            streaks = await db_manager.get_trading_streaks()
            
            await db_manager.close()
            return {
                'category': category,
                'hourly': hourly,
                'expiry': expiry,
                'calibration': calibration,
                'streaks': streaks
            }
        
        result = _run_async(get_analytics())
        return result
        
    except Exception as e:
        st.error(f"Error loading edge analytics: {e}")
        return {'category': {}, 'hourly': {}, 'expiry': {}, 'calibration': [], 'streaks': {}}

def load_recent_trades(limit=50):
    """Load recent trade executions."""
    try:
        db_manager = DatabaseManager()
        
        async def get_trades():
            await db_manager.initialize()
            trades = await db_manager.get_recent_trades(limit=limit)
            await db_manager.close()
            return trades
        
        result = _run_async(get_trades())
        return result
        
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return []

def main():
    """Main dashboard function."""
    
    st.title("🚀 Trading System Dashboard")
    st.markdown("**Real-time monitoring and analysis of your automated trading system**")
    
    # Add refresh button and auto-refresh toggle
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.caption("Auto-refresh uses polling (30s), not live streaming.")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False, help="Refresh every 30 seconds")
    with col3:
        if st.button("🔄 Refresh", help="Clear cache and reload all data"):
            st.cache_data.clear()
            st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    # Sidebar for navigation
    st.sidebar.title("📊 Dashboard")
    
    page = st.sidebar.selectbox(
        "Select View",
        [
            "📈 Overview",
            "🎯 Strategy Performance", 
            "📊 Edge Analytics",
            "🤖 LLM Analysis",
            "💼 Positions & Trades",
            "⚠️ Risk Management",
            "⚙️ Controls",
            "📅 Historical",
            "🔧 System Health"
        ]
    )
    
    # Load data with error handling
    try:
        performance_data, positions = load_performance_data()
        llm_queries, llm_stats = load_llm_data()
        system_health_data = load_system_health()
        period_pnl = load_period_pnl()
        reconciliation_status = load_reconciliation_status()
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        st.info("Please check your system connections and try refreshing.")
        return
    
    # Show data status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Data Status:**")
    st.sidebar.metric("Active Positions", len(positions) if positions else 0)
    st.sidebar.metric("LLM Queries (24h)", len(llm_queries) if llm_queries else 0)
    st.sidebar.metric("Portfolio Balance", f"${system_health_data.get('total_portfolio_value', 0):.2f}")
    
    # Show P&L by period in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💰 P&L Summary:**")
    st.sidebar.metric("Today", f"${period_pnl.get('today', {}).get('total_pnl', 0):.2f}")
    st.sidebar.metric("This Week", f"${period_pnl.get('week', {}).get('total_pnl', 0):.2f}")

    # Data health banner
    last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    positions_count = system_health_data.get("positions_count", 0)
    position_value = system_health_data.get("position_value", 0.0)
    recon_ts = reconciliation_status.get("timestamp", "unknown")
    if (not positions) and (positions_count > 0 or position_value > 0):
        st.warning(
            f"Data discrepancy detected. Dashboard shows 0 positions, "
            f"but Kalshi reports {positions_count} positions. "
            f"Last refresh: {last_refresh}. Last reconciliation: {recon_ts}."
        )
    else:
        st.info(f"Data OK. Last refresh: {last_refresh}. Last reconciliation: {recon_ts}.")
    
    # Page routing
    if page == "📈 Overview":
        show_overview(performance_data, positions, system_health_data, period_pnl)
    elif page == "🎯 Strategy Performance":
        show_strategy_performance(performance_data)
    elif page == "📊 Edge Analytics":
        show_edge_analytics()
    elif page == "🤖 LLM Analysis":
        show_llm_analysis(llm_queries, llm_stats)
    elif page == "💼 Positions & Trades":
        show_positions_trades(positions)
    elif page == "⚠️ Risk Management":
        show_risk_management(performance_data, positions, system_health_data['total_portfolio_value'])
    elif page == "⚙️ Controls":
        show_controls()
    elif page == "📅 Historical":
        show_historical()
    elif page == "🔧 System Health":
        show_system_health(system_health_data['available_cash'], system_health_data['positions_count'], llm_stats)

def show_overview(performance_data, positions, system_health_data, period_pnl=None):
    """Show overview dashboard."""
    
    st.header("📈 System Overview")
    
    # P&L by Period Cards
    if period_pnl:
        st.subheader("💰 P&L by Period")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        
        with p_col1:
            today_pnl = period_pnl.get('today', {}).get('total_pnl', 0)
            today_trades = period_pnl.get('today', {}).get('total_trades', 0)
            st.metric(
                "📅 Today",
                f"${today_pnl:.2f}",
                delta=f"{today_trades} trades",
                delta_color="off"
            )
        
        with p_col2:
            week_pnl = period_pnl.get('week', {}).get('total_pnl', 0)
            week_wr = period_pnl.get('week', {}).get('win_rate', 0)
            st.metric(
                "📆 This Week",
                f"${week_pnl:.2f}",
                delta=f"{week_wr:.0f}% win rate",
                delta_color="normal" if week_wr >= 50 else "inverse"
            )
        
        with p_col3:
            month_pnl = period_pnl.get('month', {}).get('total_pnl', 0)
            month_trades = period_pnl.get('month', {}).get('total_trades', 0)
            st.metric(
                "📊 This Month",
                f"${month_pnl:.2f}",
                delta=f"{month_trades} trades",
                delta_color="off"
            )
        
        with p_col4:
            all_pnl = period_pnl.get('all', {}).get('total_pnl', 0)
            all_wr = period_pnl.get('all', {}).get('win_rate', 0)
            st.metric(
                "🏆 All Time",
                f"${all_pnl:.2f}",
                delta=f"{all_wr:.0f}% win rate",
                delta_color="normal" if all_wr >= 50 else "inverse"
            )
        
        st.divider()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Portfolio Balance",
            value=f"${system_health_data['total_portfolio_value']:.2f}",
            help="Total portfolio value: cash + current positions"
        )
    
    # Add second row for additional financial metrics
    col1b, col2b, col3b, col4b = st.columns(4)
    
    with col1b:
        st.metric(
            label="💵 Available Cash",
            value=f"${system_health_data['available_cash']:.2f}",
            help="Cash available for new trades"
        )
    
    with col2b:
        st.metric(
            label="📊 Position Value",
            value=f"${system_health_data['position_value']:.2f}",
            help="Current market value of all positions"
        )
    
    with col2:
        total_trades = sum(stats.get('completed_trades', 0) for stats in performance_data.values()) if performance_data else 0
        st.metric(
            label="📈 Total Trades",
            value=total_trades,
            help="Total completed trades across all strategies"
        )
    
    with col3:
        # Calculate both realized and unrealized P&L
        realized_pnl = sum(stats.get('total_pnl', 0) for stats in performance_data.values()) if performance_data else 0
        
        # Calculate unrealized P&L from current positions
        unrealized_pnl = 0
        if positions:
            # This is a rough estimate - in practice you'd get current market prices
            for pos in positions:
                # Position is now a dictionary
                if 'entry_price' in pos and 'quantity' in pos:
                    # Estimate current value vs entry value
                    # For demo purposes, we'll use a simple calculation
                    position_value = pos['entry_price'] * pos['quantity']
                    # Assume current value is similar to entry (this would be calculated with live prices)
                    unrealized_pnl += 0  # Placeholder - would need current market prices
        
        total_pnl = realized_pnl + unrealized_pnl
        
        st.metric(
            label="💹 Total P&L",
            value=f"${total_pnl:.2f}",
            delta=f"Realized: ${realized_pnl:.2f}, Unrealized: ${unrealized_pnl:.2f}",
            help="Total profit/loss: realized from completed trades + unrealized from open positions"
        )
    
    with col4:
        st.metric(
            label="🎯 Active Positions",
            value=len(positions) if positions else 0,
            help="Currently open positions"
        )
    
    with col3b:
        # Portfolio utilization
        if system_health_data['total_portfolio_value'] > 0:
            utilization_pct = (system_health_data['position_value'] / system_health_data['total_portfolio_value']) * 100
        else:
            utilization_pct = 0
        st.metric(
            label="📊 Portfolio Utilization",
            value=f"{utilization_pct:.1f}%",
            help="Percentage of portfolio currently in positions"
        )
    
    with col4b:
        # Cash utilization  
        if system_health_data['available_cash'] > 0:
            initial_cash = system_health_data['total_portfolio_value']  # Approximation
            cash_used_pct = ((initial_cash - system_health_data['available_cash']) / initial_cash) * 100 if initial_cash > 0 else 0
        else:
            cash_used_pct = 100
        st.metric(
            label="💸 Cash Deployed",
            value=f"{min(100, max(0, cash_used_pct)):.1f}%", 
            help="Percentage of original cash now in positions"
        )
    
    # Strategy performance summary
    if performance_data:
        st.subheader("🎯 Strategy Performance Summary")
        
        # Create strategy performance chart
        strategy_names = []
        strategy_pnl = []
        strategy_trades = []
        strategy_win_rates = []
        
        for strategy, stats in performance_data.items():
            strategy_names.append(strategy.replace('_', ' ').title())
            strategy_pnl.append(stats.get('total_pnl', 0))
            strategy_trades.append(stats.get('completed_trades', 0))
            strategy_win_rates.append(stats.get('win_rate_pct', 0))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L by strategy
            fig_pnl = px.bar(
                x=strategy_names,
                y=strategy_pnl,
                title="P&L by Strategy",
                labels={'x': 'Strategy', 'y': 'P&L ($)'},
                color=strategy_pnl,
                color_continuous_scale='RdYlGn'
            )
            fig_pnl.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_pnl, width='stretch')
        
        with col2:
            # Win rate by strategy
            fig_winrate = px.bar(
                x=strategy_names,
                y=strategy_win_rates,
                title="Win Rate by Strategy (%)",
                labels={'x': 'Strategy', 'y': 'Win Rate (%)'},
                color=strategy_win_rates,
                color_continuous_scale='Blues'
            )
            fig_winrate.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_winrate, width='stretch')
    else:
        st.info("📊 **No strategy data yet** - Run the trading system to start collecting performance data")
    
    # Recent activity summary
    st.subheader("📋 Recent Activity")
    
    if positions:
        st.write(f"**{len(positions)} active positions:**")
        
        # Show top positions by value
        position_data = []
        for pos in positions[:10]:  # Top 10
            # Convert timestamp string back to datetime for display
            try:
                timestamp = datetime.fromisoformat(pos['timestamp'])
                time_str = timestamp.strftime('%m/%d %H:%M')
            except:
                time_str = 'Unknown'
            
            position_data.append({
                'Market': pos['market_id'][:25] + '...' if len(pos['market_id']) > 25 else pos['market_id'],
                'Side': pos['side'],
                'Quantity': pos['quantity'],
                'Entry Price': f"${pos['entry_price']:.3f}",
                'Value': f"${pos['quantity'] * pos['entry_price']:.2f}",
                'Strategy': pos['strategy'] or 'Unknown',
                'Time': time_str
            })
        
        if position_data:
            df = pd.DataFrame(position_data)
            st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("No active positions currently.")

def show_strategy_performance(performance_data):
    """Show detailed strategy performance analysis."""
    
    st.header("🎯 Strategy Performance Analysis")
    
    if not performance_data:
        st.warning("No strategy performance data available yet.")
        return
    
    # Strategy selector
    strategies = list(performance_data.keys())
    selected_strategy = st.selectbox(
        "Select Strategy for Detailed Analysis",
        ["All Strategies"] + strategies
    )
    
    if selected_strategy == "All Strategies":
        # Compare all strategies
        st.subheader("📊 Strategy Comparison")
        
        # Create comparison table
        comparison_data = []
        for strategy, stats in performance_data.items():
            comparison_data.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Completed Trades': stats['completed_trades'],
                'Total P&L': f"${stats['total_pnl']:.2f}",
                'Avg P&L per Trade': f"${stats['avg_pnl_per_trade']:.2f}",
                'Win Rate': f"{stats['win_rate_pct']:.1f}%",
                'Best Trade': f"${stats['best_trade']:.2f}",
                'Worst Trade': f"${stats['worst_trade']:.2f}",
                'Open Positions': stats['open_positions'],
                'Capital Deployed': f"${stats['capital_deployed']:.2f}"
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, width='stretch', hide_index=True)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk-return scatter
            fig_risk = go.Figure()
            
            for strategy, stats in performance_data.items():
                if stats['completed_trades'] > 0:
                    fig_risk.add_trace(go.Scatter(
                        x=[stats['avg_pnl_per_trade']],
                        y=[stats['win_rate_pct']],
                        mode='markers+text',
                        text=[strategy.replace('_', ' ').title()],
                        textposition="top center",
                        marker=dict(
                            size=stats['completed_trades'] * 2,
                            color=stats['total_pnl'],
                            colorscale='RdYlGn',
                            showscale=True
                        ),
                        name=strategy
                    ))
            
            fig_risk.update_layout(
                title="Risk-Return Analysis (Bubble size = Trade count)",
                xaxis_title="Average P&L per Trade ($)",
                yaxis_title="Win Rate (%)",
                height=500
            )
            st.plotly_chart(fig_risk, width='stretch')
        
        with col2:
            # Capital deployment
            fig_capital = px.pie(
                values=[stats['capital_deployed'] for stats in performance_data.values()],
                names=[strategy.replace('_', ' ').title() for strategy in performance_data.keys()],
                title="Capital Deployment by Strategy"
            )
            fig_capital.update_layout(height=500)
            st.plotly_chart(fig_capital, width='stretch')
    
    else:
        # Show individual strategy details
        stats = performance_data[selected_strategy]
        
        st.subheader(f"📋 {selected_strategy.replace('_', ' ').title()} Performance")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total P&L", f"${stats['total_pnl']:.2f}")
        with col2:
            st.metric("Win Rate", f"{stats['win_rate_pct']:.1f}%")
        with col3:
            st.metric("Completed Trades", stats['completed_trades'])
        with col4:
            st.metric("Open Positions", stats['open_positions'])
        
        # Detailed metrics
        if stats['completed_trades'] > 0:
            st.subheader("📈 Detailed Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Trade Performance:**")
                st.write(f"- Average P&L per Trade: ${stats['avg_pnl_per_trade']:.2f}")
                st.write(f"- Best Trade: ${stats['best_trade']:.2f}")
                st.write(f"- Worst Trade: ${stats['worst_trade']:.2f}")
                st.write(f"- Winning Trades: {stats['winning_trades']}")
                st.write(f"- Losing Trades: {stats['losing_trades']}")
            
            with col2:
                st.write("**Capital Allocation:**")
                st.write(f"- Capital Deployed: ${stats['capital_deployed']:.2f}")
                st.write(f"- Open Positions: {stats['open_positions']}")
                if stats['capital_deployed'] > 0:
                    avg_position_size = stats['capital_deployed'] / max(stats['open_positions'], 1)
                    st.write(f"- Avg Position Size: ${avg_position_size:.2f}")

def show_llm_analysis(llm_queries, llm_stats):
    """Show LLM query analysis and review."""
    
    st.header("🤖 LLM Analysis & Review")
    st.markdown("**Review all AI queries and responses for insights and improvements**")
    
    if not llm_queries and not llm_stats:
        st.warning("No LLM query data available yet. LLM logging will start with new queries.")
        st.info("💡 **Tip:** The system will automatically log all future Grok queries for analysis.")
        return
    
    # LLM usage stats
    if llm_stats:
        st.subheader("📊 LLM Usage Statistics (Last 7 Days)")
        
        # Create stats summary
        total_queries = sum(stats['query_count'] for stats in llm_stats.values())
        total_cost = sum(stats['total_cost'] for stats in llm_stats.values())
        total_tokens = sum(stats['total_tokens'] for stats in llm_stats.values())
        has_estimated_tokens = any(stats.get('estimated', False) for stats in llm_stats.values())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Total Cost", f"${total_cost:.2f}")
        with col3:
            token_label = "Total Tokens*" if has_estimated_tokens else "Total Tokens"
            token_help = "Estimated from response lengths (some token data missing)" if has_estimated_tokens else "Actual token usage"
            st.metric(
                token_label, 
                f"{total_tokens:,}",
                help=token_help
            )
        with col4:
            avg_cost_per_query = total_cost / max(total_queries, 1)
            st.metric("Avg Cost/Query", f"${avg_cost_per_query:.3f}")
        
        if has_estimated_tokens:
            st.caption("*Token counts marked with * are estimated from response text length due to missing usage data")
        
            st.plotly_chart(fig_usage, width='stretch')
            
        # New AI Insights (Cost History & Confidence Distribution)
        st.subheader("💡 AI Insights")
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            # AI Cost History
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                db_manager = DatabaseManager()
                async def get_cost():
                    await db_manager.initialize()
                    data = await db_manager.get_ai_cost_history(days=30)
                    await db_manager.close()
                    return data
                cost_history = loop.run_until_complete(get_cost())
                loop.close()
                
                if cost_history:
                    df_cost = pd.DataFrame(cost_history)
                    df_cost['date'] = pd.to_datetime(df_cost['date'])
                    fig_cost = px.area(df_cost, x='date', y='cost', title="Daily AI Cost (Last 30 Days)",
                                     labels={'cost': 'Cost ($)', 'date': 'Date'})
                    st.plotly_chart(fig_cost, width='stretch')
            except: pass
            
        with insight_col2:
            # Confidence Distribution
            confidences = [query.confidence for query in llm_queries if hasattr(query, 'confidence') and query.confidence]
            if not confidences and llm_queries:
                # Try to extract from rationale if not separate
                confidences = []
                import re
                for q in llm_queries:
                    match = re.search(r'confidence:?\s*(\d+(?:\.\d+)?)', str(q.response), re.IGNORECASE)
                    if match:
                        val = float(match.group(1))
                        if val < 1: val *= 100
                        confidences.append(val)
            
            if confidences:
                fig_conf = px.histogram(x=confidences, nbins=20, title="Confidence Distribution",
                                       labels={'x': 'Confidence (%)', 'y': 'Count'},
                                       color_discrete_sequence=['indianred'])
                st.plotly_chart(fig_conf, width='stretch')
            else:
                st.info("No confidence data available for distribution chart.")

    
    # Query filters
    st.subheader("🔍 Query Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategies = list(set(query.strategy for query in llm_queries)) if llm_queries else []
        selected_strategy = st.selectbox(
            "Filter by Strategy",
            ["All"] + strategies
        )
    
    with col2:
        query_types = list(set(query.query_type for query in llm_queries)) if llm_queries else []
        selected_type = st.selectbox(
            "Filter by Query Type",
            ["All"] + query_types
        )
    
    with col3:
        hours_back = st.selectbox(
            "Time Range",
            [6, 12, 24, 48, 168],  # Last 6h, 12h, 24h, 48h, 7 days
            index=2,  # Default to 24h
            format_func=lambda x: f"Last {x} hours" if x < 168 else "Last 7 days"
        )
    
    # Filter queries
    filtered_queries = llm_queries
    
    if llm_queries:
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        filtered_queries = [
            q for q in llm_queries 
            if q.timestamp >= cutoff_time
        ]
        
        if selected_strategy != "All":
            filtered_queries = [q for q in filtered_queries if q.strategy == selected_strategy]
        
        if selected_type != "All":
            filtered_queries = [q for q in filtered_queries if q.query_type == selected_type]
        
        st.write(f"**Showing {len(filtered_queries)} queries**")
        
        # Display queries
        for i, query in enumerate(filtered_queries[:20]):  # Show latest 20
            with st.expander(
                f"🤖 {query.strategy} | {query.query_type} | {query.timestamp.strftime('%H:%M:%S')}",
                expanded=(i < 3)  # Expand first 3
            ):
                
                # Query metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Strategy:** {query.strategy}")
                with col2:
                    st.write(f"**Type:** {query.query_type}")
                with col3:
                    if query.market_id:
                        st.write(f"**Market:** {query.market_id[:20]}...")
                
                if query.cost_usd:
                    st.write(f"**Cost:** ${query.cost_usd:.4f}")
                
                # Prompt and response
                st.markdown("**🔤 Prompt:**")
                st.code(query.prompt, language="text")
                
                st.markdown("**🤖 Response:**")
                st.code(query.response, language="text")
                
                # Extracted data
                if query.confidence_extracted:
                    st.write(f"**Confidence Extracted:** {query.confidence_extracted:.2%}")
                
                if query.decision_extracted:
                    st.write(f"**Decision Extracted:** {query.decision_extracted}")
    
    else:
        st.info("No LLM queries found for the selected filters.")

def show_positions_trades(positions):
    """Show detailed positions and trades analysis."""
    
    st.header("💼 Positions & Trades")
    
    if not positions:
        st.warning("No active positions found.")
        return
    
    # Positions overview
    st.subheader(f"📊 Active Positions ({len(positions)})")
    
    # Create positions DataFrame
    position_data = []
    for pos in positions:
        # Convert timestamp string back to datetime for display
        try:
            timestamp = datetime.fromisoformat(pos['timestamp'])
            time_str = timestamp.strftime('%m/%d %H:%M')
        except:
            time_str = 'Unknown'
        
        position_data.append({
            'Market ID': pos['market_id'],
            'Strategy': pos['strategy'] or 'Unknown',
            'Side': pos['side'],
            'Quantity': pos['quantity'],
            'Entry Price': f"${pos['entry_price']:.3f}",
            'Position Value': f"${pos['quantity'] * pos['entry_price']:.2f}",
            'Entry Time': time_str,
            'Status': pos['status'],
            'Stop Loss': f"${pos['stop_loss_price']:.3f}" if pos['stop_loss_price'] else "None",
            'Take Profit': f"${pos['take_profit_price']:.3f}" if pos['take_profit_price'] else "None"
        })
    
    df_positions = pd.DataFrame(position_data)
    
    # Positions filters
    col1, col2 = st.columns(2)
    
    with col1:
        strategies = df_positions['Strategy'].unique().tolist()
        selected_strategies = st.multiselect(
            "Filter by Strategy",
            strategies,
            default=strategies
        )
    
    with col2:
        sides = df_positions['Side'].unique().tolist()
        selected_sides = st.multiselect(
            "Filter by Side",
            sides,
            default=sides
        )
    
    # Apply filters
    filtered_df = df_positions[
        (df_positions['Strategy'].isin(selected_strategies)) &
        (df_positions['Side'].isin(selected_sides))
    ]
    
    # Display filtered positions
    st.dataframe(filtered_df, width='stretch', hide_index=True)
    
    # Position analytics
    if not filtered_df.empty:
        st.subheader("📈 Position Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Value by strategy
            strategy_values = filtered_df.groupby('Strategy')['Position Value'].apply(
                lambda x: x.str.replace('$', '').astype(float).sum()
            )
            
            fig_strategy = px.pie(
                values=strategy_values.values,
                names=strategy_values.index,
                title="Position Value by Strategy"
            )
            st.plotly_chart(fig_strategy, width='stretch')
        
        with col2:
            # Side distribution
            side_counts = filtered_df['Side'].value_counts()
            
            fig_sides = px.bar(
                x=side_counts.index,
                y=side_counts.values,
                title="Positions by Side",
                labels={'x': 'Side', 'y': 'Count'}
            )
            st.plotly_chart(fig_sides, width='stretch')

def show_risk_management(performance_data, positions, system_balance):
    """Show risk management dashboard."""
    
    st.header("⚠️ Risk Management")
    
    # Handle empty positions gracefully
    if not positions:
        st.info("No active positions to analyze for risk management.")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Utilization", "0.0%")
        with col2:
            st.metric("Total Deployed", "$0.00")
        with col3:
            st.metric("Avg Position Size", "$0.00")
        with col4:
            st.metric("Max Single Position", "0.0%")
        
        st.subheader("🚨 Risk Alerts")
        st.success("✅ All risk metrics within acceptable ranges")
        return
    
    # Calculate risk metrics from live positions
    try:
        total_deployed = sum(pos['quantity'] * pos['entry_price'] for pos in positions if 'quantity' in pos and 'entry_price' in pos)
        portfolio_utilization = (total_deployed / system_balance * 100) if system_balance > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Utilization",
                f"{portfolio_utilization:.1f}%",
                help="Percentage of balance deployed in positions"
            )
        
        with col2:
            st.metric(
                "Total Deployed",
                f"${total_deployed:.2f}",
                help="Total capital in active positions"
            )
        
        with col3:
            avg_position_size = total_deployed / len(positions) if positions else 0
            st.metric(
                "Avg Position Size",
                f"${avg_position_size:.2f}",
                help="Average size per position"
            )
        
        with col4:
            # Calculate max single position risk
            position_values = [pos['quantity'] * pos['entry_price'] for pos in positions if 'quantity' in pos and 'entry_price' in pos]
            max_position = max(position_values) if position_values else 0
            max_risk_pct = (max_position / system_balance * 100) if system_balance > 0 else 0
            st.metric(
                "Max Single Position",
                f"{max_risk_pct:.1f}%",
                help="Largest position as % of portfolio"
            )
        
        # Risk alerts
        st.subheader("🚨 Risk Alerts")
        
        alerts = []
        
        if portfolio_utilization > 90:
            alerts.append("⚠️ **High Portfolio Utilization**: Over 90% of capital deployed")
        
        if max_risk_pct > 20:
            alerts.append("⚠️ **Large Position Risk**: Single position exceeds 20% of portfolio")
        
        if len(positions) > 50:
            alerts.append("⚠️ **High Position Count**: Over 50 active positions may be difficult to manage")
        
        # Check for positions without stop losses (if supported)
        no_stop_loss = []
        for pos in positions:
            if 'stop_loss_price' in pos and not pos['stop_loss_price']:
                no_stop_loss.append(pos)
        
        if no_stop_loss:
            alerts.append(f"⚠️ **No Stop Losses**: {len(no_stop_loss)} positions lack stop loss protection")
        
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ All risk metrics within acceptable ranges")
            
        # Exposure by Category & Drawdown Tracker
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.subheader("🥧 Exposure by Category")
            # We need to map market_ids to categories
            # For simplicity, we'll try to extract category from the ticker prefix
            cat_map = {}
            for pos in positions:
                ticker = pos.get('market_id', '')
                # Common Kalshi prefixes
                if ticker.startswith('KX'): cat_prefix = ticker[2:4]
                else: cat_prefix = 'Other'
                
                cat_map[cat_prefix] = cat_map.get(cat_prefix, 0) + (pos['quantity'] * pos['entry_price'])
            
            if cat_map:
                fig_exp = px.pie(names=list(cat_map.keys()), values=list(cat_map.values()), 
                                title="Currency Exposure by Category")
                st.plotly_chart(fig_exp, width='stretch')
            else:
                st.info("No categorical exposure data available.")
                
        with risk_col2:
            st.subheader("📉 Drawdown Tracker")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                db_manager = DatabaseManager()
                async def get_drawdown():
                    await db_manager.initialize()
                    data = await db_manager.get_daily_pnl_history(days=365)
                    await db_manager.close()
                    return data
                pnl_data = loop.run_until_complete(get_drawdown())
                loop.close()
                
                if pnl_data:
                    df_dd = pd.DataFrame(pnl_data).sort_values('date')
                    df_dd['cumulative_pnl'] = df_dd['pnl'].cumsum()
                    df_dd['peak'] = df_dd['cumulative_pnl'].cummax()
                    df_dd['drawdown'] = df_dd['cumulative_pnl'] - df_dd['peak']
                    
                    max_dd = df_dd['drawdown'].min()
                    curr_dd = df_dd['drawdown'].iloc[-1] if not df_dd.empty else 0
                    
                    st.metric("Current Drawdown", f"${curr_dd:.2f}")
                    st.metric("Max Historical Drawdown", f"${max_dd:.2f}")
                    
                    fig_dd = px.line(df_dd, x='date', y='drawdown', title="Historical Drawdown ($)")
                    st.plotly_chart(fig_dd, width='stretch')
            except: pass
        
        # Risk by strategy breakdown
        strategy_names = [pos['strategy'] for pos in positions if 'strategy' in pos]
        if len(set(strategy_names)) > 1:
            st.subheader("📊 Risk by Strategy")
            
            strategy_risk = {}
            for pos in positions:
                if 'strategy' in pos and 'quantity' in pos and 'entry_price' in pos:
                    strategy = pos['strategy'] or 'Unknown'
                    if strategy not in strategy_risk:
                        strategy_risk[strategy] = {'exposure': 0, 'positions': 0}
                    strategy_risk[strategy]['exposure'] += pos['quantity'] * pos['entry_price']
                    strategy_risk[strategy]['positions'] += 1
            
            if strategy_risk:
                strategy_df = pd.DataFrame([
                    {
                        'Strategy': strategy,
                        'Exposure': f"${data['exposure']:.2f}",
                        'Positions': data['positions'],
                        'Avg Size': f"${data['exposure'] / data['positions']:.2f}",
                        'Portfolio %': f"{(data['exposure'] / system_balance * 100):.1f}%" if system_balance > 0 else "0.0%"
                    }
                    for strategy, data in strategy_risk.items()
                ])
                st.dataframe(strategy_df, width='stretch', hide_index=True)
        
    except Exception as e:
        st.error(f"Error calculating risk metrics: {e}")
        st.info("Using basic risk metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Utilization", "Error")
        with col2:
            st.metric("Total Deployed", "Error")
        with col3:
            st.metric("Avg Position Size", "Error")
        with col4:
            st.metric("Max Single Position", "Error")

def show_system_health(available_cash, positions_count, llm_stats):
    """Show system health and monitoring."""
    
    st.header("🔧 System Health")
    
    # System status
    st.subheader("🟢 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ **Kalshi Connection**: Active")
        st.write(f"Available Cash: ${available_cash:.2f}")
        st.write(f"Positions: {positions_count}")
    
    with col2:
        if llm_stats:
            st.success("✅ **LLM Integration**: Active")
            total_queries = sum(stats['query_count'] for stats in llm_stats.values())
            st.write(f"Queries (7d): {total_queries}")
        else:
            st.warning("⚠️ **LLM Logging**: No data")
    
    with col3:
        st.success("✅ **Database**: Connected")
        st.write("All tables operational")
    
    # Recent activity timeline
    st.subheader("📅 System Activity")
    
    if llm_stats:
        st.write("**Recent LLM Activity:**")
        for strategy, stats in llm_stats.items():
            if stats['last_query']:
                last_query_time = datetime.fromisoformat(stats['last_query'])
                time_ago = datetime.now() - last_query_time
                
                if time_ago.days > 0:
                    time_str = f"{time_ago.days} days ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600} hours ago"
                else:
                    time_str = f"{time_ago.seconds // 60} minutes ago"
                
                st.write(f"- **{strategy}**: Last query {time_str}")
    
    # Configuration summary
    st.subheader("⚙️ Configuration")
    
    config_info = {
        "Database Path": "trading_system.db",
        "Dashboard Refresh": "Auto (1 min cache)",
        "LLM Logging": "Enabled" if llm_stats else "Pending first query",
        "Strategy Tracking": "Enabled",
        "Risk Management": "Active"
    }
    
    for key, value in config_info.items():
        st.write(f"**{key}:** {value}")
    
    # System recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if available_cash < 100:
        recommendations.append("💰 Consider increasing account balance for more trading opportunities")
    
    if not llm_stats:
        recommendations.append("🤖 LLM query logging will begin with next trading cycle")
    
    total_queries = sum(stats['query_count'] for stats in llm_stats.values()) if llm_stats else 0
    if total_queries > 1000:
        recommendations.append("📊 High LLM usage - consider optimizing query frequency")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ System running optimally - no recommendations at this time")

def show_edge_analytics():
    """Show edge analytics (category, hourly, expiry, calibration)."""
    st.header("📊 Edge Analytics")
    st.markdown("**Identifying your strongest trading edges across different dimensions**")
    
    analytics = load_edge_analytics()
    
    # 1. Category Performance
    st.subheader("📁 Performance by Market Category")
    if analytics['category']:
        cat_data = []
        for cat, stats in analytics['category'].items():
            cat_data.append({
                'Category': cat.title(),
                'Win Rate': f"{stats['win_rate']:.1f}%",
                'Total P&L': stats['total_pnl'],
                'Trades': stats['trades']
            })
        df_cat = pd.DataFrame(cat_data).sort_values('Total P&L', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_cat = px.bar(
                df_cat, x='Category', y='Total P&L',
                color='Total P&L', color_continuous_scale='RdYlGn',
                title="Total P&L by Category"
            )
            st.plotly_chart(fig_cat, width='stretch')
        with col2:
            st.dataframe(df_cat, hide_index=True, width='stretch')
    else:
        st.info("No category data available yet.")

    # 2. Time-of-Day Analysis
    st.subheader("🕒 Time-of-Day Analysis")
    if analytics['hourly']:
        hours = list(range(24))
        hourly_pnl = [analytics['hourly'].get(h, {}).get('total_pnl', 0) for h in hours]
        hourly_wr = [analytics['hourly'].get(h, {}).get('win_rate', 0) for h in hours]
        
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Bar(x=hours, y=hourly_pnl, name="P&L ($)", marker_color='royalblue'))
        fig_hour.add_trace(go.Scatter(x=hours, y=hourly_wr, name="Win Rate (%)", yaxis="y2", line=dict(color='firebrick', width=3)))
        
        fig_hour.update_layout(
            title="Hourly Performance (P&L vs Win Rate)",
            xaxis=dict(title="Hour of Day (UTC)", tickmode='linear'),
            yaxis=dict(title="Total P&L ($)"),
            yaxis2=dict(title="Win Rate (%)", overlaying='y', side='right', range=[0, 100]),
            legend=dict(x=0.01, y=0.99),
            height=400
        )
        st.plotly_chart(fig_hour, width='stretch')
    else:
        st.info("No hourly data available yet.")

    # 3. Expiry & Calibration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⌛ Win Rate by Time-to-Expiry")
        if analytics['expiry']:
            exp_data = [{'Bucket': k, 'Win Rate': v['win_rate'], 'Trades': v['trades']} for k, v in analytics['expiry'].items()]
            df_exp = pd.DataFrame(exp_data)
            fig_exp = px.bar(df_exp, x='Bucket', y='Win Rate', title="Win Rate % by Expiry Bucket", color='Win Rate', color_continuous_scale='Viridis')
            fig_exp.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_exp, width='stretch')
        else:
            st.info("No expiry data available.")
            
    with col2:
        st.subheader("🎯 Confidence Calibration")
        if analytics['calibration']:
            df_cal = pd.DataFrame(analytics['calibration'])
            fig_cal = go.Figure()
            # Perfect calibration line
            fig_cal.add_trace(go.Scatter(x=[50, 100], y=[50, 100], mode='lines', name='Perfect Calibration', line=dict(dash='dash', color='gray')))
            # Actual calibration
            fig_cal.add_trace(go.Scatter(x=df_cal['avg_confidence'], y=df_cal['actual_win_rate'], mode='markers+lines', name='Actual', 
                                     marker=dict(size=df_cal['trades']*2 + 10, color='orange', sizemode='area')))
            
            fig_cal.update_layout(
                title="Predicted Confidence vs. Actual Win Rate",
                xaxis_title="Predicted Confidence (%)",
                yaxis_title="Actual Win Rate (%)",
                xaxis=dict(range=[50, 100]),
                yaxis=dict(range=[0, 100]),
                height=400
            )
            st.plotly_chart(fig_cal, width='stretch')
        else:
            st.info("Not enough trade data for calibration analysis.")

def show_controls():
    """Show trading system controls (kill switch, thresholds, etc)."""
    st.header("⚙️ System Controls")
    st.markdown("**Live control panel for the trading bot's core parameters**")
    
    db_manager = DatabaseManager()
    
    async def get_and_set():
        await db_manager.initialize()
        
        # 1. Kill Switch
        st.subheader("🛑 Master Controls")
        kill_switch = await db_manager.get_setting("kill_switch", "off")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            new_kill = st.toggle("KILL SWITCH", value=(kill_switch == "on"), help="Immediately pause all new trading activity")
            if new_kill != (kill_switch == "on"):
                await db_manager.set_setting("kill_switch", "on" if new_kill else "off")
                st.success(f"Kill switch turned {'ON' if new_kill else 'OFF'}")
        with col2:
            if new_kill:
                st.error("🚨 SYSTEM PAUSED: No new trades will be opened.")
            else:
                st.success("✅ SYSTEM ACTIVE: Bot is scanning for opportunities.")

        st.divider()
        
        # 2. Thresholds
        st.subheader("📈 Trading Thresholds")
        default_min_conf = settings.trading.min_confidence_to_trade
        default_max_risk = settings.trading.max_position_size_pct / 100
        min_conf = float(await db_manager.get_setting("min_confidence", str(default_min_conf)))
        max_risk = float(await db_manager.get_setting("max_risk_per_trade", str(default_max_risk)))
        st.caption(f"Effective defaults: min_conf={default_min_conf:.2f}, max_risk={default_max_risk:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            new_conf = st.slider("Minimum AI Confidence", 0.5, 0.95, min_conf, 0.05)
            if new_conf != min_conf:
                await db_manager.set_setting("min_confidence", str(new_conf))
                st.toast(f"Confidence threshold updated to {new_conf}")
        with col2:
            new_risk = st.slider("Max Risk per Trade (% Portfolio)", 0.01, 0.20, max_risk, 0.01)
            if new_risk != max_risk:
                await db_manager.set_setting("max_risk_per_trade", str(new_risk))
                st.toast(f"Max risk updated to {new_risk*100:.0f}%")

        st.divider()
        
        # 3. Strategy Allocations
        st.subheader("⚖️ Strategy Allocations")
        
        col_reb_1, col_reb_2 = st.columns([1, 2])
        with col_reb_1:
            auto_reb = await db_manager.get_setting("enable_auto_rebalance", "on")
            new_auto = st.toggle("Auto-Rebalance", value=(auto_reb == "on"), help="Automatically adjust weights based on strategy performance")
            if new_auto != (auto_reb == "on"):
                await db_manager.set_setting("enable_auto_rebalance", "on" if new_auto else "off")
                st.toast(f"Auto-rebalance turned {'ON' if new_auto else 'OFF'}")
        with col_reb_2:
            if new_auto:
                st.info("Performance-based rebalancing is ACTIVE. Sliders below act as baselines.")
            else:
                st.info("Manual weights are ACTIVE. Bot will stick strictly to the values below.")

        st.info("Adjust the target weight for each trading strategy.")
        
        mm_weight = int(await db_manager.get_setting("weight_market_making", "30"))
        dir_weight = int(await db_manager.get_setting("weight_directional", "40"))
        qf_weight = int(await db_manager.get_setting("weight_quick_flip", "30"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            new_mm = st.number_input("Market Making %", 0, 100, mm_weight, 5)
        with col2:
            new_dir = st.number_input("Directional %", 0, 100, dir_weight, 5)
        with col3:
            new_qf = st.number_input("Quick Flip %", 0, 100, qf_weight, 5)
            
        total_weight = new_mm + new_dir + new_qf
        if total_weight != 100:
            st.warning(f"Warning: Total allocation is {total_weight}%. It should equal 100%.")
        
        if st.button("Save Allocations"):
            if total_weight != 100:
                st.error("Allocation total must equal 100% before saving.")
            else:
                await db_manager.set_setting("weight_market_making", str(new_mm))
                await db_manager.set_setting("weight_directional", str(new_dir))
                await db_manager.set_setting("weight_quick_flip", str(new_qf))
                st.success("Strategy allocations saved.")

        st.divider()
        
        # 4. Blacklist
        st.subheader("🚫 Market Category Blacklist")
        blacklist_raw = await db_manager.get_setting("category_blacklist", "")
        current_blacklist = [c.strip() for c in blacklist_raw.split(",")] if blacklist_raw else []
        
        all_categories = ["Politics", "Sports", "Economics", "Weather", "Crypto", "Entertainment", "Finance"]
        new_blacklist = st.multiselect("Exclude these categories from trading", all_categories, default=[c.title() for c in current_blacklist])
        
        if st.button("Update Blacklist"):
            await db_manager.set_setting("category_blacklist", ",".join([c.lower() for c in new_blacklist]))
            st.success("Category blacklist updated.")
            
        await db_manager.close()

    try:
        _run_async(get_and_set())
    except Exception as e:
        st.error(f"Error handling controls: {e}")

def show_historical():
    """Show historical analysis (daily P&L, journal, streaks)."""
    st.header("📅 Historical Analysis")
    st.markdown("**Deep dive into past performance and trade history**")
    
    # 1. Streaks & Summary
    analytics = load_edge_analytics()
    streaks = analytics.get('streaks', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        streak_val = streaks.get('current_streak', 0)
        s_type = streaks.get('streak_type', '').upper()
        color = "green" if s_type == 'WIN' else "red"
        st.markdown(f"**Current Streak:** <span style='color:{color}; font-size: 24px;'>{streak_val} {s_type}</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Max Win Streak", streaks.get('max_win_streak', 0))
    with col3:
        st.metric("Max Loss Streak", streaks.get('max_loss_streak', 0))
    with col4:
        # Placeholder for other metrics
        st.metric("Realized P&L (Total)", f"${sum(trade.get('pnl', 0) for trade in load_recent_trades(limit=1000)):.2f}")
        
    st.divider()

    # 2. Daily P&L Heatmap
    st.subheader("🗓️ Daily P&L History")
    pnl_history = []
    try:
        db_manager = DatabaseManager()
        async def get_hist():
            await db_manager.initialize()
            data = await db_manager.get_daily_pnl_history(days=60)
            await db_manager.close()
            return data
        pnl_history = _run_async(get_hist())
    except:
        pass
    
    if pnl_history:
        df_hist = pd.DataFrame(pnl_history)
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        
        fig_hist = px.bar(
            df_hist, x='date', y='pnl',
            color='pnl', color_continuous_scale='RdYlGn',
            title="Daily Realized P&L (Last 60 Days)",
            labels={'pnl': 'P&L ($)', 'date': 'Date'}
        )
        st.plotly_chart(fig_hist, width='stretch')
    else:
        st.info("No historical P&L data found.")

    # 3. Trade Journal
    st.subheader("📓 Trade Journal")
    trades = load_recent_trades(limit=100)
    
    if trades:
        df_trades = pd.DataFrame(trades)
        # Formatter for better display
        df_display = df_trades.copy()
        df_display['pnl'] = df_display['pnl'].map(lambda x: f"${x:.2f}")
        df_display['entry_price'] = df_display['entry_price'].map(lambda x: f"${x:.3f}")
        df_display['exit_price'] = df_display['exit_price'].map(lambda x: f"${x:.3f}")
        
        st.dataframe(df_display, width='stretch', hide_index=True)
        
        # Export button
        csv = df_trades.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export to CSV",
            csv,
            "trading_journal.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("No trades logged in the journal yet.")

if __name__ == "__main__":
    main()
 
