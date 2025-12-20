#!/usr/bin/env python3
"""
Example usage of EnsembleMonitor for real-time AI model performance monitoring.

This example demonstrates how to integrate the EnsembleMonitor into the existing
trading system to track model performance, analyze contributions, and generate alerts.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

# Example of how to use EnsembleMonitor in a trading system
async def ensemble_monitoring_example():
    """
    Example showing EnsembleMonitor integration with trading decisions.
    """
    from src.intelligence.ensemble_monitor import EnsembleMonitor

    # Initialize the monitor
    monitor = EnsembleMonitor("trading_system.db")
    await monitor.initialize()

    # Example 1: Track ensemble decision with outcome
    ensemble_decision = {
        'market_id': 'tech-stock-2024-Q1',
        'models_consulted': ['grok-4', 'gpt-4', 'claude-3'],
        'final_decision': 'YES',
        'disagreement_level': 0.25,
        'selected_model': 'grok-4',
        'reasoning': 'Strong consensus on technology sector growth',
        'market_category': 'technology',
        'confidence_extracted': 0.85,
        'outcome': 'success',
        'pnl': 150.0,
        'cost_usd': 0.12,
        'response_time_ms': 2200,
        'timestamp': datetime.now()
    }

    await monitor.track_performance(ensemble_decision)

    # Example 2: Track model costs
    cost_data = {
        'model_name': 'grok-4',
        'cost_usd': 0.12,
        'market_category': 'technology',
        'tokens_used': 2400,
        'timestamp': datetime.now()
    }

    await monitor.track_model_cost(cost_data)

    # Example 3: Get real-time metrics
    realtime_metrics = await monitor.get_realtime_metrics()
    print(f"Real-time metrics: {realtime_metrics}")

    # Example 4: Analyze model contributions
    contributions = await monitor.analyze_model_contributions()
    print(f"Model contributions: {contributions}")

    # Example 5: Get cost breakdown
    cost_breakdown = await monitor.get_cost_breakdown()
    print(f"Cost breakdown: {cost_breakdown}")

    # Example 6: Generate dashboard data
    dashboard_data = await monitor.generate_dashboard_data()
    print(f"Dashboard data generated with {len(dashboard_data)} components")

    return dashboard_data

async def alert_monitoring_example():
    """
    Example showing how to monitor for performance alerts.
    """
    from src.intelligence.ensemble_monitor import EnsembleMonitor

    monitor = EnsembleMonitor("trading_system.db")
    await monitor.initialize()

    # Simulate poor performance that should trigger alerts
    poor_decisions = []
    for i in range(15):  # 15 consecutive failures to trigger alerts
        poor_decision = {
            'market_id': f'poor-trade-{i}',
            'models_consulted': ['grok-4'],
            'final_decision': 'YES',
            'disagreement_level': 0.1,
            'selected_model': 'grok-4',
            'reasoning': 'Test decision',
            'outcome': 'failure',  # All failures
            'pnl': -25.0,
            'timestamp': datetime.now() - timedelta(minutes=i*5)
        }
        poor_decisions.append(poor_decision)
        await monitor.track_performance(poor_decision)

    # Check for alerts
    alerts = await monitor.check_performance_alerts()
    print(f"Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  - {alert['type']}: {alert['message']} (Severity: {alert['severity']})")

async def agreement_analysis_example():
    """
    Example showing ensemble agreement/disagreement analysis.
    """
    from src.intelligence.ensemble_monitor import EnsembleMonitor

    monitor = EnsembleMonitor("trading_system.db")
    await monitor.initialize()

    # Simulate different agreement scenarios
    scenarios = [
        # High agreement scenario
        {
            'market_id': 'consensus-trade-1',
            'models_consulted': ['grok-4', 'gpt-4'],
            'disagreement_level': 0.1,
            'final_decision': 'YES',
            'selected_model': 'grok-4',
            'model_votes': {'grok-4': 'YES', 'gpt-4': 'YES'},
            'market_category': 'finance',
            'outcome': 'success',
            'pnl': 80.0,
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        # High disagreement scenario
        {
            'market_id': 'disagreement-trade-1',
            'models_consulted': ['grok-4', 'gpt-4', 'claude-3'],
            'disagreement_level': 0.9,
            'final_decision': 'NO',
            'selected_model': 'gpt-4',
            'model_votes': {'grok-4': 'YES', 'gpt-4': 'NO', 'claude-3': 'SKIP'},
            'market_category': 'sports',
            'outcome': 'failure',
            'pnl': -45.0,
            'timestamp': datetime.now() - timedelta(hours=1)
        }
    ]

    for scenario in scenarios:
        await monitor.track_performance(scenario)

    # Analyze agreement patterns
    agreement_analysis = await monitor.analyze_agreement_patterns()
    print(f"Agreement analysis: {agreement_analysis}")

    # Get detailed disagreement correlation
    correlation = await monitor.analyze_disagreement_correlation()
    print(f"Disagreement correlation: {correlation}")

async def performance_tracking_example():
    """
    Example showing comprehensive performance tracking.
    """
    from src.intelligence.ensemble_monitor import EnsembleMonitor

    monitor = EnsembleMonitor("trading_system.db")
    await monitor.initialize()

    # Track performance across different models and categories
    models = ['grok-4', 'gpt-4', 'claude-3']
    categories = ['technology', 'finance', 'sports', 'politics']

    for i in range(50):  # 50 sample decisions
        model = models[i % len(models)]
        category = categories[i % len(categories)]

        # Simulate varying performance
        success = (i % 3) == 0  # ~33% success rate
        pnl = 100.0 if success else -40.0

        decision = {
            'market_id': f'perf-test-{i}',
            'models_consulted': [model],
            'final_decision': 'YES' if success else 'NO',
            'disagreement_level': 0.2,
            'selected_model': model,
            'reasoning': f'Performance test {i}',
            'market_category': category,
            'outcome': 'success' if success else 'failure',
            'pnl': pnl,
            'cost_usd': 0.05 + (i % 5) * 0.01,  # Varying costs
            'timestamp': datetime.now() - timedelta(minutes=i*10)
        }

        await monitor.track_performance(decision)

    # Get comprehensive analytics
    performance_windows = await monitor.get_performance_windows()
    print(f"Performance windows: {list(performance_windows.keys())}")

    # Get model strengths analysis
    strengths = await monitor.identify_model_strengths()
    print(f"Model strengths identified for {len(strengths)} models")

    # Calculate contribution metrics
    contribution_metrics = await monitor.calculate_contribution_metrics()
    print(f"Contribution metrics: {contribution_metrics}")

def integration_with_existing_system():
    """
    Example of how to integrate EnsembleMonitor with existing decide.py logic.
    """
    integration_code = '''
# In src/jobs/decide.py, integrate monitoring like this:

async def enhanced_ensemble_decide(market_id: str, potential_investment: float):
    """Enhanced decision making with monitoring integration."""

    from src.intelligence.ensemble_monitor import EnsembleMonitor

    # Initialize monitor if not already done
    if not hasattr(enhanced_ensemble_decide, 'monitor'):
        enhanced_ensemble_decide.monitor = EnsembleMonitor()
        await enhanced_ensemble_decide.monitor.initialize()

    monitor = enhanced_ensemble_decide.monitor

    # Make ensemble decision (existing logic)
    decision_result = await ensemble_decide_logic(market_id, potential_investment)

    # Track the decision with monitoring
    tracking_data = {
        'market_id': market_id,
        'models_consulted': decision_result['models_consulted'],
        'final_decision': decision_result['decision'],
        'disagreement_level': decision_result['disagreement_level'],
        'selected_model': decision_result['selected_model'],
        'reasoning': decision_result['reasoning'],
        'market_category': decision_result.get('market_category', 'unknown'),
        'confidence_extracted': decision_result.get('confidence'),
        'timestamp': datetime.now()
    }

    await monitor.track_performance(tracking_data)

    # Check for alerts
    alerts = await monitor.check_performance_alerts()
    if alerts:
        logger.warning(f"Performance alerts triggered: {[a['type'] for a in alerts]}")

    return decision_result
    '''

    print("Integration example code for decide.py:")
    print(integration_code)

def dashboard_integration_example():
    """
    Example of how to integrate monitoring data with trading_dashboard.py.
    """
    dashboard_code = '''
# In trading_dashboard.py, add monitoring dashboard section:

import asyncio
from src.intelligence.ensemble_monitor import EnsembleMonitor

def display_ensemble_monitoring():
    """Display ensemble monitoring analytics in dashboard."""

    # Run async operation in sync context
    loop = asyncio.new_event_loop()
    try:
        monitor = EnsembleMonitor()
        loop.run_until_complete(monitor.initialize())
        dashboard_data = loop.run_until_complete(monitor.generate_dashboard_data())
    finally:
        loop.close()

    # Display real-time metrics
    if 'summary_metrics' in dashboard_data:
        metrics = dashboard_data['summary_metrics']
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Success Rate",
                f"{metrics.get('success_rate', 0):.1%}",
                delta=f"{metrics.get('recent_trend', 'stable')}"
            )

        with col2:
            st.metric(
                "Total Decisions",
                metrics.get('total_decisions', 0)
            )

        with col3:
            st.metric(
                "Total P&L",
                f"${metrics.get('total_pnl', 0):,.2f}"
            )

    # Display model performance
    if 'model_performance' in dashboard_data:
        st.subheader("Model Performance Analysis")

        for model, perf in dashboard_data['model_performance'].items():
            with st.expander(f"{model} Performance"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"Success Rate: {perf.get('success_rate', 0):.1%}")
                    st.write(f"Total Decisions: {perf.get('total_decisions', 0)}")
                    st.write(f"Average P&L: ${perf.get('avg_pnl_per_decision', 0):.2f}")

                with col2:
                    st.write(f"Top Categories:")
                    for cat, cat_data in perf.get('by_category', {}).items():
                        st.write(f"  • {cat}: {cat_data.get('success_rate', 0):.1%}")

    # Display cost analysis
    if 'cost_analysis' in dashboard_data:
        st.subheader("Cost Analysis")

        cost_data = dashboard_data['cost_analysis']
        if 'by_model' in cost_data:
            st.write("**Cost by Model:**")
            for model, costs in cost_data['by_model'].items():
                st.write(f"{model}: ${costs.get('total_cost', 0):.4f} "
                         f"({costs.get('avg_cost_per_request', 0):.4f} per request, "
                         f"ROI: {costs.get('roi_ratio', 0):.1f})")

    # Display alerts
    if 'alerts' in dashboard_data and dashboard_data['alerts']:
        st.subheader("⚠️ Performance Alerts")

        for alert in dashboard_data['alerts']:
            severity_color = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }.get(alert['severity'], '⚪')

            st.error(f"{severity_color} {alert['type'].replace('_', ' ').title()}: "
                    f"{alert['message']}")
    '''

    print("Dashboard integration example:")
    print(dashboard_code)

if __name__ == '__main__':
    print("🚀 EnsembleMonitor Integration Examples")
    print("=" * 50)

    print("\n1. Basic monitoring example:")
    print("Run this to see basic EnsembleMonitor usage:")
    print("python -c 'import asyncio; from examples.ensemble_monitoring_example import ensemble_monitoring_example; asyncio.run(ensemble_monitoring_example())'")

    print("\n2. Alert monitoring example:")
    print("Demonstrates alert generation for performance degradation")

    print("\n3. Agreement analysis example:")
    print("Shows how to analyze ensemble agreement/disagreement patterns")

    print("\n4. Performance tracking example:")
    print("Comprehensive performance tracking across models and categories")

    print("\n5. Integration examples:")
    print("- decide.py integration for real-time monitoring")
    print("- trading_dashboard.py integration for visualization")

    print("\n" + "=" * 50)
    print("📚 For complete examples, see the functions in this file.")
    print("🎯 Key Features:")
    print("   • Real-time performance tracking")
    print("   • Model contribution analysis")
    print("   • Cost breakdown and ROI analysis")
    print("   • Ensemble agreement/disagreement tracking")
    print("   • Automated alert generation")
    print("   • Dashboard data aggregation")
    print("   • Performance trend analysis")