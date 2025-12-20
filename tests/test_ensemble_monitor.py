"""
Tests for EnsembleMonitor class - Task 8.1: Monitoring and Analytics Dashboard
Tests focused on real-time ensemble performance monitoring, contribution analysis,
cost breakdown, agreement tracking, and dashboard data presentation.
"""

import pytest
import asyncio
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.intelligence.ensemble_monitor import EnsembleMonitor
from src.utils.database import DatabaseManager, ModelPerformance, EnsembleDecision


class TestEnsembleMonitorRealtimePerformance:
    """Test real-time ensemble performance monitoring."""

    @pytest.fixture
    async def monitor(self):
        """Create EnsembleMonitor instance for testing."""
        monitor = EnsembleMonitor(":memory:")
        await monitor.db.initialize()
        return monitor

    @pytest.mark.asyncio
    async def test_realtime_performance_tracking(self, monitor):
        """Test that ensemble performance is tracked in real-time."""
        # Setup test data
        now = datetime.now()
        decisions = [
            {
                'market_id': 'market_1',
                'models_consulted': ['grok-4', 'gpt-4', 'claude-3'],
                'final_decision': 'YES',
                'disagreement_level': 0.2,
                'selected_model': 'grok-4',
                'reasoning': 'Strong bullish sentiment',
                'timestamp': now,
                'outcome': 'success',
                'pnl': 50.0
            },
            {
                'market_id': 'market_2',
                'models_consulted': ['grok-3', 'gpt-4'],
                'final_decision': 'NO',
                'disagreement_level': 0.8,
                'selected_model': 'grok-3',
                'reasoning': 'High disagreement, conservative choice',
                'timestamp': now + timedelta(minutes=5),
                'outcome': 'failure',
                'pnl': -30.0
            }
        ]

        # Test performance tracking
        for decision in decisions:
            await monitor.track_performance(decision)

        # Verify real-time metrics
        realtime_metrics = await monitor.get_realtime_metrics()

        assert realtime_metrics['total_decisions'] == 2
        assert realtime_metrics['success_rate'] == 0.5
        assert realtime_metrics['avg_disagreement'] == 0.5
        assert realtime_metrics['total_pnl'] == 20.0
        assert realtime_metrics['recent_trend'] == 'stable'  # One success, one failure

    @pytest.mark.asyncio
    async def test_performance_alert_generation(self, monitor):
        """Test alert generation for performance degradation."""
        # Simulate poor recent performance
        now = datetime.now()
        for i in range(10):
            decision = {
                'market_id': f'market_{i}',
                'models_consulted': ['grok-4'],
                'final_decision': 'YES',
                'disagreement_level': 0.1,
                'selected_model': 'grok-4',
                'reasoning': 'Test decision',
                'timestamp': now - timedelta(minutes=i),
                'outcome': 'failure',  # All failures
                'pnl': -10.0
            }
            await monitor.track_performance(decision)

        # Check for alerts
        alerts = await monitor.check_performance_alerts()

        assert len(alerts) > 0
        alert_types = [alert['type'] for alert in alerts]
        assert 'low_success_rate' in alert_types
        assert 'consecutive_losses' in alert_types

        # Verify alert details
        success_rate_alert = next(a for a in alerts if a['type'] == 'low_success_rate')
        assert success_rate_alert['severity'] == 'high'
        assert success_rate_alert['current_value'] == 0.0
        assert success_rate_alert['threshold'] == 0.3

    @pytest.mark.asyncio
    async def test_realtime_performance_windows(self, monitor):
        """Test real-time performance tracking across time windows."""
        now = datetime.now()

        # Insert data with different timestamps
        for hours_ago in [1, 2, 6, 12, 24, 48]:
            decision = {
                'market_id': f'market_{hours_ago}',
                'models_consulted': ['grok-4'],
                'final_decision': 'YES',
                'disagreement_level': 0.2,
                'selected_model': 'grok-4',
                'reasoning': 'Test decision',
                'timestamp': now - timedelta(hours=hours_ago),
                'outcome': 'success' if hours_ago % 2 == 0 else 'failure',
                'pnl': 20.0 if hours_ago % 2 == 0 else -15.0
            }
            await monitor.track_performance(decision)

        # Test different time windows
        windows = await monitor.get_performance_windows()

        assert '1h' in windows
        assert '6h' in windows
        assert '24h' in windows
        assert '7d' in windows

        # Verify window calculations
        assert windows['1h']['decisions'] <= windows['6h']['decisions']
        assert windows['6h']['decisions'] <= windows['24h']['decisions']
        assert windows['24h']['decisions'] <= windows['7d']['decisions']


class TestModelContributionAnalysis:
    """Test model contribution analysis for successful trades."""

    @pytest.fixture
    async def monitor(self):
        """Create EnsembleMonitor instance for testing."""
        monitor = EnsembleMonitor(":memory:")
        await monitor.db.initialize()
        return monitor

    @pytest.mark.asyncio
    async def test_model_success_attribution(self, monitor):
        """Test that successful trades are properly attributed to models."""
        now = datetime.now()

        # Simulate decisions with known outcomes
        decisions = [
            # grok-4 successful trades
            {
                'market_id': 'market_1',
                'models_consulted': ['grok-4', 'gpt-4'],
                'selected_model': 'grok-4',
                'market_category': 'technology',
                'final_decision': 'YES',
                'outcome': 'success',
                'pnl': 100.0,
                'timestamp': now
            },
            {
                'market_id': 'market_2',
                'models_consulted': ['grok-4', 'gpt-4', 'claude-3'],
                'selected_model': 'grok-4',
                'market_category': 'technology',
                'final_decision': 'NO',
                'outcome': 'success',
                'pnl': 75.0,
                'timestamp': now + timedelta(minutes=5)
            },
            # gpt-4 mixed results
            {
                'market_id': 'market_3',
                'models_consulted': ['gpt-4'],
                'selected_model': 'gpt-4',
                'market_category': 'finance',
                'final_decision': 'YES',
                'outcome': 'success',
                'pnl': 50.0,
                'timestamp': now + timedelta(minutes=10)
            },
            {
                'market_id': 'market_4',
                'models_consulted': ['gpt-4'],
                'selected_model': 'gpt-4',
                'market_category': 'finance',
                'final_decision': 'NO',
                'outcome': 'failure',
                'pnl': -25.0,
                'timestamp': now + timedelta(minutes=15)
            }
        ]

        for decision in decisions:
            await monitor.track_performance(decision)

        # Analyze contributions
        contributions = await monitor.analyze_model_contributions()

        # Verify grok-4 attribution
        grok4_contrib = contributions['grok-4']
        assert grok4_contrib['total_successes'] == 2
        assert grok4_contrib['total_pnl'] == 175.0
        assert grok4_contrib['success_rate'] == 1.0
        assert 'technology' in grok4_contrib['by_category']

        # Verify gpt-4 attribution
        gpt4_contrib = contributions['gpt-4']
        assert gpt4_contrib['total_successes'] == 1
        assert gpt4_contrib['total_failures'] == 1
        assert gpt4_contrib['total_pnl'] == 25.0
        assert gpt4_contrib['success_rate'] == 0.5
        assert 'finance' in gpt4_contrib['by_category']

    @pytest.mark.asyncio
    async def test_model_strength_identification(self, monitor):
        """Test identification of model strengths in specific conditions."""
        now = datetime.now()

        # Create scenarios highlighting model strengths
        scenarios = [
            # grok-4 excels in technology markets
            {
                'market_id': 'tech_1',
                'selected_model': 'grok-4',
                'market_category': 'technology',
                'volatility_regime': 'high',
                'outcome': 'success',
                'pnl': 80.0,
                'timestamp': now
            },
            {
                'market_id': 'tech_2',
                'selected_model': 'grok-4',
                'market_category': 'technology',
                'volatility_regime': 'high',
                'outcome': 'success',
                'pnl': 60.0,
                'timestamp': now + timedelta(minutes=5)
            },
            # gpt-4 excels in finance markets
            {
                'market_id': 'finance_1',
                'selected_model': 'gpt-4',
                'market_category': 'finance',
                'volatility_regime': 'low',
                'outcome': 'success',
                'pnl': 40.0,
                'timestamp': now + timedelta(minutes=10)
            },
            {
                'market_id': 'finance_2',
                'selected_model': 'gpt-4',
                'market_category': 'finance',
                'volatility_regime': 'low',
                'outcome': 'success',
                'pnl': 35.0,
                'timestamp': now + timedelta(minutes=15)
            }
        ]

        for scenario in scenarios:
            decision = {
                'models_consulted': [scenario['selected_model']],
                'final_decision': 'YES',
                'disagreement_level': 0.1,
                'reasoning': 'Test decision',
                **scenario
            }
            await monitor.track_performance(decision)

        # Identify model strengths
        strengths = await monitor.identify_model_strengths()

        # Verify grok-4 technology strength
        grok4_strengths = strengths['grok-4']
        assert 'technology' in grok4_strengths['top_categories']
        assert grok4_strengths['top_categories']['technology']['success_rate'] == 1.0
        assert grok4_strengths['top_categories']['technology']['avg_pnl'] > 0

        # Verify gpt-4 finance strength
        gpt4_strengths = strengths['gpt-4']
        assert 'finance' in gpt4_strengths['top_categories']
        assert gpt4_strengths['top_categories']['finance']['success_rate'] == 1.0
        assert gpt4_strengths['top_categories']['finance']['avg_pnl'] > 0

    @pytest.mark.asyncio
    async def test_contribution_metrics_calculation(self, monitor):
        """Test contribution metrics calculations."""
        now = datetime.now()

        # Insert test data
        for i in range(20):
            outcome = 'success' if i % 3 == 0 else 'failure'  # ~33% success rate
            pnl = 50.0 if outcome == 'success' else -20.0
            model = 'grok-4' if i % 2 == 0 else 'gpt-4'

            decision = {
                'market_id': f'market_{i}',
                'models_consulted': [model],
                'selected_model': model,
                'market_category': 'mixed',
                'final_decision': 'YES',
                'outcome': outcome,
                'pnl': pnl,
                'timestamp': now + timedelta(minutes=i)
            }
            await monitor.track_performance(decision)

        # Calculate contribution metrics
        metrics = await monitor.calculate_contribution_metrics()

        # Verify overall metrics
        assert 'total_decisions' in metrics
        assert 'total_pnl' in metrics
        assert 'overall_success_rate' in metrics

        # Verify per-model metrics
        assert 'grok-4' in metrics['by_model']
        assert 'gpt-4' in metrics['by_model']

        # Check metric calculations
        for model in ['grok-4', 'gpt-4']:
            model_metrics = metrics['by_model'][model]
            assert 'success_rate' in model_metrics
            assert 'avg_pnl_per_decision' in model_metrics
            assert 'contribution_percentage' in model_metrics
            assert 'risk_adjusted_return' in model_metrics

            # Validate metric ranges
            assert 0 <= model_metrics['success_rate'] <= 1
            assert isinstance(model_metrics['avg_pnl_per_decision'], (int, float))
            assert 0 <= model_metrics['contribution_percentage'] <= 100


class TestCostBreakdownAndReporting:
    """Test cost breakdown and reporting by model and market category."""

    @pytest.fixture
    async def monitor(self):
        """Create EnsembleMonitor instance for testing."""
        monitor = EnsembleMonitor(":memory:")
        await monitor.db.initialize()
        return monitor

    @pytest.mark.asyncio
    async def test_cost_breakdown_by_model(self, monitor):
        """Test cost breakdown by AI model."""
        now = datetime.now()

        # Simulate model usage with different costs
        model_costs = [
            {
                'model_name': 'grok-4',
                'cost_usd': 0.05,
                'market_category': 'technology',
                'timestamp': now,
                'tokens_used': 1000
            },
            {
                'model_name': 'grok-4',
                'cost_usd': 0.08,
                'market_category': 'finance',
                'timestamp': now + timedelta(minutes=5),
                'tokens_used': 1600
            },
            {
                'model_name': 'gpt-4',
                'cost_usd': 0.15,
                'market_category': 'technology',
                'timestamp': now + timedelta(minutes=10),
                'tokens_used': 3000
            },
            {
                'model_name': 'claude-3',
                'cost_usd': 0.12,
                'market_category': 'technology',
                'timestamp': now + timedelta(minutes=15),
                'tokens_used': 2400
            }
        ]

        for cost_data in model_costs:
            await monitor.track_model_cost(cost_data)

        # Generate cost breakdown
        cost_breakdown = await monitor.get_cost_breakdown()

        # Verify model-level breakdown
        assert 'grok-4' in cost_breakdown['by_model']
        assert 'gpt-4' in cost_breakdown['by_model']
        assert 'claude-3' in cost_breakdown['by_model']

        # Check grok-4 costs
        grok4_costs = cost_breakdown['by_model']['grok-4']
        assert grok4_costs['total_cost'] == 0.13  # 0.05 + 0.08
        assert grok4_costs['total_requests'] == 2
        assert grok4_costs['avg_cost_per_request'] == 0.065
        assert grok4_costs['total_tokens'] == 2600

        # Check gpt-4 costs
        gpt4_costs = cost_breakdown['by_model']['gpt-4']
        assert gpt4_costs['total_cost'] == 0.15
        assert gpt4_costs['total_requests'] == 1
        assert gpt4_costs['avg_cost_per_request'] == 0.15
        assert gpt4_costs['total_tokens'] == 3000

    @pytest.mark.asyncio
    async def test_cost_breakdown_by_category(self, monitor):
        """Test cost breakdown by market category."""
        now = datetime.now()

        category_costs = [
            {
                'model_name': 'grok-4',
                'cost_usd': 0.05,
                'market_category': 'technology',
                'timestamp': now
            },
            {
                'model_name': 'gpt-4',
                'cost_usd': 0.15,
                'market_category': 'technology',
                'timestamp': now + timedelta(minutes=5)
            },
            {
                'model_name': 'grok-4',
                'cost_usd': 0.08,
                'market_category': 'finance',
                'timestamp': now + timedelta(minutes=10)
            },
            {
                'model_name': 'claude-3',
                'cost_usd': 0.12,
                'market_category': 'sports',
                'timestamp': now + timedelta(minutes=15)
            }
        ]

        for cost_data in category_costs:
            await monitor.track_model_cost(cost_data)

        # Generate category breakdown
        category_breakdown = await monitor.get_cost_breakdown_by_category()

        # Verify category-level breakdown
        assert 'technology' in category_breakdown
        assert 'finance' in category_breakdown
        assert 'sports' in category_breakdown

        # Check technology costs (should include grok-4 and gpt-4)
        tech_costs = category_breakdown['technology']
        assert tech_costs['total_cost'] == 0.20  # 0.05 + 0.15
        assert tech_costs['total_requests'] == 2
        assert tech_costs['models_used'] == ['grok-4', 'gpt-4']

        # Check finance costs
        finance_costs = category_breakdown['finance']
        assert finance_costs['total_cost'] == 0.08
        assert finance_costs['total_requests'] == 1
        assert finance_costs['models_used'] == ['grok-4']

    @pytest.mark.asyncio
    async def test_cost_efficiency_analysis(self, monitor):
        """Test cost efficiency analysis and ROI calculation."""
        now = datetime.now()

        # Track costs and outcomes together
        performance_data = [
            {
                'model_name': 'grok-4',
                'cost_usd': 0.05,
                'outcome': 'success',
                'pnl': 100.0,
                'market_category': 'technology',
                'timestamp': now
            },
            {
                'model_name': 'grok-4',
                'cost_usd': 0.08,
                'outcome': 'failure',
                'pnl': -30.0,
                'market_category': 'technology',
                'timestamp': now + timedelta(minutes=5)
            },
            {
                'model_name': 'gpt-4',
                'cost_usd': 0.15,
                'outcome': 'success',
                'pnl': 150.0,
                'market_category': 'finance',
                'timestamp': now + timedelta(minutes=10)
            }
        ]

        for data in performance_data:
            await monitor.track_model_cost(data)
            await monitor.track_performance(data)

        # Analyze cost efficiency
        efficiency = await monitor.analyze_cost_efficiency()

        # Verify grok-4 efficiency
        grok4_eff = efficiency['grok-4']
        assert grok4_eff['total_cost'] == 0.13
        assert grok4_eff['total_pnl'] == 70.0  # 100 - 30
        assert grok4_eff['roi_ratio'] > 500  # 70 / 0.13
        assert grok4_eff['success_rate'] == 0.5

        # Verify gpt-4 efficiency
        gpt4_eff = efficiency['gpt-4']
        assert gpt4_eff['total_cost'] == 0.15
        assert gpt4_eff['total_pnl'] == 150.0
        assert gpt4_eff['roi_ratio'] == 1000  # 150 / 0.15
        assert gpt4_eff['success_rate'] == 1.0

        # Check comparative efficiency
        assert 'most_efficient' in efficiency
        assert 'least_efficient' in efficiency
        assert efficiency['most_efficient']['model'] == 'gpt-4'  # Higher ROI


class TestEnsembleAgreementTracking:
    """Test ensemble agreement/disagreement tracking with decision quality correlation."""

    @pytest.fixture
    async def monitor(self):
        """Create EnsembleMonitor instance for testing."""
        monitor = EnsembleMonitor(":memory:")
        await monitor.db.initialize()
        return monitor

    @pytest.mark.asyncio
    async def test_agreement_level_tracking(self, monitor):
        """Test tracking of ensemble agreement levels."""
        now = datetime.now()

        # Simulate different agreement scenarios
        agreement_scenarios = [
            {
                'market_id': 'market_1',
                'models_consulted': ['grok-4', 'gpt-4', 'claude-3'],
                'disagreement_level': 0.1,  # High agreement
                'final_decision': 'YES',
                'selected_model': 'grok-4',
                'outcome': 'success',
                'pnl': 80.0,
                'timestamp': now
            },
            {
                'market_id': 'market_2',
                'models_consulted': ['grok-4', 'gpt-4'],
                'disagreement_level': 0.9,  # High disagreement
                'final_decision': 'NO',
                'selected_model': 'gpt-4',
                'outcome': 'failure',
                'pnl': -45.0,
                'timestamp': now + timedelta(minutes=5)
            },
            {
                'market_id': 'market_3',
                'models_consulted': ['grok-4', 'gpt-4', 'claude-3'],
                'disagreement_level': 0.2,  # Moderate agreement
                'final_decision': 'YES',
                'selected_model': 'claude-3',
                'outcome': 'success',
                'pnl': 60.0,
                'timestamp': now + timedelta(minutes=10)
            },
            {
                'market_id': 'market_4',
                'models_consulted': ['grok-4', 'gpt-4'],
                'disagreement_level': 0.8,  # High disagreement
                'final_decision': 'YES',
                'selected_model': 'grok-4',
                'outcome': 'failure',
                'pnl': -35.0,
                'timestamp': now + timedelta(minutes=15)
            }
        ]

        for scenario in agreement_scenarios:
            decision = {
                'reasoning': 'Test decision',
                **scenario
            }
            await monitor.track_performance(decision)

        # Analyze agreement patterns
        agreement_analysis = await monitor.analyze_agreement_patterns()

        # Verify agreement level tracking
        assert 'high_agreement' in agreement_analysis
        assert 'high_disagreement' in agreement_analysis
        assert 'moderate_agreement' in agreement_analysis

        # Check high agreement outcomes (should be better)
        high_agreement = agreement_analysis['high_agreement']
        assert high_agreement['avg_disagreement'] <= 0.25
        assert high_agreement['success_rate'] == 1.0  # Both high agreement cases succeeded
        assert high_agreement['avg_pnl'] == 70.0  # (80 + 60) / 2

        # Check high disagreement outcomes (should be worse)
        high_disagreement = agreement_analysis['high_disagreement']
        assert high_disagreement['avg_disagreement'] >= 0.75
        assert high_disagreement['success_rate'] == 0.0  # Both high disagreement cases failed
        assert high_disagreement['avg_pnl'] == -40.0  # (-45 - 35) / 2

    @pytest.mark.asyncio
    async def test_decision_quality_correlation(self, monitor):
        """Test correlation between agreement levels and decision quality."""
        now = datetime.now()

        # Create data showing correlation patterns
        correlation_data = []
        for i in range(50):
            disagreement_level = i / 50  # 0.0 to 0.98
            # Higher disagreement leads to lower success rate (negative correlation)
            success_probability = max(0.1, 1.0 - disagreement_level * 0.8)
            outcome = 'success' if random.random() < success_probability else 'failure'
            pnl = 100.0 if outcome == 'success' else -50.0

            correlation_data.append({
                'market_id': f'market_{i}',
                'models_consulted': ['grok-4', 'gpt-4'],
                'disagreement_level': disagreement_level,
                'final_decision': 'YES',
                'selected_model': 'grok-4',
                'outcome': outcome,
                'pnl': pnl,
                'timestamp': now + timedelta(minutes=i)
            })

        for data in correlation_data:
            decision = {
                'reasoning': 'Test decision',
                **data
            }
            await monitor.track_performance(decision)

        # Analyze correlation
        correlation = await monitor.analyze_disagreement_correlation()

        # Verify correlation analysis
        assert 'correlation_coefficient' in correlation
        assert 'agreement_buckets' in correlation
        assert 'recommendations' in correlation

        # Check correlation (should be negative - higher disagreement = worse outcomes)
        assert correlation['correlation_coefficient'] < 0
        assert abs(correlation['correlation_coefficient']) > 0.3  # Meaningful correlation

        # Verify bucket analysis
        buckets = correlation['agreement_buckets']
        assert 'low_disagreement' in buckets  # disagreement < 0.33
        assert 'medium_disagreement' in buckets  # 0.33 <= disagreement < 0.67
        assert 'high_disagreement' in buckets   # disagreement >= 0.67

        # Check that low disagreement has better outcomes
        low_disag = buckets['low_disagreement']
        high_disag = buckets['high_disagreement']
        assert low_disag['success_rate'] >= high_disag['success_rate']
        assert low_disag['avg_pnl'] >= high_disag['avg_pnl']

    @pytest.mark.asyncio
    async def test_ensemble_disagreement_patterns(self, monitor):
        """Test identification of useful vs problematic disagreement patterns."""
        now = datetime.now()

        # Create patterns showing when disagreement is valuable vs problematic
        patterns = [
            # Productive disagreement (different expertise)
            {
                'market_id': 'tech_disagreement',
                'models_consulted': ['grok-4', 'gpt-4'],
                'disagreement_level': 0.8,
                'final_decision': 'YES',
                'selected_model': 'grok-4',
                'model_votes': {'grok-4': 'YES', 'gpt-4': 'NO'},
                'market_category': 'technology',
                'outcome': 'success',
                'pnl': 120.0,
                'reasoning': 'Different model expertise, chose correctly',
                'timestamp': now
            },
            # Problematic disagreement (uncertainty)
            {
                'market_id': 'uncertain_disagreement',
                'models_consulted': ['grok-4', 'gpt-4', 'claude-3'],
                'disagreement_level': 0.9,
                'final_decision': 'NO',
                'selected_model': 'gpt-4',
                'model_votes': {'grok-4': 'YES', 'gpt-4': 'NO', 'claude-3': 'SKIP'},
                'market_category': 'finance',
                'outcome': 'failure',
                'pnl': -80.0,
                'reasoning': 'High uncertainty, poor choice',
                'timestamp': now + timedelta(minutes=5)
            },
            # Low disagreement (consistent consensus)
            {
                'market_id': 'consensus_trade',
                'models_consulted': ['grok-4', 'gpt-4'],
                'disagreement_level': 0.1,
                'final_decision': 'YES',
                'selected_model': 'grok-4',
                'model_votes': {'grok-4': 'YES', 'gpt-4': 'YES'},
                'market_category': 'sports',
                'outcome': 'success',
                'pnl': 60.0,
                'reasoning': 'Strong consensus, correct decision',
                'timestamp': now + timedelta(minutes=10)
            }
        ]

        for pattern in patterns:
            await monitor.track_performance(pattern)

        # Analyze disagreement patterns
        pattern_analysis = await monitor.analyze_disagreement_patterns()

        # Verify pattern identification
        assert 'productive_disagreement' in pattern_analysis
        assert 'problematic_disagreement' in pattern_analysis
        assert 'consensus_trades' in pattern_analysis
        assert 'recommendations' in pattern_analysis

        # Check productive disagreement characteristics
        productive = pattern_analysis['productive_disagreement']
        assert productive['avg_disagreement'] >= 0.7
        assert productive['success_rate'] == 1.0  # In our test case
        assert 'technology' in productive['categories']

        # Check problematic disagreement characteristics
        problematic = pattern_analysis['problematic_disagreement']
        assert problematic['avg_disagreement'] >= 0.8
        assert problematic['success_rate'] == 0.0  # In our test case
        assert 'finance' in problematic['categories']

        # Check consensus characteristics
        consensus = pattern_analysis['consensus_trades']
        assert consensus['avg_disagreement'] <= 0.25
        assert consensus['success_rate'] == 1.0


class TestDashboardDataAggregation:
    """Test dashboard data aggregation and presentation."""

    @pytest.fixture
    async def monitor(self):
        """Create EnsembleMonitor instance for testing."""
        monitor = EnsembleMonitor(":memory:")
        await monitor.db.initialize()
        return monitor

    @pytest.mark.asyncio
    async def test_dashboard_metrics_aggregation(self, monitor):
        """Test aggregation of metrics for dashboard presentation."""
        now = datetime.now()

        # Create comprehensive test data
        test_data = []
        for i in range(100):
            outcome = 'success' if i % 3 == 0 else 'failure'  # ~33% success rate
            model = ['grok-4', 'gpt-4', 'claude-3'][i % 3]
            category = ['technology', 'finance', 'sports'][i % 3]
            disagreement = round(random.uniform(0.0, 1.0), 2)
            cost = round(random.uniform(0.02, 0.20), 4)
            pnl = round(random.uniform(-100, 200), 2) if outcome == 'success' else round(random.uniform(-80, -20), 2)

            decision = {
                'market_id': f'market_{i}',
                'models_consulted': [model],
                'selected_model': model,
                'market_category': category,
                'disagreement_level': disagreement,
                'final_decision': 'YES',
                'outcome': outcome,
                'pnl': pnl,
                'cost_usd': cost,
                'reasoning': 'Test decision',
                'timestamp': now - timedelta(minutes=i)
            }
            test_data.append(decision)
            await monitor.track_performance(decision)
            await monitor.track_model_cost(decision)

        # Generate dashboard data
        dashboard_data = await monitor.generate_dashboard_data()

        # Verify dashboard structure
        assert 'summary_metrics' in dashboard_data
        assert 'model_performance' in dashboard_data
        assert 'cost_analysis' in dashboard_data
        assert 'agreement_analysis' in dashboard_data
        assert 'time_series_data' in dashboard_data

        # Check summary metrics
        summary = dashboard_data['summary_metrics']
        assert 'total_decisions' in summary
        assert 'success_rate' in summary
        assert 'total_pnl' in summary
        assert 'total_cost' in summary
        assert 'roi' in summary

        # Verify metric calculations
        assert summary['total_decisions'] == 100
        assert 0 <= summary['success_rate'] <= 1
        assert isinstance(summary['total_pnl'], (int, float))
        assert isinstance(summary['total_cost'], (int, float))

        # Check model performance section
        model_perf = dashboard_data['model_performance']
        assert 'grok-4' in model_perf
        assert 'gpt-4' in model_perf
        assert 'claude-3' in model_perf

        # Check cost analysis
        cost_analysis = dashboard_data['cost_analysis']
        assert 'by_model' in cost_analysis
        assert 'by_category' in cost_analysis
        assert 'efficiency' in cost_analysis

        # Check agreement analysis
        agreement = dashboard_data['agreement_analysis']
        assert 'agreement_distribution' in agreement
        assert 'disagreement_impact' in agreement

    @pytest.mark.asyncio
    async def test_time_series_data_generation(self, monitor):
        """Test generation of time series data for dashboard charts."""
        now = datetime.now()

        # Create time-ordered data
        for hours_ago in range(48, 0, -1):  # Last 48 hours
            num_decisions = random.randint(1, 10)
            for i in range(num_decisions):
                timestamp = now - timedelta(hours=hours_ago, minutes=i*10)
                outcome = 'success' if random.random() > 0.6 else 'failure'

                decision = {
                    'market_id': f'market_{hours_ago}_{i}',
                    'models_consulted': ['grok-4'],
                    'selected_model': 'grok-4',
                    'disagreement_level': 0.2,
                    'final_decision': 'YES',
                    'outcome': outcome,
                    'pnl': 50.0 if outcome == 'success' else -25.0,
                    'cost_usd': 0.05,
                    'reasoning': 'Test decision',
                    'timestamp': timestamp
                }
                await monitor.track_performance(decision)

        # Generate time series data
        time_series = await monitor.generate_time_series_data()

        # Verify time series structure
        assert 'hourly_metrics' in time_series
        assert 'daily_metrics' in time_series
        assert 'performance_trend' in time_series

        # Check hourly data
        hourly = time_series['hourly_metrics']
        assert len(hourly) <= 48  # Max 48 hours

        # Verify each hour has required fields
        for hour_data in hourly[:5]:  # Check first 5 hours
            assert 'timestamp' in hour_data
            assert 'decisions_count' in hour_data
            assert 'success_rate' in hour_data
            assert 'total_pnl' in hour_data
            assert 'avg_disagreement' in hour_data

        # Check trend analysis
        trend = time_series['performance_trend']
        assert 'overall_trend' in trend
        assert 'recent_performance' in trend
        assert 'volatility' in trend
        assert trend['overall_trend'] in ['improving', 'declining', 'stable']

    @pytest.mark.asyncio
    async def test_alert_generation_for_dashboard(self, monitor):
        """Test generation of alerts for dashboard display."""
        now = datetime.now()

        # Create scenarios that should trigger alerts
        alert_scenarios = [
            # Low success rate scenario
            *[{
                'market_id': f'failure_{i}',
                'models_consulted': ['grok-4'],
                'selected_model': 'grok-4',
                'disagreement_level': 0.1,
                'final_decision': 'YES',
                'outcome': 'failure',
                'pnl': -20.0,
                'reasoning': 'Test failure',
                'timestamp': now - timedelta(minutes=i*5)
            } for i in range(15)],  # 15 consecutive failures
            # High cost scenario
            {
                'market_id': 'expensive_trade',
                'models_consulted': ['gpt-4'],
                'selected_model': 'gpt-4',
                'disagreement_level': 0.3,
                'final_decision': 'YES',
                'outcome': 'success',
                'pnl': 10.0,
                'cost_usd': 5.0,  # High cost relative to profit
                'reasoning': 'Expensive but successful',
                'timestamp': now
            }
        ]

        for scenario in alert_scenarios:
            if isinstance(scenario, list):
                for s in scenario:
                    await monitor.track_performance(s)
            else:
                await monitor.track_performance(scenario)
                await monitor.track_model_cost(scenario)

        # Generate dashboard alerts
        alerts = await monitor.generate_dashboard_alerts()

        # Verify alert structure
        assert isinstance(alerts, list)
        assert len(alerts) > 0

        # Check alert fields
        for alert in alerts:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'timestamp' in alert
            assert alert['severity'] in ['low', 'medium', 'high', 'critical']

            # Verify required alert types are present
            alert_types = [a['type'] for a in alerts]
            assert 'low_success_rate' in alert_types or 'consecutive_failures' in alert_types

        # Check severity distribution
        critical_count = sum(1 for a in alerts if a['severity'] == 'critical')
        high_count = sum(1 for a in alerts if a['severity'] == 'high')

        # Should have high/critical alerts due to consecutive failures
        assert critical_count + high_count >= 1


# Import random for random number generation in tests
import random
import math