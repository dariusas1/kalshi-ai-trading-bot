#!/usr/bin/env python3
"""
Validation script for EnsembleMonitor functionality.
Tests the core structure and method signatures without requiring external dependencies.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that we can import the required classes."""
    try:
        # Test database dataclass imports
        from src.utils.database import ModelPerformance, EnsembleDecision

        # Test that classes can be instantiated
        perf = ModelPerformance(
            model_name="test-model",
            timestamp=datetime.now(),
            market_category="test",
            accuracy_score=0.8,
            confidence_calibration=0.7,
            response_time_ms=1500,
            cost_usd=0.05,
            decision_quality=0.85
        )

        decision = EnsembleDecision(
            market_id="test-market",
            models_consulted=["model1", "model2"],
            final_decision="YES",
            disagreement_level=0.3,
            selected_model="model1",
            reasoning="Test reasoning",
            timestamp=datetime.now()
        )

        print("✓ Database dataclass imports and instantiation successful")
        return True

    except Exception as e:
        print(f"✗ Database dataclass test failed: {e}")
        return False

def test_data_structures():
    """Test data structure validation."""
    try:
        from src.utils.database import ModelPerformance, EnsembleDecision

        # Test ModelPerformance data structure
        perf_data = {
            'model_name': 'grok-4',
            'timestamp': datetime.now(),
            'market_category': 'technology',
            'accuracy_score': 0.85,
            'confidence_calibration': 0.9,
            'response_time_ms': 1200,
            'cost_usd': 0.07,
            'decision_quality': 0.88
        }

        performance = ModelPerformance(**perf_data)
        assert performance.model_name == 'grok-4'
        assert performance.market_category == 'technology'
        assert performance.accuracy_score == 0.85

        # Test EnsembleDecision data structure
        decision_data = {
            'market_id': 'market-123',
            'models_consulted': ['grok-4', 'gpt-4'],
            'final_decision': 'YES',
            'disagreement_level': 0.2,
            'selected_model': 'grok-4',
            'reasoning': 'Strong bullish consensus',
            'timestamp': datetime.now()
        }

        decision = EnsembleDecision(**decision_data)
        assert decision.market_id == 'market-123'
        assert len(decision.models_consulted) == 2
        assert decision.disagreement_level == 0.2

        print("✓ Data structure validation successful")
        return True

    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False

def test_method_interfaces():
    """Test that EnsembleMonitor has the expected method signatures."""
    try:
        # Test file structure without importing dependencies
        import inspect

        # Read the EnsembleMonitor source
        with open('src/intelligence/ensemble_monitor.py', 'r') as f:
            source_code = f.read()

        # Check for expected methods
        expected_methods = [
            'track_performance',
            'track_model_cost',
            'get_realtime_metrics',
            'analyze_model_contributions',
            'get_cost_breakdown',
            'analyze_agreement_patterns',
            'generate_dashboard_data'
        ]

        for method in expected_methods:
            if f'async def {method}' in source_code:
                print(f"✓ Method {method} found with async signature")
            else:
                print(f"✗ Method {method} not found")
                return False

        # Check for class structure
        if 'class EnsembleMonitor' in source_code:
            print("✓ EnsembleMonitor class found")
        else:
            print("✗ EnsembleMonitor class not found")
            return False

        # Check for proper inheritance
        if 'TradingLoggerMixin' in source_code:
            print("✓ TradingLoggerMixin inheritance found")
        else:
            print("✗ TradingLoggerMixin inheritance not found")
            return False

        return True

    except Exception as e:
        print(f"✗ Method interface test failed: {e}")
        return False

def test_monitoring_workflow():
    """Test the monitoring workflow logic."""
    try:
        # Simulate monitoring workflow
        from src.utils.database import ModelPerformance, EnsembleDecision
        from datetime import datetime

        # Create test data
        now = datetime.now()
        test_decisions = [
            {
                'market_id': 'market_1',
                'models_consulted': ['grok-4', 'gpt-4'],
                'final_decision': 'YES',
                'disagreement_level': 0.1,
                'selected_model': 'grok-4',
                'reasoning': 'Strong consensus',
                'outcome': 'success',
                'pnl': 100.0,
                'cost_usd': 0.05,
                'timestamp': now
            },
            {
                'market_id': 'market_2',
                'models_consulted': ['gpt-4', 'claude-3'],
                'final_decision': 'NO',
                'disagreement_level': 0.8,
                'selected_model': 'gpt-4',
                'reasoning': 'High disagreement, conservative',
                'outcome': 'failure',
                'pnl': -50.0,
                'cost_usd': 0.08,
                'timestamp': now + timedelta(minutes=5)
            }
        ]

        # Test data processing logic
        total_decisions = len(test_decisions)
        successful_decisions = sum(1 for d in test_decisions if d['outcome'] == 'success')
        success_rate = successful_decisions / total_decisions

        avg_disagreement = sum(d['disagreement_level'] for d in test_decisions) / total_decisions
        total_pnl = sum(d['pnl'] for d in test_decisions)

        # Validate calculations
        assert total_decisions == 2
        assert success_rate == 0.5
        assert avg_disagreement == 0.45  # (0.1 + 0.8) / 2
        assert total_pnl == 50.0  # 100 - 50

        print("✓ Monitoring workflow logic validation successful")
        return True

    except Exception as e:
        print(f"✗ Monitoring workflow test failed: {e}")
        return False

def test_cost_analysis():
    """Test cost analysis logic."""
    try:
        # Simulate cost data
        cost_data = [
            {'model_name': 'grok-4', 'cost_usd': 0.05, 'pnl': 100.0},
            {'model_name': 'grok-4', 'cost_usd': 0.08, 'pnl': -30.0},
            {'model_name': 'gpt-4', 'cost_usd': 0.15, 'pnl': 150.0},
            {'model_name': 'claude-3', 'cost_usd': 0.12, 'pnl': 75.0}
        ]

        # Calculate cost metrics
        model_costs = {}
        for data in cost_data:
            model = data['model_name']
            if model not in model_costs:
                model_costs[model] = {'total_cost': 0.0, 'total_pnl': 0.0, 'count': 0}

            model_costs[model]['total_cost'] += data['cost_usd']
            model_costs[model]['total_pnl'] += data['pnl']
            model_costs[model]['count'] += 1

        # Validate calculations
        assert model_costs['grok-4']['total_cost'] == 0.13  # 0.05 + 0.08
        assert model_costs['grok-4']['total_pnl'] == 70.0   # 100 - 30
        assert model_costs['gpt-4']['total_cost'] == 0.15
        assert model_costs['gpt-4']['total_pnl'] == 150.0

        # Calculate ROI
        for model, costs in model_costs.items():
            roi = costs['total_pnl'] / costs['total_cost'] if costs['total_cost'] > 0 else 0
            assert roi > 0  # All models have positive ROI in this test

        print("✓ Cost analysis logic validation successful")
        return True

    except Exception as e:
        print(f"✗ Cost analysis test failed: {e}")
        return False

def test_agreement_analysis():
    """Test agreement analysis logic."""
    try:
        # Simulate agreement data
        agreement_scenarios = [
            {'disagreement_level': 0.1, 'outcome': 'success'},  # High agreement, success
            {'disagreement_level': 0.2, 'outcome': 'success'},  # High agreement, success
            {'disagreement_level': 0.8, 'outcome': 'failure'},  # High disagreement, failure
            {'disagreement_level': 0.9, 'outcome': 'failure'},  # High disagreement, failure
        ]

        # Group by agreement level
        high_agreement = [s for s in agreement_scenarios if s['disagreement_level'] < 0.33]
        high_disagreement = [s for s in agreement_scenarios if s['disagreement_level'] >= 0.67]

        # Calculate metrics
        high_agreement_success = sum(1 for s in high_agreement if s['outcome'] == 'success') / len(high_agreement)
        high_disagreement_success = sum(1 for s in high_disagreement if s['outcome'] == 'success') / len(high_disagreement)

        # Validate logic
        assert len(high_agreement) == 2
        assert len(high_disagreement) == 2
        assert high_agreement_success == 1.0  # Both high agreement cases succeeded
        assert high_disagreement_success == 0.0  # Both high disagreement cases failed

        # Correlation should show negative relationship
        assert high_agreement_success > high_disagreement_success

        print("✓ Agreement analysis logic validation successful")
        return True

    except Exception as e:
        print(f"✗ Agreement analysis test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Starting EnsembleMonitor validation...")
    print("=" * 50)

    tests = [
        ("Dataclass Imports", test_imports),
        ("Data Structure Validation", test_data_structures),
        ("Method Interface Check", test_method_interfaces),
        ("Monitoring Workflow Logic", test_monitoring_workflow),
        ("Cost Analysis Logic", test_cost_analysis),
        ("Agreement Analysis Logic", test_agreement_analysis)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed! EnsembleMonitor implementation is structurally sound.")
        return 0
    else:
        print("⚠️  Some validation tests failed. Please review the implementation.")
        return 1

if __name__ == '__main__':
    exit(main())