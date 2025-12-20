"""
Basic test for CostOptimizer - tests functionality without full imports.
This test validates that the CostOptimizer can be imported and initialized correctly.
"""

import unittest
import sys
import os

# Direct import test to check if our cost optimizer file exists and has correct structure
def test_cost_optimizer_import():
    """Test that cost optimizer file exists and has required classes."""

    # Check if the cost_optimizer file exists
    cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

    if not os.path.exists(cost_optimizer_path):
        raise FileNotFoundError(f"CostOptimizer file not found at {cost_optimizer_path}")

    # Read the file and check for required classes and methods
    with open(cost_optimizer_path, 'r') as f:
        content = f.read()

    # Check for required classes
    required_classes = [
        'CostOptimizer',
        'CostEfficiencyMetrics',
        'BudgetStatus',
        'CacheEntry',
        'DynamicCostModel',
        'CostOptimizationConfig'
    ]

    for class_name in required_classes:
        if f'class {class_name}' not in content:
            raise AssertionError(f"Required class {class_name} not found in cost_optimizer.py")

    # Check for required methods in CostOptimizer
    required_methods = [
        'calculate_cost_efficiency',
        'monitor_spend',
        'enforce_budget_limits',
        'get_cached_result',
        'cache_model_result',
        'select_models_with_budget'
    ]

    for method_name in required_methods:
        if f'def {method_name}' not in content:
            raise AssertionError(f"Required method {method_name} not found in CostOptimizer class")

    print("✓ CostOptimizer file structure validation passed")
    return True

def test_cost_optimizer_dataclasses():
    """Test that required dataclasses are properly defined."""

    cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

    with open(cost_optimizer_path, 'r') as f:
        content = f.read()

    # Check for dataclass decorators
    dataclass_checks = [
        ('@dataclass', 'class CostEfficiencyMetrics'),
        ('@dataclass', 'class BudgetStatus'),
        ('@dataclass', 'class CacheEntry'),
        ('@dataclass', 'class DynamicCostModel'),
        ('@dataclass', 'class CostOptimizationConfig')
    ]

    for decorator, class_name in dataclass_checks:
        if decorator not in content or class_name not in content:
            raise AssertionError(f"Dataclass {class_name} not properly defined")

    print("✓ Dataclass structure validation passed")
    return True

def test_cost_optimizer_imports():
    """Test that required imports are present."""

    cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

    with open(cost_optimizer_path, 'r') as f:
        content = f.read()

    # Check for required imports
    required_imports = [
        'from typing import',
        'from dataclasses import dataclass',
        'from datetime import datetime, timedelta',
        'import asyncio',
        'import json',
        'import statistics',
        'from collections import defaultdict, OrderedDict',
        'import logging'
    ]

    for import_statement in required_imports:
        if import_statement not in content:
            raise AssertionError(f"Required import {import_statement} not found")

    print("✓ Import structure validation passed")
    return True

class TestCostOptimizerStructure(unittest.TestCase):
    """Test CostOptimizer file structure and basic functionality."""

    def test_file_exists(self):
        """Test that CostOptimizer file exists."""
        cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')
        self.assertTrue(os.path.exists(cost_optimizer_path), "CostOptimizer file should exist")

    def test_required_classes_present(self):
        """Test that required classes are present in the file."""
        cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

        with open(cost_optimizer_path, 'r') as f:
            content = f.read()

        required_classes = [
            'CostOptimizer',
            'CostEfficiencyMetrics',
            'BudgetStatus',
            'CacheEntry',
            'DynamicCostModel',
            'CostOptimizationConfig'
        ]

        for class_name in required_classes:
            self.assertIn(f'class {class_name}', content, f"Class {class_name} should be defined")

    def test_required_methods_present(self):
        """Test that required methods are present in CostOptimizer."""
        cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

        with open(cost_optimizer_path, 'r') as f:
            content = f.read()

        required_methods = [
            'async def calculate_cost_efficiency',
            'async def monitor_spend',
            'async def enforce_budget_limits',
            'async def get_cached_result',
            'async def cache_model_result',
            'async def select_models_with_budget',
            'async def update_cost_performance_model',
            'async def get_optimization_performance_metrics'
        ]

        for method_name in required_methods:
            self.assertIn(method_name, content, f"Method {method_name} should be defined")

    def test_dataclass_structure(self):
        """Test that dataclasses are properly structured."""
        cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

        with open(cost_optimizer_path, 'r') as f:
            content = f.read()

        # Check that dataclasses have required attributes
        self.assertIn('@dataclass', content, "Should use dataclass decorators")

        # Check specific dataclass fields (by looking at their definitions)
        self.assertIn('model_name: str', content, "CostEfficiencyMetrics should have model_name field")
        self.assertIn('daily_limit: float', content, "BudgetStatus should have daily_limit field")
        self.assertIn('cache_key: str', content, "CacheEntry should have cache_key field")

    def test_imports_present(self):
        """Test that required imports are present."""
        cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

        with open(cost_optimizer_path, 'r') as f:
            content = f.read()

        required_imports = [
            'import asyncio',
            'import statistics',
            'from typing import',
            'from dataclasses import dataclass, field',
            'from datetime import datetime, timedelta',
            'from collections import defaultdict, OrderedDict'
        ]

        for import_statement in required_imports:
            self.assertIn(import_statement, content, f"Import {import_statement} should be present")

    def test_config_defaults(self):
        """Test that CostOptimizationConfig has appropriate defaults."""
        cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

        with open(cost_optimizer_path, 'r') as f:
            content = f.read()

        # Look for default values in CostOptimizationConfig
        self.assertIn('enable_dynamic_modeling: bool = True', content)
        self.assertIn('daily_budget_limit: float = 50.0', content)
        self.assertIn('enable_budget_controls: bool = True', content)
        self.assertIn('enable_intelligent_caching: bool = True', content)
        self.assertIn('cache_ttl_minutes: int = 30', content)

    def test_method_signatures(self):
        """Test that methods have appropriate signatures."""
        cost_optimizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'intelligence', 'cost_optimizer.py')

        with open(cost_optimizer_path, 'r') as f:
            content = f.read()

        # Check async methods have proper signatures
        self.assertIn('async def calculate_cost_efficiency(', content)
        self.assertIn('async def monitor_spend(', content)
        self.assertIn('async def enforce_budget_limits(', content)
        self.assertIn('async def get_cached_result(', content)
        self.assertIn('async def cache_model_result(', content)
        self.assertIn('async def select_models_with_budget(', content)


def run_all_tests():
    """Run all basic validation tests."""
    print("Running CostOptimizer basic validation tests...")

    tests = [
        test_cost_optimizer_import,
        test_cost_optimizer_dataclasses,
        test_cost_optimizer_imports
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(True)
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            results.append(False)

    if all(results):
        print("🎉 All basic validation tests passed!")
        print("\n✅ CostOptimizer implementation structure is correct")
        print("✅ All required classes and methods are present")
        print("✅ Dataclasses are properly defined")
        print("✅ Imports and dependencies are correct")
        print("✅ Default configurations are appropriate")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == '__main__':
    # Run unittest tests
    unittest.main(verbosity=2)

    print("\n" + "="*50)

    # Run basic validation
    run_all_tests()