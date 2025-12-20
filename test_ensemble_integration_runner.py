#!/usr/bin/env python3
"""
Simple test runner for ensemble integration tests.
This script runs the integration tests without requiring pytest.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_test_class(test_class_name, test_methods):
    """Run all test methods in a test class."""
    print(f"\n{'='*60}")
    print(f"Running {test_class_name}")
    print(f"{'='*60}")

    # Import the test class
    try:
        # Use __import__ to avoid module loading issues
        module = __import__("test_ensemble_integration")
        test_class = getattr(module, test_class_name)

        # Create instance with fixtures
        instance = test_class()

        passed = 0
        failed = 0

        for method_name in test_methods:
            if hasattr(instance, method_name) and method_name.startswith('test_'):
                try:
                    print(f"\n  Running {method_name}...", end=" ")

                    # Get the test method
                    test_method = getattr(instance, method_name)

                    # Run setup if it exists
                    if hasattr(instance, 'setUp'):
                        instance.setUp()

                    # Run the test
                    if asyncio.iscoroutinefunction(test_method):
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(test_method())
                    else:
                        test_method()

                    # Run teardown if it exists
                    if hasattr(instance, 'tearDown'):
                        instance.tearDown()

                    print("✅ PASSED")
                    passed += 1

                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    print(f"    {traceback.format_exc()}")
                    failed += 1

        print(f"\nResults for {test_class_name}:")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Total:  {passed + failed}")

        return failed == 0

    except Exception as e:
        print(f"❌ Error loading test class {test_class_name}: {e}")
        return False

def main():
    """Main test runner."""
    print("Ensemble Integration Test Runner")
    print("=" * 60)

    test_classes = [
        ("TestXAIClientEnsembleIntegration", [
            "test_existing_ensemble_decision_integration",
            "test_advanced_ensemble_decision_integration",
            "test_ensemble_fallback_to_basic"
        ]),
        ("TestSettingsMultiModelEnsembleFlag", [
            "test_multi_model_ensemble_disabled",
            "test_settings_ensemble_flag_default_value",
            "test_settings_flag_affects_decision_routing"
        ]),
        ("TestDecisionLogicIntegration", [
            "test_decide_py_ensemble_integration",
            "test_decide_py_single_model_fallback",
            "test_high_stakes_threshold_calculation"
        ]),
        ("TestEnsembleComponentCoordination", [
            "test_ensemble_components_initialization",
            "test_component_data_flow",
            "test_ensemble_state_management",
            "test_component_error_handling"
        ])
    ]

    all_passed = True

    for test_class_name, test_methods in test_classes:
        passed = run_test_class(test_class_name, test_methods)
        all_passed = all_passed and passed

    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("Integration tests completed successfully!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Integration tests failed - see details above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)