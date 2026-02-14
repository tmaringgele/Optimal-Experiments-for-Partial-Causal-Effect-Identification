#!/usr/bin/env python
"""Quick test to verify imports work correctly."""

try:
    from symbolic_bounds.tests.test_constraints import validate_constraints
    print("✓ test_constraints imports successfully")
except Exception as e:
    print(f"✗ test_constraints import failed: {e}")

try:
    from symbolic_bounds.tests.test_lp_solve import test_simple_chain_solve
    print("✓ test_lp_solve imports successfully")
except Exception as e:
    print(f"✗ test_lp_solve import failed: {e}")

try:
    from symbolic_bounds.tests.test_parametric_lp import test_without_experiments
    print("✓ test_parametric_lp imports successfully")
except Exception as e:
    print(f"✗ test_parametric_lp import failed: {e}")

try:
    from symbolic_bounds.tests.test_section6_1 import test_confounded_exposure_outcome
    print("✓ test_section6_1 imports successfully")
except Exception as e:
    print(f"✗ test_section6_1 import failed: {e}")

print("\nAll import tests completed!")
