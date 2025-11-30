"""
Test to validate Example 6.1 from the paper: Confounded Exposure and Outcome.

This test verifies that our implementation produces the same bounds as described
in Section 6.1 of the paper for a confounded X→Y model where:
- X is ternary: {0, 1, 2}
- Y is binary: {0, 1}
- U is an unmeasured confounder

The paper states the bounds should be:
    p{X = x1, Y = 1} + p{X = x2, Y = 0} - 1
       ≤ p{Y(X = x1) = 1} - p{Y(X = x2) = 1} ≤
    1 - p{X = x1, Y = 0} - p{X = x2, Y = 1}

for (x1, x2) ∈ {(1, 0), (2, 0), (2, 1)}.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory


def test_confounded_exposure_outcome():
    """
    Test Example 6.1: Confounded exposure and outcome with ternary X and binary Y.
    """
    print("=" * 80)
    print("TEST: Example 6.1 - Confounded Exposure and Outcome")
    print("=" * 80)
    print("\nDAG: X ← U → Y  (X ternary {0,1,2}, Y binary {0,1})")
    print("Goal: Verify bounds for risk differences")
    
    # Create DAG with confounded X and Y
    # Both X and Y are in W_R since we compute interventional queries P(Y(X=x)=1)
    # The confounding is modeled through the response types and observed distribution
    dag = DAG()
    X = dag.add_node('X', support={0, 1, 2}, partition='R')  # Ternary exposure
    Y = dag.add_node('Y', support={0, 1}, partition='R')     # Binary outcome
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    print(f"\nNodes: X (ternary), Y (binary)")
    print(f"W_L = {{}}, W_R = {{X, Y}}")
    print(f"Response types for X: {len(X.response_types)} (= |support|)")
    print(f"Response types for Y: {len(Y.response_types)} (= |support|^|parents|)")
    
    # Generate synthetic observed data
    generator = DataGenerator(dag, seed=42)
    scm = SCM(dag, generator)
    
    # Get observed joint distribution P(X, Y)
    obs_joint = scm.getObservedJoint()
    
    print("\n" + "-" * 80)
    print("Observed Joint Distribution P(X, Y):")
    print("-" * 80)
    
    # Organize by X, Y values
    p_joint = {}
    for config, prob in obs_joint.items():
        x_val = None
        y_val = None
        for node, val in config:
            if node.name == 'X':
                x_val = val
            elif node.name == 'Y':
                y_val = val
        p_joint[(x_val, y_val)] = prob
    
    # Display joint probabilities
    for x in [0, 1, 2]:
        for y in [0, 1]:
            prob = p_joint.get((x, y), 0.0)
            print(f"  P(X={x}, Y={y}) = {prob:.6f}")
    
    # Test all three risk difference contrasts
    contrasts = [
        (1, 0),  # P(Y(1)=1) - P(Y(0)=1)
        (2, 0),  # P(Y(2)=1) - P(Y(0)=1)
        (2, 1),  # P(Y(2)=1) - P(Y(1)=1)
    ]
    
    print("\n" + "=" * 80)
    print("Computing Bounds for Risk Differences")
    print("=" * 80)
    
    all_tests_passed = True
    
    for x1, x2 in contrasts:
        print(f"\n{'-' * 80}")
        print(f"Risk Difference: P(Y(X={x1})=1) - P(Y(X={x2})=1)")
        print(f"{'-' * 80}")
        
        # Compute lower bound: minimize P(Y(X=x1)=1) - P(Y(X=x2)=1)
        lp_lower = ProgramFactory.write_LP(
            scm, 
            Y={Y}, 
            X={X}, 
            Y_values=(1,), 
            X_values=(x1,)
        )
        
        # Create LP for P(Y(X=x2)=1) and subtract
        lp_x2 = ProgramFactory.write_LP(
            scm,
            Y={Y},
            X={X},
            Y_values=(1,),
            X_values=(x2,)
        )
        
        # For the difference, we need: min [P(Y(X=x1)=1) - P(Y(X=x2)=1)]
        # = min P(Y(X=x1)=1) subject to same constraints, with objective α1 - α2
        # For simplicity, compute bounds separately and then combine
        
        result_x1_lower = lp_lower.solve(verbose=False)
        lp_lower_max = ProgramFactory.write_LP(
            scm, Y={Y}, X={X}, Y_values=(1,), X_values=(x1,)
        )
        lp_lower_max.is_minimization = False
        result_x1_upper = lp_lower_max.solve(verbose=False)
        
        result_x2_lower = lp_x2.solve(verbose=False)
        lp_x2_max = ProgramFactory.write_LP(
            scm, Y={Y}, X={X}, Y_values=(1,), X_values=(x2,)
        )
        lp_x2_max.is_minimization = False
        result_x2_upper = lp_x2_max.solve(verbose=False)
        
        if all(r['status'] == 'optimal' for r in [result_x1_lower, result_x1_upper, 
                                                    result_x2_lower, result_x2_upper]):
            # Bounds on difference: [min(Y(x1)) - max(Y(x2)), max(Y(x1)) - min(Y(x2))]
            computed_lower = result_x1_lower['optimal_value'] - result_x2_upper['optimal_value']
            computed_upper = result_x1_upper['optimal_value'] - result_x2_lower['optimal_value']
            
            print(f"\nComputed bounds from our implementation:")
            print(f"  [{computed_lower:.6f}, {computed_upper:.6f}]")
            
            # Paper's formula:
            # Lower: p{X = x1, Y = 1} + p{X = x2, Y = 0} - 1
            # Upper: 1 - p{X = x1, Y = 0} - p{X = x2, Y = 1}
            
            p_x1_y1 = p_joint.get((x1, 1), 0.0)
            p_x2_y0 = p_joint.get((x2, 0), 0.0)
            p_x1_y0 = p_joint.get((x1, 0), 0.0)
            p_x2_y1 = p_joint.get((x2, 1), 0.0)
            
            paper_lower = p_x1_y1 + p_x2_y0 - 1.0
            paper_upper = 1.0 - p_x1_y0 - p_x2_y1
            
            print(f"\nPaper's formula bounds:")
            print(f"  Lower: P(X={x1},Y=1) + P(X={x2},Y=0) - 1")
            print(f"       = {p_x1_y1:.6f} + {p_x2_y0:.6f} - 1")
            print(f"       = {paper_lower:.6f}")
            print(f"  Upper: 1 - P(X={x1},Y=0) - P(X={x2},Y=1)")
            print(f"       = 1 - {p_x1_y0:.6f} - {p_x2_y1:.6f}")
            print(f"       = {paper_upper:.6f}")
            print(f"  [{paper_lower:.6f}, {paper_upper:.6f}]")
            
            # Check if bounds match (with tolerance for numerical errors)
            tolerance = 1e-4
            lower_match = abs(computed_lower - paper_lower) < tolerance
            upper_match = abs(computed_upper - paper_upper) < tolerance
            
            print(f"\nVerification:")
            print(f"  Lower bound match: {lower_match} (diff: {abs(computed_lower - paper_lower):.2e})")
            print(f"  Upper bound match: {upper_match} (diff: {abs(computed_upper - paper_upper):.2e})")
            
            if lower_match and upper_match:
                print(f"  ✓ PASSED: Bounds match paper's formula!")
            else:
                print(f"  ✗ FAILED: Bounds do not match!")
                all_tests_passed = False
        else:
            print(f"  ✗ FAILED: Could not solve LP")
            all_tests_passed = False
    
    print("\n" + "=" * 80)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED - Implementation matches Example 6.1!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)
    
    return all_tests_passed


if __name__ == "__main__":
    success = test_confounded_exposure_outcome()
    sys.exit(0 if success else 1)
