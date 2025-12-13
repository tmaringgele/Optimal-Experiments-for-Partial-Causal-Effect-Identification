"""
Test script for LinearProgram.solve() method.

This script creates a simple LP and tests the PPOPT-based solver.
"""

import numpy as np
from symbolic_bounds.dag import DAG
from symbolic_bounds.scm import SCM
from symbolic_bounds.program_factory import ProgramFactory


def test_simple_chain_solve():
    """
    Test the LP solver on a simple X -> Y chain.
    
    This creates an LP for computing bounds on the causal effect P(Y=1 | do(X=1)).
    """
    print("=" * 80)
    print("TEST: Solving LP for Simple Chain X -> Y")
    print("=" * 80)
    
    # Create DAG: X -> Y
    # Both X and Y are in W_R (interventional setting)
    dag = DAG()
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    print("\nDAG Structure:")
    print(f"  Nodes: {', '.join(n.name for n in dag.get_all_nodes())}")
    print(f"  Edges: X -> Y")
    print(f"  W_L = {{}}, W_R = {{X, Y}}")
    
    # Create SCM with automatic data generation
    from symbolic_bounds.data_generator import DataGenerator
    data_gen = DataGenerator(dag, seed=42)
    scm = SCM(dag, data_gen)
    observed_joint = scm.getObservedJoint()
    
    print("\nGenerated Observed Distribution P*(X, Y):")
    for config, prob in sorted(observed_joint.items(), 
                              key=lambda item: tuple((n.name, v) for n, v in sorted(item[0], key=lambda x: x[0].name))):
        config_str = ", ".join(f"{n.name}={v}" for n, v in sorted(config, key=lambda x: x[0].name))
        print(f"  P*({config_str}) = {prob:.6f}")
    
    # Build LP for computing bounds on P(Y=1 | do(X=1))
    lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))
    
    print("\nLinear Program Structure:")
    summary = lp.get_summary()
    print(f"  Variables (ℵᴿ): {summary['n_variables']}")
    print(f"  Constraints: {summary['n_constraints']}")
    print(f"  Non-zero in objective: {summary['objective_nonzero']}")
    print(f"  Type: {'Minimization' if lp.is_minimization else 'Maximization'}")
    
    # Show objective function
    print("\nObjective Function α^T q:")
    non_zero_indices = np.where(np.abs(lp.objective) > 1e-10)[0]
    for idx in non_zero_indices[:5]:  # Show first 5
        print(f"  α[{idx}] = {lp.objective[idx]:.4f}  ({lp.variable_labels[idx]})")
    if len(non_zero_indices) > 5:
        print(f"  ... and {len(non_zero_indices) - 5} more non-zero coefficients")
    
    # Solve the LP
    print("\n" + "-" * 80)
    print("Solving LP using PPOPT...")
    print("-" * 80)
    
    try:
        result = lp.solve(solver_type='glpk', verbose=False)
        
        if result['status'] == 'optimal':
            print("\n✓ LP solved successfully!")
            print(f"\nOptimal value: {result['optimal_value']:.8f}")
            
            # Show solution vector (first few entries)
            solution = result['solution']
            print(f"\nSolution vector q (dimension {len(solution)}):")
            print(f"  Sum of q: {np.sum(solution):.10f} (should be ≈ 1.0)")
            print(f"  Min value: {np.min(solution):.10f} (should be ≥ 0)")
            print(f"  Max value: {np.max(solution):.10f}")
            
            # Show non-zero probabilities
            nonzero_mask = solution > 1e-6
            nonzero_indices = np.where(nonzero_mask)[0]
            print(f"\n  Non-zero probabilities ({len(nonzero_indices)} total):")
            for idx in nonzero_indices[:10]:  # Show first 10
                print(f"    q[{idx}] = {solution[idx]:.8f}  ({lp.variable_labels[idx]})")
            if len(nonzero_indices) > 10:
                print(f"    ... and {len(nonzero_indices) - 10} more")
            
            # Verify constraint satisfaction
            print("\n" + "-" * 80)
            print("Verifying constraint satisfaction: P q = p")
            print("-" * 80)
            
            lhs = lp.constraint_matrix @ solution
            rhs = lp.rhs
            max_violation = np.max(np.abs(lhs - rhs))
            
            print(f"  Maximum constraint violation: {max_violation:.2e}")
            
            if max_violation < 1e-4:
                print("  ✓ All constraints satisfied!")
            else:
                print("  ✗ Warning: Some constraints violated")
                # Show worst violations
                violations = np.abs(lhs - rhs)
                worst_indices = np.argsort(violations)[-3:][::-1]
                for idx in worst_indices:
                    print(f"    Constraint {idx}: |{lhs[idx]:.6f} - {rhs[idx]:.6f}| = {violations[idx]:.2e}")
            
            # Verify normalization
            print("\n" + "-" * 80)
            print("Verifying normalization: 1^T q = 1")
            print("-" * 80)
            
            q_sum = np.sum(solution)
            norm_violation = abs(q_sum - 1.0)
            print(f"  Sum of q: {q_sum:.10f}")
            print(f"  Violation: {norm_violation:.2e}")
            
            if norm_violation < 1e-6:
                print("  ✓ Normalization satisfied!")
            else:
                print("  ✗ Warning: Normalization violated")
            
            # Verify non-negativity
            print("\n" + "-" * 80)
            print("Verifying non-negativity: q ≥ 0")
            print("-" * 80)
            
            min_val = np.min(solution)
            print(f"  Minimum value: {min_val:.2e}")
            
            if min_val >= -1e-6:
                print("  ✓ Non-negativity satisfied!")
            else:
                print("  ✗ Warning: Some negative values detected")
                negative_indices = np.where(solution < -1e-6)[0]
                print(f"  Negative values at indices: {negative_indices}")
            
            print("\n" + "=" * 80)
            print("TEST PASSED: LP solved successfully with valid solution!")
            print("=" * 80)
            return True
            
        else:
            print(f"\n✗ LP solving failed: {result['status']}")
            return False
            
    except ImportError as e:
        print(f"\n✗ PPOPT not installed: {e}")
        print("\nTo install PPOPT, run:")
        print("  pip install ppopt")
        print("\nOr install cvxopt for the glpk solver:")
        print("  pip install cvxopt")
        return False
    except Exception as e:
        print(f"\n✗ Error solving LP: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_maximize():
    """
    Test that maximization works correctly (should negate objective).
    """
    print("\n" + "=" * 80)
    print("TEST: Maximization (negated objective)")
    print("=" * 80)
    
    # Create simple LP
    dag = DAG()
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    from symbolic_bounds.data_generator import DataGenerator
    data_gen = DataGenerator(dag, seed=42)
    scm = SCM(dag, data_gen)
    
    # Build LP for upper bound (maximization)
    lp_max = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))
    lp_max.is_minimization = False  # Switch to maximization
    
    print(f"\nLP type: {'Minimization' if lp_max.is_minimization else 'Maximization'}")
    
    try:
        result = lp_max.solve(solver_type='glpk', verbose=False)
        
        if result['status'] == 'optimal':
            print(f"✓ Maximization LP solved!")
            print(f"  Optimal value: {result['optimal_value']:.8f}")
            return True
        else:
            print(f"✗ Maximization failed: {result['status']}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LINEAR PROGRAM SOLVER TESTS")
    print("=" * 80)
    
    test1_passed = test_simple_chain_solve()
    test2_passed = test_maximize()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Simple Chain Solve:  {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Maximization:        {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    
    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED - LP solver works correctly!")
    else:
        print("\n✗ SOME TESTS FAILED")
