"""
Test cases replicating examples from Section 6 of Sachs et al.
"A General Method for Deriving Tight Symbolic Bounds on Causal Effects"

This file tests the dual LP approach on standard causal inference examples:
1. Simple confounding (X → Y case)
2. Instrumental variable with confounding
3. Mediation structure
"""

import numpy as np
from symbolic_bounds.dag import DAG, Node
from symbolic_bounds.program_factory import ProgramFactory
from symbolic_bounds.linear_program import LinearProgram
from test_dual_clean import compute_symbolic_bounds_dual


def test_simple_confounding():
    """
    Example: X → Y with unobserved confounding
    Graph: X → Y, U → X, U → Y (U unobserved)
    
    Query: P(Y=1 | do(X=1))
    
    This is the basic example from the paper, should match our previous test.
    """
    print("\n" + "="*80)
    print("TEST 1: Simple Confounding (X → Y)")
    print("="*80)
    
    # Create DAG
    X = Node("X", support={0, 1})
    Y = Node("Y", support={0, 1})
    
    dag = DAG()
    dag.add_node(X)
    dag.add_node(Y)
    dag.add_edge(X, Y)
    
    # Partition: X unobserved (W_R), Y observed (W_L)
    # Actually, both are observed in this simple case
    W_L = {X}
    W_R = {Y}
    
    # Create LP
    factory = ProgramFactory()
    lp = factory.create_program(dag, W_L, W_R)
    
    # Set query: P(Y=1 | do(X=1))
    intervention_X = {X: 1}
    objective_Y = {Y}
    Y_values = (1,)
    
    alpha = factory.writeRung2(dag, objective_Y, intervention_X, Y_values, W_L, W_R)
    lp.set_objective(alpha, sense='max')
    
    # Compute symbolic bounds
    print("\nQuery: P(Y=1 | do(X=1))")
    print(f"Parameters: {lp.param_names}")
    
    bounds = compute_symbolic_bounds_dual(lp, sense='both')
    
    # Test with sample distribution
    test_dist = np.array([0.3, 0.2, 0.1, 0.4])  # p(X=0,Y=0), p(X=0,Y=1), p(X=1,Y=0), p(X=1,Y=1)
    upper_val = bounds['upper_bound'].evaluate(test_dist)
    lower_val = bounds['lower_bound'].evaluate(test_dist)
    
    print(f"\nWith test distribution: {test_dist}")
    print(f"Upper bound: {upper_val:.4f}")
    print(f"Lower bound: {lower_val:.4f}")
    print(f"Width: {upper_val - lower_val:.4f}")
    
    if lower_val <= upper_val:
        print("✓ Bounds are valid!")
    else:
        print("✗ ERROR: Lower > Upper")
    
    return bounds


def test_instrumental_variable():
    """
    Example: Instrumental Variable with Confounding
    Graph: Z → X → Y, U → X, U → Y (U unobserved)
    
    This is the classic IV structure where:
    - Z is the instrument (affects X but not Y directly)
    - X is the treatment
    - Y is the outcome
    - U is unobserved confounder between X and Y
    
    Query: P(Y=1 | do(X=1))
    """
    print("\n" + "="*80)
    print("TEST 2: Instrumental Variable with Confounding")
    print("="*80)
    
    # Create nodes
    Z = Node("Z", support={0, 1})  # Instrument
    X = Node("X", support={0, 1})  # Treatment
    Y = Node("Y", support={0, 1})  # Outcome
    
    # Build DAG: Z → X → Y
    dag = DAG()
    dag.add_node(Z)
    dag.add_node(X)
    dag.add_node(Y)
    dag.add_edge(Z, X)  # Z affects X
    dag.add_edge(X, Y)  # X affects Y
    # U → X and U → Y represented by joint response types
    
    # Partition: Z and X observed (W_L), Y partially observed (W_R)
    W_L = {Z}
    W_R = {X, Y}
    
    # Create LP
    factory = ProgramFactory()
    lp = factory.create_program(dag, W_L, W_R)
    
    # Query: P(Y=1 | do(X=1))
    intervention_X = {X: 1}
    objective_Y = {Y}
    Y_values = (1,)
    
    alpha = factory.writeRung2(dag, objective_Y, intervention_X, Y_values, W_L, W_R)
    lp.set_objective(alpha, sense='max')
    
    # Compute symbolic bounds
    print("\nQuery: P(Y=1 | do(X=1))")
    print(f"Parameters: {lp.param_names[:5]}... ({len(lp.param_names)} total)")
    
    bounds = compute_symbolic_bounds_dual(lp, sense='both')
    
    # Test with sample distribution
    # For Z,X,Y we have 2*2*2 = 8 parameters
    test_dist = np.array([0.15, 0.10, 0.12, 0.08, 0.18, 0.12, 0.10, 0.15])
    upper_val = bounds['upper_bound'].evaluate(test_dist)
    lower_val = bounds['lower_bound'].evaluate(test_dist)
    
    print(f"\nWith test distribution (8 params)")
    print(f"Upper bound: {upper_val:.4f}")
    print(f"Lower bound: {lower_val:.4f}")
    print(f"Width: {upper_val - lower_val:.4f}")
    
    if lower_val <= upper_val:
        print("✓ Bounds are valid!")
    else:
        print("✗ ERROR: Lower > Upper")
    
    return bounds


def test_mediation():
    """
    Example: Mediation Structure
    Graph: X → M → Y, X → Y (with potential confounding)
    
    Where:
    - X is the treatment
    - M is the mediator
    - Y is the outcome
    
    We can query:
    1. Total effect: P(Y=1 | do(X=1))
    2. Direct effect: P(Y=1 | do(X=1, M=m))
    3. Indirect effect (via bounds)
    
    Query: P(Y=1 | do(X=1))
    """
    print("\n" + "="*80)
    print("TEST 3: Mediation Structure")
    print("="*80)
    
    # Create nodes
    X = Node("X", support={0, 1})  # Treatment
    M = Node("M", support={0, 1})  # Mediator
    Y = Node("Y", support={0, 1})  # Outcome
    
    # Build DAG: X → M → Y and X → Y
    dag = DAG()
    dag.add_node(X)
    dag.add_node(M)
    dag.add_node(Y)
    dag.add_edge(X, M)  # X affects M
    dag.add_edge(M, Y)  # M affects Y
    dag.add_edge(X, Y)  # Direct effect X → Y
    
    # Partition: X observed (W_L), M and Y in W_R
    W_L = {X}
    W_R = {M, Y}
    
    # Create LP
    factory = ProgramFactory()
    lp = factory.create_program(dag, W_L, W_R)
    
    # Query: P(Y=1 | do(X=1)) - Total effect
    intervention_X = {X: 1}
    objective_Y = {Y}
    Y_values = (1,)
    
    alpha = factory.writeRung2(dag, objective_Y, intervention_X, Y_values, W_L, W_R)
    lp.set_objective(alpha, sense='max')
    
    # Compute symbolic bounds
    print("\nQuery: P(Y=1 | do(X=1)) - Total Effect")
    print(f"Parameters: {lp.param_names[:5]}... ({len(lp.param_names)} total)")
    
    bounds = compute_symbolic_bounds_dual(lp, sense='both')
    
    # Test with sample distribution
    # For X,M,Y we have 2*2*2 = 8 parameters
    test_dist = np.array([0.20, 0.15, 0.10, 0.05, 0.12, 0.18, 0.08, 0.12])
    upper_val = bounds['upper_bound'].evaluate(test_dist)
    lower_val = bounds['lower_bound'].evaluate(test_dist)
    
    print(f"\nWith test distribution (8 params)")
    print(f"Upper bound: {upper_val:.4f}")
    print(f"Lower bound: {lower_val:.4f}")
    print(f"Width: {upper_val - lower_val:.4f}")
    
    if lower_val <= upper_val:
        print("✓ Bounds are valid!")
    else:
        print("✗ ERROR: Lower > Upper")
    
    return bounds


def test_chain():
    """
    Example: Chain Structure Z → X → Y
    
    This is the example we've been testing.
    Query: P(Y=1 | do(X=1))
    """
    print("\n" + "="*80)
    print("TEST 4: Chain Structure (Z → X → Y)")
    print("="*80)
    
    # Create nodes
    Z = Node("Z", support={0, 1})
    X = Node("X", support={0, 1})
    Y = Node("Y", support={0, 1})
    
    # Build DAG: Z → X → Y
    dag = DAG()
    dag.add_node(Z)
    dag.add_node(X)
    dag.add_node(Y)
    dag.add_edge(Z, X)
    dag.add_edge(X, Y)
    
    # Partition: Z observed (W_L), X,Y in W_R
    W_L = {Z}
    W_R = {X, Y}
    
    # Create LP
    factory = ProgramFactory()
    lp = factory.create_program(dag, W_L, W_R)
    
    # Query: P(Y=1 | do(X=1))
    intervention_X = {X: 1}
    objective_Y = {Y}
    Y_values = (1,)
    
    alpha = factory.writeRung2(dag, objective_Y, intervention_X, Y_values, W_L, W_R)
    lp.set_objective(alpha, sense='max')
    
    # Compute symbolic bounds
    print("\nQuery: P(Y=1 | do(X=1))")
    print(f"Parameters: {lp.param_names}")
    
    bounds = compute_symbolic_bounds_dual(lp, sense='both')
    
    # Test with sample distribution
    # For Z,X,Y we have 8 parameters
    test_dist = np.array([0.15, 0.10, 0.12, 0.08, 0.18, 0.12, 0.10, 0.15])
    upper_val = bounds['upper_bound'].evaluate(test_dist)
    lower_val = bounds['lower_bound'].evaluate(test_dist)
    
    print(f"\nWith test distribution")
    print(f"Upper bound: {upper_val:.4f}")
    print(f"Lower bound: {lower_val:.4f}")
    print(f"Width: {upper_val - lower_val:.4f}")
    
    if lower_val <= upper_val:
        print("✓ Bounds are valid!")
    else:
        print("✗ ERROR: Lower > Upper")
    
    return bounds


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REPLICATING SECTION 6 EXAMPLES FROM SACHS ET AL.")
    print("="*80)
    
    # Run all tests
    results = {}
    
    try:
        results['simple'] = test_simple_confounding()
    except Exception as e:
        print(f"\n✗ Simple confounding test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['iv'] = test_instrumental_variable()
    except Exception as e:
        print(f"\n✗ IV test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['mediation'] = test_mediation()
    except Exception as e:
        print(f"\n✗ Mediation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['chain'] = test_chain()
    except Exception as e:
        print(f"\n✗ Chain test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tests completed: {len(results)}/4")
    
    for name, result in results.items():
        if result:
            print(f"✓ {name.upper()}: Successfully computed symbolic bounds")
        else:
            print(f"✗ {name.upper()}: Failed")
