"""
Test SYMBOLIC vertex enumeration - bounds as functions of parameters.
"""

import sympy as sp
from symbolic_bounds import (
    ProgramFactory,
    SymbolicVertexEnumerator,
    compute_symbolic_causal_bounds
)
from symbolic_bounds.dag import DAG


def test_simple_unconditional():
    """Test symbolic bounds for X -> Y (no confounding, no W_L)"""
    print("=" * 80)
    print("TEST 1: Symbolic Bounds for X -> Y (Unconditional)")
    print("=" * 80)
    
    # Create simple DAG
    dag = DAG()
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    print("\nDAG Structure:")
    print("  W_L = {} (no unobserved confounders)")
    print("  W_R = {X, Y}")
    print("  Edges: X -> Y")
    
    # Build LP for P(Y=1 | do(X=1))
    print("\nQuery: P(Y=1 | do(X=1))")
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
    
    print(f"\nSymbolic parameters:")
    for param in lp.rhs_params:
        print(f"  {param.name}")
    
    # Compute symbolic bounds
    print("\n" + "-" * 80)
    print("Computing symbolic bounds...")
    print("-" * 80)
    
    lower, upper = compute_symbolic_causal_bounds(
        lp, 
        "P(Y=1 | do(X=1))",
        verbose=False
    )
    
    print(f"\nFound {upper.n_vertices} vertices")
    print(f"\nSymbolic expressions:")
    for i, expr in enumerate(upper.vertex_expressions):
        print(f"  Vertex {i+1}: {expr}")
    
    print(f"\nUpper bound = max({', '.join([f'v{i+1}' for i in range(upper.n_vertices)])})")
    print(f"Lower bound = min({', '.join([f'v{i+1}' for i in range(lower.n_vertices)])})")
    
    # Test numerical evaluation
    print("\n" + "-" * 80)
    print("Testing numerical evaluation:")
    print("-" * 80)
    
    param_values = {
        'p_X=0,Y=0': 0.2,
        'p_X=0,Y=1': 0.3,
        'p_X=1,Y=0': 0.1,
        'p_X=1,Y=1': 0.4,
    }
    
    print("\nGiven parameter values:")
    for k, v in param_values.items():
        print(f"  {k} = {v}")
    
    lb_val = lower.evaluate(param_values)
    ub_val = upper.evaluate(param_values)
    
    print(f"\nEvaluated bounds:")
    print(f"  Lower bound = {lb_val:.4f}")
    print(f"  Upper bound = {ub_val:.4f}")
    print(f"  Width = {ub_val - lb_val:.4f}")
    
    assert 0 <= lb_val <= ub_val <= 1, "Bounds must be valid probabilities"
    print("\n✓ Symbolic bounds computed and validated\n")
    
    return lower, upper


def test_simple_chain():
    """Test symbolic bounds for Z -> X -> Y"""
    print("=" * 80)
    print("TEST 2: Symbolic Bounds for Z -> X -> Y (Chain)")
    print("=" * 80)
    
    # Create DAG
    dag = DAG()
    Z = dag.add_node('Z', support={0, 1}, partition='L')
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(Z, X)
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    print("\nDAG Structure:")
    print("  W_L = {Z}")
    print("  W_R = {X, Y}")
    print("  Edges: Z -> X, X -> Y")
    
    # Build LP
    print("\nQuery: P(Y=1 | do(X=1))")
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
    
    print(f"\nSymbolic parameters (conditional probabilities P(X,Y|Z)):")
    for param in lp.rhs_params:
        print(f"  {param.name}")
    
    # Compute symbolic bounds
    print("\n" + "-" * 80)
    print("Computing symbolic bounds (this may take a moment)...")
    print("-" * 80)
    
    lower, upper = SymbolicVertexEnumerator.compute_symbolic_bounds(lp, sense='both')
    
    print(f"\nFound {upper.n_vertices} vertices")
    
    # Show simplified expressions
    print(f"\nSymbolic vertex expressions (first 5):")
    for i, expr in enumerate(upper.vertex_expressions[:5]):
        simplified = sp.simplify(expr)
        print(f"  v{i+1} = {simplified}")
    if len(upper.vertex_expressions) > 5:
        print(f"  ... and {len(upper.vertex_expressions) - 5} more vertices")
    
    print(f"\nUpper bound = max(v1, v2, ..., v{upper.n_vertices})")
    print(f"Lower bound = min(v1, v2, ..., v{lower.n_vertices})")
    
    # Test numerical evaluation
    print("\n" + "-" * 80)
    print("Testing numerical evaluation:")
    print("-" * 80)
    
    param_values = {
        'p_X=0,Y=0|Z=0': 0.3,
        'p_X=0,Y=1|Z=0': 0.2,
        'p_X=1,Y=0|Z=0': 0.1,
        'p_X=1,Y=1|Z=0': 0.4,
        'p_X=0,Y=0|Z=1': 0.1,
        'p_X=0,Y=1|Z=1': 0.3,
        'p_X=1,Y=0|Z=1': 0.2,
        'p_X=1,Y=1|Z=1': 0.4,
    }
    
    print("\nGiven parameter values:")
    for k, v in sorted(param_values.items()):
        print(f"  {k} = {v}")
    
    lb_val = lower.evaluate(param_values)
    ub_val = upper.evaluate(param_values)
    
    print(f"\nEvaluated bounds:")
    print(f"  Lower bound = {lb_val:.4f}")
    print(f"  Upper bound = {ub_val:.4f}")
    print(f"  Width = {ub_val - lb_val:.4f}")
    
    assert 0 <= lb_val <= ub_val <= 1, "Bounds must be valid probabilities"
    print("\n✓ Symbolic bounds computed and validated\n")
    
    return lower, upper


def test_symbolic_bound_properties():
    """Test properties of symbolic bounds"""
    print("=" * 80)
    print("TEST 3: Symbolic Bound Properties")
    print("=" * 80)
    
    # Create simple case
    dag = DAG()
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
    
    print("\nComputing symbolic bounds...")
    lower, upper = SymbolicVertexEnumerator.compute_symbolic_bounds(lp, sense='both')
    
    print(f"\n✓ Vertices found: {upper.n_vertices}")
    
    # Test simplification
    print("\nTesting simplification...")
    lower_simp = lower.simplify()
    upper_simp = upper.simplify()
    print("✓ Simplification successful")
    
    # Test multiple evaluations
    print("\nTesting multiple evaluations...")
    test_values = [
        {'p_X=0,Y=0': 0.1, 'p_X=0,Y=1': 0.4, 'p_X=1,Y=0': 0.2, 'p_X=1,Y=1': 0.3},
        {'p_X=0,Y=0': 0.25, 'p_X=0,Y=1': 0.25, 'p_X=1,Y=0': 0.25, 'p_X=1,Y=1': 0.25},
        {'p_X=0,Y=0': 0.4, 'p_X=0,Y=1': 0.1, 'p_X=1,Y=0': 0.1, 'p_X=1,Y=1': 0.4},
    ]
    
    for i, params in enumerate(test_values):
        lb = lower.evaluate(params)
        ub = upper.evaluate(params)
        print(f"  Test {i+1}: [{lb:.4f}, {ub:.4f}]  Width={ub-lb:.4f}")
        assert 0 <= lb <= ub <= 1, f"Invalid bounds for test {i+1}"
    
    print("\n✓ All property tests passed\n")


def test_print_formatted():
    """Test formatted printing of symbolic bounds"""
    print("=" * 80)
    print("TEST 4: Formatted Symbolic Bound Output")
    print("=" * 80)
    
    dag = DAG()
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
    
    # Use the verbose function
    lower, upper = compute_symbolic_causal_bounds(
        lp,
        "P(Y=1 | do(X=1))",
        verbose=True
    )
    
    print("\n✓ Formatted output complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SYMBOLIC VERTEX ENUMERATION TESTS")
    print("=" * 80)
    print("\nComputing bounds as SYMBOLIC FUNCTIONS of parameters")
    print("(not requiring numerical values)")
    print("=" * 80 + "\n")
    
    # Run tests
    test_simple_unconditional()
    test_simple_chain()
    test_symbolic_bound_properties()
    test_print_formatted()
    
    print("=" * 80)
    print("ALL SYMBOLIC TESTS PASSED ✓")
    print("=" * 80)
    print("\nSymbolic vertex enumeration is working!")
    print("Bounds are expressed as symbolic functions of observational parameters.")
    print("\nKey capabilities:")
    print("  ✓ Vertices as symbolic expressions")
    print("  ✓ Bounds as max/min over symbolic vertices")
    print("  ✓ Numerical evaluation for any parameter values")
    print("  ✓ Simplification of symbolic expressions")
    print("\nNext steps:")
    print("  1. Optimize symbolic expression simplification")
    print("  2. Implement sensitivity analysis")
    print("  3. Experimental design based on symbolic bounds")
