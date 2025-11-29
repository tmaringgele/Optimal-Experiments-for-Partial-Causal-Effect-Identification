"""
Test and demonstration of symbolic LP construction for causal effect bounds.
"""

from symbolic_bounds import ProgramFactory, LinearProgram
from symbolic_bounds.dag import DAG


def test_simple_chain():
    """Test LP construction for simple chain: Z -> X -> Y"""
    print("="*80)
    print("TEST 1: Simple Chain Z -> X -> Y")
    print("="*80)
    
    # Create DAG
    dag = DAG()
    Z = dag.add_node('Z', support={0, 1}, partition='L')
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(Z, X)
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    print(f"\nDAG structure:")
    print(f"  W_L = {{Z}}")
    print(f"  W_R = {{X, Y}}")
    print(f"  Edges: Z -> X, X -> Y")
    
    # Build LP for P(Y=1 | do(X=1))
    print(f"\nQuery: P(Y=1 | do(X=1))")
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,), sense='max')
    
    # Print LP
    lp.print_lp(show_full_matrix=False)
    
    # Show feasible region info
    print("\n" + "="*80)
    print("FEASIBLE REGION INFORMATION")
    print("="*80)
    info = lp.get_feasible_region_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n")
    return lp


def test_unconditional():
    """Test LP construction for unconditional case (no W_L nodes)"""
    print("="*80)
    print("TEST 2: Unconditional Case X -> Y (no W_L)")
    print("="*80)
    
    # Create DAG with no W_L nodes
    dag = DAG()
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    print(f"\nDAG structure:")
    print(f"  W_L = {{}}")
    print(f"  W_R = {{X, Y}}")
    print(f"  Edges: X -> Y")
    
    # Build LP for P(Y=1 | do(X=1))
    print(f"\nQuery: P(Y=1 | do(X=1))")
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,), sense='max')
    
    # Print LP
    lp.print_lp(show_full_matrix=False)
    
    print("\n")
    return lp


def test_confounding():
    """Test LP construction with confounding: Z -> X -> Y, Z -> Y"""
    print("="*80)
    print("TEST 3: Confounding Structure")
    print("="*80)
    
    # Create DAG with confounding
    dag = DAG()
    Z = dag.add_node('Z', support={0, 1}, partition='L')
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(Z, X)
    dag.add_edge(X, Y)
    dag.add_edge(Z, Y)  # Confounding
    dag.generate_all_response_types()
    
    print(f"\nDAG structure:")
    print(f"  W_L = {{Z}}")
    print(f"  W_R = {{X, Y}}")
    print(f"  Edges: Z -> X, X -> Y, Z -> Y (confounded)")
    
    # Build LP for P(Y=1 | do(X=0))
    print(f"\nQuery: P(Y=1 | do(X=0))")
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (0,), sense='max')
    
    # Print LP
    lp.print_lp(show_full_matrix=False)
    
    print("\n")
    return lp


def test_evaluate_rhs():
    """Test evaluating the symbolic RHS with concrete probability values"""
    print("="*80)
    print("TEST 4: Evaluating Symbolic RHS with Concrete Values")
    print("="*80)
    
    # Create simple DAG
    dag = DAG()
    Z = dag.add_node('Z', support={0, 1}, partition='L')
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(Z, X)
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    # Build LP
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,), sense='max')
    
    print(f"\nSymbolic parameters in RHS:")
    for param in lp.rhs_params:
        print(f"  {param.name}")
    
    # Create concrete probability distribution
    # For Z=0: P(X,Y|Z=0)
    # For Z=1: P(X,Y|Z=1)
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
    
    print(f"\nConcrete probability values:")
    for name, value in param_values.items():
        print(f"  {name} = {value}")
    
    # Evaluate RHS
    rhs_numeric = lp.evaluate_rhs(param_values)
    
    print(f"\nEvaluated RHS vector b:")
    for i, (label, val) in enumerate(zip(lp.constraint_labels, rhs_numeric)):
        print(f"  b[{i}] = {val:.3f}  ({label})")
    
    print(f"\nVerification:")
    print(f"  Sum of P(X,Y|Z=0): {sum(rhs_numeric[:4]):.3f} (should be 1.0)")
    print(f"  Sum of P(X,Y|Z=1): {sum(rhs_numeric[4:]):.3f} (should be 1.0)")
    
    print("\n")


if __name__ == "__main__":
    # Run all tests
    lp1 = test_simple_chain()
    lp2 = test_unconditional()
    lp3 = test_confounding()
    test_evaluate_rhs()
    
    print("="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print("\nThe LP is now ready for vertex enumeration!")
    print("\nNext steps:")
    print("  1. Implement vertex enumeration algorithm")
    print("  2. Compute symbolic bounds as functions of parameters")
    print("  3. Optimize experiment design based on bound tightness")
