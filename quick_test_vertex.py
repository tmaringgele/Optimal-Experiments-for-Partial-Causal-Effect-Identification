"""Quick test of vertex enumeration - should complete in seconds"""
from symbolic_bounds import ProgramFactory, compute_causal_bounds
from symbolic_bounds.dag import DAG

print("Quick Vertex Enumeration Test\n")

# Simple case: X -> Y (no confounding)
dag = DAG()
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(X, Y)
dag.generate_all_response_types()

lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))

param_values = {
    'p_X=0,Y=0': 0.2,
    'p_X=0,Y=1': 0.3,
    'p_X=1,Y=0': 0.1,
    'p_X=1,Y=1': 0.4,
}

print(f"DAG: X -> Y (ℵᴿ = {lp.aleph_R})")
print(f"Query: P(Y=1 | do(X=1))")
print(f"Computing bounds...\n")

lb, ub = compute_causal_bounds(lp, param_values, "P(Y=1 | do(X=1))", verbose=True)

print(f"\n✓ Test completed successfully!")
print(f"  Bounds: [{lb:.4f}, {ub:.4f}]")
