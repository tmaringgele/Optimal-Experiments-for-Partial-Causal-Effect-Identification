"""Debug chain case"""
import sympy as sp
from symbolic_bounds import DAG, ProgramFactory
import numpy as np

# Create Z -> X -> Y chain
dag = DAG()
Z = dag.add_node('Z', support={0, 1}, partition='L')
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(Z, X)
dag.add_edge(X, Y)
dag.generate_all_response_types()

# Build LP for P(Y=1 | do(X=1))
lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))

print("="*80)
print("LP STRUCTURE FOR Z -> X -> Y")
print("="*80)
print(f"\nDimension n = ℵᴿ = {lp.aleph_R}")
print(f"Number of constraints m = {lp.n_constraints}")
print(f"\nConstraint matrix A (shape {lp.constraint_matrix.shape}):")
print(lp.constraint_matrix)
print(f"\nRHS symbolic names ({len(lp.rhs_symbolic)} entries):")
for i, name in enumerate(lp.rhs_symbolic):
    print(f"  {i}: {name}")

print(f"\n\nAnalysis:")
print(f"  m = {lp.n_constraints}")
print(f"  n = {lp.aleph_R}")
print(f"  Need n - m = {lp.aleph_R - lp.n_constraints} variables to be zero")

# Check if constraints sum to conditional probabilities
print(f"\n\nConstraint structure:")
print(f"We have 2 contexts (Z=0, Z=1)")
print(f"For each context, we have 4 configs of (X,Y)")
print(f"Total: 8 constraints")
print(f"\nFor each context Z=z, the 4 constraints should sum to:")
print(f"  P(X=0,Y=0|Z=z) + P(X=0,Y=1|Z=z) + P(X=1,Y=0|Z=z) + P(X=1,Y=1|Z=z) = 1")
print(f"\nSo sum(q) is IMPLIED by constraints IF RHS sums to 1 within each context")

# Check RHS structure
print(f"\n\nRHS vector structure:")
z0_params = [n for n in lp.rhs_symbolic if '|Z=0' in n]
z1_params = [n for n in lp.rhs_symbolic if '|Z=1' in n]
print(f"Z=0 context: {len(z0_params)} parameters")
for p in z0_params:
    print(f"  {p}")
print(f"Z=1 context: {len(z1_params)} parameters")
for p in z1_params:
    print(f"  {p}")
