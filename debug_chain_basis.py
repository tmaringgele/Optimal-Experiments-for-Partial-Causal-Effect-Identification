"""Debug a specific basis for chain case"""
import sympy as sp
from sympy import Matrix, simplify
from symbolic_bounds import DAG, ProgramFactory
import itertools

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

# Create symbolic variables
param_symbols = {param.name: sp.Symbol(param.name, real=True, nonnegative=True)
                for param in lp.rhs_params}

A_eq = Matrix(lp.constraint_matrix)
b_sym = Matrix([param_symbols[name] for name in lp.rhs_symbolic])

n = lp.aleph_R  # 16
m = lp.n_constraints  # 8
n_zeros_needed = n - m  # 8

print(f"n = {n}, m = {m}, need {n_zeros_needed} zeros")
print(f"\nConstraint matrix (8x16):")
print(A_eq)

# Check rank of A
print(f"\nRank of constraint matrix: {A_eq.rank()}")
print(f"Should be: {min(m, n)} = {min(m, n)}")

# Try a specific basis: set first 8 variables to zero
zero_indices = tuple(range(8))
free_indices = list(range(8, 16))

print(f"\n\nTrying basis:")
print(f"  Zero indices: {zero_indices}")
print(f"  Free indices: {free_indices}")

A_reduced = A_eq[:, free_indices]
print(f"\nReduced matrix (selecting columns {free_indices}):")
print(A_reduced)
print(f"Shape: {A_reduced.shape}")
print(f"Rank: {A_reduced.rank()}")
print(f"Determinant: {A_reduced.det()}")

# Try another basis
print("\n" + "="*80)
zero_indices2 = (0, 1, 2, 3, 4, 5, 6, 7)
free_indices2 = [8, 9, 10, 11, 12, 13, 14, 15]
A_reduced2 = A_eq[:, free_indices2]
print(f"Another basis - free indices: {free_indices2}")
print(f"Matrix:\n{A_reduced2}")
print(f"Rank: {A_reduced2.rank()}")
print(f"Det: {A_reduced2.det()}")

# Let me check the structure of the constraint matrix more carefully
print("\n" + "="*80)
print("CONSTRAINT STRUCTURE ANALYSIS")
print("="*80)
print("\nLooking at the constraint matrix rows:")
for i in range(m):
    row = A_eq.row(i)
    nonzero = [j for j in range(n) if row[j] != 0]
    print(f"Row {i}: nonzero columns = {nonzero}")

print("\nLooking at the constraint matrix columns:")
for j in range(n):
    col = A_eq.col(j)
    nonzero = [i for i in range(m) if col[i] != 0]
    print(f"Col {j}: nonzero rows = {nonzero}")
