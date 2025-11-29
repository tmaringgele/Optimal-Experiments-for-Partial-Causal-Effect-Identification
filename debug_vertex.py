"""Debug symbolic vertex enumeration"""
from symbolic_bounds import DAG, ProgramFactory, SymbolicVertexEnumerator
import sympy as sp
from sympy import Matrix, simplify
import numpy as np

# Create simple X -> Y DAG
dag = DAG()
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(X, Y)
dag.generate_all_response_types()

# Create linear program for P(Y=1 | do(X=1))
lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))

print("="*80)
print("LP STRUCTURE")
print("="*80)
print(f"\nDimension n = ℵᴿ = {lp.aleph_R}")
print(f"Number of constraints m = {lp.n_constraints}")
print(f"\nConstraint matrix A (shape {lp.constraint_matrix.shape}):")
print(lp.constraint_matrix)
print(f"\nRHS symbolic names: {lp.rhs_symbolic}")
print(f"\nRHS parameters: {[p.name for p in lp.rhs_params]}")

if lp.objective is not None:
    print(f"\nObjective vector c for P(Y=1 | do(X=1)):")
    print(f"Shape: {lp.objective.shape}")
    print(lp.objective)
else:
    print("\nNo objective vector set")

# Manually compute n_zeros_needed
m = lp.n_constraints
n = lp.aleph_R
total_constraints = m + 1  # +1 for normalization

print("\n" + "="*80)
print("VERTEX ENUMERATION CALCULATION")
print("="*80)
print(f"\nEquality constraints (A q = b): {m}")
print(f"Normalization (Σq = 1): 1")
print(f"Total equality constraints: {total_constraints}")
print(f"Number of variables: {n}")
print(f"\nFor a vertex, need exactly {n} active constraints")
print(f"Already have {total_constraints} equality constraints")
print(f"Need {n - total_constraints} non-negativity constraints to be active (i.e., variables = 0)")

# Try manual vertex enumeration
import itertools

# Create symbolic variables
param_symbols = {param.name: sp.Symbol(param.name, real=True, nonnegative=True)
                for param in lp.rhs_params}

print(f"\nParameter symbols created: {list(param_symbols.keys())}")
print(f"RHS symbolic has {len(lp.rhs_symbolic)} entries: {lp.rhs_symbolic}")

A_eq = Matrix(lp.constraint_matrix)
b_list = [param_symbols[name] for name in lp.rhs_symbolic]
print(f"\nBuilding RHS vector with {len(b_list)} entries")
print(f"Entries: {b_list}")
b_sym = Matrix(b_list)

# Add normalization
A_full = A_eq.row_insert(m, Matrix([[1]*n]))
b_full = b_sym.row_insert(m, Matrix([1]))

print(f"\nAugmented system A_full (with normalization):")
print(f"Shape: {A_full.shape}")
print(A_full)
print(f"\nAugmented RHS b_full:")
print(b_full)

n_zeros_needed = n - total_constraints

print(f"\n" + "="*80)
print(f"TRYING ALL BASES")
print("="*80)
print(f"\nNeed to set {n_zeros_needed} variables to zero")
print(f"Trying all C({n}, {n_zeros_needed}) = {len(list(itertools.combinations(range(n), n_zeros_needed)))} combinations\n")

vertices_found = 0
for idx, zero_indices in enumerate(itertools.combinations(range(n), n_zeros_needed)):
    free_indices = [i for i in range(n) if i not in zero_indices]
    
    print(f"Attempt {idx+1}: Zero indices = {zero_indices}, Free indices = {free_indices}")
    
    # Extract submatrix
    A_reduced = A_full[:, free_indices]
    
    print(f"  A_reduced shape: {A_reduced.shape}")
    
    if A_reduced.shape[0] != A_reduced.shape[1]:
        print(f"  ❌ Not square - skip")
        continue
    
    try:
        det = A_reduced.det()
        print(f"  Determinant: {det}")
        
        if det == 0:
            print(f"  ❌ Singular - skip")
            continue
        
        # Solve
        q_free = A_reduced.inv() * b_full
        
        print(f"  ✅ VERTEX FOUND!")
        vertices_found += 1
        print(f"  Solution for free variables:")
        for fidx, free_idx in enumerate(free_indices):
            print(f"    q[{free_idx}] = {simplify(q_free[fidx])}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()

print("="*80)
print(f"Total vertices found: {vertices_found}")
print("="*80)
