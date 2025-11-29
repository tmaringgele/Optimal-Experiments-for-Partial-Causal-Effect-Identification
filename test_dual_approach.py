"""
Test dual LP approach with simple example
"""
import numpy as np
import cdd
from fractions import Fraction

# Simple example: max c^T q subject to A q = b, q >= 0
# c = [1, 2, 3, 4]
# A = [[1, 1, 0, 0],
#      [0, 0, 1, 1]]
# b = [p1, p2]  # symbolic

# Primal LP:
# max c^T q
# s.t. A q = b
#      q >= 0

# Dual LP (standard form):
# min b^T y
# s.t. A^T y >= c
#      y free (unrestricted)

# To use CDD, we need to convert to:
# min/max objective
# s.t. a1 * x <= b1
#      (all inequalities, variables >= 0)

# For the dual, we have:
# A^T y >= c  (m_e equality constraint duals, unrestricted)
# This becomes: -A^T y <= -c

# But y is unrestricted. We can split y = y+ - y- where y+, y- >= 0
# Or we can use CDD's representation directly

# Let's follow the R code more carefully:
print("="*80)
print("UNDERSTANDING THE R CODE")
print("="*80)

print("""
The R code does:

# Primal LP (for MAX):
# max c0^T q
# s.t. A_e q = b (symbolic)
#      A_l q <= b_l (numeric, if any)
#      q >= 0

# Dual LP:
# min b^T y_e + b_l^T y_l
# s.t. A_e^T y_e + A_l^T y_l >= c0
#      y_l >= 0, y_e free

# For MAX primal, we want MIN dual.
# To convert to CDD format (which wants <= inequalities):
# We need: a1 * [y_l, y_e]^T <= b1

# From A_e^T y_e + A_l^T y_l >= c0, we get:
# -A_l^T y_l - A_e^T y_e <= -c0

# And y_l >= 0 becomes:
# -y_l <= 0

# So the dual constraint matrix is:
# a1 = [ -A_l^T  -A_e^T ]
#      [ -I      0      ]
# b1 = [ -c0            ]
#      [ 0              ]

# CDD then finds vertices of this polytope.
# For each vertex [y_l, y_e], the dual objective is:
# b^T y_e + b_l^T y_l  (where b is symbolic!)
""")

print("\nLet me implement this step by step...")

# Example problem
c0 = np.array([[1.0], [2.0], [3.0], [4.0]])  # objective coefficients
A_e = np.array([[1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0]])  # equality constraints
# b = [p1, p2] is symbolic

n = c0.shape[0]  # number of variables (4)
m_e = A_e.shape[0]  # number of equality constraints (2)
m_l = 0  # no inequality constraints
A_l = np.zeros((0, n))
b_l = np.zeros((0, 1))

# Build dual LP in CDD format
# Variables: [y_l (m_l), y_e (m_e)] = [y_e1, y_e2]
# Constraints:
# -A_l^T y_l - A_e^T y_e <= -c0
# -I y_l <= 0

a1_top = -A_e.T  # shape (n, m_e) = (4, 2)
if m_l > 0:
    a1_top = np.hstack([-A_l.T, -A_e.T])
    a1_bottom = np.hstack([-np.eye(m_l), np.zeros((m_l, m_e))])
    a1 = np.vstack([a1_top, a1_bottom])
    b1 = np.vstack([-c0, np.zeros((m_l, 1))])
else:
    a1 = a1_top
    b1 = -c0

print(f"\nDual constraint matrix a1 (shape {a1.shape}):")
print(a1)
print(f"\nDual RHS b1 (shape {b1.shape}):")
print(b1)

# Convert to CDD H-representation
# Format: [b1 | -a1] for the representation a1 * x <= b1
# becomes: -a1 * x + b1 >= 0 in CDD format
mat = np.hstack([b1, -a1])
print(f"\nCDD H-representation matrix (shape {mat.shape}):")
print(mat)

# Convert to Fraction for exact arithmetic
mat_frac = [[Fraction(float(x)).limit_denominator(1000) for x in row] for row in mat]

# Create CDD matrix
h_mat = cdd.matrix_from_array(mat_frac)
h_mat.rep_type = cdd.RepType.INEQUALITY

print(f"\nCDD matrix created:")
print(h_mat)

# Convert H-representation to V-representation
print("\nComputing V-representation (vertices)...")
poly = cdd.polyhedron_from_matrix(h_mat)
v_mat = cdd.copy_generators(poly)

print(f"\nV-representation:")
print(v_mat)

print("\nExtracting vertices:")
v_array = v_mat.array
for i, row in enumerate(v_array):
    if row[0] == 1:  # vertex (not ray)
        vertex = [float(row[j]) for j in range(1, len(row))]
        print(f"  Vertex {i}: {vertex}")
        
print("\n✓ Dual approach with CDD works!")
