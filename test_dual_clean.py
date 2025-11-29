"""
New clean implementation using dual LP approach
"""
import numpy as np
import sympy as sp
from fractions import Fraction
import cdd
from typing import List, Tuple
from symbolic_bounds import LinearProgram

def compute_symbolic_bounds_dual(lp: LinearProgram, sense: str = 'both') -> Tuple[List[sp.Expr], List[sp.Expr]]:
    """
    Compute symbolic bounds using dual LP approach (following Sachs et al.)
    
    Args:
        lp: LinearProgram with symbolic RHS
        sense: 'max', 'min', or 'both'
    
    Returns:
        Tuple of (upper_bound_expressions, lower_bound_expressions)
    """
    # Extract LP components
    c0 = lp.objective  # objective vector
    A_e = lp.constraint_matrix  # equality constraints
    n = lp.aleph_R  # number of variables
    m_e = lp.n_constraints  # number of equality constraints
    
    # No inequality constraints for now (can be added later)
    m_l = 0
    A_l = np.zeros((0, n))
    b_l = np.zeros((0, 1))
    
    # Create symbolic variables for parameters
    param_symbols = {param.name: sp.Symbol(param.name, real=True, nonnegative=True)
                    for param in lp.rhs_params}
    b_sym = [param_symbols[name] for name in lp.rhs_symbolic]
    
    print(f"\nPrimal LP: {n} variables, {m_e} equality constraints")
    print(f"Objective coefficients: {c0.flatten()[:5]}...")  # show first 5
    
    # Compute for both MAX and MIN
    upper_vertex_expressions = []
    lower_vertex_expressions = []
    
    for opt_sense in ['max', 'min']:
        print(f"\n--- Computing for {opt_sense.upper()} ---")
        
        # Build dual LP constraints
        # For MAX primal: Dual is min b^T y subject to A^T y >= c
        # For MIN primal: Dual is max b^T y subject to A^T y <= c
        
        if opt_sense == 'max':
            # Convert A^T y >= c to -A^T y <= -c
            a1 = -A_e.T  # shape (n, m_e)
            b1 = -c0.reshape(-1, 1)  # shape (n, 1)
        else:
            # A^T y <= c stays as is
            a1 = A_e.T
            b1 = c0.reshape(-1, 1)
        
        print(f"Dual LP: {m_e} variables, {n} constraints")
        
        # Convert to CDD H-representation format: [b1 | -a1]
        mat = np.hstack([b1, -a1])
        
        # Convert to exact arithmetic using Fractions
        mat_frac = [[Fraction(float(x)).limit_denominator(10000) for x in row] for row in mat]
        
        # Create CDD matrix
        h_mat = cdd.matrix_from_array(mat_frac)
        h_mat.rep_type = cdd.RepType.INEQUALITY
        
        # Convert H-representation to V-representation
        poly = cdd.polyhedron_from_matrix(h_mat)
        v_mat = cdd.copy_generators(poly)
        
        # Extract vertices
        dual_vertices = []
        v_array = v_mat.array
        for i, row in enumerate(v_array):
            if row[0] == 1:  # vertex (not ray)
                vertex = [float(row[j]) for j in range(1, len(row))]
                dual_vertices.append(vertex)
        
        print(f"Found {len(dual_vertices)} dual vertices")
        
        # For each dual vertex, compute the dual objective symbolically
        # Dual objective: b^T y (where b is symbolic)
        vertex_expressions = []
        
        for i, y in enumerate(dual_vertices):
            # Compute b^T y symbolically
            expr = sp.Integer(0)
            for j in range(m_e):
                expr += b_sym[j] * y[j]
            vertex_expressions.append(sp.simplify(expr))
        
        print(f"Symbolic expressions (first 3):")
        for i, expr in enumerate(vertex_expressions[:3]):
            print(f"  v{i+1} = {expr}")
        if len(vertex_expressions) > 3:
            print(f"  ... and {len(vertex_expressions) - 3} more")
        
        if opt_sense == 'max':
            upper_vertex_expressions = vertex_expressions
        else:
            lower_vertex_expressions = vertex_expressions
    
    if sense == 'max':
        return upper_vertex_expressions, None
    elif sense == 'min':
        return None, lower_vertex_expressions
    else:
        return upper_vertex_expressions, lower_vertex_expressions


# Test with simple X -> Y case
if __name__ == "__main__":
    from symbolic_bounds import DAG, ProgramFactory
    
    print("="*80)
    print("TEST: Symbolic Bounds for X -> Y using Dual Approach")
    print("="*80)
    
    # Create simple DAG
    dag = DAG()
    X = dag.add_node('X', support={0, 1}, partition='R')
    Y = dag.add_node('Y', support={0, 1}, partition='R')
    dag.add_edge(X, Y)
    dag.generate_all_response_types()
    
    # Build LP for P(Y=1 | do(X=1))
    lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
    
    print(f"\nQuery: P(Y=1 | do(X=1))")
    print(f"Parameters: {[p.name for p in lp.rhs_params]}")
    
    upper_exprs, lower_exprs = compute_symbolic_bounds_dual(lp, sense='both')
    
    print("\n" + "="*80)
    print("UPPER BOUND")
    print("="*80)
    print(f"max({', '.join([str(e) for e in upper_exprs[:5]])}{'...' if len(upper_exprs) > 5 else ''})")
    
    print("\n" + "="*80)
    print("LOWER BOUND")
    print("="*80)
    print(f"min({', '.join([str(e) for e in lower_exprs[:5]])}{'...' if len(lower_exprs) > 5 else ''})")
    
    # Test numerical evaluation
    print("\n" + "="*80)
    print("NUMERICAL EVALUATION")
    print("="*80)
    param_values = {
        'p_X=0,Y=0': 0.2,
        'p_X=0,Y=1': 0.3,
        'p_X=1,Y=0': 0.1,
        'p_X=1,Y=1': 0.4
    }
    
    # Evaluate
    def eval_expr(expr):
        subs_dict = {sp.Symbol(k, real=True, nonnegative=True): v for k, v in param_values.items()}
        result = expr.subs(subs_dict)
        return float(result)
    
    upper_vals = [eval_expr(expr) for expr in upper_exprs]
    lower_vals = [eval_expr(expr) for expr in lower_exprs]
    
    ub = max(upper_vals)
    lb = min(lower_vals)
    
    print(f"Upper bound: {ub:.4f}")
    print(f"Lower bound: {lb:.4f}")
    print(f"Width: {ub - lb:.4f}")
    
    if 0 <= lb <= ub <= 1:
        print("\n✓ Bounds are valid!")
    else:
        print(f"\n✗ Invalid bounds: {lb} to {ub}")
