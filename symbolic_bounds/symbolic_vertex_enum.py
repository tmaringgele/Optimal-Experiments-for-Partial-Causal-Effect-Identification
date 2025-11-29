"""
Symbolic vertex enumeration for parametric linear programs using dual LP approach.

This module computes vertices and bounds as SYMBOLIC EXPRESSIONS of parameters,
without requiring numerical values for the conditional probabilities.

Algorithm (following Sachs et al.):
1. Formulate the DUAL LP problem from the primal
2. Use CDD library to convert H-representation (inequalities) to V-representation (vertices)
3. For each dual vertex, evaluate the dual objective as a symbolic expression
4. Bounds are max/min over these symbolic expressions

The key insight: Dual vertices are easy to compute (constraints are numeric),
while primal vertices would be hard (RHS is symbolic).
"""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from dataclasses import dataclass
import sympy as sp
from sympy import symbols, simplify, Matrix, lambdify
import cdd
from fractions import Fraction

from .linear_program import LinearProgram, SymbolicParameter


@dataclass
class SymbolicVertex:
    """
    Represents a vertex as a symbolic expression.
    
    Attributes:
        expression: Dictionary mapping decision variable indices to symbolic expressions
        active_constraints: Indices of active constraints at this vertex
        basis: The basis defining this vertex
    """
    expression: Dict[int, sp.Expr]
    active_constraints: List[int]
    basis: Optional[List[int]] = None
    
    def evaluate(self, param_values: Dict[str, float]) -> np.ndarray:
        """Evaluate the symbolic vertex with concrete parameter values."""
        n = len(self.expression)
        result = np.zeros(n)
        for i, expr in self.expression.items():
            if isinstance(expr, (int, float)):
                result[i] = float(expr)
            else:
                # Substitute parameter values
                subs_dict = {sp.Symbol(k): v for k, v in param_values.items()}
                result[i] = float(expr.subs(subs_dict))
        return result


@dataclass
class SymbolicBound:
    """
    Represents a bound as a piecewise symbolic expression.
    
    The bound is the max (for upper) or min (for lower) over all vertices,
    each vertex contributing a symbolic expression.
    
    Attributes:
        vertex_expressions: List of symbolic expressions (one per vertex)
        vertex_labels: Human-readable labels for each vertex
        sense: 'max' or 'min'
        n_vertices: Number of vertices
        vertices: Optional list of SymbolicVertex objects (for validation)
        lp: Optional LinearProgram (for validation)
    """
    vertex_expressions: List[sp.Expr]
    vertex_labels: List[str]
    sense: str
    n_vertices: int
    vertices: Optional[List[SymbolicVertex]] = None
    lp: Optional[LinearProgram] = None
    
    def __repr__(self) -> str:
        return (f"SymbolicBound(sense='{self.sense}', "
                f"n_vertices={self.n_vertices})")
    
    def evaluate(self, param_values: Dict[str, float]) -> float:
        """
        Evaluate the bound with concrete parameter values.
        
        Args:
            param_values: Dictionary mapping parameter names to numerical values
        
        Returns:
            Numerical value of the bound
        """
        values = []
        for idx, expr in enumerate(self.vertex_expressions):
            # Evaluate the objective value
            if isinstance(expr, (int, float)):
                obj_value = float(expr)
            else:
                # Get all free symbols in the expression
                free_syms = expr.free_symbols
                subs_dict = {}
                for sym in free_syms:
                    sym_name = str(sym)
                    if sym_name in param_values:
                        subs_dict[sym] = param_values[sym_name]
                
                # Substitute and convert to float
                result = expr.subs(subs_dict)
                try:
                    obj_value = float(result)
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not evaluate expression {expr} with {param_values}")
                    print(f"After substitution: {result}")
                    print(f"Free symbols: {free_syms}")
                    raise
            
            # If we have vertex info, validate that the vertex is feasible (q >= 0)
            if self.vertices and idx < len(self.vertices):
                vertex = self.vertices[idx]
                # Evaluate all components of the vertex
                is_feasible = True
                for var_idx, var_expr in vertex.expression.items():
                    if isinstance(var_expr, (int, float)):
                        val = float(var_expr)
                    else:
                        free_syms = var_expr.free_symbols
                        subs_dict = {sym: param_values.get(str(sym), 0) for sym in free_syms}
                        val = float(var_expr.subs(subs_dict))
                    
                    if val < -1e-10:  # Small tolerance for numerical errors
                        is_feasible = False
                        break
                
                if is_feasible:
                    values.append(obj_value)
            else:
                values.append(obj_value)
        
        if len(values) == 0:
            raise ValueError("No feasible vertices found for given parameter values")
        
        # Debug: show how many feasible vertices
        if self.vertices:
            print(f"  ({len(values)}/{len(self.vertex_expressions)} vertices are feasible for this evaluation)")
        
        if self.sense == 'max':
            return max(values)
        else:
            return min(values)
    
    def simplify(self) -> 'SymbolicBound':
        """Simplify all symbolic expressions."""
        simplified_exprs = [simplify(expr) for expr in self.vertex_expressions]
        return SymbolicBound(
            vertex_expressions=simplified_exprs,
            vertex_labels=self.vertex_labels,
            sense=self.sense,
            n_vertices=self.n_vertices
        )


class SymbolicVertexEnumerator:
    """
    Enumerates vertices symbolically using dual LP approach (Sachs et al. method).
    
    Algorithm:
    1. Formulate the DUAL LP problem from the primal
    2. Use CDD library to convert H-representation to V-representation
    3. For each dual vertex, evaluate the dual objective symbolically
    4. Bounds are max/min over these symbolic expressions
    
    For the primal LP:
        optimize:   c^T q
        subject to: A q = b  (b contains symbolic parameters)
                   q >= 0
    
    The dual LP is:
        optimize:   b^T y
        subject to: A^T y >= c  (for max primal)
                   y free
    
    The key insight: dual vertices are easy to compute (numeric constraints),
    while primal vertices would be hard (symbolic RHS).
    """
    
    @staticmethod
    def enumerate_dual_vertices(lp: LinearProgram, sense: str = 'max') -> Tuple[np.ndarray, List[sp.Expr]]:
        """
        Enumerate all vertices symbolically.
        
        Args:
            lp: LinearProgram with symbolic RHS
        
        Returns:
            List of SymbolicVertex objects, each expressing q as function of parameters
        
        Algorithm:
            Each vertex is defined by a basis - exactly n linearly independent
            active constraints where n = ℵᴿ (dimension of q).
            We iterate over all possible bases and solve symbolically.
        """
        # Create symbolic variables for parameters
        param_symbols = {param.name: sp.Symbol(param.name, real=True, nonnegative=True)
                        for param in lp.rhs_params}
        
        # Build symbolic constraint matrix
        A_eq = Matrix(lp.constraint_matrix)
        b_sym = Matrix([param_symbols[name] for name in lp.rhs_symbolic])
        
        n = lp.aleph_R  # dimension of q
        m_original = lp.n_constraints  # number of equality constraints (may include redundant rows)
        
        # Check rank - if constraint matrix is rank-deficient, we need to remove redundant rows
        rank = A_eq.rank()
        if rank < m_original:
            print(f"Warning: Constraint matrix has rank {rank} < {m_original} rows. Removing redundant constraints...")
            # Find independent rows using row echelon form
            # Use rref to identify pivot columns, which correspond to independent rows
            rref_form, pivot_cols = A_eq.rref()
            
            # The number of pivot columns equals the rank
            # We need to select 'rank' independent rows from A_eq
            # Strategy: select rows that contribute to the row echelon form
            independent_rows = []
            for i in range(m_original):
                # Check if row i is independent of previous rows
                test_matrix = Matrix([A_eq.row(j) for j in independent_rows + [i]])
                if test_matrix.rank() == len(independent_rows) + 1:
                    independent_rows.append(i)
                if len(independent_rows) == rank:
                    break
            
            print(f"Selected independent rows: {independent_rows}")
            A_eq = Matrix([A_eq.row(i) for i in independent_rows])
            b_sym = Matrix([b_sym[i] for i in independent_rows])
            m = rank
        else:
            m = m_original
        
        # We need to find vertices of:
        # A q = b, q >= 0
        # 
        # NOTE: We do NOT add sum(q) = 1 as an explicit constraint because
        # it is already implied by the structure of our constraints.
        # The RHS vector b contains conditional probabilities that sum to 1,
        # so the equality constraints Aq = b already enforce sum(q) = 1.
        
        A_full = A_eq
        b_full = b_sym
        total_constraints = m
        
        vertices = []
        
        import itertools
        
        # A vertex is where exactly n constraints are active (tight).
        # The constraints can be:
        # - equality constraints (always active)
        # - non-negativity q[i] >= 0 (active when q[i] = 0)
        
        # Since we have m equality constraints and n variables,
        # if m < n, we need (n - m) additional active non-negativity constraints
        # if m == n, we just solve the equality system
        # if m > n, system is over-determined (shouldn't happen for valid LP)
        
        if total_constraints > n:
            # Over-determined system - shouldn't happen
            raise ValueError(f"Over-determined system: {total_constraints} constraints > {n} variables")
        
        n_zeros_needed = n - total_constraints  # number of variables that must be zero
        
        # Iterate over all ways to choose which variables are zero
        total_combinations = len(list(itertools.combinations(range(n), n_zeros_needed)))
        print(f"Checking {total_combinations} bases...")
        
        checked = 0
        for zero_indices in itertools.combinations(range(n), n_zeros_needed):
            checked += 1
            if checked % 1000 == 0:
                print(f"  Progress: {checked}/{total_combinations} bases checked, {len(vertices)} vertices found")
            
            free_indices = [i for i in range(n) if i not in zero_indices]
            
            if len(free_indices) != total_constraints:
                continue
            
            # Extract columns for free variables
            A_reduced = A_full[:, free_indices]
            
            # Check if system is solvable (square and non-singular)
            if A_reduced.shape[0] != A_reduced.shape[1]:
                continue
            
            try:
                det = A_reduced.det()
                if det == 0:
                    continue
                
                # Solve: A_reduced * q_free = b_full
                q_free = A_reduced.inv() * b_full
                
                # Build full vertex
                vertex_expr = {}
                for idx, free_idx in enumerate(free_indices):
                    vertex_expr[free_idx] = simplify(q_free[idx])
                for zero_idx in zero_indices:
                    vertex_expr[zero_idx] = sp.Integer(0)
                
                vertices.append(SymbolicVertex(
                    expression=vertex_expr,
                    active_constraints=list(range(total_constraints)) + [n + zidx for zidx in zero_indices],
                    basis=free_indices
                ))
                
            except Exception as e:
                # Singular or unsolvable system
                continue
        
        print(f"Total: {len(vertices)} vertices found from {total_combinations} bases")
        return vertices
    
    @staticmethod
    def compute_symbolic_bounds(lp: LinearProgram,
                                sense: str = 'both') -> Tuple[Optional[SymbolicBound], 
                                                               Optional[SymbolicBound]]:
        """
        Compute symbolic upper and/or lower bounds.
        
        Args:
            lp: LinearProgram with symbolic RHS
            sense: 'max', 'min', or 'both'
        
        Returns:
            Tuple of (upper_bound, lower_bound) as SymbolicBound objects
            Either can be None if not requested
        
        Example:
            >>> lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
            >>> upper, lower = SymbolicVertexEnumerator.compute_symbolic_bounds(lp)
            >>> print(upper.vertex_expressions[0])  # Symbolic expression in terms of p_*
        """
        # Enumerate vertices symbolically
        vertices = SymbolicVertexEnumerator.enumerate_symbolic_vertices(lp)
        
        if len(vertices) == 0:
            raise ValueError("No vertices found - feasible region may be empty")
        
        # Create symbolic objective vector
        c = lp.objective
        
        # Compute objective value at each vertex symbolically
        vertex_expressions = []
        vertex_labels = []
        
        for i, vertex in enumerate(vertices):
            # Compute c^T q symbolically
            obj_expr = sp.Integer(0)
            for j in range(lp.aleph_R):
                if c[j] != 0:
                    obj_expr += c[j] * vertex.expression[j]
            
            vertex_expressions.append(simplify(obj_expr))
            vertex_labels.append(f"Vertex {i+1}")
        
        # Create bound objects
        upper_bound = None
        lower_bound = None
        
        if sense in ['max', 'both']:
            upper_bound = SymbolicBound(
                vertex_expressions=vertex_expressions,
                vertex_labels=vertex_labels,
                sense='max',
                n_vertices=len(vertices),
                vertices=vertices,
                lp=lp
            )
        
        if sense in ['min', 'both']:
            lower_bound = SymbolicBound(
                vertex_expressions=vertex_expressions,
                vertex_labels=vertex_labels,
                sense='min',
                n_vertices=len(vertices),
                vertices=vertices,
                lp=lp
            )
        
        return upper_bound, lower_bound
    
    @staticmethod
    def print_symbolic_bound(bound: SymbolicBound,
                           query_description: str,
                           simplify_exprs: bool = True) -> None:
        """
        Print symbolic bound in readable format.
        
        Args:
            bound: SymbolicBound object
            query_description: Human-readable query description
            simplify_exprs: Whether to simplify expressions before printing
        """
        bound_type = "Upper Bound" if bound.sense == 'max' else "Lower Bound"
        
        print("=" * 80)
        print(f"SYMBOLIC {bound_type.upper()}: {query_description}")
        print("=" * 80)
        print(f"  Number of vertices: {bound.n_vertices}")
        print(f"  Bound = {bound.sense}(" + ", ".join([f"v{i+1}" for i in range(bound.n_vertices)]) + ")")
        print()
        
        for i, (expr, label) in enumerate(zip(bound.vertex_expressions, bound.vertex_labels)):
            if simplify_exprs:
                expr = simplify(expr)
            print(f"  v{i+1} = {expr}")
        
        print()
        print(f"  {bound_type} = {bound.sense}({{v1, v2, ..., v{bound.n_vertices}}})")
        print("=" * 80)


def compute_symbolic_causal_bounds(lp: LinearProgram,
                                  query_description: str = "Causal Query",
                                  verbose: bool = True) -> Tuple[SymbolicBound, SymbolicBound]:
    """
    Convenience function to compute both symbolic bounds.
    
    Args:
        lp: LinearProgram with symbolic parameters
        query_description: Human-readable query description
        verbose: Whether to print results
    
    Returns:
        Tuple of (lower_bound, upper_bound) as SymbolicBound objects
    
    Example:
        >>> dag = DAG()
        >>> # ... construct dag ...
        >>> lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
        >>> lower, upper = compute_symbolic_causal_bounds(lp, "P(Y=1 | do(X=1))")
        >>> print(f"Lower bound expression: {lower.vertex_expressions[0]}")
    """
    upper_bound, lower_bound = SymbolicVertexEnumerator.compute_symbolic_bounds(
        lp, sense='both'
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"SYMBOLIC CAUSAL EFFECT BOUNDS: {query_description}")
        print("=" * 80)
        print(f"  Vertices enumerated: {upper_bound.n_vertices}")
        print(f"  Bounds expressed as functions of parameters: {lp.n_params}")
        print()
        
        SymbolicVertexEnumerator.print_symbolic_bound(
            lower_bound, query_description, simplify_exprs=True
        )
        print()
        SymbolicVertexEnumerator.print_symbolic_bound(
            upper_bound, query_description, simplify_exprs=True
        )
    
    return lower_bound, upper_bound
