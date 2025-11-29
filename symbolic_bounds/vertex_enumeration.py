"""
Vertex enumeration for symbolic linear programs.

This module implements vertex enumeration to compute tight bounds on causal effects.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass
import pypoman

from .linear_program import LinearProgram


@dataclass
class BoundResult:
    """
    Result of computing bounds via vertex enumeration.
    
    Attributes:
        optimal_value: The optimal value (upper or lower bound)
        optimal_vertex: The vertex achieving the optimal value
        all_vertices: All vertices of the feasible polytope
        all_values: Objective values at all vertices
        n_vertices: Number of vertices enumerated
        sense: 'max' or 'min'
    """
    optimal_value: float
    optimal_vertex: np.ndarray
    all_vertices: List[np.ndarray]
    all_values: List[float]
    n_vertices: int
    sense: str
    
    def __repr__(self) -> str:
        return (f"BoundResult(optimal_value={self.optimal_value:.6f}, "
                f"n_vertices={self.n_vertices}, sense='{self.sense}')")


class VertexEnumerator:
    """
    Vertex enumerator for linear programs with equality and inequality constraints.
    
    Handles LPs of the form:
        optimize:  c^T q
        subject to: A q = b  (equality constraints)
                   q >= 0    (non-negativity)
                   sum(q) = 1 (normalization)
    
    The method converts equality constraints to the form needed by pypoman,
    enumerates vertices, and computes optimal bounds.
    """
    
    @staticmethod
    def enumerate_vertices(lp: LinearProgram, 
                          param_values: Dict[str, float],
                          tol: float = 1e-10) -> List[np.ndarray]:
        """
        Enumerate all vertices of the feasible polytope for given parameter values.
        
        Args:
            lp: LinearProgram object with symbolic parameters
            param_values: Dictionary mapping parameter names to numerical values
            tol: Tolerance for numerical comparisons
        
        Returns:
            List of vertices (each vertex is a numpy array of length ℵᴿ)
        
        Notes:
            The feasible region is defined by:
            - A q = b (equality constraints from LP)
            - q >= 0 (non-negativity)
            - sum(q) = 1 (normalization)
            
            We convert equalities A q = b to inequalities:
            - A q <= b and -A q <= -b
            
            Then combine with:
            - -q <= 0 (i.e., q >= 0)
            - sum(q) <= 1 and -sum(q) <= -1 (i.e., sum(q) = 1)
        """
        # Evaluate RHS with concrete parameter values
        b = lp.evaluate_rhs(param_values)
        A_eq = lp.constraint_matrix
        
        # Convert to halfspace representation: A_ineq q <= b_ineq
        # 1. Equality constraints: A q = b becomes A q <= b and -A q <= -b
        A_from_eq_upper = A_eq
        b_from_eq_upper = b
        A_from_eq_lower = -A_eq
        b_from_eq_lower = -b
        
        # 2. Non-negativity: q >= 0 becomes -q <= 0
        n = lp.aleph_R
        A_nonneg = -np.eye(n)
        b_nonneg = np.zeros(n)
        
        # 3. Normalization: sum(q) = 1 becomes sum(q) <= 1 and -sum(q) <= -1
        A_norm_upper = np.ones((1, n))
        b_norm_upper = np.array([1.0])
        A_norm_lower = -np.ones((1, n))
        b_norm_lower = np.array([-1.0])
        
        # Stack all constraints
        A_ineq = np.vstack([
            A_from_eq_upper,
            A_from_eq_lower,
            A_nonneg,
            A_norm_upper,
            A_norm_lower
        ])
        
        b_ineq = np.hstack([
            b_from_eq_upper,
            b_from_eq_lower,
            b_nonneg,
            b_norm_upper,
            b_norm_lower
        ])
        
        # Enumerate vertices using pypoman
        vertices = pypoman.compute_polytope_vertices(A_ineq, b_ineq)
        
        # Filter vertices to remove numerical noise
        # (sometimes pypoman returns near-duplicate vertices)
        filtered_vertices = []
        for v in vertices:
            # Check if vertex is not too close to existing ones
            is_new = True
            for existing_v in filtered_vertices:
                if np.allclose(v, existing_v, atol=tol):
                    is_new = False
                    break
            if is_new:
                filtered_vertices.append(v)
        
        return filtered_vertices
    
    @staticmethod
    def compute_bounds(lp: LinearProgram,
                      param_values: Dict[str, float],
                      sense: str = 'both',
                      tol: float = 1e-10) -> Tuple[Optional[BoundResult], Optional[BoundResult]]:
        """
        Compute upper and/or lower bounds by enumerating vertices.
        
        Args:
            lp: LinearProgram object with objective and constraints
            param_values: Dictionary mapping parameter names to numerical values
            sense: 'max', 'min', or 'both' - which bounds to compute
            tol: Tolerance for numerical comparisons
        
        Returns:
            Tuple of (upper_bound_result, lower_bound_result)
            Either can be None if not requested
        
        Example:
            >>> lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,), sense='max')
            >>> param_values = {'p_X=0,Y=0|Z=0': 0.2, ...}
            >>> upper, lower = VertexEnumerator.compute_bounds(lp, param_values, sense='both')
            >>> print(f"P(Y=1 | do(X=1)) ∈ [{lower.optimal_value:.3f}, {upper.optimal_value:.3f}]")
        """
        # Enumerate vertices
        vertices = VertexEnumerator.enumerate_vertices(lp, param_values, tol)
        
        if len(vertices) == 0:
            raise ValueError("No vertices found - feasible region may be empty")
        
        # Evaluate objective at each vertex
        c = lp.objective
        values = [float(np.dot(c, v)) for v in vertices]
        
        # Compute bounds
        upper_result = None
        lower_result = None
        
        if sense in ['max', 'both']:
            max_idx = np.argmax(values)
            upper_result = BoundResult(
                optimal_value=values[max_idx],
                optimal_vertex=vertices[max_idx],
                all_vertices=vertices,
                all_values=values,
                n_vertices=len(vertices),
                sense='max'
            )
        
        if sense in ['min', 'both']:
            min_idx = np.argmin(values)
            lower_result = BoundResult(
                optimal_value=values[min_idx],
                optimal_vertex=vertices[min_idx],
                all_vertices=vertices,
                all_values=values,
                n_vertices=len(vertices),
                sense='min'
            )
        
        return upper_result, lower_result
    
    @staticmethod
    def print_bound_result(result: BoundResult, 
                          query_description: str,
                          lp: LinearProgram,
                          show_vertex: bool = True,
                          show_all_values: bool = False) -> None:
        """
        Print bound result in a readable format.
        
        Args:
            result: BoundResult from compute_bounds
            query_description: Human-readable description of the query
            lp: LinearProgram object for accessing response type labels
            show_vertex: Whether to show the optimal vertex
            show_all_values: Whether to show objective values at all vertices
        """
        bound_type = "Upper Bound" if result.sense == 'max' else "Lower Bound"
        
        print("=" * 80)
        print(f"{bound_type}: {query_description}")
        print("=" * 80)
        print(f"  Optimal value: {result.optimal_value:.6f}")
        print(f"  Number of vertices: {result.n_vertices}")
        
        if show_vertex:
            print(f"\n  Optimal vertex q*:")
            non_zero_indices = np.where(np.abs(result.optimal_vertex) > 1e-10)[0]
            if len(non_zero_indices) <= 20:
                for i in non_zero_indices:
                    print(f"    q[{i}] = {result.optimal_vertex[i]:.6f}  ({lp.response_type_labels[i]})")
            else:
                print(f"    {len(non_zero_indices)} non-zero entries (showing first 10):")
                for i in non_zero_indices[:10]:
                    print(f"    q[{i}] = {result.optimal_vertex[i]:.6f}  ({lp.response_type_labels[i]})")
                print(f"    ... ({len(non_zero_indices) - 10} more)")
        
        if show_all_values:
            print(f"\n  Objective values at all {result.n_vertices} vertices:")
            sorted_values = sorted(result.all_values)
            if len(sorted_values) <= 20:
                for i, val in enumerate(sorted_values):
                    marker = " ← optimal" if abs(val - result.optimal_value) < 1e-10 else ""
                    print(f"    Vertex {i+1}: {val:.6f}{marker}")
            else:
                print(f"    Min: {sorted_values[0]:.6f}")
                print(f"    Q1:  {sorted_values[len(sorted_values)//4]:.6f}")
                print(f"    Med: {sorted_values[len(sorted_values)//2]:.6f}")
                print(f"    Q3:  {sorted_values[3*len(sorted_values)//4]:.6f}")
                print(f"    Max: {sorted_values[-1]:.6f}")


def compute_causal_bounds(lp: LinearProgram,
                         param_values: Dict[str, float],
                         query_description: str = "Causal Query",
                         verbose: bool = True) -> Tuple[float, float]:
    """
    Convenience function to compute both bounds for a causal effect query.
    
    Args:
        lp: LinearProgram object with objective and constraints
        param_values: Dictionary mapping parameter names to numerical values
        query_description: Human-readable description of the query
        verbose: Whether to print detailed results
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    
    Example:
        >>> dag = DAG()
        >>> # ... construct dag ...
        >>> lp = ProgramFactory.build_lp(dag, {Y}, {X}, (1,), (1,))
        >>> param_values = {'p_X=0,Y=0|Z=0': 0.2, ...}
        >>> lb, ub = compute_causal_bounds(lp, param_values, "P(Y=1 | do(X=1))")
        >>> print(f"Bounds: [{lb:.3f}, {ub:.3f}]")
    """
    upper_result, lower_result = VertexEnumerator.compute_bounds(
        lp, param_values, sense='both'
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"CAUSAL EFFECT BOUNDS: {query_description}")
        print("=" * 80)
        print(f"  Lower bound: {lower_result.optimal_value:.6f}")
        print(f"  Upper bound: {upper_result.optimal_value:.6f}")
        print(f"  Width: {upper_result.optimal_value - lower_result.optimal_value:.6f}")
        print(f"  Vertices enumerated: {upper_result.n_vertices}")
        print("=" * 80)
    
    return lower_result.optimal_value, upper_result.optimal_value
