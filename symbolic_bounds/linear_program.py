"""
LinearProgram class for representing symbolic linear programs for causal effect bounds.

This module provides structures for representing LPs with symbolic parameters,
preparing for vertex enumeration and parametric optimization.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from .node import Node
from .constraints import Constraints


@dataclass
class SymbolicParameter:
    """
    Represents a symbolic parameter in the linear program.
    
    For causal effect identification, these are typically conditional probabilities
    P(W_R | W_L) for specific configurations of W_L and W_R.
    
    Attributes:
        name: Human-readable name (e.g., "p_W1=0,W2=1|Z=0")
        w_r_config: Configuration of W_R nodes as tuple of (Node, value) pairs
        w_l_config: Configuration of W_L nodes as tuple of (Node, value) pairs
        index: Index in the parameter vector
    """
    name: str
    w_r_config: Tuple[Tuple[Node, int], ...]
    w_l_config: Tuple[Tuple[Node, int], ...]
    index: int
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"SymbolicParameter('{self.name}')"


class LinearProgram:
    """
    Represents a linear program with symbolic parameters for causal effect bounds.
    
    The LP has the form:
        minimize/maximize: c^T q
        subject to:       A q = b
                         q >= 0
                         sum(q) = 1
    
    where:
    - q: decision variable (response type probabilities), dimension ℵᴿ
    - c: objective function vector (from writeRung2), dimension ℵᴿ
    - A: constraint matrix (from P matrix), dimension (# constraints) × ℵᴿ
    - b: RHS vector with symbolic parameters (conditional probabilities P(W_R|W_L))
    
    The RHS vector b contains symbolic parameters that represent conditional
    probabilities P(W_R | W_L) for different configurations. This enables:
    1. Symbolic optimization (vertex enumeration with parametric bounds)
    2. Sensitivity analysis with respect to observational distributions
    3. Computing bounds as functions of observable probabilities
    
    Attributes:
        objective: Objective function vector c (length ℵᴿ)
        constraint_matrix: Constraint matrix A
        rhs_symbolic: Right-hand side vector with symbolic expressions
        rhs_params: List of symbolic parameters appearing in RHS
        param_dict: Dictionary mapping parameter names to SymbolicParameter objects
        response_type_labels: Human-readable labels for decision variables q
        constraint_labels: Human-readable labels for constraints
        sense: 'min' or 'max' for optimization direction
    """
    
    def __init__(self):
        """Initialize empty linear program."""
        self.objective: Optional[np.ndarray] = None  # c vector (length ℵᴿ)
        self.constraint_matrix: Optional[np.ndarray] = None  # A matrix
        self.rhs_symbolic: Optional[List[str]] = None  # Symbolic RHS expressions
        self.rhs_params: List[SymbolicParameter] = []  # Unique parameters in RHS
        self.param_dict: Dict[str, SymbolicParameter] = {}  # Name -> parameter
        
        # Labels for interpretability
        self.response_type_labels: List[str] = []
        self.constraint_labels: List[str] = []
        
        # Optimization direction
        self.sense: str = 'max'  # 'min' or 'max'
        
        # Dimension info
        self.aleph_R: int = 0  # Number of response types
        self.n_constraints: int = 0
        self.n_params: int = 0
    
    def set_objective(self, objective: np.ndarray, sense: str = 'max') -> None:
        """
        Set the objective function c^T q.
        
        Args:
            objective: Objective vector c (from writeRung2)
            sense: 'min' or 'max' for optimization direction
        """
        self.objective = objective
        self.sense = sense
        self.aleph_R = len(objective)
    
    def add_constraints_from_matrix(self, 
                                   constraint_matrix: np.ndarray,
                                   rhs_symbolic: List[str],
                                   constraint_labels: List[str]) -> None:
        """
        Add equality constraints A q = b with symbolic RHS.
        
        Args:
            constraint_matrix: Matrix A (rows = constraints, cols = decision vars)
            rhs_symbolic: List of symbolic expressions for RHS (one per constraint)
            constraint_labels: Human-readable labels for each constraint
        """
        if self.constraint_matrix is None:
            self.constraint_matrix = constraint_matrix
            self.rhs_symbolic = rhs_symbolic
            self.constraint_labels = constraint_labels
        else:
            # Stack additional constraints
            self.constraint_matrix = np.vstack([self.constraint_matrix, constraint_matrix])
            self.rhs_symbolic.extend(rhs_symbolic)
            self.constraint_labels.extend(constraint_labels)
        
        self.n_constraints = len(self.constraint_labels)
    
    def register_parameter(self, param: SymbolicParameter) -> None:
        """Register a symbolic parameter that appears in the RHS."""
        if param.name not in self.param_dict:
            self.param_dict[param.name] = param
            self.rhs_params.append(param)
            self.n_params = len(self.rhs_params)
    
    def evaluate_rhs(self, param_values: Dict[str, float]) -> np.ndarray:
        """
        Evaluate the RHS vector b by substituting concrete values for parameters.
        
        This is useful for:
        1. Testing the LP with specific probability distributions
        2. Computing bounds for a given observational distribution
        3. Validating the symbolic structure
        
        Args:
            param_values: Dictionary mapping parameter names to numerical values
        
        Returns:
            Numerical RHS vector b
        """
        # For now, assume rhs_symbolic contains parameter names directly
        # In future, could support arithmetic expressions
        rhs_numeric = np.zeros(self.n_constraints)
        
        for i, symbolic_expr in enumerate(self.rhs_symbolic):
            if symbolic_expr in param_values:
                rhs_numeric[i] = param_values[symbolic_expr]
            else:
                raise ValueError(f"Parameter '{symbolic_expr}' not provided in param_values")
        
        return rhs_numeric
    
    def print_lp(self, show_full_matrix: bool = False) -> None:
        """
        Print the linear program in a readable format.
        
        Args:
            show_full_matrix: If True, print full constraint matrix
        """
        print("=" * 80)
        print("LINEAR PROGRAM FOR CAUSAL EFFECT BOUNDS")
        print("=" * 80)
        
        # Dimensions
        print(f"\n{'Dimensions':^80}")
        print("-" * 80)
        print(f"  Decision variables (q): {self.aleph_R} (response type probabilities)")
        print(f"  Constraints           : {self.n_constraints}")
        print(f"  Symbolic parameters   : {self.n_params}")
        
        # Objective function
        print(f"\n{'Objective Function':^80}")
        print("-" * 80)
        print(f"  {self.sense.upper()}  c^T q")
        print(f"\n  Objective vector c:")
        if self.objective is not None:
            non_zero_indices = np.where(self.objective != 0)[0]
            if len(non_zero_indices) <= 20:
                for i in non_zero_indices:
                    print(f"    c[{i}] = {self.objective[i]:.6f}  ({self.response_type_labels[i]})")
            else:
                print(f"    {len(non_zero_indices)} non-zero entries")
                for i in non_zero_indices[:10]:
                    print(f"    c[{i}] = {self.objective[i]:.6f}  ({self.response_type_labels[i]})")
                print(f"    ... ({len(non_zero_indices) - 10} more)")
        
        # Constraints
        print(f"\n{'Constraints: A q = b (symbolic)':^80}")
        print("-" * 80)
        if show_full_matrix and self.constraint_matrix is not None:
            print("\nConstraint matrix A:")
            print(self.constraint_matrix)
            print("\nRHS vector b (symbolic):")
            for i, (label, rhs) in enumerate(zip(self.constraint_labels, self.rhs_symbolic)):
                print(f"  b[{i}] = {rhs}  (for {label})")
        else:
            # Show sample constraints
            n_show = min(5, self.n_constraints)
            print(f"\nShowing first {n_show} constraints (out of {self.n_constraints}):\n")
            for i in range(n_show):
                # Format constraint equation
                lhs_terms = []
                for j in range(self.aleph_R):
                    coef = self.constraint_matrix[i, j]
                    if coef != 0:
                        if len(lhs_terms) < 3:  # Show first 3 terms
                            lhs_terms.append(f"{coef:.3f}*q[{j}]")
                
                if len(lhs_terms) < self.aleph_R:
                    lhs_str = " + ".join(lhs_terms) + " + ..."
                else:
                    lhs_str = " + ".join(lhs_terms)
                
                print(f"  {self.constraint_labels[i]}:")
                print(f"    {lhs_str} = {self.rhs_symbolic[i]}")
            
            if self.n_constraints > n_show:
                print(f"\n  ... ({self.n_constraints - n_show} more constraints)")
        
        # Symbolic parameters
        print(f"\n{'Symbolic Parameters (Conditional Probabilities)':^80}")
        print("-" * 80)
        if self.n_params > 0:
            print(f"\nTotal: {self.n_params} parameters representing P(W_R | W_L)\n")
            for param in self.rhs_params:
                # Format W_R configuration
                w_r_str = ", ".join([f"{node.name}={val}" for node, val in param.w_r_config])
                # Format W_L configuration
                w_l_str = ", ".join([f"{node.name}={val}" for node, val in param.w_l_config])
                print(f"  {param.name}:")
                print(f"    P({w_r_str} | {w_l_str})")
        else:
            print("  No symbolic parameters (unconditional case)")
        
        # Additional constraints (implicit)
        print(f"\n{'Additional Constraints':^80}")
        print("-" * 80)
        print("  q >= 0           (non-negativity: all response type probabilities >= 0)")
        print("  sum(q) = 1       (normalization: response type probabilities sum to 1)")
        
        print("\n" + "=" * 80)
        print(f"LP ready for vertex enumeration and parametric optimization")
        print("=" * 80)
    
    def get_feasible_region_info(self) -> Dict:
        """
        Get information about the feasible region for vertex enumeration.
        
        Returns:
            Dictionary with:
            - n_variables: Number of decision variables
            - n_constraints: Number of equality constraints
            - n_parameters: Number of symbolic parameters
            - dimension: Expected dimension of feasible polytope
        """
        # Dimension of feasible region = n_variables - rank(A)
        # For response types: typically full rank, so dimension = aleph_R - n_constraints
        
        if self.constraint_matrix is not None:
            rank = np.linalg.matrix_rank(self.constraint_matrix)
            expected_dim = self.aleph_R - rank
        else:
            rank = 0
            expected_dim = self.aleph_R
        
        return {
            'n_variables': self.aleph_R,
            'n_constraints': self.n_constraints,
            'n_parameters': self.n_params,
            'constraint_matrix_rank': rank,
            'expected_polytope_dimension': expected_dim,
            'notes': 'For vertex enumeration, feasible region is intersection of simplex with equality constraints'
        }
