"""
LinearProgram class for representing linear programming problems.

This module defines the LP structure for computing bounds on causal effects:
    minimize/maximize    α^T q
    subject to           P q = p
                         q ≥ 0
                         1^T q = 1
"""

import numpy as np
from typing import Optional


class LinearProgram:
    """
    Represents a linear program for computing causal effect bounds.
    
    The LP has the form:
        minimize/maximize    α^T q
        subject to           P q = p
                             q ≥ 0
                             1^T q = 1
    
    Where:
        - q is the decision variable (probabilities of response types)
        - α is the objective function coefficient vector
        - P is the constraint matrix (from Algorithm 1)
        - p is the right-hand-side vector (from observed distribution)
    
    Attributes:
        objective: Coefficient vector α for the objective function
        constraint_matrix: Matrix P mapping response types to observations
        rhs: Right-hand-side vector p (observed probabilities)
        variable_labels: Labels for decision variables (response type names)
        constraint_labels: Labels for constraints (observed configuration names)
        is_minimization: True for minimization, False for maximization
    """
    
    def __init__(
        self,
        objective: np.ndarray,
        constraint_matrix: np.ndarray,
        rhs: np.ndarray,
        variable_labels: list[str],
        constraint_labels: list[str],
        is_minimization: bool = True
    ):
        """
        Create a linear program.
        
        Args:
            objective: Coefficient vector α (length ℵᴿ)
            constraint_matrix: Matrix P (B × ℵᴿ)
            rhs: Right-hand-side vector p (length B)
            variable_labels: Names of decision variables q_γ
            constraint_labels: Names of constraints (configurations)
            is_minimization: True for min, False for max
            
        Raises:
            ValueError: If dimensions don't match
        """
        # Validate dimensions
        n_vars = len(objective)
        n_constraints = len(rhs)
        
        if constraint_matrix.shape != (n_constraints, n_vars):
            raise ValueError(
                f"Constraint matrix shape {constraint_matrix.shape} doesn't match "
                f"({n_constraints}, {n_vars})"
            )
        
        if len(variable_labels) != n_vars:
            raise ValueError(
                f"Number of variable labels ({len(variable_labels)}) doesn't match "
                f"number of variables ({n_vars})"
            )
        
        if len(constraint_labels) != n_constraints:
            raise ValueError(
                f"Number of constraint labels ({len(constraint_labels)}) doesn't match "
                f"number of constraints ({n_constraints})"
            )
        
        self.objective = objective
        self.constraint_matrix = constraint_matrix
        self.rhs = rhs
        self.variable_labels = variable_labels
        self.constraint_labels = constraint_labels
        self.is_minimization = is_minimization
    
    def print_lp(self, show_full_matrices: bool = False) -> None:
        """
        Print the linear program in a readable format.
        
        Args:
            show_full_matrices: If True, show full matrices; otherwise show dimensions only
        """
        print("\n" + "=" * 80)
        print("LINEAR PROGRAM")
        print("=" * 80)
        
        # Objective
        obj_type = "minimize" if self.is_minimization else "maximize"
        print(f"\n{obj_type}    α^T q")
        print(f"\nObjective vector α:")
        print(f"  Dimension: {len(self.objective)}")
        
        if show_full_matrices:
            non_zero_count = np.sum(np.abs(self.objective) > 1e-10)
            print(f"  Non-zero entries: {non_zero_count} / {len(self.objective)}")
            print(f"\n  α values:")
            for i, (val, label) in enumerate(zip(self.objective, self.variable_labels)):
                if abs(val) > 1e-10:
                    print(f"    α[{i}] = {val:8.4f}  ({label})")
        
        # Constraints
        print(f"\nsubject to    P q = p")
        print(f"\nConstraint matrix P:")
        print(f"  Shape: {self.constraint_matrix.shape} (rows × columns)")
        print(f"  Rows (configurations): {self.constraint_matrix.shape[0]}")
        print(f"  Columns (response types): {self.constraint_matrix.shape[1]}")
        
        if show_full_matrices:
            print(f"\n  P matrix:")
            # Show first few rows
            n_show = min(5, len(self.constraint_labels))
            for i in range(n_show):
                row = self.constraint_matrix[i]
                non_zero = np.sum(np.abs(row) > 1e-10)
                print(f"    Row {i} ({self.constraint_labels[i]}): {non_zero} non-zero entries")
            if len(self.constraint_labels) > n_show:
                print(f"    ... and {len(self.constraint_labels) - n_show} more rows")
        
        print(f"\nRight-hand-side vector p (observed probabilities):")
        print(f"  Dimension: {len(self.rhs)}")
        print(f"  Sum: {np.sum(self.rhs):.6f} (should be 1.0)")
        
        if show_full_matrices:
            print(f"\n  p values:")
            for i, (val, label) in enumerate(zip(self.rhs, self.constraint_labels)):
                print(f"    p[{i}] = {val:8.6f}  ({label})")
        
        # Probability constraints
        print(f"\n              q ≥ 0")
        print(f"              1^T q = 1")
        print(f"\nDecision variable q:")
        print(f"  Dimension: {len(self.variable_labels)} (= ℵᴿ)")
        print(f"  Interpretation: Probabilities of response type combinations")
        
        print("\n" + "=" * 80)
    
    def get_summary(self) -> dict:
        """
        Get a summary of the LP dimensions and properties.
        
        Returns:
            Dictionary with LP statistics
        """
        return {
            "n_variables": len(self.objective),
            "n_constraints": len(self.rhs),
            "constraint_matrix_shape": self.constraint_matrix.shape,
            "is_minimization": self.is_minimization,
            "rhs_sum": float(np.sum(self.rhs)),
            "objective_nonzero": int(np.sum(np.abs(self.objective) > 1e-10)),
            "constraint_matrix_nonzero": int(np.sum(np.abs(self.constraint_matrix) > 1e-10))
        }
