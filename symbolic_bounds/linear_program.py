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


class ParametricSolution:
    """
    Wrapper around PPOPT's Solution object that provides a cleaner interface
    for evaluating parametric linear programs.
    
    This class handles:
    - Automatic array shape conversion (1D -> 2D column vector)
    - Consistent interface regardless of whether experiments exist
    """
    
    def __init__(self, ppopt_solution, num_experiments: int):
        """
        Create a parametric solution wrapper.
        
        Args:
            ppopt_solution: The Solution object from PPOPT
            num_experiments: Number of experimental constraints (0 if none)
        """
        self.solution = ppopt_solution
        self.num_experiments = num_experiments
    
    def evaluate_objective(self, theta):
        """
        Evaluate the objective function at a specific parameter value.
        
        Args:
            theta: Parameter value(s). Can be:
                - Scalar: converted to np.array([[theta]])
                - 1D array: converted to column vector
                - 2D array: used directly
                
        Returns:
            float: The objective value at the given parameter value,
                   or None if the parameter is outside the feasible region.
        """
        # Convert theta to proper shape
        theta_arr = np.asarray(theta)
        
        # Handle different input shapes
        if theta_arr.ndim == 0:
            # Scalar input
            theta_arr = theta_arr.reshape(1, 1)
        elif theta_arr.ndim == 1:
            # 1D array - reshape to column vector
            theta_arr = theta_arr.reshape(-1, 1)
        elif theta_arr.ndim == 2:
            # Already 2D - use as is
            pass
        else:
            raise ValueError(f"theta must be scalar, 1D, or 2D array, got shape {theta_arr.shape}")
        
        # Ensure correct number of parameters
        expected_params = self.num_experiments if self.num_experiments > 0 else 1
        if theta_arr.shape[0] != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameter(s), got {theta_arr.shape[0]}. "
                f"{'One parameter per experimental constraint.' if self.num_experiments > 0 else 'Use any scalar for dummy parameter.'}"
            )
        
        # Evaluate using PPOPT's solution
        return self.solution.evaluate_objective(theta_arr)
    
    def get_region(self, theta):
        """Get the critical region containing the given parameter value."""
        theta_arr = np.asarray(theta)
        if theta_arr.ndim <= 1:
            theta_arr = theta_arr.reshape(-1, 1)
        return self.solution.get_region(theta_arr)
    
    def evaluate(self, theta):
        """Evaluate the decision variables at a specific parameter value."""
        theta_arr = np.asarray(theta)
        if theta_arr.ndim <= 1:
            theta_arr = theta_arr.reshape(-1, 1)
        return self.solution.evaluate(theta_arr)
    
    @property
    def critical_regions(self):
        """Access the underlying critical regions."""
        return self.solution.critical_regions
    
    def __len__(self):
        """Number of critical regions."""
        return len(self.solution.critical_regions)


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
        q_labels: Enumeration of response type configurations (decision variables)
        variable_labels: Labels for decision variables (response type names)
        constraint_labels: Labels for constraints (observed configuration names)
        is_minimization: True for minimization, False for maximization
    """
    
    def __init__(
        self,
        objective: np.ndarray,
        constraint_matrix: np.ndarray,
        rhs: np.ndarray,
        q_labels: list[tuple],
        variable_labels: list[str],
        constraint_labels: list[str],
        experiment_matrix: Optional[np.ndarray] = None,
        experiment_labels: Optional[list[str]] = None,
        is_minimization: bool = True
    ):
        """
        Create a linear program.
        
        Args:
            objective: Coefficient vector α (length ℵᴿ)
            constraint_matrix: Matrix P (B × ℵᴿ)
            rhs: Right-hand-side vector p (length B)
            q_labels: Enumeration of response type configurations for W_R nodes
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
        
        if len(q_labels) != n_vars:
            raise ValueError(
                f"Number of q labels ({len(q_labels)}) doesn't match "
                f"number of variables ({n_vars})"
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
        self.q_labels = q_labels
        self.variable_labels = variable_labels
        self.constraint_labels = constraint_labels
        self.is_minimization = is_minimization
        self.experiment_matrix = experiment_matrix
        self.experiment_labels = experiment_labels
    
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
        
        # Experiment matrix information
        if self.experiment_matrix is not None:
            print(f"\nExperiment matrix E:")
            print(f"  Shape: {self.experiment_matrix.shape} (experiments × configurations)")
            print(f"  Number of experiments: {self.experiment_matrix.shape[0]}")
            
            if self.experiment_labels:
                print(f"  Experiment labels available: {len(self.experiment_labels)}")
            
            if show_full_matrices:
                non_zero_count = np.sum(np.abs(self.experiment_matrix) > 1e-10)
                total_entries = self.experiment_matrix.shape[0] * self.experiment_matrix.shape[1]
                print(f"  Non-zero entries: {non_zero_count} / {total_entries}")
                
                if self.experiment_labels:
                    print(f"\n  Experiments:")
                    n_show = min(5, len(self.experiment_labels))
                    for i in range(n_show):
                        non_zero = np.sum(np.abs(self.experiment_matrix[i]) > 1e-10)
                        print(f"    Exp {i} ({self.experiment_labels[i]}): {non_zero} non-zero entries")
                    if len(self.experiment_labels) > n_show:
                        print(f"    ... and {len(self.experiment_labels) - n_show} more experiments")
        
        # Probability constraints
        print(f"\n              q ≥ 0")
        print(f"              1^T q = 1")
        print(f"\nDecision variable q:")
        print(f"  Dimension: {len(self.variable_labels)} (= ℵᴿ)")
        print(f"  Interpretation: Probabilities of response type combinations from W_R")
        
        if show_full_matrices:
            print(f"\n  q enumeration (first 10 entries):")
            n_show = min(10, len(self.q_labels))
            for i in range(n_show):
                rt_config = self.q_labels[i]
                print(f"    q[{i}]: {self.variable_labels[i]}")
            if len(self.q_labels) > n_show:
                print(f"    ... and {len(self.q_labels) - n_show} more entries")
        
        print("\n" + "=" * 80)
    
    def print_decision_variables(self) -> None:
        """
        Print all decision variables q with their response type mappings.
        """
        print("\n" + "=" * 80)
        print("DECISION VARIABLES q")
        print("=" * 80)
        print(f"\nTotal variables: {len(self.q_labels)} (= ℵᴿ)")
        print("Each q[i] represents P(response type configuration i)")
        print("\nComplete enumeration:")
        print("-" * 80)
        
        for i, (rt_config, label) in enumerate(zip(self.q_labels, self.variable_labels)):
            # Print index and label
            print(f"\nq[{i}]: {label}")
            
            # Print detailed response type mapping
            if rt_config:
                print("  Response type configuration:")
                for rt in rt_config:
                    node_name = rt.node.name
                    # Format the response type mapping
                    if not rt.mapping:
                        print(f"    {node_name}: (empty mapping)")
                    else:
                        # Check if node has no parents
                        first_key = next(iter(rt.mapping.keys()))
                        if len(first_key) == 0:
                            # No parents - constant value
                            value = rt.mapping[first_key]
                            print(f"    {node_name} = {value}")
                        else:
                            # Has parents - show full mapping
                            print(f"    {node_name}:")
                            for parent_config, output in sorted(rt.mapping.items(),
                                                               key=lambda x: tuple(v for _, v in x[0])):
                                config_str = ", ".join(f"{n.name}={v}" for n, v in parent_config)
                                print(f"      [{config_str}] → {output}")
        
        print("\n" + "=" * 80)
    
    def print_objective(self) -> None:
        """
        Print the objective function α^T q with all coefficients.
        """
        print("\n" + "=" * 80)
        print("OBJECTIVE FUNCTION")
        print("=" * 80)
        
        obj_type = "minimize" if self.is_minimization else "maximize"
        print(f"\n{obj_type}  α^T q")
        print(f"\nObjective vector α (dimension: {len(self.objective)})")
        
        non_zero_indices = np.where(np.abs(self.objective) > 1e-10)[0]
        zero_count = len(self.objective) - len(non_zero_indices)
        
        print(f"Non-zero entries: {len(non_zero_indices)}")
        print(f"Zero entries: {zero_count}")
        
        if len(non_zero_indices) > 0:
            print("\nNon-zero coefficients:")
            print("-" * 80)
            for idx in non_zero_indices:
                print(f"  α[{idx}] = {self.objective[idx]:10.6f}    ({self.variable_labels[idx]})")
        
        if zero_count > 0 and zero_count <= 20:
            print("\nZero coefficients:")
            print("-" * 80)
            zero_indices = np.where(np.abs(self.objective) <= 1e-10)[0]
            for idx in zero_indices:
                print(f"  α[{idx}] = {self.objective[idx]:10.6f}    ({self.variable_labels[idx]})")
        elif zero_count > 20:
            print(f"\n(Omitting {zero_count} zero coefficients)")
        
        print("\n" + "=" * 80)
    
    def print_rhs(self) -> None:
        """
        Print the right-hand-side vector p (observed probabilities).
        """
        print("\n" + "=" * 80)
        print("RIGHT-HAND-SIDE VECTOR p")
        print("=" * 80)
        
        print(f"\nObserved probability vector (dimension: {len(self.rhs)})")
        print(f"Sum: {np.sum(self.rhs):.10f} (should equal 1.0)")
        print("\nAll entries:")
        print("-" * 80)
        
        for i, (prob, label) in enumerate(zip(self.rhs, self.constraint_labels)):
            print(f"  p[{i}] = {prob:12.10f}    ({label})")
        
        print("\n" + "=" * 80)
    
    def print_constraints(self, max_terms_per_row: Optional[int] = None, include_matrix: bool = False) -> None:
        """
        Print all constraints P q = p in equation form.
        
        Args:
            max_terms_per_row: Maximum terms to show per equation. None shows all.
        """
        print("\n" + "=" * 80)
        print("CONSTRAINT EQUATIONS: P q = p")
        print("=" * 80)
        
        print(f"\nConstraint matrix P: {self.constraint_matrix.shape[0]} × {self.constraint_matrix.shape[1]}")
        print(f"Total constraints: {len(self.constraint_labels)}")
        print("\nAll constraint equations:")
        print("-" * 80)
        
        for i in range(len(self.constraint_labels)):
            row = self.constraint_matrix[i]
            rhs_val = self.rhs[i]
            label = self.constraint_labels[i]
            
            # Find non-zero coefficients
            nonzero_indices = np.where(np.abs(row) > 1e-10)[0]
            
            print(f"\nConstraint {i}: {label}")
            print(f"  p[{i}] = {rhs_val:.10f}")
            
            # Format left-hand side
            terms = []
            indices_to_show = nonzero_indices if max_terms_per_row is None else nonzero_indices[:max_terms_per_row]
            
            for idx in indices_to_show:
                coef = row[idx]
                if abs(coef - 1.0) < 1e-10:
                    terms.append(f"q[{idx}]")
                else:
                    terms.append(f"{coef:.6f}*q[{idx}]")
            
            if max_terms_per_row is not None and len(nonzero_indices) > max_terms_per_row:
                terms.append(f"... +{len(nonzero_indices) - max_terms_per_row} more terms")
            
            lhs = " + ".join(terms) if terms else "0"
            print(f"  LHS = {lhs}")
            print(f"  Non-zero terms: {len(nonzero_indices)}")
        if include_matrix:
            print(f" \nFull constraint matrix P:")
            print(self.constraint_matrix)
        
        print("\n" + "=" * 80)
    
    def print_constraint_matrix(self) -> None:
        """
        Print the full constraint matrix P with row and column labels.
        """
        print("\n" + "=" * 80)
        print("CONSTRAINT MATRIX P")
        print("=" * 80)
        
        print(f"\nShape: {self.constraint_matrix.shape[0]} rows × {self.constraint_matrix.shape[1]} columns")
        print(f"Rows: observed configurations")
        print(f"Columns: response type combinations")
        
        non_zero_count = np.sum(np.abs(self.constraint_matrix) > 1e-10)
        total_entries = self.constraint_matrix.shape[0] * self.constraint_matrix.shape[1]
        sparsity = 100 * (1 - non_zero_count / total_entries)
        
        print(f"\nNon-zero entries: {non_zero_count} / {total_entries} ({100*non_zero_count/total_entries:.2f}%)")
        print(f"Sparsity: {sparsity:.2f}%")
        
        print("\nFull matrix:")
        print("-" * 80)
        print(self.constraint_matrix)
        
        print("\n" + "=" * 80)
    
    def print_experiment_matrix(self, include_matrix: bool = False) -> None:
        """
        Print the experiment matrix E if available.
        """
        if self.experiment_matrix is not None:
            print(f"\nExperiment matrix E:")
            print(f"  Shape: {self.experiment_matrix.shape} (experiments × configurations)")
            print(f"  Number of experiments: {self.experiment_matrix.shape[0]}")
            
            if self.experiment_labels:
                print(f"  Experiment labels available: {len(self.experiment_labels)}")
            
            non_zero_count = np.sum(np.abs(self.experiment_matrix) > 1e-10)
            total_entries = self.experiment_matrix.shape[0] * self.experiment_matrix.shape[1]
            print(f"  Non-zero entries: {non_zero_count} / {total_entries}")
            
            # Print details of each experiment
            print(f"\n  Experiments:")
            n_show = min(5, len(self.experiment_labels))
            for i in range(n_show):
                non_zero = np.sum(np.abs(self.experiment_matrix[i]) > 1e-10)
                print(f"    Exp {i} ({self.experiment_labels[i]}): {non_zero} non-zero entries")
            if len(self.experiment_labels) > n_show:
                print(f"    ... and {len(self.experiment_labels) - n_show} more experiments")
            if include_matrix:
                print(f"\nFull experiment matrix:")
                print(self.experiment_matrix)
    
    def print_full_lp(self, include_matrix: bool = False) -> None:
        """
        Print complete LP with all components.
        
        Args:
            include_matrix: If True, also print the full constraint matrix
        """
        print("\n" + "=" * 100)
        print(" " * 35 + "COMPLETE LINEAR PROGRAM")
        print("=" * 100)
        
        # Summary
        print(f"\nProblem type: {'Minimization' if self.is_minimization else 'Maximization'}")
        print(f"Variables: {len(self.objective)}")
        print(f"Constraints: {len(self.rhs)}")
        
        # All components
        self.print_decision_variables()
        self.print_objective()
        self.print_rhs()
        self.print_constraints(include_matrix=include_matrix)
        self.print_experiment_matrix(include_matrix=include_matrix)

        
        print("\n" + "=" * 100)
        print(" " * 38 + "END OF LINEAR PROGRAM")
        print("=" * 100)
    
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
            "constraint_matrix_nonzero": int(np.sum(np.abs(self.constraint_matrix) > 1e-10)),
            "q_dimension": len(self.q_labels)
        }
    
    def solve(self, solver_type: str = 'glpk', verbose: bool = False):
        """
        Solve the linear program using PPOPT's MPLP_Program with automatic constraint processing.
        
        The LP is converted from our format:
            minimize/maximize    α^T q
            subject to           P q = p  (equality constraints)
                                Exp q = theta (experiment constraints)
                                q ≥ 0    (non-negativity)
        
        To PPOPT's format:
            minimize    c^T x
            subject to  A x ≤ b + F θ
        
        When experimental constraints exist (experiment_matrix is not None), the solution
        is a parametric solution where θ represents the experimental results. To evaluate
        the objective at a specific experimental result, use:
            solution = lp.solve()
            result = solution.evaluate_objective(np.array([[theta_value]]))
        
        When no experimental constraints exist, a dummy parameter is created for PPOPT
        compatibility, and you can evaluate using any dummy value.
        
        PPOPT's process_constraints() automatically removes strongly and weakly redundant
        constraints and rescales them, leading to significant performance increases and
        improved numerical stability.
        
        Args:
            solver_type: Solver to use ('glpk' or 'gurobi'). Default 'glpk'.
            verbose: If True, print detailed solver output.
        
        Returns:
            ParametricSolution: A wrapper around PPOPT's Solution object that handles
                evaluation at specific parameter values. Use .evaluate_objective(theta)
                to get the objective value at a specific parameter value.
        
        Raises:
            ImportError: If PPOPT is not installed
            ValueError: If solver_type is not supported
        """
        try:
            import sys
            import os
            # Add PPOPT to path
            ppopt_path = os.path.join(os.path.dirname(__file__), 'ppopt_repo', 'PPOPT', 'src')
            if ppopt_path not in sys.path:
                sys.path.insert(0, ppopt_path)
            
            from ppopt.mplp_program import MPLP_Program
            from ppopt.mpmodel import MPModeler
            from ppopt.solver import Solver
            from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
        except ImportError as e:
            raise ImportError(
                "PPOPT is required to solve LPs. Install it with:\n"
                "  pip install ppopt\n"
                f"Error: {e}"
            )
        
        
        # Validate solver
        if solver_type not in ['glpk', 'gurobi']:
            raise ValueError(f"Solver '{solver_type}' not supported. Use 'glpk' or 'gurobi'.")
        
        n_vars = len(self.objective)
        n_constraints = len(self.rhs)
        
        model = MPModeler()


        #Step 1: Define variables
        q = []
        for i in range(n_vars):
            q.append(model.add_var(name=f"q_{i}"))

        #Step 2: If no experiments, define dummy parameter theta
        # This is required by the MPLP_Program interface
        if self.experiment_matrix is None:
            theta = model.add_param(name="theta")
        else:
            #Define experiment constraints
            theta = []
            for exp_idx in range(self.experiment_matrix.shape[0]):
                theta.append(model.add_param(name=f"theta_{exp_idx}"))
                lhs = sum(self.experiment_matrix[exp_idx, col] * q[col] for col in range(n_vars))
                model.add_constr(lhs == theta[exp_idx])  #Assuming each experiment sums to 1

        
        #Step 3: Define obs constraints
        for row in range(self.constraint_matrix.shape[0]):
            lhs = sum(self.constraint_matrix[row, col] * q[col] for col in range(n_vars))
            rhs = self.rhs[row]
            model.add_constr(lhs == rhs)

        #Step 4: Define non-negativity constraints
        for i in range(n_vars):
            model.add_constr(q[i] >= 0)
        
        #Step 5: Define objective
        if self.is_minimization:
            model.set_objective(sum(self.objective[i] * q[i] for i in range(n_vars)))
        else:
            model.set_objective(sum(-self.objective[i] * q[i] for i in range(n_vars)))
        
    
        if verbose:
            print(f"Building MPLP_Program:")
            print(f"  Variables: {n_vars}")
        prog = model.formulate_problem()

        # Process constraints: remove redundant constraints and rescale
        if verbose:
            print("\nProcessing constraints (removing redundancies)...")
        
        prog.process_constraints()

        solution = solve_mpqp(prog, mpqp_algorithm.combinatorial)

        # Wrap the solution for easier usage
        num_experiments = 0 if self.experiment_matrix is None else self.experiment_matrix.shape[0]
        return ParametricSolution(solution, num_experiments)
