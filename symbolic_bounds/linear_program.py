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
                                 q ≥ 0    (non-negativity)
        
        To PPOPT's format:
            minimize    c^T x
            subject to  A x ≤ b
        
        PPOPT's process_constraints() automatically removes strongly and weakly redundant
        constraints and rescales them, leading to significant performance increases and
        improved numerical stability.
        
        Args:
            solver_type: Solver to use ('glpk' or 'gurobi'). Default 'glpk'.
            verbose: If True, print detailed solver output.
        
        Returns:
            dict: Solution dictionary with keys:
                - 'optimal_value': The optimal objective value
                - 'solution': The optimal q vector (decision variables)
                - 'status': 'optimal', 'infeasible', or 'error'
                - 'solver_output': Raw solution object from PPOPT (if successful)
        
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
        
        

        A_rows = []
        b_rows = []
        
        # Equality constraints: P q = p
        for rows in range(self.constraint_matrix.shape[0]):
            A_rows.append(self.constraint_matrix[rows:rows+1, :])
            b_rows.append(np.array(self.rhs[rows]))
        

        # Non-negativity: q ≥ 0 becomes -q ≤ 0
        A_rows.append(-np.ones(n_vars))
        b_rows.append(np.zeros(1))
        
        # Combine all constraints
        A = np.vstack(A_rows)
        b = np.vstack(b_rows)
        
        eq_indices = list(range(self.constraint_matrix.shape[0]))
        # Objective: handle minimization vs maximization
        # PPOPT minimizes by default, so for maximization, negate the objective
        c = self.objective if self.is_minimization else -self.objective
        c = c.reshape(-1, 1)
        
        # MPLP_Program requires H, F, A_t, b_t for parametric programming
        # Here we have no parameters, so set them to empty
        # H = np.zeros((2, 1))
        # F = np.zeros((1, 2))
        # A_t = np.zeros((1, 2))
        # b_t = np.zeros((2, 1))
        H = np.zeros((0, 0))
        F = np.zeros((0, 0))
        A_t = np.zeros((0, 0))
        b_t = np.zeros((0, 0))

        
        if verbose:
            print(f"Building MPLP_Program:")
            print(f"  Variables: {n_vars}")
            print(f"  Constraints: {A.shape[0]} (before redundancy removal)")
            print(f"  Original P matrix: {self.constraint_matrix.shape}")
        
        # Create MPLP_Program with Solver object
        solver_obj = Solver(solvers={'lp': solver_type})
        prog = MPLP_Program(A, b, c, H, A_t, b_t, F, solver=solver_obj, equality_indices=eq_indices)
        
        # Process constraints: remove redundant constraints and rescale
        if verbose:
            print("\nProcessing constraints (removing redundancies)...")
        
        prog.process_constraints()
        
        if verbose:
            print(f"  Constraints after processing: {prog.A.shape[0]}")
        
        # For a standard LP (no parameters), we solve at theta = empty vector
        # Use the prog.solver directly to solve the processed constraints
        if verbose:
            print("\nSolving LP with processed constraints...")
        
        try:
            # Solve the LP: min c^T x s.t. A x ≤ b, using the processed constraints
            result = prog.solver.solve_lp(
                prog.c,
                prog.A,
                prog.b,
                equality_constraints=prog.equality_indices,
                verbose=verbose,
                get_duals=False
            )
        except Exception as e:
            if verbose:
                print(f"Solver failed: {e}")
            return {
                'status': 'error',
                'optimal_value': None,
                'solution': None,
                'solver_output': None,
                'error': str(e)
            }
        
        # Check if solution was found
        if result is None or not hasattr(result, 'sol') or result.sol is None:
            return {
                'status': 'infeasible',
                'optimal_value': None,
                'solution': None,
                'solver_output': result
            }
        
        # Extract solution
        optimal_solution = result.sol.flatten()
        optimal_obj = float(result.obj)
        
        # If we maximized, negate the objective back
        if not self.is_minimization:
            optimal_obj = -optimal_obj
        
        if verbose:
            print(f"\n✓ Solution found")
            print(f"  Optimal value: {optimal_obj:.8f}")
            print(f"  Solution sum: {optimal_solution.sum():.6f}")
            print(f"  Solution range: [{optimal_solution.min():.2e}, {optimal_solution.max():.2e}]")
        
        return {
            'status': 'optimal',
            'optimal_value': optimal_obj,
            'solution': optimal_solution,
            'solver_output': result
        }

    def solve_model(self, solver_type: str = 'glpk', verbose: bool = False):
        """
        Solve the linear program using PPOPT's MPLP_Program with automatic constraint processing.
        
        The LP is converted from our format:
            minimize/maximize    α^T q
            subject to           P q = p  (equality constraints)
                                q ≥ 0    (non-negativity)
        
        To PPOPT's format:
            minimize    c^T x
            subject to  A x ≤ b
        
        PPOPT's process_constraints() automatically removes strongly and weakly redundant
        constraints and rescales them, leading to significant performance increases and
        improved numerical stability.
        
        Args:
            solver_type: Solver to use ('glpk' or 'gurobi'). Default 'glpk'.
            verbose: If True, print detailed solver output.
        
        Returns:
            dict: Solution dictionary with keys:
                - 'optimal_value': The optimal objective value
                - 'solution': The optimal q vector (decision variables)
                - 'status': 'optimal', 'infeasible', or 'error'
                - 'solver_output': Raw solution object from PPOPT (if successful)
        
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

        #Step 1.5: Define (dummy) parameters
        # This is required by the MPLP_Program interface
        p = model.add_param(name="p")
        
        #Step 2: Define obs constraints
        for row in range(self.constraint_matrix.shape[0]):
            lhs = sum(self.constraint_matrix[row, col] * q[col] for col in range(n_vars))
            rhs = self.rhs[row]
            model.add_constr(lhs == rhs)

        #Step 3: Define non-negativity constraints
        for i in range(n_vars):
            model.add_constr(q[i] >= 0)
        
        #Step 4: Define objective
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

        return solution
