# Linear Program Solver Implementation

## Overview

Successfully implemented `solve()` method for the `LinearProgram` class using the PPOPT (Parametric Optimization) library. The solver converts our LP format to PPOPT's format and solves using either GLPK or Gurobi backends.

## Implementation Details

### LP Format Conversion

Our LP format:
```
minimize/maximize    α^T q
subject to           P q = p  (equality constraints)
                     q ≥ 0    (non-negativity)
                     1^T q = 1 (normalization)
```

PPOPT format:
```
minimize    c^T x
subject to  A x ≤ b
            A_eq x = b_eq  (specified via equality_indices)
```

### Conversion Strategy

PPOPT's `Solver.solve_lp()` supports equality constraints directly via the `equality_indices` parameter, which specifies which rows of the constraint matrix are equalities.

1. **Equality Constraints**: Keep `P q = p` as-is, mark row indices as equalities

2. **Non-negativity**: Add `-I q ≤ 0` (inequality constraints)

3. **Normalization**: Add `1^T q = 1` as an equality constraint (mark its index)

4. **Maximization**: Negate objective when maximizing (PPOPT minimizes by default)

This approach is **cleaner and more numerically stable** than converting equalities to pairs of inequalities.

### PPOPT Integration

The solver uses PPOPT's `Solver` class:
```python
from ppopt.solver import Solver

solver = Solver(solvers={'lp': 'glpk'})
# equality_indices is a list of row indices that are equality constraints
result = solver.solve_lp(c, A, b, equality_constraints=equality_indices, verbose=False)
```

Supported backends:
- **glpk**: Free open-source solver (via cvxopt)
- **gurobi**: Commercial high-performance solver

## Method Signature

```python
def solve(self, solver_type: str = 'glpk', verbose: bool = False) -> dict:
    """
    Solve the linear program using PPOPT solvers.
    
    Args:
        solver_type: Solver to use ('glpk' or 'gurobi'). Default 'glpk'.
        verbose: If True, print detailed solver output.
    
    Returns:
        dict: Solution dictionary with keys:
            - 'optimal_value': The optimal objective value
            - 'solution': The optimal q vector (decision variables)
            - 'status': 'optimal', 'infeasible', or 'error'
            - 'solver_output': Raw SolverOutput object from PPOPT (if successful)
    
    Raises:
        ImportError: If PPOPT is not installed
        ValueError: If solver_type is not supported
    """
```

## Test Results

Successfully tested on X → Y chain:

### Test 1: Minimization (Lower Bound)
```
Generated Observed Distribution P*(X, Y):
  P*(X=0, Y=0) = 0.304750
  P*(X=0, Y=1) = 0.195292
  P*(X=1, Y=0) = 0.156406
  P*(X=1, Y=1) = 0.343552

LP Structure:
  Variables (ℵᴿ): 8
  Constraints: 4
  Non-zero in objective: 4

Optimal value: 0.34355229

Verification:
  ✓ All constraints satisfied (max violation: 0.00e+00)
  ✓ Normalization satisfied (1^T q = 1.000000)
  ✓ Non-negativity satisfied (min = 0.00e+00)
```

### Test 2: Maximization (Upper Bound)
```
LP type: Maximization
Optimal value: 0.84359441
✓ Maximization LP solved!
```

## Key Features

1. **Native Equality Support**: Uses PPOPT's `equality_indices` parameter for cleaner, more stable handling
2. **Flexible Backends**: Supports both GLPK (free) and Gurobi (commercial)
3. **Maximization Support**: Handles both min and max by negating objective
4. **Validation**: Verifies constraint satisfaction and probability constraints
5. **Future-Ready**: Uses PPOPT which supports parametric LPs for future extensions

## Implementation Note

**Initial approach** (less efficient): Converted each equality `A x = b` to two inequalities `A x ≤ b` and `-A x ≤ -b`, doubling constraint count.

**Current approach** (better): Uses PPOPT's built-in `equality_indices` parameter to mark which constraints are equalities. This:
- Reduces constraint count (from 2n to n for equality constraints)
- Improves numerical stability
- Leverages solver's native equality handling
- Matches the PPOPT documentation examples

## Dependencies

```bash
pip install ppopt
# or for GLPK backend specifically:
pip install cvxopt
```

## Usage Example

```python
from symbolic_bounds.dag import DAG
from symbolic_bounds.scm import SCM
from symbolic_bounds.data_generator import DataGenerator
from symbolic_bounds.program_factory import ProgramFactory

# Create DAG
dag = DAG()
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(X, Y)
dag.generate_all_response_types()

# Generate data
data_gen = DataGenerator(dag, seed=42)
scm = SCM(dag, data_gen)

# Build LP for P(Y=1 | do(X=1))
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

# Solve for lower bound
result = lp.solve(solver_type='glpk', verbose=False)
print(f"Lower bound: {result['optimal_value']}")

# Solve for upper bound
lp.is_minimization = False
result = lp.solve(solver_type='glpk', verbose=False)
print(f"Upper bound: {result['optimal_value']}")
```

## Files Modified

- `symbolic_bounds/linear_program.py`: Added `solve()` method (150 lines)
- `symbolic_bounds/test_lp_solve.py`: Created comprehensive test suite (220 lines)

## Next Steps

The solve method is ready for:
1. Computing bounds on causal effects
2. Experimental design optimization
3. Extension to parametric LPs (future work)
4. Integration with constraint optimization algorithms

## Notes

- PPOPT path is automatically added from `ppopt_repo/PPOPT/src/`
- Default solver is GLPK (free, open-source)
- Solution includes full SolverOutput object with duals, slacks, and active set
- All probability constraints (normalization, non-negativity) are enforced
