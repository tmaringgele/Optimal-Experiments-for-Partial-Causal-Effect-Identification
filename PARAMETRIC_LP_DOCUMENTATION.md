# Parametric Linear Program Implementation

## Overview

The `solve()` method in `linear_program.py` now correctly handles parametric linear programs with experimental constraints. This allows you to compute causal bounds given experimental results P(V=v|do(Z=z)) = θ.

## What Changed

### 1. Added `ParametricSolution` Wrapper Class

A new wrapper class `ParametricSolution` provides a user-friendly interface for evaluating parametric solutions. It handles:

- **Automatic array shape conversion**: Accepts scalars, 1D arrays, or 2D arrays
- **Consistent interface**: Works the same whether experiments exist or not
- **Clear error messages**: Validates input dimensions and provides helpful feedback

### 2. Modified `solve()` Return Type

The `solve()` method now returns a `ParametricSolution` object instead of a raw PPOPT `Solution`. This makes the API more consistent and easier to use.

### 3. Improved Documentation

Updated docstrings to explain:
- How parametric solutions work
- How to evaluate at specific parameter values
- The difference between cases with/without experimental constraints

## Usage Examples

### Without Experimental Constraints (Original Working Case)

```python
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import numpy as np

# Create DAG
dag = DAG()
X = dag.add_node('X', support={0, 1})
Y = dag.add_node('Y', support={0, 1})
dag.add_edge(X, Y)
dag.generate_all_response_types()

# Create SCM
generator = DataGenerator(dag, seed=42)
scm = SCM(dag, generator)

# Compute bounds on P(Y=1|do(X=1))
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

lp.is_minimization = True
lb = lp.solve().evaluate_objective(1)  # Any value works (dummy parameter)

lp.is_minimization = False
ub = lp.solve().evaluate_objective(1)

print(f"Bounds: [{lb:.6f}, {-ub:.6f}]")
```

### With Experimental Constraints (NEW!)

```python
# Suppose we perform an experiment and observe P(Y=1|do(X=1)) = 0.6

# Compute bounds on P(Y=1|do(X=1)) given this experimental result
lp = ProgramFactory.write_LP(
    scm,
    Y={Y}, X={X}, Y_values=(1,), X_values=(1,),    # Query
    V={Y}, Z={X}, V_values=(1,), Z_values=(1,)     # Experimental constraint
)

# Evaluate at the experimental result
experimental_result = 0.6

lp.is_minimization = True
lb = lp.solve().evaluate_objective(experimental_result)

lp.is_minimization = False
ub = lp.solve().evaluate_objective(experimental_result)

print(f"Bounds given experiment: [{lb:.6f}, {-ub:.6f}]")
# Output: Bounds given experiment: [0.600000, 0.600000]
# (Bounds collapse to experimental value, as expected)
```

### Multiple Input Formats

The `evaluate_objective()` method accepts various input formats:

```python
result = lp.solve()

# All of these work:
obj1 = result.evaluate_objective(0.6)                  # Scalar
obj2 = result.evaluate_objective(np.array([0.6]))      # 1D array
obj3 = result.evaluate_objective(np.array([[0.6]]))    # 2D array

# All return the same value: 0.6
```

### Computing Different Quantities

You can compute one causal effect given constraints on another:

```python
# Compute P(Y=0|do(X=1)) given we observed P(Y=1|do(X=1)) = 0.6
lp = ProgramFactory.write_LP(
    scm,
    Y={Y}, X={X}, Y_values=(0,), X_values=(1,),    # Query: P(Y=0|do(X=1))
    V={Y}, Z={X}, V_values=(1,), Z_values=(1,)     # Constraint: P(Y=1|do(X=1)) = θ
)

lp.is_minimization = True
p_y0 = lp.solve().evaluate_objective(0.6)
print(f"P(Y=0|do(X=1)) = {p_y0:.6f}")  # Output: 0.400000

# Verify they sum to 1: 0.6 + 0.4 = 1.0 ✓
```

## Technical Details

### How Parametric LPs Work

When you add experimental constraints via the `V`, `Z`, `V_values`, and `Z_values` parameters:

1. An experiment matrix `Exp` is created where each row represents one experimental constraint
2. PPOPT creates a parameter θ for each experimental constraint
3. The constraint `Exp q = θ` is added to the LP
4. PPOPT solves the parametric LP and returns critical regions covering the feasible θ space
5. You evaluate the solution at specific θ values to get bounds given those experimental results

### Underlying PPOPT Format

The LP is converted from our format:
```
minimize    α^T q
subject to  P q = p          (observational constraints)
            Exp q = θ         (experimental constraints)
            q ≥ 0
```

To PPOPT's format:
```
minimize    c^T x
subject to  A x ≤ b + F θ
            A_θ θ ≤ b_θ
```

Where:
- `x` corresponds to our decision variable `q` (response type probabilities)
- `θ` represents the experimental results
- `F` encodes how experiments depend on the parameter

### Critical Regions

PPOPT's solution contains multiple critical regions, each representing a portion of the parameter space where the optimal solution has a specific form. The `ParametricSolution` wrapper:

1. Finds which region contains your parameter value
2. Evaluates the optimal solution in that region
3. Returns `None` if the parameter value is outside all feasible regions

## Error Handling

The `ParametricSolution` class validates inputs and provides clear error messages:

```python
# Wrong number of parameters
result.evaluate_objective([0.5, 0.6])  # Two values when expecting one
# ValueError: Expected 1 parameter(s), got 2

# Invalid shape
result.evaluate_objective(np.array([[[0.5]]]))  # 3D array
# ValueError: theta must be scalar, 1D, or 2D array, got shape (1, 1, 1)
```

## Backward Compatibility

The changes are fully backward compatible:
- Existing code without experimental constraints continues to work
- The dummy parameter approach for non-parametric cases is preserved
- All existing tests pass without modification

## Testing

Run the comprehensive test suite:

```bash
python test_parametric_lp.py
```

This tests:
1. LPs without experimental constraints (original functionality)
2. LPs with experimental constraints (new functionality)
3. Different input formats (scalar, 1D, 2D arrays)
4. Complementary probabilities
5. Feasible region detection

## Future Enhancements

Possible future improvements:
- Support for multiple experimental constraints simultaneously
- Visualization of critical regions in parameter space
- Caching of parametric solutions to avoid re-solving
- Integration with experimental design optimization
