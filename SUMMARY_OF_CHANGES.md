# Summary of Changes: Parametric LP Implementation

## Problem

The `solve()` method in `linear_program.py` was designed to handle parametric linear programs with experimental constraints P(V=v|do(Z=z)) = θ, but there was an inconsistency in the interface:

- **Without experiments**: `solve().evaluate_objective(np.array([1]))` worked (1D array)
- **With experiments**: `solve().evaluate_objective(np.array([1]))` returned `None` (needed 2D array)

This made the API confusing and error-prone for users.

## Solution

Created a `ParametricSolution` wrapper class that:

1. **Handles array shapes automatically**: Accepts scalars, 1D arrays, or 2D column vectors
2. **Provides consistent interface**: Works the same way with or without experimental constraints
3. **Validates inputs**: Checks parameter dimensions and provides clear error messages
4. **Maintains backward compatibility**: All existing code continues to work

## Files Changed

### `symbolic_bounds/linear_program.py`

**Added:**
- `ParametricSolution` class (lines 15-106)
  - `__init__()`: Initialize with PPOPT solution and number of experiments
  - `evaluate_objective()`: Evaluate with automatic shape handling
  - `get_region()`: Get critical region (with shape handling)
  - `evaluate()`: Get decision variables (with shape handling)
  - `critical_regions` property: Access underlying regions
  - `__len__()`: Number of critical regions

**Modified:**
- `solve()` method (line 549):
  - Now returns `ParametricSolution` instead of raw PPOPT `Solution`
  - Updated docstring to explain parametric evaluation
  - Passes number of experiments to wrapper

## Key Features

### 1. Automatic Shape Conversion

```python
result = lp.solve()

# All of these now work:
result.evaluate_objective(0.6)                  # Scalar
result.evaluate_objective(np.array([0.6]))      # 1D array
result.evaluate_objective(np.array([[0.6]]))    # 2D array
```

### 2. Clear Error Messages

```python
# If you pass wrong number of parameters:
result.evaluate_objective([0.5, 0.6])
# ValueError: Expected 1 parameter(s), got 2. One parameter per experimental constraint.
```

### 3. Backward Compatible

```python
# Original code still works:
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))
lb = lp.solve().evaluate_objective(1)  # Still works!
```

## Usage Examples

### Basic Usage (No Experiments)

```python
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

lp.is_minimization = True
lb = lp.solve().evaluate_objective(1)

lp.is_minimization = False
ub = lp.solve().evaluate_objective(1)

print(f"Bounds: [{lb:.6f}, {-ub:.6f}]")
```

### With Experimental Constraints (NEW!)

```python
# Add experimental constraint P(Y=1|do(X=1)) = θ
lp = ProgramFactory.write_LP(
    scm,
    Y={Y}, X={X}, Y_values=(1,), X_values=(1,),
    V={Y}, Z={X}, V_values=(1,), Z_values=(1,)
)

# Evaluate at experimental result
experimental_result = 0.6
obj = lp.solve().evaluate_objective(experimental_result)
print(f"Objective at θ={experimental_result}: {obj:.6f}")
```

## Testing

Created `test_parametric_lp.py` with comprehensive tests:

1. ✓ LP without experimental constraints
2. ✓ LP with experimental constraints  
3. ✓ Different input formats (scalar, 1D, 2D)
4. ✓ Complementary probabilities
5. ✓ Feasible region detection

**All tests pass!**

```bash
python test_parametric_lp.py
# ================================================================================
# ALL TESTS PASSED ✓
# ================================================================================
```

## Documentation

Created two documentation files:

1. **`PARAMETRIC_LP_DOCUMENTATION.md`**: Comprehensive guide with:
   - Overview of parametric LPs
   - Usage examples
   - Technical details
   - Error handling
   - Testing instructions

2. **`SUMMARY_OF_CHANGES.md`**: This file, quick reference for changes

## Impact

### Benefits

- ✓ **Easier to use**: No need to worry about array shapes
- ✓ **More robust**: Automatic validation and clear error messages
- ✓ **Backward compatible**: Existing code continues to work
- ✓ **Well documented**: Clear examples and explanations
- ✓ **Well tested**: Comprehensive test suite

### No Breaking Changes

- All existing tests pass without modification
- Original `complete_workflow.ipynb` works as before
- API is enhanced, not changed

## Technical Notes

### How It Works

1. User calls `lp.solve()` → returns `ParametricSolution` wrapper
2. User calls `result.evaluate_objective(theta)` → wrapper:
   - Converts theta to proper shape (column vector)
   - Validates dimensions
   - Calls PPOPT's `solution.evaluate_objective(theta_2d)`
   - Returns the result

### PPOPT Integration

PPOPT's `solve_mpqp()` returns a `Solution` object with critical regions. Each region represents a portion of parameter space where the optimal solution has a specific form. The wrapper:

1. Takes the PPOPT solution
2. Provides a cleaner interface for evaluation
3. Handles shape conversion automatically
4. Delegates to PPOPT for actual computation

### Why This Approach?

**Alternatives considered:**

1. Modify PPOPT directly → Too invasive, hard to maintain
2. Auto-reshape in `solve()` → Can't know shape at solve time
3. Add separate methods → More complex API
4. **Wrapper class** → ✓ Clean, maintainable, backward compatible

## Verification

Verified the implementation works correctly:

- ✓ Original `complete_workflow.ipynb` runs successfully
- ✓ Original `symb_bounds_test.ipynb` runs successfully
- ✓ All custom tests in `test_parametric_lp.py` pass
- ✓ Both with and without experimental constraints work
- ✓ All input formats (scalar, 1D, 2D) work correctly

## Next Steps

The parametric LP implementation is now complete and ready for use. Users can:

1. Use experimental constraints by passing `V`, `Z`, `V_values`, `Z_values` to `write_LP()`
2. Evaluate the parametric solution at specific experimental results
3. Compute bounds given experimental data
4. Use any input format (scalar, 1D array, or 2D array) interchangeably

## Questions?

See `PARAMETRIC_LP_DOCUMENTATION.md` for detailed examples and explanations.
