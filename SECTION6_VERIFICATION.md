# Verification of Sachs et al. (2022) Section 6 Examples

## Summary

This document summarizes the verification of the dual LP approach for computing symbolic bounds on causal effects, as described in Sachs et al. (2022) "A General Method for Deriving Tight Symbolic Bounds on Causal Effects".

## What Was Implemented

### 1. Dual LP Approach (✅ Complete)
- **File**: `test_dual_clean.py`
- **Status**: Fully working implementation
- **Method**: Uses `pycddlib` for vertex enumeration in dual space
- **Result**: Successfully computes symbolic bounds as linear combinations of observed parameters

### 2. Section 6.1: Confounded Exposure and Outcome

#### Binary Case (✅ Verified)
- **File**: `verify_section6_1.py`
- **Setup**: X, Y ∈ {0,1} with unobserved confounder U
- **Query**: Average Treatment Effect (ATE) = P{Y(X=1)=1} - P{Y(X=0)=1}
- **Result**: Successfully derived **Balke-Pearl bounds**:
  - **Lower bound**: `p{X=1,Y=1} + p{X=0,Y=0} - 1`
  - **Upper bound**: `1 - p{X=1,Y=0} - p{X=0,Y=1}`

**Numerical Example**:
```
Given: p(X=0,Y=0) = 0.3, p(X=0,Y=1) = 0.2, p(X=1,Y=0) = 0.1, p(X=1,Y=1) = 0.4
Bounds: [-0.30, 0.70]
```

#### Ternary Case (⚠️ Pending)
- **Setup**: X ∈ {0,1,2}, Y ∈ {0,1}
- **Parameters**: 6 observed probabilities (3×2)
- **Query**: Risk difference P{Y(X=x₁)=1} - P{Y(X=x₂)=1}
- **Expected Bounds** (from paper):
  ```
  p{X=x₁,Y=1} + p{X=x₂,Y=0} - 1 
    ≤ p{Y(X=x₁)=1} - p{Y(X=x₂)=1} ≤ 
  1 - p{X=x₁,Y=0} - p{X=x₂,Y=1}
  ```
- **Status**: Formula documented, implementation pending
- **Requirements**: 
  - Extend response type enumeration to handle ternary variables
  - Generate constraint matrix for 3×2 case

### 3. Section 6.2: Two Instruments (📋 Documented)
- **File**: `test_section6_examples.py`
- **Setup**: Z₁, Z₂ → X → Y (all binary)
- **Complexity**: 
  - 16 constraints (conditional probabilities)
  - 64 parameters (response function distribution)
  - 112 vertices in dual polytope
- **Query**: P{Y(X=1)=1} - P{Y(X=0)=1}
- **Status**: Paper notes bounds are too long to present simply
- **Note**: Code for this example is in supplementary materials

### 4. Section 6.3: Measurement Error (📋 Documented)
- **File**: `test_section6_examples.py`
- **Setup**: X → Y → Y₂, where Y is unobserved
- **Constraint**: Monotonicity Y₂(Y=1) ≥ Y₂(Y=0)
- **Parameters**: 12 parameters, 4 constraints
- **Query**: P{Y(X=1)=1} - P{Y(X=0)=1}
- **Expected Bounds** (from paper):
  ```
  max{-1, 2·p{Y₂=0|X=0} - 2·p{Y₂=0|X=1} - 1}
    ≤ P{Y(X=1)=1} - P{Y(X=0)=1} ≤
  min{1, 2·p{Y₂=0|X=0} - 2·p{Y₂=0|X=1} + 1}
  ```
- **Status**: Formula documented, implementation pending
- **Requirements**:
  - Handle latent variables (Y unobserved)
  - Implement monotonicity constraints

## Key Achievements

### ✅ Completed
1. **Dual LP Implementation**: Working code in `test_dual_clean.py`
2. **Binary Confounding**: Verified Balke-Pearl bounds match Section 6.1
3. **Documentation**: All three Section 6 examples documented with expected formulas
4. **Numerical Verification**: Confirmed bounds are valid with concrete distributions

### ⚠️ In Progress
1. **Ternary Variables**: Need to extend response type enumeration
2. **Constraint Generation**: Automate constraint matrix construction for general DAGs

### 📋 Future Work
1. **Two Instruments**: Implement full 112-vertex enumeration
2. **Measurement Error**: Add support for latent variables and monotonicity constraints
3. **General Framework**: Integrate with existing `symbolic_bounds` package

## Technical Details

### Method: Dual LP Approach

The dual LP method transforms the problem of finding symbolic bounds:

**Primal LP**:
```
maximize/minimize: c^T θ
subject to: A θ = p
            θ ≥ 0
```

**Dual LP**:
```
maximize/minimize: p^T y
subject to: A^T y ≤ c  (for max)
            A^T y ≥ c  (for min)
```

Where:
- `θ`: response function distribution (hidden parameters)
- `p`: observed distribution parameters
- `c`: causal query coefficients
- `A`: constraint matrix relating θ to p

The symbolic bounds are found by:
1. Enumerating vertices of the dual feasible region using `pycddlib`
2. Computing `p^T y` for each vertex y
3. Taking max/min over vertices to get upper/lower bounds

### Key Formula (Balke-Pearl Bounds)

For binary confounded X→Y, the ATE bounds are:

```python
lower = p11 + p00 - 1
upper = 1 - p10 - p01
```

This is a special case of the general formula from Section 6.1:
```
p{X=x₁,Y=1} + p{X=x₂,Y=0} - 1 ≤ ATE ≤ 1 - p{X=x₁,Y=0} - p{X=x₂,Y=1}
```

## Files

### Implementation Files
- `test_dual_clean.py`: Main dual LP implementation (✅ working)
- `verify_section6_1.py`: Section 6.1 binary case verification (✅ working)
- `test_section6_examples.py`: Documentation of all Section 6 examples (📋 reference)

### Supporting Files
- `section6.md`: Paper content provided by user
- `symbolic_bounds/`: Package with DAG, node, LP construction tools
- `VERTEX_ENUMERATION_SUMMARY.md`: Documentation of vertex enumeration approach

## References

Sachs, M. C., et al. (2022). "A General Method for Deriving Tight Symbolic Bounds on Causal Effects." *Journal of Causal Inference*, 10(1), 223-245.

## Next Steps

1. **Extend to ternary variables**: Modify response type enumeration in `symbolic_bounds/response_type.py`
2. **Automate constraint generation**: Use `symbolic_bounds/program_factory.py` to generate A matrix
3. **Integrate with dual solver**: Connect ProgramFactory output to `test_dual_clean.py` solver
4. **Test on all Section 6 examples**: Verify symbolic results match paper formulas

## Conclusion

The dual LP approach has been successfully implemented and verified for the binary confounded case (Section 6.1). The results match the expected Balke-Pearl bounds. The method is general and can be extended to handle:
- Ternary and higher-cardinality variables
- Multiple instruments
- Latent variables
- Additional constraints (monotonicity, etc.)

The core algorithm is working correctly. Remaining work is primarily engineering: automating constraint generation and extending response type enumeration to handle more complex scenarios.
