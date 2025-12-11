# Solving Linear Programs: Complete Guide

This document provides comprehensive guidance on the three methods available for solving LinearProgram objects, optimized for LLM coding agents.

## Overview of Solving Methods

The `LinearProgram` class provides three solving methods, each with different capabilities and use cases:

| Method | Library | Parametric? | Experimental Constraints? | Stability | Use Case |
|--------|---------|-------------|---------------------------|-----------|----------|
| `solve()` | PPOPT | ✅ Yes | ✅ Yes | ⚠️ Buggy | Parametric optimization with experiments |
| `solve_with_highs()` | HiGHS | ❌ No | ❌ No | ✅ Excellent | Fast, reliable bounds without experiments |
| `solve_with_autobound()` | autobound | ❌ No | ✅ Yes | ✅ Good | Integration with existing autobound workflows |

---

## Method 1: `solve()` - PPOPT-based Parametric Solver

**File**: `linear_program.py`  
**Library**: PPOPT (Parametric Programming Optimization Toolbox)  
**Status**: ⚠️ Known to be buggy, use with caution

### Purpose

Solves parametric linear programs where experimental constraints introduce parameters. The solution is a function of experimental results θ.

### Mathematical Formulation

Solves:
```
minimize/maximize    α^T q
subject to           P q = p           (observational constraints)
                     E q = θ           (experimental constraints - parametric)
                     q ≥ 0             (non-negativity)
```

Where θ represents experimental outcomes (e.g., P(Y | do(X=x))).

### Method Signature

```python
def solve(self, solver_type: str = 'glpk', verbose: bool = False) -> ParametricSolution
```

### Parameters

- **solver_type** (`str`, default='glpk'): 
  - Options: `'glpk'` or `'gurobi'`
  - Backend solver for LP subproblems
  - GLPK is free and usually sufficient

- **verbose** (`bool`, default=False):
  - If True, prints solver progress and diagnostics

### Returns

**ParametricSolution** object with methods:
- `evaluate_objective(theta)`: Evaluate objective at parameter value θ
- `evaluate(theta)`: Get decision variables at θ
- `get_region(theta)`: Get critical region containing θ
- `critical_regions`: List of all critical regions

### Usage Examples

#### Example 1: No Experimental Constraints

```python
# Create LP without experiments
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

# Solve for lower bound
lp.is_minimization = True
result_lb = lp.solve(verbose=False)
lb = result_lb.evaluate_objective(np.array([1]))  # Dummy parameter

# Solve for upper bound
lp.is_minimization = False
result_ub = lp.solve(verbose=False)
ub = result_ub.evaluate_objective(np.array([1]))

print(f"Bounds: [{lb:.6f}, {ub:.6f}]")
```

#### Example 2: With Experimental Constraints

```python
# Create LP with experimental constraints
# V={Y}, Z={X} means we observe P(Y | do(X))
lp_exp = ProgramFactory.write_LP(
    scm, 
    Y={Y}, X={X}, 
    Y_values=(1,), X_values=(1,),
    V={Y}, Z={X},           # Experimental observation
    V_values=(1,), Z_values=(1,)
)

# Solve (parametric in experimental result)
lp_exp.is_minimization = True
result_lb = lp_exp.solve(verbose=False)

# Evaluate at different experimental values
for theta in np.linspace(0, 1, 11):
    lb = result_lb.evaluate_objective(np.array([theta]))
    if lb is not None:
        print(f"θ={theta:.2f}: lower bound = {lb:.6f}")
    else:
        print(f"θ={theta:.2f}: infeasible")
```

### How It Works

1. **Convert to PPOPT format**: 
   - Creates `MPModeler` and adds variables/constraints
   - Experimental constraints become parameters
   - If no experiments, adds dummy parameter for PPOPT compatibility

2. **Build constraint system**:
   - Observational: `P q = p` (equality constraints)
   - Experimental: `E q = θ` (parametric equality constraints)
   - Non-negativity: `q ≥ 0`

3. **Solve parametrically**:
   - Uses `solve_mpqp()` with combinatorial algorithm
   - Computes critical regions in parameter space
   - Returns piecewise-affine solution

4. **Handle maximization**:
   - Internally converts max to min by negating objective
   - `ParametricSolution` wrapper handles sign correction

### Known Issues

⚠️ **PPOPT is extremely buggy** - this is why alternative methods were created:
- Numerical instability with some constraint matrices
- Incorrect results in certain cases
- Poor error messages when failing

**Recommendation**: Use `solve_with_highs()` or `solve_with_autobound()` when possible.

### ParametricSolution Wrapper

The return value wraps PPOPT's solution for easier use:

```python
class ParametricSolution:
    def evaluate_objective(self, theta):
        """
        Evaluate objective at parameter value(s).
        
        Args:
            theta: Scalar, 1D array, or 2D array
                - Scalar: converted to [[theta]]
                - 1D array: converted to column vector
                - 2D array: used directly
        
        Returns:
            float: Objective value (or None if infeasible)
        """
        # Handles array shape conversion
        # Automatically corrects sign for maximization
        # Returns None if theta outside feasible region
```

---

## Method 2: `solve_with_highs()` - HiGHS-based Solver

**File**: `linear_program.py`  
**Library**: HiGHS (via highspy)  
**Status**: ✅ Recommended for non-parametric problems

### Purpose

Fast, stable solver for LPs without experimental constraints. Uses state-of-the-art HiGHS optimizer.

### Mathematical Formulation

Solves:
```
minimize/maximize    α^T q
subject to           p - slack ≤ P q ≤ p + slack    (equality with tolerance)
                     q ≥ 0                          (non-negativity)
```

Note: Equality constraints are relaxed with slack to improve numerical stability.

### Method Signature

```python
def solve_with_highs(self, verbose: bool = False, slack: float = 1e-6) -> dict
```

### Parameters

- **verbose** (`bool`, default=False):
  - If True, prints solver diagnostics and results

- **slack** (`float`, default=1e-6):
  - Tolerance for equality constraints
  - Converts `P q = p` to `p - slack ≤ P q ≤ p + slack`
  - Increase if solver reports infeasibility
  - Typical range: 1e-8 to 1e-4

### Returns

Dictionary with keys:
- **status** (`str`): Solution status (e.g., "HighsModelStatus.kOptimal")
- **objective_value** (`float` or `None`): Optimal objective value
- **solution** (`ndarray` or `None`): Optimal decision variables q
- **dual** (`ndarray` or `None`): Dual variables for constraints

### Usage Examples

#### Example 1: Basic Bounds Computation

```python
# Create LP without experiments
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

# Lower bound
lp.is_minimization = True
result_lb = lp.solve_with_highs(verbose=True)
lb = result_lb['objective_value']

# Upper bound
lp.is_minimization = False
result_ub = lp.solve_with_highs(verbose=True)
ub = result_ub['objective_value']

print(f"Bounds: [{lb:.6f}, {ub:.6f}]")
print(f"Width: {ub - lb:.6f}")
```

#### Example 2: Handling Numerical Issues

```python
# If solver reports infeasible, try larger slack
result = lp.solve_with_highs(slack=1e-4, verbose=True)

if result['status'] != 'HighsModelStatus.kOptimal':
    print(f"Warning: {result['status']}")
    # Try even larger slack
    result = lp.solve_with_highs(slack=1e-3, verbose=True)
```

#### Example 3: Accessing Solution Details

```python
result = lp.solve_with_highs(verbose=False)

if result['objective_value'] is not None:
    print(f"Objective: {result['objective_value']:.6f}")
    print(f"Solution vector q:")
    print(f"  Length: {len(result['solution'])}")
    print(f"  Sum: {np.sum(result['solution']):.10f}")  # Should ≈ 1.0
    print(f"  Non-zero entries: {np.sum(result['solution'] > 1e-10)}")
    
    # Check which response types have non-zero probability
    for i, (prob, label) in enumerate(zip(result['solution'], lp.variable_labels)):
        if prob > 1e-6:
            print(f"  q[{i}] = {prob:.6f}: {label}")
```

### How It Works

1. **Initialize HiGHS model**:
   ```python
   h = highspy.Highs()
   h.setOptionValue("log_to_console", verbose)
   ```

2. **Add variables**:
   - n_vars variables with bounds `0 ≤ q ≤ ∞`
   - Uses `h.addVars(n_vars, lb, ub)`

3. **Set objective**:
   - For minimization: use α directly
   - For maximization: negate α (HiGHS only minimizes)

4. **Add constraints**:
   - For each row of P: `p - slack ≤ Σ P_ij q_j ≤ p + slack`
   - Only adds non-zero entries for efficiency

5. **Solve and extract**:
   - Call `h.run()`
   - Extract solution with `h.getSolution()`
   - Negate objective back for maximization

### Why Slack is Needed

**Problem**: Strict equality `P q = p` can be numerically infeasible
- Floating-point arithmetic introduces tiny errors
- Constraint matrix may have near-zero eigenvalues
- Solver may report infeasibility even when theoretically feasible

**Solution**: Relax to inequalities with small tolerance
- `p - 1e-6 ≤ P q ≤ p + 1e-6`
- Mathematically equivalent for practical purposes
- Dramatically improves solver stability

### Advantages vs PPOPT

✅ **Much more stable**: Rarely fails on well-formed LPs  
✅ **Faster**: Optimized C++ implementation  
✅ **Better error messages**: Clear diagnostics when issues occur  
✅ **Industry standard**: HiGHS used in production systems  

### Limitations

❌ **No parametric solutions**: Cannot handle experimental constraints  
❌ **No critical regions**: Only provides point solution  

---

## Method 3: `solve_with_autobound()` - Autobound Integration

**File**: `linear_program.py`  
**Library**: autobound package  
**Status**: ✅ Good for integration with existing autobound workflows

### Purpose

Integrates with the autobound package's `causalProblem` interface. Useful when you need autobound's features or want to compare methods. DAG structure, node domains, and unobserved nodes are automatically extracted from the stored DAG reference.

### Mathematical Formulation

Converts the LP to autobound's format:
```
minimize/maximize    P(Y=y | do(X=x))
subject to           Observational data P(W_L, W_R)
                     Optional: Interventional data P(V | do(Z))
```

### Method Signature

```python
def solve_with_autobound(
    self,
    intervention_data: dict = None,
    verbose: bool = False,
    solver: str = 'glpk'
) -> dict
```

### Parameters

- **intervention_data** (`dict` or `None`, default=None):
  - If provided, adds interventional constraints
  - Structure:
    ```python
    {
        'data': pd.DataFrame,           # Intervention results
        'intervention_node': str,        # Node intervened on (e.g., 'M')
        'intervention_col': str,         # Column with intervention values
        'observed_cols': list[str]       # Columns with observed variables
    }
    ```
  - DataFrame must have columns: [intervention_col] + observed_cols + ['prob']

- **verbose** (`bool`, default=False):
  - If True, prints detailed progress

- **solver** (`str`, default='glpk'):
  - Pyomo solver to use: `'glpk'`, `'ipopt'`, etc.

### Returns

Dictionary with keys:
- **lower_bound** (`float`): Lower bound on causal effect
- **upper_bound** (`float`): Upper bound on causal effect
- **width** (`float`): Width of identification region (upper - lower)
- **status** (`str`): 'success' if solved successfully

### Automatic DAG Parameter Extraction

The method automatically extracts DAG information:
- **dag_structure**: Edges formatted as "A -> B, C -> D" with U_L/U_R confounders added
- **node_domains**: Domain sizes from node.support
- **unobserved_nodes**: "U_L,U_R" based on W_L/W_R partition

This happens through the `dag.get_autobound_info()` method called internally.

### Usage Examples

#### Example 1: Without Interventions

```python
# Create LP (DAG info automatically stored)
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

# Solve (no DAG parameters needed!)
result = lp.solve_with_autobound(
    verbose=True,
    solver='glpk'
)

print(f"Lower bound: {result['lower_bound']:.6f}")
print(f"Upper bound: {result['upper_bound']:.6f}")
print(f"Width: {result['width']:.6f}")
```

#### Example 2: With Interventions

```python
import pandas as pd

# Create intervention data
# This represents P(Y=y | do(X=x)) for all x, y
intervention_rows = []

for x_val in [0, 1]:
    for y_val in [0, 1]:
        # Compute true intervention probability
        true_prob = generator.computeTrueIntervention(
            Y={Y}, X={X}, Y_values=(y_val,), X_values=(x_val,)
        )
        intervention_rows.append({
            'X_do': x_val,      # Intervention value
            'Y': y_val,         # Observed outcome
            'prob': true_prob   # Probability
        })

df_intervention = pd.DataFrame(intervention_rows)

# Define intervention data structure
intervention_data = {
    'data': df_intervention,
    'intervention_node': 'X',
    'intervention_col': 'X_do',
    'observed_cols': ['Y']
}

# Solve with interventions (DAG info extracted automatically)
result = lp.solve_with_autobound(
    intervention_data=intervention_data,
    verbose=True,
    solver='glpk'
)

print(f"With intervention: Width = {result['width']:.6f}")
```

#### Example 3: Complex Multi-Node Case (from 3Hedges.ipynb)

```python
# Load intervention data from CSV
df_doM = pd.read_csv('data/doM.csv')

intervention_data_M = {
    'data': df_doM,
    'intervention_node': 'M',
    'intervention_col': 'M_do',
    'observed_cols': ['Z', 'X', 'Y']  # All other observed variables
}

# DAG structure automatically extracted from LP
result = lp.solve_with_autobound(
    intervention_data=intervention_data_M,
    verbose=False,
    solver='glpk'
)
```

### How It Works

1. **Convert observational data**:
   - Parse constraint labels to extract variable configurations
   - Convert to DataFrame with columns for each variable + 'prob'
   - Write to temporary CSV file

2. **Create autobound DAG**:
   ```python
   from autobound.DAG import DAG as AutoboundDAG
   from autobound.causalProblem import causalProblem
   
   dag = AutoboundDAG()
   dag.from_structure(dag_structure, unob=unobserved_nodes)
   problem = causalProblem(dag, number_values=node_domains)
   ```

3. **Load observational data**:
   ```python
   problem.load_data(temp_csv_path)
   problem.add_prob_constraints()
   ```

4. **Add intervention constraints** (if provided):
   ```python
   for _, row in df_intervention.iterrows():
       # Build query string: "Y(X=x)=y & Z(X=x)=z & ..."
       query_str = build_query(row, intervention_node, observed_cols)
       prob_val = row['prob']
       
       lhs = problem.query(query_str)
       problem.add_constraint(lhs - Query(prob_val))
   ```

5. **Set estimand and solve**:
   ```python
   # Infer estimand (e.g., "Y(X=1)=1")
   estimand = infer_estimand(node_domains)
   problem.set_estimand(problem.query(estimand))
   
   prog = problem.write_program()
   lower, upper = prog.run_pyomo(solver, verbose=verbose)
   ```

6. **Clean up**:
   - Temporary CSV file is automatically deleted

### Intervention Data Format

The intervention DataFrame must follow this structure:

```python
# For single intervention node
df = pd.DataFrame([
    {'X_do': 0, 'Y': 0, 'prob': 0.3},  # P(Y=0 | do(X=0))
    {'X_do': 0, 'Y': 1, 'prob': 0.7},  # P(Y=1 | do(X=0))
    {'X_do': 1, 'Y': 0, 'prob': 0.2},  # P(Y=0 | do(X=1))
    {'X_do': 1, 'Y': 1, 'prob': 0.8},  # P(Y=1 | do(X=1))
])

# For intervention with additional observed variables
df = pd.DataFrame([
    {'M_do': 0, 'Z': 0, 'X': 0, 'Y': 0, 'prob': 0.1},
    {'M_do': 0, 'Z': 0, 'X': 0, 'Y': 1, 'prob': 0.2},
    # ... all combinations ...
])
```

**Key Rules**:
- One row per configuration
- Must enumerate ALL combinations of (intervention_val, observed_vals)
- Probabilities must sum to 1 within each intervention value
- Column names must match what you specify in intervention_data dict

### Generating Intervention Data from DataGenerator

```python
# Using the symbolic_bounds DataGenerator
generator = DataGenerator(dag, seed=42)

intervention_rows = []

# For each intervention value
for x_val in X.support:
    # For each possible outcome configuration
    for y_val in Y.support:
        # Compute true probability
        prob = generator.computeTrueIntervention(
            Y={Y}, X={X}, 
            Y_values=(y_val,), X_values=(x_val,)
        )
        
        intervention_rows.append({
            'X_do': x_val,
            'Y': y_val,
            'prob': prob
        })

df_intervention = pd.DataFrame(intervention_rows)
```

### Advantages

✅ **Handles interventions**: Unlike `solve_with_highs()`  
✅ **Familiar interface**: For users already using autobound  
✅ **Well-tested**: autobound is mature and stable  
✅ **Flexible**: Easy to add multiple types of constraints  

### Limitations

❌ **Requires CSV files**: Less efficient than direct matrix solving  
❌ **Extra dependency**: Must have autobound installed  
❌ **No parametric solutions**: Like HiGHS, returns point estimates  
❌ **Estimand inference**: Currently uses heuristics, may need manual specification  

---

## Comparison and Selection Guide

### Decision Tree for Choosing a Method

```
Do you need parametric solutions (bounds as function of experiment)?
├─ YES → Use solve() with PPOPT
│         ⚠️ Warning: buggy, validate results carefully
│
└─ NO → Do you have experimental constraints?
         ├─ YES → Use solve_with_autobound()
         │         ✅ Reliable, handles interventions
         │
         └─ NO → Use solve_with_highs()
                   ✅ Fastest, most stable
```

### Performance Comparison

For a typical problem (4 nodes, 64 response types):

| Method | Time | Stability | Features |
|--------|------|-----------|----------|
| solve_with_highs() | ~0.01s | Excellent | Basic bounds only |
| solve_with_autobound() | ~0.1s | Good | Interventions, CSV-based |
| solve() | ~0.5s | Poor | Parametric, experimental |

### When to Use Each Method

#### Use `solve_with_highs()` when:
- ✅ You only need observational bounds
- ✅ Speed is important
- ✅ You want maximum reliability
- ✅ No experimental constraints needed

#### Use `solve_with_autobound()` when:
- ✅ You have intervention data to incorporate
- ✅ You're already using autobound in your workflow
- ✅ You want to compare with autobound's direct approach
- ✅ You need a reliable alternative to PPOPT for interventions

#### Use `solve()` when:
- ✅ You absolutely need parametric solutions
- ✅ You're plotting bounds vs experimental results
- ✅ You're willing to validate results carefully
- ⚠️ Be prepared for potential bugs

---

## Common Patterns and Idioms

### Pattern 1: Computing Tight Bounds

```python
def compute_bounds(lp, method='highs'):
    """Compute tight lower and upper bounds."""
    
    if method == 'highs':
        lp.is_minimization = True
        result_lb = lp.solve_with_highs(verbose=False)
        lb = result_lb['objective_value']
        
        lp.is_minimization = False
        result_ub = lp.solve_with_highs(verbose=False)
        ub = result_ub['objective_value']
        
    elif method == 'autobound':
        result = lp.solve_with_autobound(
            dag_structure=...,
            node_domains=...,
            unobserved_nodes=...,
            verbose=False
        )
        lb = result['lower_bound']
        ub = result['upper_bound']
    
    return lb, ub
```

### Pattern 2: Parametric Bounds Plotting

```python
def plot_parametric_bounds(lp_exp, theta_range=np.linspace(0, 1, 101)):
    """Plot bounds as function of experimental parameter."""
    
    lp_exp.is_minimization = True
    result_lb = lp_exp.solve(verbose=False)
    
    lp_exp.is_minimization = False
    result_ub = lp_exp.solve(verbose=False)
    
    bounds_data = []
    for theta in theta_range:
        lb = result_lb.evaluate_objective(np.array([theta]))
        ub = result_ub.evaluate_objective(np.array([theta]))
        
        if lb is not None and ub is not None:
            bounds_data.append({
                'theta': theta,
                'lower': lb,
                'upper': ub,
                'width': ub - lb
            })
    
    df = pd.DataFrame(bounds_data)
    df.plot(x='theta', y=['lower', 'upper'], 
            xlabel='Experimental Result θ',
            ylabel='Causal Effect Bounds')
```

### Pattern 3: Comparing Methods

```python
def compare_solvers(lp):
    """Compare results from different solvers."""
    
    print("Comparing solver methods...")
    
    # Method 1: HiGHS
    lp.is_minimization = True
    result_highs_lb = lp.solve_with_highs(verbose=False)
    lp.is_minimization = False
    result_highs_ub = lp.solve_with_highs(verbose=False)
    
    # Method 2: PPOPT
    lp.is_minimization = True
    result_ppopt_lb = lp.solve(verbose=False)
    lp.is_minimization = False
    result_ppopt_ub = lp.solve(verbose=False)
    
    # Method 3: autobound
    result_autobound = lp.solve_with_autobound(
        dag_structure=...,
        node_domains=...,
        verbose=False
    )
    
    # Compare
    print(f"HiGHS:     [{result_highs_lb['objective_value']:.6f}, "
          f"{result_highs_ub['objective_value']:.6f}]")
    print(f"PPOPT:     [{result_ppopt_lb.evaluate_objective(np.array([1])):.6f}, "
          f"{result_ppopt_ub.evaluate_objective(np.array([1])):.6f}]")
    print(f"autobound: [{result_autobound['lower_bound']:.6f}, "
          f"{result_autobound['upper_bound']:.6f}]")
```

### Pattern 4: Robust Solving with Fallback

```python
def solve_robust(lp, **kwargs):
    """Solve with automatic fallback if primary method fails."""
    
    # Try HiGHS first (fastest, most stable)
    try:
        lp.is_minimization = True
        result_lb = lp.solve_with_highs(slack=1e-6, verbose=False)
        lp.is_minimization = False
        result_ub = lp.solve_with_highs(slack=1e-6, verbose=False)
        
        if (result_lb['objective_value'] is not None and 
            result_ub['objective_value'] is not None):
            return {
                'lower': result_lb['objective_value'],
                'upper': result_ub['objective_value'],
                'method': 'highs'
            }
    except Exception as e:
        print(f"HiGHS failed: {e}")
    
    # Fall back to HiGHS with larger slack
    try:
        lp.is_minimization = True
        result_lb = lp.solve_with_highs(slack=1e-4, verbose=False)
        lp.is_minimization = False
        result_ub = lp.solve_with_highs(slack=1e-4, verbose=False)
        
        if (result_lb['objective_value'] is not None and 
            result_ub['objective_value'] is not None):
            return {
                'lower': result_lb['objective_value'],
                'upper': result_ub['objective_value'],
                'method': 'highs_relaxed'
            }
    except Exception as e:
        print(f"HiGHS (relaxed) failed: {e}")
    
    # Last resort: autobound
    try:
        result = lp.solve_with_autobound(**kwargs, verbose=False)
        return {
            'lower': result['lower_bound'],
            'upper': result['upper_bound'],
            'method': 'autobound'
        }
    except Exception as e:
        print(f"autobound failed: {e}")
        raise RuntimeError("All solving methods failed")
```

---

## Troubleshooting

### Issue 1: "Infeasible" Status from HiGHS

**Symptoms**: `solve_with_highs()` returns status other than `kOptimal`

**Causes**:
- Equality constraints too strict
- Numerical precision issues
- Genuinely infeasible LP (bug in constraint generation)

**Solutions**:
```python
# 1. Increase slack tolerance
result = lp.solve_with_highs(slack=1e-4, verbose=True)

# 2. Check constraint matrix for issues
lp.print_lp(show_full_matrices=True)
print(f"RHS sum: {np.sum(lp.rhs):.10f}")  # Should be 1.0
print(f"Matrix rank: {np.linalg.matrix_rank(lp.constraint_matrix)}")

# 3. Validate constraint generation
from symbolic_bounds.test_constraints import validate_constraints
validate_constraints(scm.dag, verbose=True)
```

### Issue 2: PPOPT Returns Wrong Results

**Symptoms**: Bounds don't contain true causal effect

**Solutions**:
```python
# 1. Cross-validate with other methods
lb_highs, ub_highs = compute_bounds(lp, method='highs')
lb_ppopt = lp.solve().evaluate_objective(np.array([1]))
ub_ppopt = lp.solve().evaluate_objective(np.array([1]))

if abs(lb_highs - lb_ppopt) > 1e-4:
    print("⚠️ PPOPT result differs from HiGHS!")
    print(f"Using HiGHS result: {lb_highs:.6f}")

# 2. Use alternative method
result = lp.solve_with_autobound(...)
```

### Issue 3: autobound "Cannot Find Estimand"

**Symptoms**: `solve_with_autobound()` fails with query-related error

**Cause**: Estimand inference heuristic doesn't match your problem

**Solution**: The current implementation has a simplified estimand inference. For complex cases, you may need to modify `_infer_estimand_query()`:

```python
# In linear_program.py, modify _infer_estimand_query method
def _infer_estimand_query(self, node_domains: dict) -> str:
    # Custom logic for your specific problem
    # Example: If you know you're always estimating P(Y(X=1)=1)
    return 'Y(X=1)=1'
```

### Issue 4: Intervention Data Format Errors

**Symptoms**: autobound complains about constraint format

**Checklist**:
```python
# Verify DataFrame structure
print(df_intervention.columns)  # Should include intervention_col + observed_cols + ['prob']
print(df_intervention['prob'].sum())  # Should be 1.0
print(df_intervention.groupby('X_do')['prob'].sum())  # Each group should sum to 1.0

# Check for missing combinations
expected_rows = np.prod([len(node_domains[col]) for col in ['X_do'] + observed_cols])
actual_rows = len(df_intervention)
if expected_rows != actual_rows:
    print(f"⚠️ Expected {expected_rows} rows, got {actual_rows}")
```

---

## Best Practices for LLM Agents

### 1. Default to `solve_with_highs()`

Unless explicitly told otherwise:
```python
# ✅ Good default choice
result = lp.solve_with_highs(verbose=True)
lb, ub = result['objective_value'], ...

# ❌ Avoid as default (buggy)
result = lp.solve()  # Only use if parametric needed
```

### 2. Always Validate Bounds

```python
# Compute bounds
lb, ub = compute_bounds(lp)

# Validate
true_value = generator.computeTrueIntervention(...)
assert lb <= true_value <= ub, f"Bounds [{lb}, {ub}] don't contain {true_value}"
```

### 3. Handle Both Minimization and Maximization

```python
# ✅ Correct pattern
lp.is_minimization = True
result_lb = solver(lp)
lb = extract_objective(result_lb)

lp.is_minimization = False  # Don't forget to flip!
result_ub = solver(lp)
ub = extract_objective(result_ub)

# ❌ Common mistake
result = solver(lp)  # What is lp.is_minimization?
```

### 4. Provide Verbose Output During Development

```python
# During debugging/development
result = lp.solve_with_highs(verbose=True)  # See what's happening

# In production/final code
result = lp.solve_with_highs(verbose=False)  # Clean output
```

### 5. Document Which Method You're Using

```python
# ✅ Clear and explicit
print("Solving using HiGHS (recommended for stability)...")
result = lp.solve_with_highs()

# ❌ Ambiguous
result = lp.solve()  # Which solve method?
```

---

## Summary Table

| Feature | solve() | solve_with_highs() | solve_with_autobound() |
|---------|---------|-------------------|----------------------|
| **Stability** | ⚠️ Poor | ✅ Excellent | ✅ Good |
| **Speed** | Slow | ✅ Fast | Medium |
| **Parametric** | ✅ Yes | ❌ No | ❌ No |
| **Interventions** | ✅ Yes | ❌ No | ✅ Yes |
| **Installation** | PPOPT | highspy | autobound |
| **Returns** | ParametricSolution | dict | dict |
| **Recommended** | ❌ No | ✅ Yes | ✅ When needed |

**General Recommendation**: Start with `solve_with_highs()`, fall back to `solve_with_autobound()` if you need interventions, only use `solve()` if you absolutely need parametric solutions and are willing to carefully validate results.
