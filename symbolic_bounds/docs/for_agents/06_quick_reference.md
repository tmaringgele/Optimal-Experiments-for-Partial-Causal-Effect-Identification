# Quick Reference Guide

Quick lookup for common tasks and answers to frequent questions.

## Common Tasks

### Create a Simple DAG
```python
from symbolic_bounds import DAG

dag = DAG()
X = dag.add_node('X', {0, 1}, 'R')  # Binary node in W_R
Y = dag.add_node('Y', {0, 1}, 'R')
dag.add_edge(X, Y)
dag.generate_all_response_types()
```

### Compute Bounds on Causal Effect
```python
from symbolic_bounds import DataGenerator, SCM, ProgramFactory
import numpy as np

generator = DataGenerator(dag, seed=42)
scm = SCM(dag, generator)

lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

lp.is_minimization = True
lower = lp.solve(verbose=False).evaluate_objective(np.array([1]))

lp.is_minimization = False
upper = lp.solve(verbose=False).evaluate_objective(np.array([1]))

print(f"Bounds: [{lower:.4f}, {upper:.4f}]")
```

### Validate Constraint System
```python
from symbolic_bounds.test_constraints import validate_constraints
validate_constraints(dag, verbose=True)
```

### Print LP Details
```python
lp.print_lp(show_full_matrices=True)
lp.print_objective()
lp.print_decision_variables()
```

---

## Quick Answers to Common Questions

### Q: What partition should I use for node X?
**A**: 
- Use `'R'` if you can/will intervene on X or if X is an outcome
- Use `'L'` if X is purely observational (e.g., unmeasured confounder)
- Rule: For query P(Y | do(X)), both X and Y must be in W_R

### Q: How many response types will my node have?
**A**: For node W with parents Pa(W):
```
# No parents
|response_types_W| = |support_W|

# Has parents
|response_types_W| = |support_W|^(Π_{P ∈ Pa(W)} |support_P|)
```

Examples:
- Binary node, no parents: 2 response types
- Binary node, one binary parent: 4 response types
- Binary node, two binary parents: 16 response types
- Ternary node, one ternary parent: 27 response types

### Q: What's the dimension of the decision variable q?
**A**: ℵᴿ = Π_{W ∈ W_R} |response_types_W|

Example: W_R = {X, Y} with X binary (no parents), Y binary (parent X):
- X: 2 response types
- Y: 4 response types  
- ℵᴿ = 2 × 4 = 8

### Q: How many constraints will the LP have?
**A**: B = Π_{W ∈ W_L ∪ W_R} |support_W|

Example: W_L = {Z}, W_R = {X, Y}, all binary:
- B = 2 × 2 × 2 = 8 constraints

### Q: Why are my bounds [0, 1]?
**A**: Two common reasons:
1. **No confounding**: Causal effect is fully identified from data
   - Bounds should collapse to point estimate
   - Check if you have confounders in W_L
2. **No constraints**: DAG structure doesn't restrict the query
   - Add more observed variables or edges
   - Check partition assignment

### Q: Why do I get an error "X must be subset of W_R"?
**A**: You're trying to intervene on a node in W_L:
```python
# WRONG
X = dag.add_node('X', partition='L')
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, ...)  # Error!

# CORRECT
X = dag.add_node('X', partition='R')
```

### Q: Do I need to call `generate_all_response_types()`?
**A**: Yes, always! Do it after building the DAG, before creating DataGenerator:
```python
dag = DAG()
# ... add nodes and edges ...
dag.generate_all_response_types()  # REQUIRED
generator = DataGenerator(dag)
```

### Q: Can I have cycles in my DAG?
**A**: No, it must be acyclic. The code doesn't check this automatically, so ensure:
- No node is its own ancestor
- Topological ordering exists

### Q: How do I model an unmeasured confounder?
**A**: Two approaches:

**Approach 1**: Implicit (no explicit U node)
```python
dag = DAG()
X = dag.add_node('X', partition='R')
Y = dag.add_node('Y', partition='R')
dag.add_edge(X, Y)
# Confounding handled through response type distribution
```

**Approach 2**: Explicit (add U to W_L)
```python
dag = DAG()
U = dag.add_node('U', partition='L')  # Unmeasured
X = dag.add_node('X', partition='R')
Y = dag.add_node('Y', partition='R')
dag.add_edge(U, X)
dag.add_edge(U, Y)
dag.add_edge(X, Y)
```

### Q: Can I intervene on multiple nodes simultaneously?
**A**: Yes, just include them all in X:
```python
lp = ProgramFactory.write_LP(
    scm,
    Y={Y1, Y2},      # Multiple outcomes
    X={X1, X2},      # Multiple interventions
    Y_values=(1, 0), # Y1=1, Y2=0
    X_values=(1, 1)  # X1=1, X2=1 (both set to 1)
)
```

### Q: What's the difference between P and P*?
**A**: 
- **P**: Constraint matrix (compatibility indicators, 0s and 1s)
- **P***: Weighted version used in Algorithm 1 (includes marginal probabilities)
- For solving LPs, use P with observed probabilities as RHS

### Q: Can I specify a custom data distribution?
**A**: Yes, bypass DataGenerator:
```python
# Option 1: Use DataGenerator but with fixed seed
generator = DataGenerator(dag, seed=42)

# Option 2: Manually specify distribution (advanced)
# Create your own joint distribution dict
custom_dist = {
    frozenset({(X, 0), (Y, 0)}): 0.25,
    frozenset({(X, 0), (Y, 1)}): 0.25,
    frozenset({(X, 1), (Y, 0)}): 0.25,
    frozenset({(X, 1), (Y, 1)}): 0.25
}
# Pass to ProgramFactory directly
```

### Q: How do I compute Average Treatment Effect (ATE)?
**A**: Solve two LPs and subtract:
```python
# P(Y=1 | do(X=1))
lp1 = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))
lp1.is_minimization = True
p1_lower = lp1.solve().evaluate_objective(np.array([1]))
lp1.is_minimization = False
p1_upper = lp1.solve().evaluate_objective(np.array([1]))

# P(Y=1 | do(X=0))
lp0 = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(0,))
lp0.is_minimization = True
p0_lower = lp0.solve().evaluate_objective(np.array([1]))
lp0.is_minimization = False
p0_upper = lp0.solve().evaluate_objective(np.array([1]))

# ATE = P(Y=1|do(X=1)) - P(Y=1|do(X=0))
ate_lower = p1_lower - p0_upper
ate_upper = p1_upper - p0_lower
```

---

## Error Messages and Solutions

### `ValueError: Node X has no response types`
**Solution**: Call `dag.generate_all_response_types()` before creating DataGenerator

### `ValueError: Y must be subset of W_R`
**Solution**: Change partition when creating node: `dag.add_node('Y', partition='R')`

### `KeyError` in response type mapping
**Solution**: Parent configuration doesn't exist in response type. Check:
1. Parent nodes are correctly added
2. Parent support matches usage
3. Response types generated after all edges added

### `LinAlgError: Singular matrix`
**Solution**: Constraint matrix is rank-deficient. This is handled automatically by `solve()` using PPOPT's `process_constraints()`. If error persists:
1. Check for duplicate constraints
2. Verify DAG is acyclic
3. Check observed distribution sums to 1.0

### LP solver returns "infeasible"
**Solution**: Observed distribution is incompatible with DAG structure. Check:
1. Observed probabilities sum to 1.0
2. No negative probabilities
3. DAG structure matches data generation

---

## Performance Tips

### Limit Problem Size
- Binary variables: max ~6-8 nodes
- Ternary variables: max ~4-5 nodes
- Use binary_only=True when generating random DAGs

### Reuse Constraints
```python
# Generate once
constraints = ProgramFactory.write_constraints(dag)

# Reuse for multiple queries
for query in queries:
    lp = ProgramFactory.write_LP(...)  # Uses cached constraints
```

### Parallelize Bound Computation
```python
# Can solve min and max independently
from concurrent.futures import ThreadPoolExecutor

def solve_min(lp):
    lp.is_minimization = True
    return lp.solve().evaluate_objective(np.array([1]))

def solve_max(lp):
    lp.is_minimization = False
    return lp.solve().evaluate_objective(np.array([1]))

with ThreadPoolExecutor() as executor:
    future_min = executor.submit(solve_min, lp)
    future_max = executor.submit(solve_max, lp)
    lower = future_min.result()
    upper = future_max.result()
```

---

## Debugging Checklist

When something doesn't work:

1. ✅ Called `dag.generate_all_response_types()`?
2. ✅ All intervention/query nodes in W_R partition?
3. ✅ DAG is acyclic?
4. ✅ Node support contains only natural numbers?
5. ✅ Observed distribution sums to 1.0?
6. ✅ Number of values matches number of nodes in Y and X?
7. ✅ Using correct numpy array format for solve: `np.array([1])`?

---

## File Organization

```
symbolic_bounds/
├── dag.py                  # Start here: DAG creation
├── node.py                 # Rarely need to import directly
├── response_type.py        # Rarely need to import directly
├── data_generator.py       # Step 2: Create data
├── scm.py                  # Step 3: Combine DAG + data
├── program_factory.py      # Step 4: Create LP
├── linear_program.py       # Step 5: Solve LP
└── constraints.py          # Used internally by program_factory
```

**Typical import**:
```python
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import numpy as np
```

---

## Version Information

**Current Implementation Status**:
- ✅ Binary and multi-valued discrete variables
- ✅ Arbitrary DAG structures with W_L/W_R partition
- ✅ Interventional queries P(Y | do(X))
- ✅ LP solving with automatic redundancy removal
- ✅ Validation against paper Example 6.1
- ⚠️ Parametric LP solving (work in progress)
- ❌ Mediation queries (not yet implemented)
- ❌ Selection bias models (not yet implemented)

**Dependencies**:
- NumPy >= 1.20
- PPOPT (from git: symbolic_bounds/ppopt_repo/PPOPT)
- Matplotlib, NetworkX (for visualization only)

**Python Version**: 3.8+
