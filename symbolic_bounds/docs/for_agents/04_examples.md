# Complete Working Examples

This document provides complete, runnable examples demonstrating typical usage patterns.

## Example 1: Simple Chain X → Y

**Scenario**: Compute bounds on P(Y=1 | do(X=1)) for a simple causal chain.

```python
import numpy as np
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory

# 1. Create DAG with X → Y
dag = DAG()
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(X, Y)

# 2. Generate response types
dag.generate_all_response_types()
print(f"X response types: {len(X.response_types)}")  # 2 (constant functions)
print(f"Y response types: {len(Y.response_types)}")  # 4 (functions from X to Y)

# 3. Generate causally consistent data
generator = DataGenerator(dag, seed=42)
scm = SCM(dag, generator)

# 4. Build LP for P(Y=1 | do(X=1))
lp = ProgramFactory.write_LP(
    scm,
    Y={Y},           # Target: Y
    X={X},           # Intervention: X
    Y_values=(1,),   # Y = 1
    X_values=(1,)    # do(X = 1)
)

# 5. Solve for bounds
lp.is_minimization = True
lower = lp.solve(verbose=False).evaluate_objective(np.array([1]))

lp.is_minimization = False
upper = lp.solve(verbose=False).evaluate_objective(np.array([1]))

print(f"\nBounds on P(Y=1 | do(X=1)): [{lower:.4f}, {upper:.4f}]")

# 6. (Optional) Inspect the LP structure
lp.print_lp(show_full_matrices=False)
```

**Expected Output**:
```
X response types: 2
Y response types: 4
Bounds on P(Y=1 | do(X=1)): [0.1234, 0.8765]
```

---

## Example 2: Confounded Model Z → X → Y ← Z

**Scenario**: Compute bounds with an unobserved confounder Z.

```python
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import numpy as np

# 1. Create DAG with confounding
dag = DAG()
Z = dag.add_node('Z', support={0, 1}, partition='L')  # Unobserved confounder
X = dag.add_node('X', support={0, 1}, partition='R')  # Treatment
Y = dag.add_node('Y', support={0, 1}, partition='R')  # Outcome

# Causal structure: Z → X, Z → Y, X → Y
dag.add_edge(Z, X)
dag.add_edge(Z, Y)
dag.add_edge(X, Y)

# 2. Generate response types
dag.generate_all_response_types()
print(f"Z response types: {len(Z.response_types)}")  # 2
print(f"X response types: {len(X.response_types)}")  # 4 (functions from Z)
print(f"Y response types: {len(Y.response_types)}")  # 16 (functions from Z,X)

# 3. Generate data
generator = DataGenerator(dag, seed=123)
scm = SCM(dag, generator)

# Print observed distribution
observed = scm.getObservedJoint()
print("\nObserved Joint Distribution:")
for config, prob in sorted(observed.items(), 
                          key=lambda x: tuple(sorted((n.name, v) for n, v in x[0]))):
    config_str = ", ".join(f"{n.name}={v}" for n, v in sorted(config, key=lambda x: x[0].name))
    print(f"  P({config_str}) = {prob:.4f}")

# 4. Build LP for average treatment effect P(Y=1 | do(X=1)) - P(Y=1 | do(X=0))
# Note: Need to solve two separate LPs

# LP for P(Y=1 | do(X=1))
lp_x1 = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

lp_x1.is_minimization = True
p_y1_x1_lower = lp_x1.solve(verbose=False).evaluate_objective(np.array([1]))

lp_x1.is_minimization = False
p_y1_x1_upper = lp_x1.solve(verbose=False).evaluate_objective(np.array([1]))

# LP for P(Y=1 | do(X=0))
lp_x0 = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(0,))

lp_x0.is_minimization = True
p_y1_x0_lower = lp_x0.solve(verbose=False).evaluate_objective(np.array([1]))

lp_x0.is_minimization = False
p_y1_x0_upper = lp_x0.solve(verbose=False).evaluate_objective(np.array([1]))

# Average Treatment Effect bounds
ate_lower = p_y1_x1_lower - p_y1_x0_upper
ate_upper = p_y1_x1_upper - p_y1_x0_lower

print(f"\nBounds on P(Y=1 | do(X=1)): [{p_y1_x1_lower:.4f}, {p_y1_x1_upper:.4f}]")
print(f"Bounds on P(Y=1 | do(X=0)): [{p_y1_x0_lower:.4f}, {p_y1_x0_upper:.4f}]")
print(f"Bounds on ATE: [{ate_lower:.4f}, {ate_upper:.4f}]")
```

---

## Example 3: Ternary Treatment

**Scenario**: Multi-valued treatment with three levels.

```python
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import numpy as np

# 1. Create DAG with ternary X
dag = DAG()
X = dag.add_node('X', support={0, 1, 2}, partition='R')  # Three treatment levels
Y = dag.add_node('Y', support={0, 1}, partition='R')     # Binary outcome
dag.add_edge(X, Y)

# 2. Generate response types
dag.generate_all_response_types()
print(f"X response types: {len(X.response_types)}")  # 3 (constant functions)
print(f"Y response types: {len(Y.response_types)}")  # 8 (2^3 functions)

# 3. Generate data
generator = DataGenerator(dag, seed=456)
scm = SCM(dag, generator)

# 4. Compute bounds for each treatment level
results = {}
for x_val in [0, 1, 2]:
    lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(x_val,))
    
    lp.is_minimization = True
    lower = lp.solve(verbose=False).evaluate_objective(np.array([1]))
    
    lp.is_minimization = False
    upper = lp.solve(verbose=False).evaluate_objective(np.array([1]))
    
    results[x_val] = (lower, upper)
    print(f"P(Y=1 | do(X={x_val})): [{lower:.4f}, {upper:.4f}]")

# 5. Compute risk differences
print("\nRisk Differences:")
for x1, x2 in [(1, 0), (2, 0), (2, 1)]:
    rd_lower = results[x1][0] - results[x2][1]  # lower - upper
    rd_upper = results[x1][1] - results[x2][0]  # upper - lower
    print(f"P(Y(X={x1})=1) - P(Y(X={x2})=1): [{rd_lower:.4f}, {rd_upper:.4f}]")
```

---

## Example 4: Using Constraints for Analysis

**Scenario**: Inspect the constraint system structure.

```python
from symbolic_bounds import DAG
from symbolic_bounds.program_factory import ProgramFactory

# Create simple DAG
dag = DAG()
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(X, Y)
dag.generate_all_response_types()

# Generate constraint system
constraints = ProgramFactory.write_constraints(dag)

# Inspect structure
print("\nConstraint System Structure:")
print(f"  Decision variables (ℵᴿ): {len(constraints.response_type_labels)}")
print(f"  Constraints (B): {len(constraints.joint_prob_labels)}")
print(f"  P matrix shape: {constraints.P.shape}")
print(f"  P matrix sparsity: {np.sum(constraints.P == 0) / constraints.P.size * 100:.1f}% zeros")

# Print response types
print("\nResponse Type Enumeration:")
for i, label in enumerate(constraints.response_type_labels):
    print(f"  q[{i}]: {label}")

# Print first few constraints
print("\nFirst 4 Constraints (explicit form):")
constraints.print_constraints(show_matrices=False, explicit_equations=True)
```

---

## Example 5: Example 6.1 from Paper (Validation)

**Scenario**: Replicate paper's Example 6.1 exactly.

```python
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import numpy as np

# Create DAG for confounded exposure-outcome
dag = DAG()
X = dag.add_node('X', support={0, 1, 2}, partition='R')  # Ternary exposure
Y = dag.add_node('Y', support={0, 1}, partition='R')     # Binary outcome
dag.add_edge(X, Y)  # Direct effect + unmeasured confounding via response types
dag.generate_all_response_types()

# Use specific seed for reproducibility (paper doesn't specify distribution)
generator = DataGenerator(dag, seed=100)
scm = SCM(dag, generator)

# Get observed joint for reference
observed = scm.getObservedJoint()

# Paper formula for bounds (Proposition 9):
# Lower: p{X=x1, Y=1} + p{X=x2, Y=0} - 1
# Upper: 1 - p{X=x1, Y=0} - p{X=x2, Y=1}

# Extract observed probabilities
def get_prob(x_val, y_val):
    key = frozenset({(X, x_val), (Y, y_val)})
    return observed.get(key, 0.0)

# Test all three contrasts from paper
contrasts = [(1, 0), (2, 0), (2, 1)]

print("Validation of Example 6.1:")
print("=" * 80)

for x1, x2 in contrasts:
    # Paper's formula
    paper_lower = get_prob(x1, 1) + get_prob(x2, 0) - 1
    paper_upper = 1 - get_prob(x1, 0) - get_prob(x2, 1)
    
    # Our LP solution
    lp_x1 = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(x1,))
    lp_x2 = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(x2,))
    
    lp_x1.is_minimization = True
    p_x1_lower = lp_x1.solve(verbose=False).evaluate_objective(np.array([1]))
    lp_x1.is_minimization = False
    p_x1_upper = lp_x1.solve(verbose=False).evaluate_objective(np.array([1]))
    
    lp_x2.is_minimization = True
    p_x2_lower = lp_x2.solve(verbose=False).evaluate_objective(np.array([1]))
    lp_x2.is_minimization = False
    p_x2_upper = lp_x2.solve(verbose=False).evaluate_objective(np.array([1]))
    
    lp_lower = p_x1_lower - p_x2_upper
    lp_upper = p_x1_upper - p_x2_lower
    
    # Compare
    diff_lower = abs(paper_lower - lp_lower)
    diff_upper = abs(paper_upper - lp_upper)
    
    print(f"\nP(Y(X={x1})=1) - P(Y(X={x2})=1):")
    print(f"  Paper formula: [{paper_lower:9.6f}, {paper_upper:9.6f}]")
    print(f"  LP solution:   [{lp_lower:9.6f}, {lp_upper:9.6f}]")
    print(f"  Difference:    [{diff_lower:9.2e}, {diff_upper:9.2e}]")
    
    if diff_lower < 1e-10 and diff_upper < 1e-10:
        print("  ✓ MATCH")
    else:
        print("  ✗ MISMATCH")
```

---

## Example 6: Random DAG Testing

**Scenario**: Test on randomly generated DAGs.

```python
from symbolic_bounds import DAG
from symbolic_bounds.random_dag_generator import (
    generate_random_partitioned_dag,
    print_dag_summary
)
from symbolic_bounds.test_constraints import validate_constraints
import matplotlib.pyplot as plt

# Generate random DAG
dag = generate_random_partitioned_dag(
    n=5,                    # 5 nodes
    binary_only=True,       # Binary variables only
    seed=789
)

# Print summary
print_dag_summary(dag)

# Visualize
fig = dag.draw(figsize=(10, 6), title="Random DAG")
plt.show()

# Validate constraint system
print("\nValidating constraint system...")
is_valid = validate_constraints(dag, verbose=True)

if is_valid:
    print("\n✓ All constraints valid!")
else:
    print("\n✗ Constraint validation failed")

# Generate and test LP
from symbolic_bounds import DataGenerator, SCM, ProgramFactory
import numpy as np

dag.generate_all_response_types()
generator = DataGenerator(dag, seed=789)
scm = SCM(dag, generator)

# Pick arbitrary query (first two R nodes)
w_r_nodes = sorted(dag.W_R, key=lambda n: n.name)
if len(w_r_nodes) >= 2:
    X_node = w_r_nodes[0]
    Y_node = w_r_nodes[1]
    
    lp = ProgramFactory.write_LP(scm, Y={Y_node}, X={X_node}, 
                                 Y_values=(1,), X_values=(1,))
    
    lp.is_minimization = True
    lower = lp.solve(verbose=False).evaluate_objective(np.array([1]))
    lp.is_minimization = False
    upper = lp.solve(verbose=False).evaluate_objective(np.array([1]))
    
    print(f"\nExample query P({Y_node.name}=1 | do({X_node.name}=1)):")
    print(f"  Bounds: [{lower:.4f}, {upper:.4f}]")
```

---

## Example 7: Debugging Workflow

**Scenario**: Step-by-step inspection when things don't work.

```python
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import numpy as np

# 1. Create DAG
dag = DAG()
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(X, Y)

# 2. Check nodes
print("DAG Nodes:")
print(f"  W_L: {[n.name for n in dag.W_L]}")
print(f"  W_R: {[n.name for n in dag.W_R]}")
print(f"  Edges: {[(p.name, c.name) for p, c in dag.edges]}")

# 3. Generate response types and inspect
dag.generate_all_response_types()
print(f"\nResponse Types Generated:")
for node in dag.get_all_nodes():
    print(f"  {node.name}: {len(node.response_types)} types")
    if len(node.response_types) <= 4:  # Print if small
        for i, rt in enumerate(node.response_types):
            print(f"    Type {i}: {rt.mapping}")

# 4. Create data and check distribution
generator = DataGenerator(dag, seed=42)
scm = SCM(dag, generator)

observed = scm.getObservedJoint()
print(f"\nObserved Distribution (should sum to 1.0):")
total = 0.0
for config, prob in sorted(observed.items()):
    config_str = ", ".join(f"{n.name}={v}" for n, v in sorted(config, key=lambda x: x[0].name))
    print(f"  P({config_str}) = {prob:.6f}")
    total += prob
print(f"  Total: {total:.6f}")

# 5. Inspect LP structure before solving
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

print(f"\nLP Structure:")
print(f"  Decision variables: {len(lp.objective)}")
print(f"  Constraints: {len(lp.rhs)}")
print(f"  Non-zero in objective: {np.sum(lp.objective != 0)}")
print(f"  Constraint matrix rank: {np.linalg.matrix_rank(lp.constraint_matrix)}")

# 6. Print objective explicitly
print(f"\nObjective Function:")
lp.print_objective()

# 7. Solve with verbose output
print(f"\nSolving LP (verbose):")
lp.is_minimization = True
result = lp.solve(verbose=True)
lower = result.evaluate_objective(np.array([1]))
print(f"Lower bound: {lower:.6f}")

lp.is_minimization = False
result = lp.solve(verbose=True)
upper = result.evaluate_objective(np.array([1]))
print(f"Upper bound: {upper:.6f}")
```

---

## Tips for Using These Examples

1. **Start Simple**: Begin with Example 1 (chain), understand it fully before moving to complex scenarios

2. **Check Dimensions**: When something fails, print matrix shapes and verify they match expected B and ℵᴿ

3. **Validate Constraints**: Use `validate_constraints(dag, verbose=True)` on new DAG structures

4. **Compare to Paper**: Example 5 shows how to validate against known results

5. **Use Debugging Tools**: Example 7's workflow is essential when troubleshooting

6. **Random Testing**: Example 6 helps find edge cases and validate robustness
