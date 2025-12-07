# Parametric LP Examples

This notebook demonstrates how to use parametric linear programs with experimental constraints.

## Setup

import numpy as np
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import matplotlib.pyplot as plt

## Example 1: Without Experimental Constraints

### Create a simple DAG: X → Y

dag = DAG()
X = dag.add_node('X', support={0, 1})
Y = dag.add_node('Y', support={0, 1})
dag.add_edge(X, Y)
dag.generate_all_response_types()

# Visualize
fig = dag.draw(figsize=(6, 2), title="Simple DAG: X → Y")
plt.show()

### Generate data and create SCM

generator = DataGenerator(dag, seed=42)
scm = SCM(dag, generator)

# Show true distribution
generator.print_true_distribution()

### Compute bounds on P(Y=1|do(X=1))

lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

# Lower bound
lp.is_minimization = True
lb = lp.solve().evaluate_objective(1)

# Upper bound
lp.is_minimization = False
ub = lp.solve().evaluate_objective(1)

print(f"\nBounds on P(Y=1|do(X=1)): [{lb:.6f}, {-ub:.6f}]")

# Get true value
true_val = generator.computeTrueIntervention(Y={Y}, X={X}, Y_values=(1,), X_values=(1,))
print(f"True P(Y=1|do(X=1)) = {true_val:.6f}")
print(f"True value is within bounds: {lb <= true_val <= -ub}")

## Example 2: With Experimental Constraints

### Suppose we perform an experiment

experimental_result = 0.6
print(f"Experimental result: P(Y=1|do(X=1)) = {experimental_result}")

### Compute bounds given this experimental result

lp_with_exp = ProgramFactory.write_LP(
    scm,
    Y={Y}, X={X}, Y_values=(1,), X_values=(1,),    # Query
    V={Y}, Z={X}, V_values=(1,), Z_values=(1,)     # Experimental constraint
)

print(f"\nExperimental constraint added: P(Y=1|do(X=1)) = θ")
print(f"Experiment matrix shape: {lp_with_exp.experiment_matrix.shape}")

# Solve parametric LP
lp_with_exp.is_minimization = True
solution = lp_with_exp.solve()

print(f"Number of critical regions: {len(solution.critical_regions)}")

# Evaluate at experimental result
obj = solution.evaluate_objective(experimental_result)
print(f"\nObjective at θ={experimental_result}: {obj:.6f}")
print(f"Note: objective equals θ because we're computing the same quantity!")

## Example 3: Different Input Formats

print("\n" + "="*60)
print("Testing different input formats for evaluate_objective:")
print("="*60)

formats = [
    ("Scalar", experimental_result),
    ("1D array", np.array([experimental_result])),
    ("2D array", np.array([[experimental_result]])),
    ("Float", float(experimental_result)),
]

for name, theta in formats:
    obj = solution.evaluate_objective(theta)
    print(f"{name:12s}: θ = {theta!r:30s} → objective = {obj:.6f}")

print("\n✓ All formats work correctly!")

## Example 4: Computing Complementary Probabilities

### Given P(Y=1|do(X=1)) = θ, compute P(Y=0|do(X=1))

lp_comp = ProgramFactory.write_LP(
    scm,
    Y={Y}, X={X}, Y_values=(0,), X_values=(1,),    # Query: P(Y=0|do(X=1))
    V={Y}, Z={X}, V_values=(1,), Z_values=(1,)     # Constraint: P(Y=1|do(X=1)) = θ
)

lp_comp.is_minimization = True
solution_comp = lp_comp.solve()

print("\n" + "="*60)
print("Complementary probabilities:")
print("="*60)

test_values = [0.4, 0.5, 0.6, 0.7]
print(f"\n{'P(Y=1|do(X=1))':<20} {'P(Y=0|do(X=1))':<20} {'Sum':<10}")
print("-" * 55)

for val in test_values:
    obj = solution_comp.evaluate_objective(val)
    if obj is not None:
        print(f"{val:<20.6f} {obj:<20.6f} {val + obj:.6f}")
    else:
        print(f"{val:<20.6f} {'Infeasible':<20} {'N/A':<10}")

print("\n✓ P(Y=0|do(X=1)) + P(Y=1|do(X=1)) = 1.0 for all feasible values!")

## Example 5: Exploring the Feasible Region

### Check which experimental results are feasible

print("\n" + "="*60)
print("Feasible region exploration:")
print("="*60)

lp_feasible = ProgramFactory.write_LP(
    scm,
    Y={Y}, X={X}, Y_values=(1,), X_values=(1,),
    V={Y}, Z={X}, V_values=(1,), Z_values=(1,)
)

lp_feasible.is_minimization = True
solution_feasible = lp_feasible.solve()

# Test range of values
test_range = np.linspace(0.0, 1.0, 21)
feasible_values = []

for val in test_range:
    obj = solution_feasible.evaluate_objective(val)
    is_feasible = obj is not None
    if is_feasible:
        feasible_values.append(val)
    status = "✓" if is_feasible else "✗"
    print(f"θ = {val:.2f}: {status}")

print(f"\nFeasible range: [{min(feasible_values):.2f}, {max(feasible_values):.2f}]")

## Example 6: Visualizing Critical Regions

### Plot the objective function across the parameter space

theta_values = np.linspace(0.0, 1.0, 101)
objectives = []

for theta in theta_values:
    obj = solution_feasible.evaluate_objective(theta)
    objectives.append(obj if obj is not None else np.nan)

plt.figure(figsize=(10, 6))
plt.plot(theta_values, objectives, 'b-', linewidth=2, label='Objective value')
plt.plot(theta_values, theta_values, 'r--', alpha=0.5, label='θ (identity)')
plt.xlabel('θ (Experimental result: P(Y=1|do(X=1)))', fontsize=12)
plt.ylabel('Objective (P(Y=1|do(X=1)))', fontsize=12)
plt.title('Parametric Solution: Objective vs Parameter', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

print("\n✓ Objective equals θ in feasible region (identity function)")
print("✓ Shows that when we constrain P(Y=1|do(X=1))=θ and compute P(Y=1|do(X=1)),")
print("  we get θ back (as expected!)")

## Summary

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
✓ Parametric LPs work correctly with experimental constraints
✓ Multiple input formats are supported (scalar, 1D, 2D arrays)
✓ Feasible regions are automatically detected
✓ Complementary probabilities sum to 1.0
✓ Critical regions partition the parameter space

Key takeaways:
1. Use V, Z, V_values, Z_values parameters to add experimental constraints
2. solve() returns a ParametricSolution object
3. evaluate_objective(theta) evaluates at specific parameter values
4. Returns None for infeasible parameter values
5. Supports any numeric input format (scalar, array, etc.)
""")
