# Linear Programming Formulation for Causal Effect Bounds

## Overview

This document explains the symbolic linear programming formulation used for computing bounds on causal effects through vertex enumeration.

## Problem Formulation

We construct a linear program of the form:

```
maximize/minimize:  c^T q
subject to:        A q = b
                   q >= 0
                   sum(q) = 1
```

Where:
- **q**: Decision variable (dimension ℵᴿ) representing probabilities over response type combinations
- **c**: Objective function vector from Algorithm 2 (represents causal query like P(Y=y | do(X=x)))
- **A**: Constraint matrix from Algorithm 1 (rows = constraints, cols = ℵᴿ)
- **b**: Right-hand side vector with **symbolic parameters** representing P(W_R | W_L)

## Key Components

### 1. Decision Variable q

The decision variable **q** is a probability distribution over response type combinations for W_R nodes:

- Dimension: ℵᴿ = |supp(R_R)| = ∏_{i ∈ R} |supp(R_i)|
- Each q[γ] represents P(r_γ), the probability of response type combination γ
- Constraints:
  - q >= 0 (non-negativity)
  - sum(q) = 1 (normalization)

**Example**: For X, Y ∈ W_R with binary support:
- If X has 2 response types and Y has 4 response types
- Then ℵᴿ = 2 × 4 = 8
- q is an 8-dimensional probability vector

### 2. Objective Vector c

The objective vector **c** is constructed by Algorithm 2 (writeRung2) to represent a causal query:

```python
c = ProgramFactory.writeRung2(dag, Y_nodes, X_nodes, Y_values, X_values)
```

- Dimension: ℵᴿ (same as q)
- c[γ] = 1 if response type r_γ produces Y=y under intervention do(X=x)
- c[γ] = 0 otherwise
- The objective c^T q = ∑_γ c[γ] q[γ] = P(Y=y | do(X=x))

**Interpretation**: Maximizing c^T q finds the upper bound on P(Y=y | do(X=x)), minimizing finds the lower bound.

### 3. Constraint Matrix A

The constraint matrix **A** comes from Algorithm 1 (write_constraints):

```python
constraints = ProgramFactory.write_constraints(dag)
A = constraints.P  # or stack of constraints.Lambda matrices
```

- Dimension: (# constraints) × ℵᴿ
- Each row corresponds to one equality constraint
- A[i,γ] = 1 if configuration i is compatible with response type γ
- A[i,γ] = 0 otherwise

The constraints relate q to observable conditional probabilities P(W_R | W_L).

### 4. Symbolic RHS Vector b

The right-hand side vector **b** contains **symbolic parameters** representing conditional probabilities:

- Dimension: (# constraints)
- Each entry b[i] is a symbolic parameter like "p_X=0,Y=1|Z=0"
- These parameters represent P(W_R = w_r | W_L = w_l) for specific configurations

**Example for Z -> X -> Y**:
```
b = [p_X=0,Y=0|Z=0,
     p_X=0,Y=1|Z=0,
     p_X=1,Y=0|Z=0,
     p_X=1,Y=1|Z=0,
     p_X=0,Y=0|Z=1,
     p_X=0,Y=1|Z=1,
     p_X=1,Y=0|Z=1,
     p_X=1,Y=1|Z=1]
```

## Why Symbolic Parameters?

### Key Insight: Parametric Bounds

By keeping parameters **symbolic** rather than substituting numerical values, we can:

1. **Compute bounds as functions**: The optimal value c^T q becomes a function of the parameters
2. **Vertex enumeration**: Each vertex of the polytope gives a symbolic bound expression
3. **Experimental design**: Identify which parameters most affect bound tightness
4. **Sensitivity analysis**: See how bounds change with observational distributions

### Comparison

**Traditional approach** (substitute numbers first):
```
Given: P(X,Y|Z=0) = [0.2, 0.3, 0.1, 0.4]
       P(X,Y|Z=1) = [0.1, 0.2, 0.3, 0.4]
Solve LP → Get: P(Y=1|do(X=1)) ∈ [0.35, 0.72]
```

**Symbolic approach** (our implementation):
```
Keep: b = [p₀, p₁, p₂, p₃, p₄, p₅, p₆, p₇] as symbols
Enumerate vertices → Get: 
  Upper bound = max{f₁(p), f₂(p), ..., fₖ(p)}
  Lower bound = min{g₁(p), g₂(p), ..., gₘ(p)}
where fᵢ, gⱼ are linear functions of parameters
```

## Constructing the LP

### Step 1: Create DAG and Generate Response Types

```python
from symbolic_bounds import ProgramFactory
from symbolic_bounds.dag import DAG

# Create DAG
dag = DAG()
Z = dag.add_node('Z', support={0, 1}, partition='L')
X = dag.add_node('X', support={0, 1}, partition='R')
Y = dag.add_node('Y', support={0, 1}, partition='R')
dag.add_edge(Z, X)
dag.add_edge(X, Y)
dag.generate_all_response_types()
```

### Step 2: Build Complete LP

```python
# Build LP for P(Y=1 | do(X=1))
lp = ProgramFactory.build_lp(
    dag, 
    objective_Y={Y},      # Outcome nodes
    intervention_X={X},   # Intervention nodes
    Y_values=(1,),        # Target Y value
    X_values=(1,),        # Intervention X value
    sense='max'           # Maximize for upper bound
)
```

### Step 3: Inspect the LP

```python
# Print LP structure
lp.print_lp(show_full_matrix=False)

# Access components
print(f"Decision variables: {lp.aleph_R}")
print(f"Constraints: {lp.n_constraints}")
print(f"Symbolic parameters: {lp.n_params}")

# Get feasible region info
info = lp.get_feasible_region_info()
print(f"Expected polytope dimension: {info['expected_polytope_dimension']}")
```

### Step 4: Evaluate with Concrete Values (Optional)

```python
# For testing or specific distribution
param_values = {
    'p_X=0,Y=0|Z=0': 0.2,
    'p_X=0,Y=1|Z=0': 0.3,
    # ... etc
}

rhs_numeric = lp.evaluate_rhs(param_values)
# Now solve numerically: max c^T q s.t. A q = rhs_numeric, q >= 0, sum(q) = 1
```

## Mathematical Properties

### Feasible Region

The feasible region is:
```
F = {q ∈ ℝ^ℵᴿ : A q = b, q >= 0, sum(q) = 1}
```

This is the intersection of:
1. The probability simplex: {q >= 0, sum(q) = 1}
2. Affine subspace defined by: A q = b

### Polytope Dimension

Expected dimension = ℵᴿ - rank(A) - 1

The "-1" comes from the normalization constraint sum(q) = 1.

### Vertices and Bounds

- The feasible region F is a polytope (bounded polyhedron)
- The optimal value is attained at a vertex of F
- Each vertex gives a candidate bound value
- Upper bound = max over all vertices
- Lower bound = min over all vertices

## Vertex Enumeration (Next Step)

For symbolic vertex enumeration, we need to:

1. **Find all vertices** of the polytope F parametrically
   - Each vertex is expressed in terms of parameters b
   - Typically: q_vertex = f(b) for some function f

2. **Evaluate objective** at each vertex
   - Bound_vertex = c^T q_vertex = c^T f(b)
   - This is a linear function of parameters b

3. **Take envelope**
   - Upper bound = max{c^T f₁(b), c^T f₂(b), ...}
   - Lower bound = min{c^T g₁(b), c^T g₂(b), ...}

4. **Result**: Bounds as piecewise-linear functions of parameters

## Example: Simple Chain

Consider Z -> X -> Y where all variables are binary.

```
Decision variables: q has dimension 8 (2 × 4 response types)
Constraints: 8 equations (4 for Z=0, 4 for Z=1)
Parameters: 8 conditional probabilities P(X,Y|Z)

For query P(Y=1 | do(X=1)):
- Objective: c has 4 ones (response types producing Y=1 when X=1)
- Feasible region: 9-dimensional polytope in 8-dimensional space
- Vertices: Enumerate to get symbolic bounds
```

Expected bound form:
```
Upper bound = max{
    p₁ + p₃ + p₅ + p₇,
    p₃ + p₇,
    ...
}
```

## Implementation Details

### SymbolicParameter Class

Represents a single parameter in the LP:

```python
@dataclass
class SymbolicParameter:
    name: str                              # e.g., "p_X=0,Y=1|Z=0"
    w_r_config: Tuple[Tuple[Node, int], ...]  # (X, 0), (Y, 1)
    w_l_config: Tuple[Tuple[Node, int], ...]  # (Z, 0)
    index: int                             # Position in parameter vector
```

### LinearProgram Class

Main container for the LP:

```python
class LinearProgram:
    objective: np.ndarray           # c vector (length ℵᴿ)
    constraint_matrix: np.ndarray   # A matrix
    rhs_symbolic: List[str]         # Symbolic RHS expressions
    rhs_params: List[SymbolicParameter]  # Parameter objects
    param_dict: Dict[str, SymbolicParameter]
    
    # Methods
    def set_objective(objective, sense)
    def add_constraints_from_matrix(A, b_symbolic, labels)
    def register_parameter(param)
    def evaluate_rhs(param_values)  # For testing
    def print_lp(show_full_matrix)
    def get_feasible_region_info()
```

## Next Steps

To complete the symbolic bound computation:

1. **Implement vertex enumeration algorithm**
   - Use parametric polytope methods
   - Or: Double description method with symbolic arithmetic
   - Or: Reverse search for vertex enumeration

2. **Symbolic optimization**
   - For each vertex v(b), compute c^T v(b)
   - Take max/min over vertices to get bounds

3. **Experimental design**
   - Analyze bound width as function of parameters
   - Identify experiments that tighten bounds
   - Optimize allocation of experimental budget

## References

- Algorithm 1: System of linear equations (paper Section X)
- Algorithm 2: Objective function construction (paper Section Y)
- Vertex enumeration: See polytope computation literature
- Parametric linear programming: See Gal & Greenberg (1997)
