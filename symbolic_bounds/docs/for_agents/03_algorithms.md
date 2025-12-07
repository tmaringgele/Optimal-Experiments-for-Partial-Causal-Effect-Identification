# Algorithm Implementation Details

This document explains how the paper's algorithms are implemented in code.

## Algorithm 1: Constraint Generation

**Paper Reference**: Section 5, Algorithm 1  
**Implementation**: `ProgramFactory.write_constraints(dag)` in `program_factory.py`

### Purpose
Generate a system of linear equations that relates:
- **p\*** = P(W_L, W_R): Joint probabilities (observable)
- **p** = P(W_R | W_L): Conditional probabilities (observable given W_L)
- **q**: Response type probabilities (unobservable, what we optimize over)

### Mathematical Framework

The constraint system has form:
```
P * q = p*              (joint constraints)
P_Λ * q = p             (conditional constraints for each W_L configuration)
q ≥ 0, 1^T q = 1        (probability constraints)
```

### Algorithm 1 Pseudocode (from paper)

```
Input: Causal graph G with vertex partition (W_L, W_R)
Output: Systems of linear equations relating p* and p to q

1. For each vertex in G, enumerate all response types
2. Initialize P as a B × ℵᴿ matrix of 0s
3. Initialize P* as a B × ℵᴿ matrix of 0s
4. Initialize Λ as a B × B matrix of 0s

5. for b ∈ {1, …, B} do                    # For each configuration
6.     for γ ∈ {1, …, ℵᴿ} do               # For each response type combo
7.         Initialize ω as empty vector
8.         for i ∈ R do                     # R indexes W_R nodes
9.             Set ωᵢ := gᵂⁱ(w_{b,L}, r_γ) # Simulate W_R value
10.        end
11.        if ω = w_{b,R} then              # Check compatibility
12.            P_{b,γ} := 1
13.            Λ_{b,b} := p{W_L = w_{b,L}}
14.            P*_{b,γ} := p{W_L = w_{b,L}}
15.        end
16.    end
17. end
```

### Implementation Details

#### Step 1: Response Type Enumeration
```python
# In dag.py: generate_all_response_types()
for node in all_nodes:
    parents = self.get_parents(node)
    if not parents:
        # No parents: |support| constant functions
        for value in node.support:
            rt = ResponseType(node)
            rt.mapping[()] = value  # Empty parent config
            node.response_types.append(rt)
    else:
        # Has parents: enumerate all functions
        parent_configs = product(*[p.support for p in parents])
        output_assignments = product(node.support, repeat=len(parent_configs))
        
        for output_values in output_assignments:
            rt = ResponseType(node)
            for parent_config, output in zip(parent_configs, output_values):
                rt.mapping[parent_config] = output
            node.response_types.append(rt)
```

**Key Points**:
- For node W with parents Pa(W):
  - Number of parent configurations: Π_{P ∈ Pa(W)} |support_P|
  - Number of response types: |support_W|^(# parent configs)
- Response types represent ALL possible deterministic functions

#### Step 2-4: Matrix Initialization

```python
# In program_factory.py: _generate_joint_constraints()
B = len(all_configs)  # Number of (W_L, W_R) configurations
aleph_R = len(w_r_response_type_combinations)

P = np.zeros((B, aleph_R))
P_star = np.zeros((B, aleph_R))
Lambda = np.zeros((B, B))
```

**Dimensions**:
- B = Π_{W ∈ W_L ∪ W_R} |support_W|
- ℵᴿ = Π_{W ∈ W_R} |response_types_W|
- Note: q only includes response types for W_R, not W_L

#### Step 5-17: Compatibility Checking (Core Loop)

```python
# In program_factory.py: _is_compatible_wr()
def _is_compatible_wr(dag, all_nodes, all_response_types,
                      w_r_nodes, r_gamma, config):
    """
    Check if response type combination r_γ produces configuration config.
    
    This implements: ωᵢ := gᵂⁱ(w_{b,L}, r_γ) and checks ω = w_{b,R}
    """
    config_map = dict(config)  # (W_L ∪ W_R) values
    
    for node, rt in zip(w_r_nodes, r_gamma):
        target_value = config_map[node]  # Desired w_{b,R} value
        parents = dag.get_parents(node)
        
        if not parents:
            # No parents: ωᵢ = constant from response type
            if rt.get(()) != target_value:
                return False
        else:
            # Build parent configuration from (w_{b,L}, w_{b,R})
            parent_config = tuple((p, config_map[p]) for p in parents)
            
            # Simulate: ωᵢ := gᵂⁱ(parent values, r_γ)
            if rt.get(parent_config) != target_value:
                return False
    
    return True  # All W_R nodes match: ω = w_{b,R}
```

**What Compatibility Means**:
- Given W_L values w_{b,L} and response types r_γ
- Simulate what W_R values would be produced
- If simulated W_R matches actual w_{b,R}, they're compatible
- Only compatible (r_γ, configuration) pairs get P_{b,γ} = 1

#### Conditional Constraints

```python
# For each W_L configuration, generate P_Λ matrix
for w_l_values in w_l_configs:
    P_lambda, p_lambda = _generate_conditional_constraints(
        dag, ..., w_l_values, ...
    )
    constraints.Lambda[condition_name] = P_lambda
    constraints.p_Lambda[condition_name] = p_lambda
```

**Purpose**: Encode conditional probabilities P(W_R | W_L = w_l)
- One matrix per distinct W_L configuration
- Same compatibility check, but fixed W_L values

---

## Algorithm 2: Objective Function for Interventional Queries

**Paper Reference**: Section 5, Algorithm 2  
**Implementation**: `ProgramFactory.writeRung2()` in `program_factory.py`

### Purpose
Construct coefficient vector α such that α^T q = P(Y=y | do(X=x))

### Mathematical Framework

For interventional query P(Y=y | do(X=x)):
```
P(Y=y | do(X=x)) = Σ_{γ: r_γ compatible with (X=x, Y=y)} q_γ
                 = α^T q
```

where α_γ = 1 if r_γ is compatible, 0 otherwise.

### Algorithm 2 Logic (adapted from paper)

```
Input: 
  - DAG G with partition (W_L, W_R)
  - Target nodes Y ⊆ W_R, target values y
  - Intervention nodes X ⊆ W_R, intervention values x
  
Output: Coefficient vector α where α^T q = P(Y=y | do(X=x))

1. Initialize α as ℵᴿ-dimensional zero vector
2. for γ ∈ {1, …, ℵᴿ} do
3.     compatible := TRUE
4.     for each W_L configuration w_l do
5.         Set X = x (intervention)
6.         Simulate W_R \ X values using r_γ and w_l
7.         if simulated Y ≠ y then
8.             compatible := FALSE
9.         end
10.    end
11.    if compatible then α_γ := 1
12. end
```

### Implementation Details

#### Validation
```python
# Check query is well-formed
if not Y.issubset(dag.W_R):
    raise ValueError("Y must be subset of W_R")
if not X.issubset(dag.W_R):
    raise ValueError("X must be subset of W_R")
if not Y.isdisjoint(X):
    raise ValueError("Y and X must be disjoint")
```

**Critical Rule**: For interventional queries, X and Y must both be in W_R
- W_L variables cannot be intervened on
- W_L variables cannot be query targets for do() queries

#### Core Loop: Checking Compatibility

```python
for gamma, r_gamma in enumerate(w_r_response_type_combinations):
    compatible_for_all = True
    
    # Check across ALL W_L configurations
    for w_l_config_values in w_l_configs:
        # 1. Set intervention: X = x
        simulated_values = {}
        for node in X_nodes:
            simulated_values[node] = X_config[node]
        
        # 2. Simulate W_R \ X in topological order
        topo_order = _topological_sort_wr(dag, w_r_nodes)
        for node in topo_order:
            if node in X_nodes:
                continue  # Already set by intervention
            
            rt = rt_map[node]  # Response type for this node
            parents = dag.get_parents(node)
            
            # Build parent configuration from W_L + already-simulated W_R
            parent_config = tuple(
                (p, w_l_config[p] if p in w_l_nodes else simulated_values[p])
                for p in parents
            )
            
            # Simulate: node_value := response_function(parent_config)
            simulated_values[node] = rt.get(parent_config)
        
        # 3. Check if Y = y
        for y_node, y_target in Y_target.items():
            if simulated_values[y_node] != y_target:
                compatible_for_all = False
                break
    
    if compatible_for_all:
        alpha[gamma] = 1.0
```

**Key Insight**: Under do(X=x), we override X's response types
- X values are set to x regardless of r_γ
- Other W_R nodes use their response types from r_γ
- Must check across all W_L configurations (marginalizing over W_L)

#### Topological Sorting

```python
def _topological_sort_wr(dag, w_r_nodes):
    """
    Sort W_R nodes in topological order for simulation.
    Needed because child nodes depend on parent values.
    """
    # Implementation uses depth-first search
    # Returns: List[Node] in topological order
```

**Why Needed**: When simulating, must compute parent values before child values

---

## ProgramFactory: Main Entry Point

`ProgramFactory` combines both algorithms to create complete LinearProgram objects.

### Method: `write_constraints(dag)`

**Returns**: Constraints object
- Implements Algorithm 1
- Independent of specific query
- Generated once per DAG structure

### Method: `write_LP(scm, Y, X, Y_values, X_values)`

**Returns**: LinearProgram object ready to solve

**Steps**:
1. Generate constraints using Algorithm 1
2. Generate objective using Algorithm 2
3. Extract observed distribution from SCM
4. Combine into LinearProgram

**Full Implementation Flow**:
```python
# 1. Get constraint system
constraints = ProgramFactory.write_constraints(scm.dag)

# 2. Get observed distribution
observed_joint = scm.getObservedJoint()

# 3. Build constraint matrix P and RHS p
P = constraints.P
p = _extract_joint_probabilities(observed_joint, constraints.joint_prob_index)

# 4. Build objective function α
alpha = ProgramFactory.writeRung2(scm.dag, Y, X, Y_values, X_values)

# 5. Create LinearProgram
lp = LinearProgram(
    objective=alpha,
    constraint_matrix=P,
    rhs=p,
    q_labels=constraints.q_labels,
    ...
)
```

---

## Solving the Linear Program

**Implementation**: `LinearProgram.solve()` in `linear_program.py`

### Approach: PPOPT Library

Uses PPOPT (Parametric Programming Optimization Toolbox):
```python
from ppopt.mplp_program import MPLP_Program
from ppopt.solver import Solver

# Convert to PPOPT format
prog = MPLP_Program(A, b, c, H, CRa, CRb, F)

# Automatic redundancy removal
prog.process_constraints()

# Solve LP
solver = Solver(solvers={'lp': 'glpk'})
result = prog.solver.solve_lp(c, A, b, C, d)
```

**Key Features**:
- Automatic detection and removal of redundant constraints
- Handles equality constraints by converting to inequality pairs
- Robust numerical solving using GLPK backend

### Redundancy Removal

**Problem**: Constraint matrices often rank-deficient
- Example: 32 constraints, but only 20 are linearly independent
- Redundant constraints cause numerical issues

**Solution**: `process_constraints()` method
- Performs QR decomposition
- Removes linearly dependent rows
- Preserves solution space

### Computing Bounds

To get tight bounds on P(Y=y | do(X=x)):
```python
# Lower bound
lp.is_minimization = True
result_lb = lp.solve()
lb = result_lb.evaluate_objective(np.array([1]))

# Upper bound
lp.is_minimization = False
result_ub = lp.solve()
ub = result_ub.evaluate_objective(np.array([1]))
```

**Why Two Solves**: 
- min α^T q gives lower bound
- max α^T q gives upper bound
- Together they form the identification region [lb, ub]

---

## Validation and Testing

### Constraint Validation (`test_constraints.py`)

Automated validation checks:
1. **Matrix Dimensions**: Verify B, ℵᴿ computed correctly
2. **Compatibility Matrix**: Check P_{b,γ} entries match simulation
3. **Probability Constraints**: Verify each row has correct # of non-zero entries
4. **Conditional Consistency**: Check Λ matrices consistent with P

### Paper Example Validation (`test_section6_1.py`)

Validates against Example 6.1 from paper:
- Confounded X→Y model
- X ternary {0,1,2}, Y binary {0,1}
- Computes bounds for three risk differences
- Verifies bounds match paper's closed-form formula

**Result**: All three bounds match to machine precision (<1e-15 error)

---

## Common Pitfalls and Debugging

### Pitfall 1: Wrong Partition Assignment
```python
# WRONG: X in W_L but trying to intervene
X = dag.add_node('X', partition='L')
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, ...)  # Error!

# CORRECT: X must be in W_R for interventions
X = dag.add_node('X', partition='R')
```

### Pitfall 2: Forgetting Response Type Generation
```python
dag = DAG()
# ... add nodes and edges ...
# WRONG: directly create DataGenerator
generator = DataGenerator(dag)  # Error: no response types!

# CORRECT: generate response types first
dag.generate_all_response_types()
generator = DataGenerator(dag)
```

### Pitfall 3: Dimension Mismatch in Values
```python
# WRONG: Number of values doesn't match nodes
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X1, X2}, 
                             Y_values=(1,), X_values=(1,))  # Error!

# CORRECT: One value per node
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X1, X2}, 
                             Y_values=(1,), X_values=(1, 0))  # X1=1, X2=0
```

### Debugging Tools

**Print LP Structure**:
```python
lp.print_lp(show_full_matrices=True)
lp.print_decision_variables()
lp.print_objective()
```

**Print Constraints**:
```python
constraints = ProgramFactory.write_constraints(dag)
constraints.print_constraints(show_matrices=False, explicit_equations=True)
```

**Validate Constraints**:
```python
from symbolic_bounds.test_constraints import validate_constraints
validate_constraints(dag, verbose=True)
```
