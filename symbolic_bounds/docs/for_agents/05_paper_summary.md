# Paper Summary and Theoretical Background

This document summarizes the key theoretical concepts from the paper that motivate the implementation.

## Paper Information

**Title**: "A General Method for Deriving Tight Symbolic Bounds on Causal Effects"

**Core Contribution**: A systematic method to compute tight bounds on causal effects by:
1. Formulating causal inference as linear programming over response type probabilities
2. Providing algorithms that work for arbitrary DAG structures
3. Proving the bounds are tight (i.e., cannot be improved without additional assumptions)

## Key Theoretical Concepts

### 1. Structural Causal Models (SCMs)

An SCM consists of:
- **Endogenous variables** V = W_L ∪ W_R (observed/intervened)
- **Exogenous variables** U (unobserved)
- **Structural equations** V_i := f_i(Pa(V_i), U)

**Key Insight**: Instead of specifying unknown functions f_i explicitly, we enumerate all possible functions (response types) and optimize over their probability distribution.

### 2. Response Types (R)

For each variable V_i with parents Pa(V_i):
- A **response type** r_i is a deterministic function: Pa(V_i) → domain(V_i)
- The set of all response types R_i has size |domain(V_i)|^|domain(Pa(V_i))|
- A **response type configuration** r is a tuple (r_1, ..., r_n) specifying one response type per variable

**Example**: For binary Y with binary parent X:
- Response types for Y: {constant-0, constant-1, copy-X, flip-X}
- Each represents a different causal mechanism

### 3. Identification and Bounds

**Perfect Identification**: When P(Y | do(X=x)) has a unique value determined by the observed distribution P(V)

**Partial Identification**: When we can only bound the causal effect:
```
L(P) ≤ P(Y | do(X=x)) ≤ U(P)
```

**Tight Bounds**: Bounds that cannot be improved:
- For every P compatible with the DAG, the true causal effect lies in [L(P), U(P)]
- For every value θ ∈ [L(P), U(P)], there exists a compatible SCM where P(Y | do(X=x)) = θ

### 4. Partition (W_L, W_R)

The paper introduces a key structural assumption:

**Definition (Proposition 2)**: Variables are partitioned:
- **W_L** (Left): Variables we only observe
- **W_R** (Right): Variables we can intervene on

**Properties**:
- No edges from W_R to W_L (all edges point right or within partitions)
- We have observational data: P(W_L, W_R)
- We have conditional experimental data: P(W_R | W_L)
- Queries are of form: P(Y | do(X)) where Y, X ⊆ W_R

**Practical Meaning**:
- W_L can include unmeasured confounders (modeled through response types)
- W_R includes treatment and outcome variables
- This structure is less restrictive than requiring a fully specified causal ordering

### 5. Linear Programming Formulation

**Theorem 1 (Main Result)**: The causal query P(Y=y | do(X=x)) can be written as:
```
P(Y=y | do(X=x)) = α^T q
```
where:
- q is a probability distribution over response types: q_γ = P(R_R = r_γ)
- R_R represents response types for W_R variables only
- α is a coefficient vector determined by the query
- q must satisfy: P q = p (observational constraints)

**This transforms causal inference to LP**:
```
minimize/maximize    α^T q
subject to           P q = p
                     q ≥ 0
                     1^T q = 1
```

### 6. Algorithm 1: Constraint Matrix Construction

**Purpose**: Generate matrix P and vector p such that P q = p encodes observational constraints.

**Key Equation**: P(W_L = w_l, W_R = w_r) = Σ_{γ: g(r_γ, w_l) = w_r} q_γ

Where:
- g(r_γ, w_l) simulates W_R values given W_L = w_l and response types r_γ
- Sum is over compatible response type configurations

**Matrix Structure**:
- Rows: One per observable configuration (w_l, w_r)
- Columns: One per response type configuration r_γ
- Entry P_{b,γ} = 1 if r_γ is compatible with configuration b, else 0
- **Sparse**: Typically only 10-30% non-zero entries

### 7. Algorithm 2: Objective Function Construction

**Purpose**: Generate vector α such that α^T q = P(Y=y | do(X=x)).

**Key Equation**: Under intervention do(X=x):
```
P(Y=y | do(X=x)) = Σ_{γ: r_γ produces Y=y when X=x} q_γ
```

**Algorithm**: 
1. For each response type configuration r_γ:
2. Simulate: Set X = x, use r_γ to compute other W_R values
3. If Y = y under simulation (for all W_L), set α_γ = 1
4. Otherwise, set α_γ = 0

**Result**: α^T q counts probability mass of response types compatible with the query.

### 8. Tightness and Soundness

**Theorem 2 (Soundness)**: For any SCM compatible with the DAG and data:
```
L(P) ≤ P(Y | do(X=x)) ≤ U(P)
```

**Theorem 3 (Tightness)**: The bounds are tight:
- For every θ ∈ [L(P), U(P)], there exists an SCM with:
  - P_SCM(W_L, W_R) = P(W_L, W_R) (matches data)
  - P_SCM(Y | do(X=x)) = θ (achieves the value)

**Practical Implication**: Cannot improve bounds without:
- Additional parametric assumptions (e.g., monotonicity)
- Additional measurements (e.g., measuring W_L variables)
- Additional experiments (e.g., interventions on other variables)

## Section 6: Examples from Paper

### Example 6.1: Confounded Exposure and Outcome

**Setup**:
- X is ternary exposure: {0, 1, 2}
- Y is binary outcome: {0, 1}
- U is unmeasured confounder (modeled via response types)
- DAG: X ← U → Y (X and Y confounded)

**Query**: Risk differences P(Y(X=x₁)=1) - P(Y(X=x₂)=1)

**Closed-form Bounds (Proposition 9)**:
```
p{X=x₁, Y=1} + p{X=x₂, Y=0} - 1
  ≤ p{Y(X=x₁)=1} - p{Y(X=x₂)=1} ≤
1 - p{X=x₁, Y=0} - p{X=x₂, Y=1}
```

**Implementation Status**: ✅ Validated (see `test_section6_1.py`)
- All three contrasts (1,0), (2,0), (2,1) match formula
- Numerical precision: < 1e-15 error

### Example 6.2: Mediation Analysis

**Setup**:
- X → M → Y (X affects Y through mediator M)
- W is observed covariate
- Query: Natural direct/indirect effects

**Implementation Status**: ⚠️ Not yet implemented
- Requires extending Algorithm 2 for mediation queries
- Need counterfactual nesting: Y(X=x, M(X=x'))

### Example 6.3: Selection Bias

**Setup**:
- X → Y with sample selection S
- S depends on X and Y: X → S ← Y
- Only observe (X, Y) when S = 1

**Implementation Status**: ⚠️ Not yet implemented
- Requires conditioning on selection variable
- Need to model selection mechanism in constraints

## Comparison to Other Methods

### vs. Do-Calculus (Pearl)
- **Do-calculus**: Symbolic manipulation to derive identifiability
- **This method**: Numerical bounds when identification fails
- **Advantage**: Works when do-calculus returns "not identifiable"

### vs. Instrumental Variables
- **IV**: Requires finding valid instrument
- **This method**: Works with arbitrary confounding structure
- **Advantage**: No need for instrument assumptions

### vs. Sensitivity Analysis
- **Sensitivity**: Varies parameter (e.g., confounding strength)
- **This method**: Worst-case bounds over all possible confounding
- **Advantage**: No need to specify confounding parameter

### vs. Balke-Pearl Bounds
- **Balke-Pearl**: Linear programming for binary variables with IV
- **This method**: Generalizes to arbitrary DAGs, multi-valued variables
- **Advantage**: More general structure

## Important Theoretical Limitations

### 1. Computational Complexity
- Number of response types grows exponentially: O(|domain|^|parents|)
- For node with k binary parents: 2^(2^k) response types
- Practical limit: ~6-8 binary nodes or ~4-5 ternary nodes

### 2. Parametric Information Not Used
- Method ignores functional form assumptions
- E.g., "Y increases monotonically in X" could tighten bounds
- Trade-off: Generality vs. efficiency

### 3. Identifiability Conditions
- When bounds collapse to point [θ, θ], effect is identified
- No automatic check if simpler identification formula exists
- May solve LP when closed-form exists (inefficient but correct)

### 4. Experimental Design
- Method assumes data P(W_L, W_R) is given
- Doesn't optimize which experiments to run
- Extension: Use bounds to decide optimal experiments (future work)

## Extensions and Future Directions

### Implemented
- ✅ Binary and multi-valued variables
- ✅ Arbitrary DAG structures (with W_L, W_R partition)
- ✅ Interventional queries P(Y | do(X))
- ✅ Automatic constraint generation (Algorithm 1)
- ✅ Automatic objective generation (Algorithm 2)

### In Progress
- ⚠️ Parametric LP solving (θ-dependent bounds)
- ⚠️ Experimental design optimization

### Future Work
- ❌ Mediation queries (natural direct/indirect effects)
- ❌ Selection bias / sample selection models
- ❌ Incorporating parametric assumptions (monotonicity, additivity)
- ❌ Continuous variables (requires discretization or integration)
- ❌ High-dimensional settings (dimensionality reduction)

## Key Takeaways for Implementation

1. **Response types are central**: Everything reduces to optimization over P(R)

2. **Sparsity is key**: Matrices are sparse; exploit this for efficiency

3. **Validation is essential**: Compare to known examples (Section 6) to verify correctness

4. **Partitioning matters**: Proper W_L/W_R assignment is critical for valid queries

5. **LP solving is standard**: Once formulated, any LP solver works (we use PPOPT/GLPK)

6. **Tightness guarantees soundness**: Bounds are conservative but never wrong

7. **Computational limits exist**: Exponential blowup limits problem size

## References to Paper Sections

When reading the paper:
- **Section 2**: Background on SCMs and causal inference
- **Section 3**: Response types and linear programming formulation
- **Section 4**: Partition structure and observational constraints
- **Section 5**: Algorithms 1 and 2 (implemented here)
- **Section 6**: Worked examples and applications
- **Section 7**: Experimental design extensions
- **Appendix**: Proofs of tightness and soundness

## Mathematical Notation Mapping

| Paper | Code | Meaning |
|-------|------|---------|
| W_L | `dag.W_L` | Left partition (observational) |
| W_R | `dag.W_R` | Right partition (interventional) |
| r_γ | `q[gamma]` | Response type configuration γ |
| ℵᴿ | `aleph_R` | Number of response type combinations |
| B | `B` | Number of observable configurations |
| p* | `p_star` | Joint probabilities P(W_L, W_R) |
| p | `p` | Observed probabilities |
| q | `q` | Decision variable (response type probabilities) |
| α | `alpha` | Objective coefficients |
| P | `P` | Constraint matrix |
| Λ | `Lambda` | Marginal probability matrix |
| g^{W_i} | `rt.get(parent_config)` | Response function for W_i |
| ω | `simulated_values` | Simulated W_R values |
