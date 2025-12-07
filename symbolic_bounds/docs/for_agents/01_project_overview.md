# Project Overview: Symbolic Bounds for Causal Effects

## Purpose

This codebase implements algorithms from the paper **"A General Method for Deriving Tight Symbolic Bounds on Causal Effects"** to compute tight bounds on causal effects from observational and interventional data. The implementation focuses on **symbolic optimization** - computing bounds that hold for all possible data distributions consistent with a given causal structure.

## Research Context

### The Core Problem

In causal inference, we often want to estimate causal effects like P(Y | do(X=x)) (the effect of intervention X=x on outcome Y), but:
- We may only have observational data P(X, Y)
- Unmeasured confounders may exist
- Direct randomized experiments may be infeasible

### The Solution Approach

The paper provides a method to compute **tight symbolic bounds** on causal effects by:
1. Representing the causal structure as a DAG (Directed Acyclic Graph)
2. Partitioning variables into W_L (observed) and W_R (can be intervened on)
3. Enumerating all possible **response types** (ways variables respond to their parents)
4. Formulating a **linear program (LP)** where:
   - Decision variables q represent probabilities over response type configurations
   - Constraints P q = p encode the observed distribution
   - Objective α^T q represents the causal query of interest

## Key Innovation: Response Types

**Response types** are the central abstraction. For a node W with parents Pa(W) and support S:
- A response type r_W is a function mapping each parent configuration to an output value in S
- Example: If Y has parent X with X ∈ {0,1} and Y ∈ {0,1}, one response type might be:
  - r_Y: {(X=0)→1, (X=1)→0}  (Y responds oppositely to X)
- The number of response types is |S|^(Π|Pa_i|)

## Repository Structure

```
symbolic_bounds/
├── dag.py                    # DAG structure with W_L/W_R partition
├── node.py                   # Individual nodes in the DAG
├── response_type.py          # Response type representations
├── data_generator.py         # Generate causally consistent distributions
├── joint_distribution.py     # Probability distribution container
├── scm.py                    # Structural Causal Model (DAG + data)
├── constraints.py            # Constraint system from Algorithm 1
├── program_factory.py        # Generates LPs from DAGs (Algorithm 1 & 2)
├── linear_program.py         # LP representation and solving
└── docs/for_agents/          # This documentation
```

## Workflow

The typical usage follows this sequence:

1. **Define Causal Structure**: Create a DAG with nodes and edges
2. **Generate Response Types**: Enumerate all possible response functions
3. **Generate Data**: Create a causally consistent distribution
4. **Build SCM**: Combine DAG + data generator
5. **Create LP**: Use ProgramFactory to build the linear program
6. **Solve**: Compute bounds by minimizing/maximizing the LP

## Key Algorithms Implemented

### Algorithm 1: Constraint Generation
- Input: DAG with partition (W_L, W_R)
- Output: System of linear equations relating:
  - p* (joint probabilities P(W_L, W_R))
  - p (conditional probabilities P(W_R | W_L))
  - q (response type probabilities)
- Implemented in: `ProgramFactory.write_constraints()`

### Algorithm 2: Objective Function for Interventional Queries
- Input: Causal query P(Y=y | do(X=x))
- Output: Coefficient vector α where α^T q = P(Y=y | do(X=x))
- Implemented in: `ProgramFactory.writeRung2()`

## Current Implementation Status

### ✅ Fully Implemented
- DAG construction with W_L/W_R partitioning
- Response type generation for all node configurations
- Data generation with causally consistent distributions
- Constraint system generation (Algorithm 1)
- Objective function generation for interventional queries (Algorithm 2)
- LP solving using PPOPT library with automatic redundancy removal
- Validation against paper examples (e.g., Example 6.1)

### ⚠️ Work in Progress
- Parametric LP solving (when objective depends on parameter θ)
- Experimental design optimization
- Additional paper examples (6.2, 6.3)

## Dependencies

- **NumPy**: Matrix operations, numerical computation
- **PPOPT**: Parametric optimization library for LP solving
- **Matplotlib/NetworkX**: DAG visualization
- **itertools**: Combinatorial enumeration of response types

## Validation

The implementation has been validated against:
- Example 6.1 from paper (confounded X→Y with ternary X, binary Y)
- All three risk difference bounds match paper's closed-form formula to machine precision
- Automated constraint validation on randomly generated DAGs

## Important Conventions

1. **Variable Naming matches the paper**:
   - ℵᴿ (aleph_R): Number of response type combinations for W_R
   - B: Number of joint configurations (W_L, W_R)
   - γ (gamma): Index for response type combination
   - r_γ: A specific response type combination

2. **Partitioning Rules**:
   - W_L: Variables we only observe (observational data)
   - W_R: Variables we can intervene on (interventional queries)
   - For interventional queries P(Y | do(X=x)), both X and Y must be in W_R

3. **Matrix Dimensions**:
   - q: dimension ℵᴿ (decision variable)
   - P: B × ℵᴿ (constraint matrix)
   - p: dimension B (observed probabilities)
   - α: dimension ℵᴿ (objective coefficients)

## Next Steps for Development

When extending this codebase:
1. Study the paper sections 5-6 for theoretical foundations
2. Review existing test files for usage patterns
3. Validate new features against paper examples
4. Use `test_constraints.py` for automated validation
