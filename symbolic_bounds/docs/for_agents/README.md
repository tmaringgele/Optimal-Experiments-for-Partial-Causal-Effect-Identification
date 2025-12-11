# Documentation for LLM Code Assistants

This folder contains comprehensive documentation optimized for LLM code assistants (like GitHub Copilot, Claude, GPT-4) to quickly understand and work with this codebase.

## Purpose

When you (an LLM agent) are asked to help with this project, read these documents to get complete context about:
- The research problem and theoretical foundation
- Code architecture and design patterns
- How algorithms from the paper are implemented
- Common usage patterns and examples
- Debugging strategies

## How to Use This Documentation

### For New Tasks
1. **Start with**: `01_project_overview.md` - Get the big picture
2. **Then read**: `05_paper_summary.md` - Understand the theory
3. **Then read**: `02_core_classes.md` - Learn the data structures
4. **Finally**: `03_algorithms.md` - Understand the implementations

### For Specific Questions
- **"How do I...?"** → `06_quick_reference.md`
- **"What does class X do?"** → `02_core_classes.md`
- **"How is Algorithm Y implemented?"** → `03_algorithms.md`
- **"Show me an example of Z"** → `04_examples.md`
- **"What does the paper say about...?"** → `05_paper_summary.md`

### For Debugging
1. Check `06_quick_reference.md` → Error Messages section
2. Review `04_examples.md` → Example 7 (Debugging Workflow)
3. Consult `03_algorithms.md` → Common Pitfalls section

## Document Overview

### 01_project_overview.md
- Research context and motivation
- High-level architecture
- Workflow and key algorithms
- Current implementation status
- Dependencies and conventions

**When to read**: Always start here for new agents

### 02_core_classes.md
- Detailed class documentation
- Attributes, methods, and usage patterns
- Relationships between classes
- Matrix dimensions and computations

**When to read**: When working with specific classes or debugging type issues

### 03_algorithms.md
- Algorithm 1: Constraint generation (paper → code)
- Algorithm 2: Objective function construction
- ProgramFactory main entry points
- LP solving with PPOPT
- Validation and testing strategies

**When to read**: When implementing new features or understanding existing algorithms

### 04_examples.md
- 7 complete, runnable examples
- Progressive complexity (simple chain → random DAGs)
- Validation against paper examples
- Debugging workflows

**When to read**: When you need working code templates or debugging patterns

### 05_paper_summary.md
- Theoretical foundations
- Key concepts (SCMs, response types, identification)
- Mathematical formulation
- Paper examples and their status
- Comparison to other methods
- Notation mapping (paper ↔ code)

**When to read**: When the user asks about theory or paper-specific questions

### 06_quick_reference.md
- Quick lookup for common tasks
- FAQ with code snippets
- Error messages and solutions
- Performance tips
- Debugging checklist

**When to read**: For quick answers to specific questions

### 07_solving_linear_programs.md
- Complete guide to all three solving methods
- solve() using PPOPT (parametric, experimental constraints)
- solve_with_highs() using HiGHS (fast, stable, no experiments)
- solve_with_autobound() using autobound (interventions, CSV-based)
- Decision trees for choosing the right method
- Common patterns, troubleshooting, and best practices

**When to read**: When you need to solve an LP or encounter solver issues

## Key Concepts to Understand

Before working on this codebase, ensure you understand:

1. **Response Types**: Central abstraction - functions mapping parent values to node values
2. **Partition (W_L, W_R)**: W_L = observational, W_R = interventional
3. **Linear Programming**: Causal queries reduce to LP over response type probabilities
4. **Compatibility**: Response types must be compatible with observed configurations
5. **Tightness**: Computed bounds cannot be improved without additional assumptions

## Quick Start Template

For most tasks, this is the standard workflow:

```python
from symbolic_bounds import DAG, DataGenerator, SCM, ProgramFactory
import numpy as np

# 1. Build DAG
dag = DAG()
X = dag.add_node('X', {0, 1}, 'R')
Y = dag.add_node('Y', {0, 1}, 'R')
dag.add_edge(X, Y)

# 2. Generate response types (REQUIRED)
dag.generate_all_response_types()

# 3. Create data and SCM
generator = DataGenerator(dag, seed=42)
scm = SCM(dag, generator)

# 4. Build LP for query P(Y=1 | do(X=1))
lp = ProgramFactory.write_LP(scm, Y={Y}, X={X}, Y_values=(1,), X_values=(1,))

# 5. Solve for bounds
lp.is_minimization = True
lower = lp.solve(verbose=False).evaluate_objective(np.array([1]))

lp.is_minimization = False
upper = lp.solve(verbose=False).evaluate_objective(np.array([1]))

print(f"Bounds: [{lower:.4f}, {upper:.4f}]")
```

## Common Pitfalls to Avoid

1. **Forgetting response type generation**: Always call `dag.generate_all_response_types()`
2. **Wrong partition**: For interventional queries, nodes must be in W_R
3. **Dimension mismatch**: Number of values must match number of nodes
4. **Acyclicity**: DAG must be acyclic (not automatically validated)

## Testing and Validation

Always validate new DAG structures:
```python
from symbolic_bounds.test_constraints import validate_constraints
validate_constraints(dag, verbose=True)
```

Compare to paper examples when possible:
```python
# See test_section6_1.py for validation against Example 6.1
```

## Important Notes for LLM Agents

### What's Working
- ✅ Binary and multi-valued discrete variables
- ✅ Arbitrary DAG structures with W_L/W_R partition
- ✅ Interventional queries P(Y | do(X))
- ✅ Automatic constraint generation (Algorithm 1)
- ✅ Automatic objective generation (Algorithm 2)
- ✅ LP solving with automatic redundancy removal

### What's Not Working Yet
- ⚠️ Parametric LP solving (θ-dependent bounds) - still being developed
- ❌ Mediation queries (natural direct/indirect effects)
- ❌ Selection bias models

### What to Document
When the user asks about parametric solving or asks you to "document everything", note:
- The parametric solve() method is work in progress
- Don't document it as fully functional
- Focus on what works: single-point LP solving for specific causal queries

## Notation Quick Reference

| Paper | Code | Meaning |
|-------|------|---------|
| W_L | `dag.W_L` | Observational variables |
| W_R | `dag.W_R` | Interventional variables |
| r_γ | `q[gamma]` | Response type configuration |
| ℵᴿ | `aleph_R` | # response type combinations |
| B | `B` | # observable configurations |
| q | `q` | Decision variable |
| α | `alpha` | Objective coefficients |
| P | `P` | Constraint matrix |

## Getting Help

When stuck:
1. Check `06_quick_reference.md` → Debugging Checklist
2. Review relevant example in `04_examples.md`
3. Validate constraints: `validate_constraints(dag, verbose=True)`
4. Print LP structure: `lp.print_lp(show_full_matrices=True)`

## Contact and Context

This documentation was created to support LLM agents in understanding and extending the implementation of algorithms from the paper "A General Method for Deriving Tight Symbolic Bounds on Causal Effects". The codebase implements symbolic optimization for partial identification of causal effects using linear programming over response type probabilities.

Last Updated: December 2025
