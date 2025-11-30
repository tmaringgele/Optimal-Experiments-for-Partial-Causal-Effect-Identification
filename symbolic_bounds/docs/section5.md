

# 5. Optimization via Vertex Enumeration



After applying Algorithms 1 and 2, we have a linear objective
and a system of linear constraints. We also have the probabilistic
constraints:

```
∀ wL ∈ ν(WL),     ∑_{wR ∈ ν(WR)} p{WR = wR | WL = wL} = 1
```

and

```
∑_{γ = 1}^{ℵR} qγ = ∑_{γ=1}^{ℵR} p{RR = rγ} = ∑_{rR ∈ ν(RR)} p{RR = rR} = 1.
```

Additional linear constraints on **q** can be optionally given as
**B q ≥ d** where **B** and **d** are respectively a matrix and vector
of real constants. These constraints can be used to encode
assumptions about the response functions that are not possible
to encode in a DAG, for example, restricting the probabilities of
implausible response patterns. We thus arrive at the following
linear programming problem for the lower bound; the upper
bound is given by the corresponding maximization problem.

```
minimize      Q = αᵀ q
subject to    P q = p,
              B q ≥ d,
              q ≥ 0,   and   1ᵀ q = 1.
```

Note that the constraint space constitutes a bounded (due
to the probabilistic constraints) convex polytope. By the fun-
damental theorem of linear programming, the global extrema
must occur at one of the vertices of the polytope. We can
thus solve this problem symbolically by applying an efficient
vertex enumeration algorithm, such as the double description
algorithm (Motzkin et al. 1953; Fukuda 2018) to enumerate the
vertices of the polytope of the dual linear program. For instance,
the dual of the minimization problem above is given by

```
maximize      [ dᵀ   1   pᵀ ] y
subject to    [ Bᵀ   1   Pᵀ
                I    0      ] y ≤ [ α
                                       0 ].
```

So by the strong duality theorem, the optimum of the dual, and
thus also of the primal problem, is of the form

```
[ dᵀ   1   pᵀ ]  ȳ
```

where **ȳ** is a vertex of the polytope

```
{ y :  [ Bᵀ   1   Pᵀ
         I    0 ]  y  ≤  [ α
                               0 ] }.
```

This gives a lower bound on the causal effect of interest as
the maximum of a set of expressions involving only observable
probabilities. Similarly, the upper bound is given by reversing
the dual inequality and minimizing over the corresponding
polytope.

**Proposition 4.** Under Conditions 1–6 and subject to any addi-
tional linear constraints of the form **B q ≥ d**, the procedure
above yields valid and tight symbolic bounds for a causal query
that is a linear combination of atomic queries.

**Corollary 2.** If Condition 4 does not hold, then the bounds
derived using the above procedure are still valid.

See the supplementary materials for proof. The Conditions 3
and 4 represent a worst-case scenario of confounding and ensure
that the decompositions giving rise to the linear constraints can-
not be further factorized to yield more granular but nonlinear
constraints. If however there is any known (partial) absence of
such confounding, then these bounds are still valid, and may be
narrow enough to be informative, while not necessarily tight.
Such an absence of confounding on the R-side implies some
independence among the RR variables, and hence additional
constraints on their distribution. Thus, the true feasible space
may be smaller than the one considered in our algorithm, but
completely contained inside it.
