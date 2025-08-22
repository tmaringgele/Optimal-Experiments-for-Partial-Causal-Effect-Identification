# ============================================================
# Bounds for P(Y=1 | do(X=1)) under DAG:
#   W->Z, Z->X, U->X, X->M, (X,M,W,U)->Y; W,U unobserved; Z,X,M,Y observed (binary)
#
# Two solvers:
#   1) bounds_PY_doX1(..., p_doZ=None, p_doM=None)
#      -> numeric lower/upper bounds (optional numeric experiments)
#
#   2) parametric_bounds_PY_doX1(..., use_doZ={True|False}, use_doM={True|False})
#      -> piecewise-linear functions (lists of affine pieces) in p_doZ / p_doM
#
# Dependencies: numpy, pulp
# ============================================================

from itertools import product
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pulp


# ---------------------------
# Response-type universe
# ---------------------------
# Compact response-function parameterization:
# t = (x0, x1, m0, m1, y00, y01, y10, y11) ∈ {0,1}^8
# Meaning:
#   X = x_z, for z∈{0,1}
#   M = m_x, for x∈{0,1}
#   Y = y_{m,x}, for (m,x)∈{0,1}^2
TYPE_SPACE = list(product([0, 1], repeat=8))


# ---------------------------
# Data helpers
# ---------------------------
def empirical_joint(Z, X, M, Y) -> Tuple[Dict[Tuple[int, int, int, int], float], Dict[int, float]]:
    """Return (pj, nZ), where
       pj[(z,x,m,y)] = empirical P(Z=z,X=x,M=m,Y=y),
       nZ[z] = empirical P(Z=z).
    """
    Z = np.asarray(Z).astype(int).ravel()
    X = np.asarray(X).astype(int).ravel()
    M = np.asarray(M).astype(int).ravel()
    Y = np.asarray(Y).astype(int).ravel()
    n = len(Z)
    if not (len(X) == len(M) == len(Y) == n):
        raise ValueError("Z,X,M,Y must have same length")

    counts = defaultdict(int)
    for z, x, m, y in zip(Z, X, M, Y):
        if z not in (0, 1) or x not in (0, 1) or m not in (0, 1) or y not in (0, 1):
            raise ValueError("Z,X,M,Y must be binary (0/1)")
        counts[(z, x, m, y)] += 1

    pj = {k: v / n for k, v in counts.items()}
    nZ = {z: sum(p for (zz, _, _, _), p in pj.items() if zz == z) for z in [0, 1]}
    # ensure missing cells are present with prob 0 for convenience
    for z, x, m, y in product([0, 1], repeat=4):
        pj.setdefault((z, x, m, y), 0.0)
    for z in [0, 1]:
        nZ.setdefault(z, 0.0)
    return pj, nZ


# ---------------------------
# Robust constraint helpers
# ---------------------------
def add_eq_with_tol(prob: pulp.LpProblem, lhs, rhs: float, name: str, tol: float = 1e-9):
    """Encode lhs == rhs as two inequalities with a small tolerance band."""
    prob += (lhs <= rhs + tol), f"{name}__le"
    prob += (lhs >= rhs - tol), f"{name}__ge"


def add_soft_experiment_constraint(
    prob: pulp.LpProblem,
    lhs,
    target: float,
    tag: str,
    penalty: float = 1e-6,
    tol: float = 1e-9,
):
    """Soft equality for experiments: lhs == target, with slacks s+ and s- and an L1 penalty.
       Returns (s_plus, s_minus, penalty_term) so you can include penalty in the objective.
    """
    s_plus = pulp.LpVariable(f"splus_{tag}", lowBound=0)
    s_minus = pulp.LpVariable(f"sminus_{tag}", lowBound=0)
    # Implement lhs - target = s_plus - s_minus (with tolerance on equality)
    add_eq_with_tol(prob, lhs - target, s_plus - s_minus, f"EXP_{tag}", tol=tol)
    penalty_term = penalty * (s_plus + s_minus)
    return s_plus, s_minus, penalty_term


# ---------------------------
# Core LP builder (numeric RHS)
# ---------------------------
def _build_numeric_lp(
    Z, X, M, Y,
    p_doZ: Optional[float],
    p_doM: Optional[float],
    sense: str = "max",
    tol: float = 1e-9,
    soft_experiments: bool = False,
    soft_penalty: float = 1e-6,
):
    """Build the LP for numeric experiments (or none).
       - Observational equalities are added with tolerance (two-sided band).
       - Experimental constraints can be exact equalities or soft (with small penalty).
    """
    pj, nZ = empirical_joint(Z, X, M, Y)
    prob = pulp.LpProblem("Bounds_doX1", pulp.LpMaximize if sense == "max" else pulp.LpMinimize)

    # Decision variables: p_{type, z} ≥ 0; their sums by z equal P(Z=z)
    p_vars = {(i, z): pulp.LpVariable(f"p_{i}_{z}", lowBound=0)
              for i, _ in enumerate(TYPE_SPACE) for z in [0, 1]}

    # P(Z=z) (two-sided with tolerance)
    for z in [0, 1]:
        add_eq_with_tol(prob, pulp.lpSum(p_vars[(i, z)] for i, _ in enumerate(TYPE_SPACE)), nZ[z], f"PZ_{z}", tol=tol)

    # Observational P(Z,X,M,Y)
    for z, x, m, y in product([0, 1], repeat=4):
        mask = []
        for i, t in enumerate(TYPE_SPACE):
            x0, x1, m0, m1, y00, y01, y10, y11 = t
            x_z = x1 if z == 1 else x0
            m_x = m1 if x == 1 else m0
            y_mx = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(m, x)]
            if x_z == x and m_x == m and y_mx == y:
                mask.append(p_vars[(i, z)])
        add_eq_with_tol(prob, pulp.lpSum(mask), pj[(z, x, m, y)], f"OBS_{z}{x}{m}{y}", tol=tol)

    # Objective target: P(Y=1 | do(X=1)) = Σ_t mass_t * y_{m1,1}
    obj_terms = []
    for i, t in enumerate(TYPE_SPACE):
        x0, x1, m0, m1, y00, y01, y10, y11 = t
        y_m1_1 = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(m1, 1)]
        obj_terms.append((p_vars[(i, 0)] + p_vars[(i, 1)]) * y_m1_1)
    target_expr = pulp.lpSum(obj_terms)

    # Optional experimental constraints (exact or soft)
    penalty_terms = []

    if p_doZ is not None:
        expr = []
        for i, t in enumerate(TYPE_SPACE):
            x0, x1, m0, m1, y00, y01, y10, y11 = t
            m_x1 = m1 if x1 == 1 else m0
            y_mx1 = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(m_x1, x1)]
            expr.append((p_vars[(i, 0)] + p_vars[(i, 1)]) * y_mx1)
        expr_doZ = pulp.lpSum(expr)
        if soft_experiments:
            _, _, pen = add_soft_experiment_constraint(prob, expr_doZ, float(p_doZ), "DOZ1", penalty=soft_penalty, tol=tol)
            penalty_terms.append(pen)
        else:
            # exact equality (single constraint; easier to reason about)
            prob += (expr_doZ == float(p_doZ)), "DOZ1"

    if p_doM is not None:
        expr = []
        for i, t in enumerate(TYPE_SPACE):
            x0, x1, m0, m1, y00, y01, y10, y11 = t
            for z in [0, 1]:
                x_z = x1 if z == 1 else x0
                y_1xz = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(1, x_z)]
                expr.append(p_vars[(i, z)] * y_1xz)
        expr_doM = pulp.lpSum(expr)
        if soft_experiments:
            _, _, pen = add_soft_experiment_constraint(prob, expr_doM, float(p_doM), "DOM1", penalty=soft_penalty, tol=tol)
            penalty_terms.append(pen)
        else:
            prob += (expr_doM == float(p_doM)), "DOM1"

    # Objective with tiny slack penalty if soft constraints used
    if sense == "max":
        prob += target_expr - (pulp.lpSum(penalty_terms) if penalty_terms else 0), "OBJ_MAX"
    else:
        prob += target_expr + (pulp.lpSum(penalty_terms) if penalty_terms else 0), "OBJ_MIN"

    return prob


def bounds_PY_doX1(
    Z, X, M, Y,
    p_doZ: Optional[float] = None,
    p_doM: Optional[float] = None,
    solver: Optional[pulp.LpSolver_CMD] = None,
    tol: float = 1e-9,
    soft_experiments: bool = False,
    soft_penalty: float = 1e-6,
) -> Tuple[float, float]:
    """Numeric bounds for P(Y=1 | do(X=1)).
       If p_doZ / p_doM are provided, they are enforced (exactly by default, or softly if soft_experiments=True).
    """
    solver = solver or pulp.PULP_CBC_CMD(msg=False)

    # Minimize
    prob_min = _build_numeric_lp(Z, X, M, Y, p_doZ, p_doM, sense="min", tol=tol,
                                 soft_experiments=soft_experiments, soft_penalty=soft_penalty)
    status = prob_min.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"(min) LP status: {pulp.LpStatus[status]}")
    lb = float(pulp.value(prob_min.objective))

    # Maximize
    prob_max = _build_numeric_lp(Z, X, M, Y, p_doZ, p_doM, sense="max", tol=tol,
                                 soft_experiments=soft_experiments, soft_penalty=soft_penalty)
    status = prob_max.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"(max) LP status: {pulp.LpStatus[status]}")
    ub = float(pulp.value(prob_max.objective))

    # Clamp small numerical noise
    if -1e-9 < lb < 0: lb = 0.0
    if 1 < lb < 1 + 1e-9: lb = 1.0
    if -1e-9 < ub < 0: ub = 0.0
    if 1 < ub < 1 + 1e-9: ub = 1.0
    return lb, ub


# ---------------------------
# Parametric (symbolic) bounds via duals
# ---------------------------
def _build_parametric_lp(
    Z, X, M, Y,
    include_doZ: bool,
    include_doM: bool,
    p_doZ_val: Optional[float],
    p_doM_val: Optional[float],
    sense: str = "max",
    tol: float = 1e-9,
):
    """Build the primal LP with observational constraints (two-sided tolerance)
       and (anchor) experimental equalities (single equality each so we can read duals).
    """
    pj, nZ = empirical_joint(Z, X, M, Y)
    prob = pulp.LpProblem("ParametricBounds_doX1", pulp.LpMaximize if sense == "max" else pulp.LpMinimize)

    # variables
    p_vars = {(i, z): pulp.LpVariable(f"p_{i}_{z}", lowBound=0)
              for i, _ in enumerate(TYPE_SPACE) for z in [0, 1]}

    constr_refs = {}  # we'll keep references for experimental constraints only

    # P(Z=z) (two-sided tolerance)
    for z in [0, 1]:
        add_eq_with_tol(prob, pulp.lpSum(p_vars[(i, z)] for i, _ in enumerate(TYPE_SPACE)), nZ[z], f"PZ_{z}", tol=tol)

    # Observational P(Z,X,M,Y) (two-sided tolerance)
    for z, x, m, y in product([0, 1], repeat=4):
        mask = []
        for i, t in enumerate(TYPE_SPACE):
            x0, x1, m0, m1, y00, y01, y10, y11 = t
            x_z = x1 if z == 1 else x0
            m_x = m1 if x == 1 else m0
            y_mx = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(m, x)]
            if x_z == x and m_x == m and y_mx == y:
                mask.append(p_vars[(i, z)])
        add_eq_with_tol(prob, pulp.lpSum(mask), pj[(z, x, m, y)], f"OBS_{z}{x}{m}{y}", tol=tol)

    # Experimental equalities (single equality each for duals). Use anchors if included.
    if include_doZ:
        expr = []
        for i, t in enumerate(TYPE_SPACE):
            x0, x1, m0, m1, y00, y01, y10, y11 = t
            m_x1 = m1 if x1 == 1 else m0
            y_mx1 = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(m_x1, x1)]
            expr.append((p_vars[(i, 0)] + p_vars[(i, 1)]) * y_mx1)
        rhs = 0.0 if p_doZ_val is None else float(p_doZ_val)
        c = pulp.lpSum(expr) == rhs
        prob += c
        constr_refs["DOZ1"] = list(prob.constraints.values())[-1]

    if include_doM:
        expr = []
        for i, t in enumerate(TYPE_SPACE):
            x0, x1, m0, m1, y00, y01, y10, y11 = t
            for z in [0, 1]:
                x_z = x1 if z == 1 else x0
                y_1xz = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(1, x_z)]
                expr.append(p_vars[(i, z)] * y_1xz)
        rhs = 0.0 if p_doM_val is None else float(p_doM_val)
        c = pulp.lpSum(expr) == rhs
        prob += c
        constr_refs["DOM1"] = list(prob.constraints.values())[-1]

    # Objective target: P(Y=1 | do(X=1))
    obj_terms = []
    for i, t in enumerate(TYPE_SPACE):
        x0, x1, m0, m1, y00, y01, y10, y11 = t
        y_m1_1 = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}[(m1, 1)]
        obj_terms.append((p_vars[(i, 0)] + p_vars[(i, 1)]) * y_m1_1)
    prob += pulp.lpSum(obj_terms)

    return prob, constr_refs


def solve_with_duals(prob: pulp.LpProblem, constr_refs: Dict[str, pulp.LpConstraint],
                     include_doZ: bool, include_doM: bool, solver: Optional[pulp.LpSolver_CMD] = None):
    """Solve and return (value_at_anchor, lamZ, lamM). Duals are only read for experimental equalities."""
    solver = solver or pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP status: {pulp.LpStatus[status]}")
    val = float(pulp.value(prob.objective))
    lamZ = float(constr_refs["DOZ1"].pi) if include_doZ else 0.0
    lamM = float(constr_refs["DOM1"].pi) if include_doM else 0.0
    return val, lamZ, lamM


def _piece_to_string(aZ: float, aM: float, b: float, use_doZ: bool, use_doM: bool, precision: int = 10) -> str:
    terms = [f"{b:.{precision}f}"]
    if use_doZ:
        terms.append(f"{aZ:.{precision}f}*p_doZ")
    if use_doM:
        terms.append(f"{aM:.{precision}f}*p_doM")
    return " + ".join(terms)


def parametric_bounds_PY_doX1(
    Z, X, M, Y,
    use_doZ: bool = False,
    use_doM: bool = False,
    probe_gridZ: Tuple[float, ...] = (0.0, 0.5, 1.0),
    probe_gridM: Tuple[float, ...] = (0.0, 0.5, 1.0),
    solver: Optional[pulp.LpSolver_CMD] = None,
    tol: float = 1e-9,
) -> Dict[str, Dict[str, List]]:
    """Return piecewise-linear envelopes for lower/upper bounds as functions of p_doZ / p_doM.

    Output:
      {
        'upper': {'pieces': [(aZ,aM,b), ...], 'strings': [...]},
        'lower': {'pieces': [(aZ,aM,b), ...], 'strings': [...]},
        'note': 'Bounds are envelopes...',
      }
    Such that:
        upper(pZ,pM) = max_j (b_j + aZ_j * pZ + aM_j * pM)
        lower(pZ,pM) = min_j (b_j + aZ_j * pZ + aM_j * pM)
    """
    Z_vals = probe_gridZ if use_doZ else (0.0,)
    M_vals = probe_gridM if use_doM else (0.0,)

    upper_pieces = []
    lower_pieces = []

    for pZ0 in Z_vals:
        for pM0 in M_vals:
            # Max piece at anchor (pZ0, pM0)
            try:
                prob_max, cref_max = _build_parametric_lp(
                    Z, X, M, Y,
                    include_doZ=use_doZ, include_doM=use_doM,
                    p_doZ_val=pZ0 if use_doZ else None,
                    p_doM_val=pM0 if use_doM else None,
                    sense="max", tol=tol,
                )
                v0, lamZ, lamM = solve_with_duals(prob_max, cref_max, use_doZ, use_doM, solver)
                b = v0 - lamZ * (pZ0 if use_doZ else 0.0) - lamM * (pM0 if use_doM else 0.0)
                upper_pieces.append((round(lamZ, 10), round(lamM, 10), round(b, 10)))
            except RuntimeError:
                # Anchor infeasible with current observational data → skip
                pass

            # Min piece at same anchor
            try:
                prob_min, cref_min = _build_parametric_lp(
                    Z, X, M, Y,
                    include_doZ=use_doZ, include_doM=use_doM,
                    p_doZ_val=pZ0 if use_doZ else None,
                    p_doM_val=pM0 if use_doM else None,
                    sense="min", tol=tol,
                )
                v0, lamZ, lamM = solve_with_duals(prob_min, cref_min, use_doZ, use_doM, solver)
                b = v0 - lamZ * (pZ0 if use_doZ else 0.0) - lamM * (pM0 if use_doM else 0.0)
                lower_pieces.append((round(lamZ, 10), round(lamM, 10), round(b, 10)))
            except RuntimeError:
                pass

    if not upper_pieces or not lower_pieces:
        raise RuntimeError("No feasible anchor points found; cannot form parametric envelopes. "
                           "Try different probe grids, or check observational data consistency.")

    # Deduplicate pieces
    def dedup(pcs):
        seen, out = set(), []
        for aZ, aM, b in pcs:
            key = (aZ, aM, b)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    upper_pieces = dedup(upper_pieces)
    lower_pieces = dedup(lower_pieces)

    return {
        "upper": {
            "pieces": upper_pieces,
            "strings": [_piece_to_string(aZ, aM, b, use_doZ, use_doM) for (aZ, aM, b) in upper_pieces],
        },
        "lower": {
            "pieces": lower_pieces,
            "strings": [_piece_to_string(aZ, aM, b, use_doZ, use_doM) for (aZ, aM, b) in lower_pieces],
        },
        "note": "Bounds are envelopes (upper = max of pieces, lower = min of pieces).",
    }


def evaluate_bound_piecewise(
    pZ: Optional[float],
    pM: Optional[float],
    pieces: List[Tuple[float, float, float]],
    sense: str = "upper",
) -> float:
    """Evaluate a piecewise-linear envelope at (pZ, pM) given pieces [(aZ, aM, b), ...]."""
    z = 0.0 if pZ is None else float(pZ)
    m = 0.0 if pM is None else float(pM)
    vals = [b + aZ * z + aM * m for (aZ, aM, b) in pieces]
    return max(vals) if sense == "upper" else min(vals)


# ---------------------------
# (Optional) quick sanity test
# ---------------------------
if __name__ == "__main__":
    # Tiny demo using a deterministic SCM for reproducibility (same as earlier messages)
    def gen_data(n=200_000, seed=123, pW=0.3, pU=0.6):
        rng = np.random.default_rng(seed)
        W = (rng.random(n) < pW).astype(int)
        U = (rng.random(n) < pU).astype(int)
        Z = W
        X = Z * (1 - U)
        M = X
        Y = ((M & W) | U).astype(int)
        return Z, X, M, Y, pW, pU

    def truths(pW, pU):
        p_doX1 = 1 - (1 - pW) * (1 - pU)
        p_doZ1 = pU + (1 - pU) * pW
        p_doM1 = 1 - (1 - pW) * (1 - pU)
        return p_doX1, p_doZ1, p_doM1

    Z, X, M, Y, pW, pU = gen_data()
    tX, tZ, tM = truths(pW, pU)
    print(f"Truths: doX1={tX:.4f}, doZ1={tZ:.4f}, doM1={tM:.4f}")

    # Numeric bounds
    print("Numeric bounds:")
    print("  obs only   :", bounds_PY_doX1(Z, X, M, Y))
    print("  + doZ      :", bounds_PY_doX1(Z, X, M, Y, p_doZ=tZ))
    print("  + doM      :", bounds_PY_doX1(Z, X, M, Y, p_doM=tM))
    print("  + both     :", bounds_PY_doX1(Z, X, M, Y, p_doZ=tZ, p_doM=tM))

    # Parametric (symbolic) envelopes
    print("\nParametric bounds (use_doZ only):")
    resZ = parametric_bounds_PY_doX1(Z, X, M, Y, use_doZ=True, use_doM=False)
    print("  upper pieces:", resZ["upper"]["strings"])
    print("  lower pieces:", resZ["lower"]["strings"])
    ub = evaluate_bound_piecewise(tZ, None, resZ["upper"]["pieces"], "upper")
    lb = evaluate_bound_piecewise(tZ, None, resZ["lower"]["pieces"], "lower")
    print(f"  evaluated at p_doZ={tZ:.4f} -> ({lb:.4f}, {ub:.4f})")

    print("\nParametric bounds (use_doZ & use_doM):")
    resZM = parametric_bounds_PY_doX1(Z, X, M, Y, use_doZ=True, use_doM=True)
    print("  upper pieces:", resZM["upper"]["strings"])
    print("  lower pieces:", resZM["lower"]["strings"])
    ub = evaluate_bound_piecewise(tZ, tM, resZM["upper"]["pieces"], "upper")
    lb = evaluate_bound_piecewise(tZ, tM, resZM["lower"]["pieces"], "lower")
    print(f"  evaluated at (p_doZ={tZ:.4f}, p_doM={tM:.4f}) -> ({lb:.4f}, {ub:.4f})")
