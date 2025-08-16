# benders_bounds_budget.py
# Exact Benders decomposition for budgeted tightest-bounds via LP dual cuts.
# Requires: numpy, scipy (linprog with method='highs'), pulp

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import linprog
import pulp


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class ConstraintBlock:
    """
    A linear-inequality block: A_ub q <= b_ub, with an optional vector M_ub >= 0
    used to relax the RHS when package is *inactive*:
        A_ub q <= b_ub * x_i + M_ub * (1 - x_i)
    For "interval" constraints L <= a^T q <= U, encode as two rows in A_ub, b_ub
    with M_ub chosen to safely deactivate them when x_i=0.
    """
    A_ub: np.ndarray  # shape (m_ub, 8)
    b_ub: np.ndarray  # shape (m_ub,)
    M_ub: Optional[np.ndarray] = None  # shape (m_ub,). Defaults to safe values if None.

    # (Optional equality block if you need exact equalities; typically we use intervals instead)
    A_eq: Optional[np.ndarray] = None  # shape (m_eq, 8)
    b_eq: Optional[np.ndarray] = None  # shape (m_eq,)
    M_eq: Optional[np.ndarray] = None  # shape (m_eq,). Used like b_eq * x_i + M_eq * (1 - x_i)


@dataclass
class ProblemData:
    """
    Full problem description in the 8-dimensional q vector (your response-type probs).
    """
    theta: np.ndarray              # objective coefficients for P(Y|do(...)) etc. shape (8,)
    # Observational constraints (always active)
    obs_ineq: ConstraintBlock      # G_obs q <= h_obs
    obs_eq: ConstraintBlock        # E_obs q = f_obs  (use only A_eq,b_eq)
    # Candidate interventions
    packages: List[ConstraintBlock]
    costs: np.ndarray              # cost per package, shape (n,)
    budget: float                  # budget B


# ---------------------------
# Utilities
# ---------------------------

def stack_ineq(obs: ConstraintBlock,
               packages: List[ConstraintBlock],
               x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[int, slice], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Build A_ub, b_ub for subproblem given selection x.
    For each package row j: rhs = b_ub[j]*x_i + M_ub[j]*(1-x_i).
    Returns:
      A_ub, b_ub,
      row_slices: mapping i -> slice of rows belonging to package i (to read duals),
      rhs_decomp: mapping i -> (const_part, coeff_part) where
                  rhs = const_part + coeff_part * x_i (vectorized per row).
    """
    # Observational inequalities (no x-dependence)
    A_list = [obs.A_ub] if obs.A_ub is not None and obs.A_ub.size else []
    b_list = [obs.b_ub] if obs.b_ub is not None and obs.b_ub.size else []

    row_slices: Dict[int, slice] = {}
    rhs_decomp: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # Track current row count
    r0 = sum(a.shape[0] for a in A_list) if A_list else 0

    # Packages
    for i, pkg in enumerate(packages):
        if pkg.A_ub is None or pkg.b_ub is None or pkg.A_ub.size == 0:
            continue
        m = pkg.A_ub.shape[0]
        M = pkg.M_ub if pkg.M_ub is not None else np.ones(m)

        A_list.append(pkg.A_ub)
        # rhs = b*x_i + M*(1-x_i) = M + (b - M)*x_i
        const = M.copy()
        coef = (pkg.b_ub - M)
        rhs_decomp[i] = (const, coef)
        b_block = const + coef * float(x[i])
        b_list.append(b_block)

        row_slices[i] = slice(r0, r0 + m)
        r0 += m

    A_ub = np.vstack(A_list) if A_list else np.zeros((0, 8))
    b_ub = np.concatenate(b_list) if b_list else np.zeros((0,))
    return A_ub, b_ub, row_slices, rhs_decomp


def stack_eq(obs: ConstraintBlock,
             packages: List[ConstraintBlock],
             x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[int, slice], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Build A_eq, b_eq for subproblem given x.
    For each package equality row j: rhs = b_eq[j]*x_i + M_eq[j]*(1-x_i).
    If no package equalities are used, returns obs equalities only.
    """
    # Observational equalities
    A_list = [obs.A_eq] if obs.A_eq is not None and obs.A_eq.size else []
    b_list = [obs.b_eq] if obs.b_eq is not None and obs.b_eq.size else []

    row_slices: Dict[int, slice] = {}
    rhs_decomp: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    r0 = sum(a.shape[0] for a in A_list) if A_list else 0

    for i, pkg in enumerate(packages):
        if pkg.A_eq is None or pkg.b_eq is None or pkg.A_eq.size == 0:
            continue
        m = pkg.A_eq.shape[0]
        M = pkg.M_eq if pkg.M_eq is not None else np.zeros(m)  # 0 relaxes lower-eq safely if x_i=0

        A_list.append(pkg.A_eq)
        const = M.copy()
        coef = (pkg.b_eq - M)
        rhs_decomp[i] = (const, coef)
        b_block = const + coef * float(x[i])
        b_list.append(b_block)

        row_slices[i] = slice(r0, r0 + m)
        r0 += m

    A_eq = np.vstack(A_list) if A_list else np.zeros((0, 8))
    b_eq = np.concatenate(b_list) if b_list else np.zeros((0,))
    return A_eq, b_eq, row_slices, rhs_decomp


def solve_subproblem(theta: np.ndarray,
                     obs_ineq: ConstraintBlock,
                     obs_eq: ConstraintBlock,
                     packages: List[ConstraintBlock],
                     x: np.ndarray,
                     sense: str) -> Tuple[float, np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Solve UB (sense='max') or LB (sense='min') subproblem with linprog (HiGHS).
    Returns:
      value, y_ineq (m_ineq,), z_eq (m_eq,), pkg_y[i] (dual slice for pkg i inequalities),
      pkg_z[i] (dual slice for pkg i equalities)
    Note: SciPy 'highs' returns duals in res.ineqlin.marginals (size m_ineq) and res.eqlin.marginals (size m_eq).
    """
    # Build constraints given x
    A_ub, b_ub, row_slices_ineq, _ = stack_ineq(obs_ineq, packages, x)
    A_eq, b_eq, row_slices_eq, _ = stack_eq(obs_eq, packages, x)

    # Objective for linprog is minimization.
    c = theta.copy()
    if sense == 'max':
        c = -c

    bounds = [(0, None)] * 8  # q >= 0

    res = linprog(c=c, A_ub=A_ub if A_ub.size else None, b_ub=b_ub if b_ub.size else None,
                  A_eq=A_eq if A_eq.size else None, b_eq=b_eq if b_eq.size else None,
                  bounds=bounds, method='highs')

    if res.status != 0:
        raise RuntimeError(f"Subproblem infeasible or failed (sense={sense}). Message: {res.message}")

    # Objective value
    val = res.fun
    if sense == 'max':
        val = -val

    # Duals (shadow prices / marginals)
    y = res.ineqlin.marginals if hasattr(res, 'ineqlin') else np.zeros(A_ub.shape[0] if A_ub.size else 0)
    z = res.eqlin.marginals if hasattr(res, 'eqlin') else np.zeros(A_eq.shape[0] if A_eq.size else 0)

    # Split package dual slices
    pkg_y: Dict[int, np.ndarray] = {}
    for i, sl in row_slices_ineq.items():
        pkg_y[i] = y[sl] if y.size else np.zeros(0)

    pkg_z: Dict[int, np.ndarray] = {}
    for i, sl in row_slices_eq.items():
        pkg_z[i] = z[sl] if z.size else np.zeros(0)

    return val, y, z, pkg_y, pkg_z


def build_benders_cut_coeffs(obs_ineq: ConstraintBlock,
                             obs_eq: ConstraintBlock,
                             packages: List[ConstraintBlock],
                             x: np.ndarray,
                             y_ineq: np.ndarray,
                             z_eq: np.ndarray,
                             pkg_y: Dict[int, np.ndarray],
                             pkg_z: Dict[int, np.ndarray]) -> Tuple[float, np.ndarray]:
    """
    Given duals at selection x, compute affine coefficients (alpha0, alpha) such that
        value >= alpha0 + sum_i alpha[i] * x_i    (UB cut: lower-bounds UB)
    or
        value <= beta0 + sum_i beta[i] * x_i     (LB cut: upper-bounds LB)
    depending on which duals we pass in (use same function for both).
    """
    # Observational contributions (no x dependence)
    const = 0.0
    if obs_ineq.A_ub is not None and obs_ineq.A_ub.size:
        const += float(np.dot(obs_ineq.b_ub, y_ineq[:obs_ineq.A_ub.shape[0]]))
        y_tail = y_ineq[obs_ineq.A_ub.shape[0]:]
    else:
        y_tail = y_ineq

    if obs_eq.A_eq is not None and obs_eq.A_eq.size:
        const += float(np.dot(obs_eq.b_eq, z_eq[:obs_eq.A_eq.shape[0]]))
        z_tail = z_eq[obs_eq.A_eq.shape[0]:]
    else:
        z_tail = z_eq

    # Package contributions
    n = len(packages)
    alpha = np.zeros(n)

    # For each package, reconstruct RHS split used in subproblem:
    # For inequalities: rhs = M + (b - M) * x_i  -> contribution = M^T y + ((b - M)^T y) * x_i
    # For equalities:   rhs = M + (b - M) * x_i  -> likewise
    # We don't need the exact slices of y_tail/z_tail here because we already split via pkg_y/pkg_z.
    for i in range(n):
        # Inequalities
        if packages[i].A_ub is not None and packages[i].A_ub.size:
            m = packages[i].A_ub.shape[0]
            M = packages[i].M_ub if packages[i].M_ub is not None else np.ones(m)
            b = packages[i].b_ub
            yi = pkg_y.get(i, np.zeros(m))
            const += float(np.dot(M, yi))
            alpha[i] += float(np.dot(b - M, yi))

        # Equalities (optional)
        if packages[i].A_eq is not None and packages[i].A_eq.size:
            m = packages[i].A_eq.shape[0]
            M = packages[i].M_eq if packages[i].M_eq is not None else np.zeros(m)
            b = packages[i].b_eq
            zi = pkg_z.get(i, np.zeros(m))
            const += float(np.dot(M, zi))
            alpha[i] += float(np.dot(b - M, zi))

    return const, alpha


# ---------------------------
# Benders driver
# ---------------------------

class BendersBounds:
    def __init__(self, data: ProblemData, tol: float = 1e-6, max_iters: int = 200):
        self.data = data
        self.tol = tol
        self.max_iters = max_iters
        self.n = len(data.packages)
        # Cut lists: u >= a0 + a^T x   and   l <= b0 + b^T x
        self.ub_cuts: List[Tuple[float, np.ndarray]] = []
        self.lb_cuts: List[Tuple[float, np.ndarray]] = []

    def _solve_master(self) -> Tuple[np.ndarray, float, float, float]:
        prob = pulp.LpProblem("Master", pulp.LpMinimize)
        x_vars = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(self.n)]
        # Bound u,l to [0,1]
        u = pulp.LpVariable("u", lowBound=0.0, upBound=1.0, cat="Continuous")
        l = pulp.LpVariable("l", lowBound=0.0, upBound=1.0, cat="Continuous")

        prob += u - l
        prob += pulp.lpSum(self.data.costs[i] * x_vars[i] for i in range(self.n)) <= self.data.budget
        # enforce UB >= LB
        prob += u >= l

        for k, (a0, a) in enumerate(self.ub_cuts):
            prob += u >= a0 + pulp.lpSum(a[i] * x_vars[i] for i in range(self.n)), f"ub_cut_{k}"
        for m, (b0, b) in enumerate(self.lb_cuts):
            prob += l <= b0 + pulp.lpSum(b[i] * x_vars[i] for i in range(self.n)), f"lb_cut_{m}"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError(f"Master did not solve cleanly: {pulp.LpStatus[prob.status]}")

        x_val = np.array([pulp.value(v) for v in x_vars], dtype=float).round().astype(int)
        u_val, l_val = float(pulp.value(u)), float(pulp.value(l))
        return x_val, u_val, l_val, u_val - l_val


    def run(self, verbose: bool = True):
        best_x, best_width = None, np.inf

        # --- Seed with x = 0 cuts (no packages) ---
        x0 = np.zeros(self.n, dtype=int)
        U0, yU0, zU0, pkg_yU0, pkg_zU0 = solve_subproblem(
            theta=self.data.theta, obs_ineq=self.data.obs_ineq, obs_eq=self.data.obs_eq,
            packages=self.data.packages, x=x0, sense='max'
        )
        L0, yL0, zL0, pkg_yL0, pkg_zL0 = solve_subproblem(
            theta=self.data.theta, obs_ineq=self.data.obs_ineq, obs_eq=self.data.obs_eq,
            packages=self.data.packages, x=x0, sense='min'
        )
        a0, a = build_benders_cut_coeffs(self.data.obs_ineq, self.data.obs_eq,
                                        self.data.packages, x0, yU0, zU0, pkg_yU0, pkg_zU0)
        b0, b = build_benders_cut_coeffs(self.data.obs_ineq, self.data.obs_eq,
                                        self.data.packages, x0, yL0, zL0, pkg_yL0, pkg_zL0)
        self.ub_cuts.append((a0, a))
        self.lb_cuts.append((b0, b))
        best_width = U0 - L0
        best_x = x0.copy()
        # --- Extra seeding: evaluate each affordable singleton e_i ---
        for i in range(self.n):
            if self.data.costs[i] <= self.data.budget:
                x1 = np.zeros(self.n, dtype=int)
                x1[i] = 1
                U1, yU1, zU1, pkg_yU1, pkg_zU1 = solve_subproblem(
                    theta=self.data.theta, obs_ineq=self.data.obs_ineq, obs_eq=self.data.obs_eq,
                    packages=self.data.packages, x=x1, sense='max'
                )
                L1, yL1, zL1, pkg_yL1, pkg_zL1 = solve_subproblem(
                    theta=self.data.theta, obs_ineq=self.data.obs_ineq, obs_eq=self.data.obs_eq,
                    packages=self.data.packages, x=x1, sense='min'
                )
                a0_1, a_1 = build_benders_cut_coeffs(self.data.obs_ineq, self.data.obs_eq,
                                                     self.data.packages, x1, yU1, zU1, pkg_yU1, pkg_zU1)
                b0_1, b_1 = build_benders_cut_coeffs(self.data.obs_ineq, self.data.obs_eq,
                                                     self.data.packages, x1, yL1, zL1, pkg_yL1, pkg_zL1)
                self.ub_cuts.append((a0_1, a_1))
                self.lb_cuts.append((b0_1, b_1))
                width1 = U1 - L1
                if width1 < best_width - self.tol:
                    best_width, best_x = width1, x1.copy()


        # --- Standard loop ---
        for it in range(1, self.max_iters + 1):
            x, u_star, l_star, master_obj = self._solve_master()

            U, yU, zU, pkg_yU, pkg_zU = solve_subproblem(
                theta=self.data.theta, obs_ineq=self.data.obs_ineq, obs_eq=self.data.obs_eq,
                packages=self.data.packages, x=x, sense='max'
            )
            L, yL, zL, pkg_yL, pkg_zL = solve_subproblem(
                theta=self.data.theta, obs_ineq=self.data.obs_ineq, obs_eq=self.data.obs_eq,
                packages=self.data.packages, x=x, sense='min'
            )
            width = U - L
            if width < best_width - self.tol:
                best_width, best_x = width, x.copy()

            a0, a = build_benders_cut_coeffs(self.data.obs_ineq, self.data.obs_eq,
                                            self.data.packages, x, yU, zU, pkg_yU, pkg_zU)
            b0, b = build_benders_cut_coeffs(self.data.obs_ineq, self.data.obs_eq,
                                            self.data.packages, x, yL, zL, pkg_yL, pkg_zL)
            self.ub_cuts.append((a0, a))
            self.lb_cuts.append((b0, b))

            if verbose:
                print(f"[Iter {it:03d}] x={x.tolist()} | sub width={width:.6f} | "
                    f"master u-l={master_obj:.6f} | incumbent={best_width:.6f}")
            gap = best_width - master_obj   # upper bound - lower bound (must be ≥ 0)
            if gap <= self.tol:
                if verbose:
                    print(f"Converged (gap ≤ {self.tol}).")
                break

        return best_x, best_width



# ---------------------------
# Example build helpers
# ---------------------------

def build_observational_blocks(p1: float, p11: float, p10: float) -> Tuple[ConstraintBlock, ConstraintBlock]:
    """
    Observational constraints for the 8-d q:
      sum q = 1
      q1 + q3 + q5 + q7 = p1
      q5 + q7 = p11
      q2 + q6 = p10
    All as equalities -> go into obs_eq block.
    No obs inequalities in this tiny example (but you could add CI intervals there).
    """
    # Equalities
    E = np.array([
        [1,1,1,1,1,1,1,1],      # sum q = 1
        [0,1,0,1,0,1,0,1],      # X=1
        [0,0,0,0,0,1,0,1],      # Y=1, X=1
        [0,0,1,0,0,0,1,0],      # Y=1, X=0
    ], dtype=float)
    f = np.array([1.0, p1, p11, p10], dtype=float)

    obs_eq = ConstraintBlock(A_ub=None, b_ub=None, A_eq=E, b_eq=f)
    # Inequalities (none here)
    obs_ineq = ConstraintBlock(A_ub=None, b_ub=None)
    return obs_ineq, obs_eq


def pkg_interval_constraint(rows: List[np.ndarray],
                            L: float, U: float,
                            M_upper: Optional[float] = 1.0, M_lower: Optional[float] = 0.0) -> ConstraintBlock:
    """
    Build a package that enforces:  L <= sum_j rows[j] · q <= U,
    where each 'rows[j]' is a 1x8 selector row (they are summed).
    Encoded as:
       +a q ≤ U * x + M_upper * (1 - x)
       -a q ≤ -L * x + M_lower * (1 - x)
    with safe defaults M_upper=1.0 (since sum of q-entries ≤ 1), M_lower=0.0.
    """
    a = np.sum(np.vstack(rows), axis=0)  # 1x8
    A_ub = np.vstack([ a, -a ])
    b_ub = np.array([ U, -L ], dtype=float)
    M_ub = np.array([ M_upper, M_lower ], dtype=float)
    return ConstraintBlock(A_ub=A_ub, b_ub=b_ub, M_ub=M_ub)


def build_toy_problem() -> ProblemData:
    """
    Toy instance from the discussion:
      theta corresponds to P(Y1=1) = q4+q5+q6+q7.
      Observational equalities with p1=0.50, p11=0.25, p10=0.20.
      Two candidate interventions:
        i=1: P(Y=1 | do(X=0)) in [0.30, 0.30]
        i=2: P(Y=1 | do(X=1)) in [0.60, 0.60]
      Costs: c1=3, c2=5; Budget B=5.
    """
    # Target: theta^T q = q4 + q5 + q6 + q7
    theta = np.array([0,0,0,0,1,1,1,1], dtype=float)

    # Observational constraints
    obs_ineq, obs_eq = build_observational_blocks(p1=0.50, p11=0.25, p10=0.20)

    # Package 1: P(Y=1 | do(X=0)) = q2+q3+q6+q7 in [0.30, 0.30]
    row_Y0 = np.array([0,0,1,1,0,0,1,1], dtype=float)  # (q2,q3,q6,q7)
    pkg1 = pkg_interval_constraint([row_Y0], L=0.30, U=0.30, M_upper=1.0, M_lower=0.0)

    # Package 2: P(Y=1 | do(X=1)) = q4+q5+q6+q7 in [0.60, 0.60]
    row_Y1 = np.array([0,0,0,0,1,1,1,1], dtype=float)  # (q4,q5,q6,q7)
    pkg2 = pkg_interval_constraint([row_Y1], L=0.60, U=0.60, M_upper=1.0, M_lower=0.0)

    costs = np.array([3.0, 5.0], dtype=float)
    budget = 5.0
    return ProblemData(theta=theta, obs_ineq=obs_ineq, obs_eq=obs_eq,
                       packages=[pkg1, pkg2], costs=costs, budget=budget)

def build_toy_problem_tightening() -> ProblemData:
    """
    Toy problem that *does* favor an intervention:
    Target: theta^T q = q4+q5+q6+q7 = P(Y=1 | do(X=1))
    Observational equalities:
        p1 = 0.50, p11 = 0.25, p10 = 0.20
    Packages:
        - pkg0 (cost 5): P(Y=1 | do(X=1)) in [0.55, 0.65]  -> strong direct tightening
        - pkg1 (cost 3): P(Y=1 | do(X=0)) in [0.25, 0.35]  -> weaker indirect tightening
    Budget: B = 5  (so it can afford pkg0 OR pkg1, but not both)
    """
    theta = np.array([0,0,0,0,1,1,1,1], dtype=float)  # P(Y1=1)

    # Observational constraints (equalities)
    obs_ineq, obs_eq = build_observational_blocks(p1=0.50, p11=0.25, p10=0.20)

    # Package 0: P(Y=1 | do(X=1)) in [0.55, 0.65]
    row_Y1 = np.array([0,0,0,0,1,1,1,1], dtype=float)  # q4+q5+q6+q7
    pkg0 = pkg_interval_constraint([row_Y1], L=0.55, U=0.65, M_upper=1.0, M_lower=0.0)

    # Package 1: P(Y=1 | do(X=0)) in [0.25, 0.35]
    row_Y0 = np.array([0,0,1,1,0,0,1,1], dtype=float)  # q2+q3+q6+q7
    pkg1 = pkg_interval_constraint([row_Y0], L=0.25, U=0.35, M_upper=1.0, M_lower=0.0)

    costs  = np.array([5.0, 3.0], dtype=float)  # cost(pkg0)=5, cost(pkg1)=3
    budget = 5.0

    return ProblemData(theta=theta, obs_ineq=obs_ineq, obs_eq=obs_eq,
                    packages=[pkg0, pkg1], costs=costs, budget=budget)



# ---------------------------
# Main (demo)
# ---------------------------

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=120)
    data = build_toy_problem()
    solver = BendersBounds(data, tol=1e-6, max_iters=100)
    x_star, width_star = solver.run(verbose=True)
    print("\n=== RESULT ===")
    print(f"Chosen interventions x*: {x_star.tolist()}")
    print(f"Tightest achievable width within budget: {width_star:.6f}")
