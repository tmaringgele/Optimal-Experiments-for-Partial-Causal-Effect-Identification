import numpy as np
import itertools
import pulp


def bounds_PY_doX1(Z, X, M, Y, p_doZ=None, p_doM=None, solver=None):
    """
    Compute bounds on P(Y=1 | do(X=1)) for the DAG:
        W->Z, Z->X, U->X, X->M, (X,M,W,U)->Y; W,U unobserved; Z,X,M,Y observed (binary).
    
    Inputs
    ------
    Z, X, M, Y : 1D numpy arrays of shape (N,), entries in {0,1}.
        Observational samples of the four observed variables.
    p_doZ : float in [0,1] or None
        Optional interventional result p_(Z=1) = P(Y=1 | do(Z=1)).
    p_doM : float in [0,1] or None
        Optional interventional result p_(M=1) = P(Y=1 | do(M=1)).
    solver : a PuLP solver instance (optional)
        e.g. pulp.PULP_CBC_CMD(msg=False). If None, uses CBC with msg=False.

    Returns
    -------
    (lower, upper) : tuple of floats
        Lower/upper bounds on P(Y=1 | do(X=1)) implied by the data and DAG.
    
    Notes
    -----
    • Response-function parameterization:
        - Z | W uses types σ ∈ {0,1}^{W}  (4 types)
        - X | (Z,U) uses types τ ∈ {0,1}^{Z×U} (16 types)
        - M | X uses types μ ∈ {0,1}^{X} (4 types)
        - Y | (X,M,W,U) has 16 entries; instead of enumerating all 2^16 Y-types,
          for each (w,u,σ,τ,μ) we only materialize Y-bits that appear in:
            (i) the observational tuple (z_nat, x_nat, m_nat, y),
            (ii) do(Z=1),
            (iii) do(M=1),
            (iv) the target do(X=1).
          That’s ≤ 4 distinct (x,m,w,u) positions, so ≤ 2^4 = 16 columns per base tuple.
        - Total columns ≤ 2*2*4*16*4*16 = 16384 (usually fewer).
    • Builds one LP and solves it twice (max/min) by reconstructing the objective.
    """

    # ---------- basic checks ----------
    Z = np.asarray(Z).astype(int).ravel()
    X = np.asarray(X).astype(int).ravel()
    M = np.asarray(M).astype(int).ravel()
    Y = np.asarray(Y).astype(int).ravel()
    n = len(Z)
    assert len(X) == len(M) == len(Y) == n, "Z,X,M,Y must have same length"
    assert set(np.unique(Z)).issubset({0,1})
    assert set(np.unique(X)).issubset({0,1})
    assert set(np.unique(M)).issubset({0,1})
    assert set(np.unique(Y)).issubset({0,1})
    if p_doZ is not None:
        assert 0.0 <= p_doZ <= 1.0
    if p_doM is not None:
        assert 0.0 <= p_doM <= 1.0

    # ---------- empirical joint π(z,x,m,y) ----------
    pi = np.zeros((2,2,2,2), dtype=float)
    for z, x, m, y in zip(Z, X, M, Y):
        pi[z, x, m, y] += 1.0
    if n == 0:
        raise ValueError("Empty data.")
    pi /= float(n)

    # If some (z,x,m) cells are unobserved (both y=0 and y=1 zero),
    # the equalities below will enforce 0 on those rows automatically.

    # ---------- enumerate response types ----------
    # Z|W types σ = (σ0, σ1)
    SZ = list(itertools.product([0,1], repeat=2))            # 4 types
    # X|Z,U types τ : order entries as (z,u) in [(0,0),(0,1),(1,0),(1,1)]
    SX = list(itertools.product([0,1], repeat=4))            # 16 types
    # M|X types μ = (μ0, μ1)
    SM = list(itertools.product([0,1], repeat=2))            # 4 types

    def x_from_tau(tau, z, u):
        # tau indexed by (z,u) -> idx = 2*z + u  in {0,1,2,3}
        return tau[2*z + u]

    # Map each (z,x,m,y) to a constraint key/id
    obs_keys = [(z,x,m,y) for z in [0,1] for x in [0,1] for m in [0,1] for y in [0,1]]
    obs_key_to_idx = {k:i for i,k in enumerate(obs_keys)}

    # Precompute observational RHS per key
    obs_rhs = {k: pi[k] for k in obs_keys}

    # ---------- build column set ----------
    # Each column corresponds to a tuple (w,u,σ,τ,μ, s) where s assigns bits to
    # the DISTINCT set of Y-positions actually used.
    # We store for each column:
    #   - obs_row: one of (z,x,m,y) where the column contributes 1
    #   - coef_Z1: {0,1} coefficient in the do(Z=1) constraint
    #   - coef_M1: {0,1} coefficient in the do(M=1) constraint
    #   - coef_tar:{0,1} coefficient in the objective (P(Y=1|do(X=1)))
    # We also keep a compact string id for readability.

    columns = []  # list of dicts with fields: name, obs_row, coef_Z1, coef_M1, coef_tar

    for w in [0,1]:
        for u in [0,1]:
            for sigma in SZ:       # sigma[w] gives z
                for tau in SX:     # tau[(z,u)] gives x
                    for mu in SM:  # mu[x] gives m
                        # Natural world parents and values
                        z_nat = sigma[w]
                        x_nat = x_from_tau(tau, z_nat, u)
                        m_nat = mu[x_nat]

                        # Indices of Y entries that matter (as (x,m,w,u) tuples)
                        idx_obs   = (x_nat, m_nat, w, u)
                        # do(Z=1):
                        x_doZ = x_from_tau(tau, 1, u)
                        m_doZ = mu[x_doZ]
                        idx_Z1   = (x_doZ, m_doZ, w, u)
                        # do(M=1):
                        idx_M1   = (x_nat, 1, w, u)
                        # target do(X=1):
                        idx_tar  = (1, mu[1], w, u)

                        # Collect distinct indices and give them local positions 0..d-1
                        distinct = []
                        for idx in (idx_obs, idx_Z1, idx_M1, idx_tar):
                            if idx not in distinct:
                                distinct.append(idx)
                        d = len(distinct)
                        pos = {distinct[j]: j for j in range(d)}

                        # For each assignment s in {0,1}^d, create a column
                        for mask in range(1 << d):
                            # bits for each distinct Y-entry
                            def bit(idx):  # value of Y at that (x,m,w,u)
                                return (mask >> pos[idx]) & 1

                            y_obs = bit(idx_obs)
                            obs_row = (z_nat, x_nat, m_nat, y_obs)
                            coef_Z1 = bit(idx_Z1)
                            coef_M1 = bit(idx_M1)
                            coef_tar = bit(idx_tar)

                            name = f"q_w{w}u{u}_s{sigma}_t{tau}_m{mu}_S{mask}"
                            columns.append({
                                "name": name,
                                "obs_row": obs_row,
                                "coef_Z1": coef_Z1,
                                "coef_M1": coef_M1,
                                "coef_tar": coef_tar
                            })

    # Group columns by observational row for fast constraint assembly
    cols_by_obs = {k: [] for k in obs_keys}
    for j, col in enumerate(columns):
        cols_by_obs[col["obs_row"]].append(j)

    # ---------- helper to solve one sense (max/min) ----------
    def solve_with_sense(sense="max"):
        if solver is None:
            used_solver = pulp.PULP_CBC_CMD(msg=False)
        else:
            used_solver = solver

        prob = pulp.LpProblem("Bounds_PY_doX1", pulp.LpMaximize if sense=="max" else pulp.LpMinimize)

        # Variables: one nonnegative var per column
        q_vars = [pulp.LpVariable(col["name"], lowBound=0) for col in columns]

        # Objective: sum coef_tar * q
        prob += pulp.lpSum(col["coef_tar"] * q_vars[j] for j, col in enumerate(columns))

        # Observational equalities: for each (z,x,m,y), sum of q on that row equals pi[z,x,m,y]
        for k in obs_keys:
            prob += pulp.lpSum(q_vars[j] for j in cols_by_obs[k]) == obs_rhs[k], f"obs_{k}"

        # Interventional constraints (optional)
        if p_doZ is not None:
            prob += pulp.lpSum(col["coef_Z1"] * q_vars[j] for j, col in enumerate(columns)) == float(p_doZ), "doZ1"
        if p_doM is not None:
            prob += pulp.lpSum(col["coef_M1"] * q_vars[j] for j, col in enumerate(columns)) == float(p_doM), "doM1"

        # (Normalization is redundant: sum over all obs rows already equals 1 by construction)

        status = prob.solve(used_solver)
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"LP {sense}imize did not solve to optimality: {pulp.LpStatus[status]}")

        val = pulp.value(prob.objective)
        # Clamp small numerical noise
        if val < 0 and val > -1e-9:
            val = 0.0
        if val > 1 and val < 1 + 1e-9:
            val = 1.0
        return float(val)

    upper = solve_with_sense("max")
    lower = solve_with_sense("min")
    return (lower, upper)
