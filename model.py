# -*- coding: utf-8 -*-
"""
可分离情形的一般凸约化模型（Gurobi）。

核心求解器 solve_separable_general 只接受数值系数（b,h、目标中 x_0 的线性项等）。

原问题（最小化）：
    inf  ∑_{k} w_k ∑_{n} η_{k,n} f_{k,n}
         + λ (θ + θ̲)
         + ∑_{i} ε_i μ_i
         + ∑_{d} g_d x_{0,d}        （可选，由 obj_x0_linear 给出）

s.t. (x_i, μ_i) ∈ \\bar{X}_i, λ ≥ 0, f, ω_{α,d}，
     ∑_d ω_{α,d} ≤ ∑_k w_k f_{k,α_k}  ∀α ∈ A，
     以及正文中的 b,h 与 y,y' 线性化约束。

\\hat{ζ}_{t,α,d}：
    \\hat{ξ}_{t,α_t,d},  t = 1,…,K
    \\hat{x}_{t-K,d},   t = K+1,…,K+I
    \\underline{Ξ}_d,   t = K+I+1
    \\overline{Ξ}_d,   t = K+I+2
"""

from __future__ import annotations

from itertools import product as itertools_product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _normalize_xi(Xi: Mapping) -> Dict[str, Sequence[float]]:
    out = {}
    for k, v in Xi.items():
        out[str(k)] = v
    return out


def solve_separable_general(
    empirical_distributions: Optional[Mapping[int, np.ndarray]],
    order_data: Mapping[int, Mapping[int, float]],
    K: int,
    I: int,
    D: int,
    N: Sequence[int],
    mathcal_D: Mapping[str, Any],
    Xi: Mapping,
    A_0: Sequence[Sequence[float]],
    b_0: Sequence[float],
    A_I: Mapping[str, Any],
    b_I: Mapping[str, Any],
    b_0_coef: Optional[Mapping[int, float]] = None,
    h_0_coef: Optional[Mapping[int, float]] = None,
    b_i_coef: Optional[Mapping[int, Mapping[int, float]]] = None,
    h_i_coef: Optional[Mapping[int, Mapping[int, float]]] = None,
    obj_x0_linear: Optional[Mapping[int, float]] = None,
    w_k: Optional[np.ndarray] = None,
    eta_k_n: Optional[Mapping[int, np.ndarray]] = None,
    theta: float = 0.0,
    theta_bar: float = 0.0,
    epsilon_i: Optional[Mapping[int, float]] = None,
    output_flag: int = 0,
) -> Tuple[
    Optional[Dict[int, float]],
    Optional[Dict[int, Dict[int, float]]],
    Optional[Dict[int, float]],
    float,
    Optional[Dict[int, List[float]]],
    Optional[float],
    Any,
]:
    """
    文中全局 t 的可分离凸约化；须显式 b/h 与可选 obj_x0_linear。
    """
    if K < 0 or I < 0:
        raise ValueError("K,I 必须非负")
    if K > 0 and empirical_distributions is None:
        raise ValueError("K>0 时必须提供 empirical_distributions")
    if I > 0 and not order_data:
        raise ValueError("I>0 时必须提供 order_data")
    if b_0_coef is None or h_0_coef is None:
        raise ValueError("必须提供 b_0_coef 与 h_0_coef")
    if I > 0 and (b_i_coef is None or h_i_coef is None):
        raise ValueError("I>0 时必须提供 b_i_coef 与 h_i_coef")

    Xi_n = _normalize_xi(Xi)
    Xi_lower = {str(d): float(Xi_n[str(d)][0]) for d in range(1, D + 1)}
    Xi_upper = {str(d): float(Xi_n[str(d)][1]) for d in range(1, D + 1)}

    if K > 0:
        assert empirical_distributions is not None
        if w_k is None:
            w_k = np.ones(K) / K
        else:
            w_k = np.asarray(w_k, dtype=float)
            if not np.isclose(w_k.sum(), 1.0, atol=1e-6):
                raise ValueError(f"w_k 之和须为 1，当前为 {w_k.sum():.6f}")
        if eta_k_n is None:
            eta_k_n = {k: np.ones(N[k]) / N[k] for k in range(K)}
        else:
            for k in range(K):
                eta_k_n[k] = np.asarray(eta_k_n[k], dtype=float)
                if not np.isclose(eta_k_n[k].sum(), 1.0, atol=1e-6):
                    raise ValueError(f"eta_k_n[{k}] 之和须为 1")
    else:
        w_k = np.array([])
        eta_k_n = {}

    if epsilon_i is None:
        epsilon_i = {i: 0.0 for i in range(1, I + 1)}

    b_i_coef = b_i_coef or {}
    h_i_coef = h_i_coef or {}

    T = K + I + 2
    if K == 0:
        all_alpha: List[Tuple[int, ...]] = [()]
    else:
        all_alpha = list(itertools_product(*[range(N[k]) for k in range(K)]))

    model = gp.Model("SeparableGeneral")
    model.setParam("OutputFlag", output_flag)

    x_0 = {d: model.addVar(lb=0.0, name=f"x_0_{d}") for d in range(1, D + 1)}

    x_i: Dict[int, Dict[int, Any]] = {}
    mu: Dict[int, Any] = {}
    if I > 0:
        for i in range(1, I + 1):
            sk = str(i)
            plist = sorted(mathcal_D[sk])
            x_i[i] = {d: model.addVar(lb=0.0, name=f"x_{i}_{d}") for d in plist}
            mu[i] = model.addVar(lb=0.0, name=f"mu_{i}")

    if K == 0:
        lambda_var = model.addVar(lb=0.0, ub=0.0, name="lambda")
    else:
        lambda_var = model.addVar(lb=0.0, name="lambda")

    f: Dict[int, Any] = {}
    if K > 0:
        assert empirical_distributions is not None
        for k in range(K):
            f[k] = model.addVars(N[k], lb=-GRB.INFINITY, name=f"f_{k}")

    omega: Dict[Tuple[Tuple[int, ...], int], Any] = {}
    for alpha in all_alpha:
        for d in range(1, D + 1):
            omega[(alpha, d)] = model.addVar(lb=-GRB.INFINITY, name=f"omega_a{alpha}_d{d}")

    terms = []
    if obj_x0_linear is not None:
        terms.append(
            gp.quicksum(float(obj_x0_linear[d]) * x_0[d] for d in range(1, D + 1))
        )

    if K > 0:
        assert empirical_distributions is not None
        terms.append(
            gp.quicksum(
                w_k[k] * gp.quicksum(eta_k_n[k][n] * f[k][n] for n in range(N[k])) for k in range(K)
            )
        )

    terms.append(lambda_var * (theta + theta_bar))
    if I > 0:
        terms.append(gp.quicksum(epsilon_i[i] * mu[i] for i in range(1, I + 1)))

    model.setObjective(gp.quicksum(terms), GRB.MINIMIZE)

    for constraint_idx, (A_row, b_val) in enumerate(zip(A_0, b_0)):
        model.addConstr(
            gp.quicksum(A_row[j] * x_0[j + 1] for j in range(D)) <= b_val,
            name=f"retailer_feas_{constraint_idx}",
        )
    if I > 0:
        for i in range(1, I + 1):
            sk = str(i)
            A_i = A_I[sk]
            b_i_row = b_I[sk]
            plist = sorted(mathcal_D[sk])
            for constraint_idx, (A_row, b_val) in enumerate(zip(A_i, b_i_row)):
                model.addConstr(
                    gp.quicksum(A_row[idx] * x_i[i][plist[idx]] for idx in range(len(plist)))
                    <= b_val * mu[i],
                    name=f"store_{i}_feas_{constraint_idx}",
                )

    hat_zeta: Dict[Tuple[Tuple[int, ...], int, int], float] = {}
    abs_diff: Dict[Tuple[Tuple[int, ...], int, int], List[float]] = {}
    hat_y: Dict[Tuple[int, int, Tuple[int, ...], int, int], Tuple[float, float]] = {}

    for alpha in all_alpha:
        for t in range(1, T + 1):
            for d in range(1, D + 1):
                d_idx = d - 1
                if t <= K:
                    kk = t - 1
                    n_t = alpha[kk]
                    assert empirical_distributions is not None
                    zv = float(empirical_distributions[kk][n_t, d_idx])
                elif t <= K + I:
                    store = t - K
                    zv = float(order_data[store].get(d, 0.0))
                elif t == K + I + 1:
                    zv = Xi_lower[str(d)]
                else:
                    zv = Xi_upper[str(d)]

                key = (alpha, d, t)
                hat_zeta[key] = zv

                if K > 0:
                    assert empirical_distributions is not None
                    diffs = []
                    for k in range(K):
                        nk = alpha[k]
                        xik = float(empirical_distributions[k][nk, d_idx])
                        diffs.append(abs(zv - xik))
                    abs_diff[key] = diffs
                else:
                    abs_diff[key] = []

                if I > 0:
                    for i in range(1, I + 1):
                        if d not in mathcal_D[str(i)]:
                            continue
                        hx = float(order_data[i].get(d, 0.0))
                        hat_y[(i, d, alpha, t)] = (
                            max(0.0, zv - hx),
                            max(0.0, hx - zv),
                        )

    y0: Dict[Tuple[Tuple[int, ...], int, int], Any] = {}
    y0p: Dict[Tuple[Tuple[int, ...], int, int], Any] = {}
    yi: Dict[Tuple[int, int, Tuple[int, ...], int, int], Any] = {}
    yip: Dict[Tuple[int, int, Tuple[int, ...], int, int], Any] = {}

    for alpha in all_alpha:
        for t in range(1, T + 1):
            for d in range(1, D + 1):
                key = (alpha, d, t)
                zv = hat_zeta[key]
                y0[key] = model.addVar(lb=0.0, name=f"y0_a{alpha}_t{t}_d{d}")
                y0p[key] = model.addVar(lb=0.0, name=f"y0p_a{alpha}_t{t}_d{d}")
                model.addConstr(y0[key] >= zv - x_0[d], name=f"c_y0_{alpha}_{t}_{d}")
                model.addConstr(y0p[key] >= x_0[d] - zv, name=f"c_y0p_{alpha}_{t}_{d}")

                if I > 0:
                    for i in range(1, I + 1):
                        if d not in mathcal_D[str(i)]:
                            continue
                        ikey = (i, d, alpha, t)
                        yi[ikey] = model.addVar(lb=0.0, name=f"yi_i{i}_a{alpha}_t{t}_d{d}")
                        yip[ikey] = model.addVar(lb=0.0, name=f"yip_i{i}_a{alpha}_t{t}_d{d}")
                        model.addConstr(
                            yi[ikey] >= mu[i] * zv - x_i[i][d],
                            name=f"c_yi_{i}_{alpha}_{t}_{d}",
                        )
                        model.addConstr(
                            yip[ikey] >= x_i[i][d] - mu[i] * zv,
                            name=f"c_yip_{i}_{alpha}_{t}_{d}",
                        )

    model.update()

    for aidx, alpha in enumerate(all_alpha):
        lhs = gp.quicksum(omega[(alpha, d)] for d in range(1, D + 1))
        if K > 0:
            rhs = gp.quicksum(w_k[k] * f[k][alpha[k]] for k in range(K))
        else:
            rhs = 0.0
        model.addConstr(lhs <= rhs, name=f"omega_agg_{aidx}")

    for aidx, alpha in enumerate(all_alpha):
        for t in range(1, T + 1):
            for d in range(1, D + 1):
                key = (alpha, d, t)
                b0 = float(b_0_coef[d])
                h0 = float(h_0_coef[d])

                lhs = b0 * y0[key] + h0 * y0p[key]

                if I > 0:
                    for i in range(1, I + 1):
                        if d not in mathcal_D[str(i)]:
                            continue
                        ikey = (i, d, alpha, t)
                        bi = float(b_i_coef[i][d])
                        hi = float(h_i_coef[i][d])
                        hy, hyp = hat_y[(i, d, alpha, t)]
                        lhs -= bi * (mu[i] * hy - yi[ikey])
                        lhs -= hi * (mu[i] * hyp - yip[ikey])

                if K > 0:
                    lhs -= gp.quicksum(
                        w_k[k] * lambda_var * abs_diff[key][k] for k in range(K)
                    )

                model.addConstr(
                    lhs <= omega[(alpha, d)],
                    name=f"main_a{aidx}_t{t}_d{d}",
                )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        ox0 = {d: x_0[d].X for d in range(1, D + 1)}
        oxi = None
        omu = None
        if I > 0:
            oxi = {
                i: {d: x_i[i][d].X for d in sorted(mathcal_D[str(i)])}
                for i in range(1, I + 1)
            }
            omu = {i: mu[i].X for i in range(1, I + 1)}
        of = None
        if K > 0:
            of = {k: [f[k][n].X for n in range(N[k])] for k in range(K)}
        return ox0, oxi, omu, lambda_var.X, of, model.ObjVal, model

    print(f"警告：优化未收敛。状态码：{model.status}")
    return None, None, None, float("nan"), None, None, model


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D_ = 2
    N_ = [3]
    K_ = 1
    I_ = 0
    emp = {0: rng.uniform(10, 50, size=(N_[0], D_))}
    Xi_ = {str(d): [0.0, 100.0] for d in range(1, D_ + 1)}
    b0 = {d: 5.0 for d in range(1, D_ + 1)}
    h0 = {d: 0.0 for d in range(1, D_ + 1)}
    g0 = {d: -3.0 for d in range(1, D_ + 1)}
    res = solve_separable_general(
        empirical_distributions=emp,
        order_data={1: {1: 0.0, 2: 0.0}},
        K=K_,
        I=I_,
        D=D_,
        N=N_,
        mathcal_D={"1": {1, 2}},
        Xi=Xi_,
        A_0=[[1.0, 1.0]],
        b_0=[200.0],
        A_I={"1": [[1.0, 0.0], [0.0, 1.0]]},
        b_I={"1": [100.0, 100.0]},
        b_0_coef=b0,
        h_0_coef=h0,
        b_i_coef={},
        h_i_coef={},
        obj_x0_linear=g0,
        theta=1.0,
        output_flag=0,
    )
    print("smoke I=0,K=1 obj:", res[5], "lambda:", res[3])
