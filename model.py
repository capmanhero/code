# -*- coding: utf-8 -*-
"""
可分离情形的一般凸约化模型（Gurobi）。

核心求解器 solve_separable_general 只接受数值系数（b,h、目标中 x_0 的线性项等）。

原问题（最小化）：
    inf  ∑_{k} w_k ∑_{n} η_{k,n} f_{k,n}
         + λ (θ + θ̲)
         + ∑_{i} ε_i μ_i
         + ∑_{d} g_d x_{0,d}        （可选，由 obj_x0_linear 给出）

另见 ``solve_multi_reference_l1_only``：多源 L1 半径向量 ``(θ_k)_{k∈[K]}``、各源独立 ``λ_k``，
且 ``∑_d ω_{α,d} ≤ ∑_k f_{k,α_k}``（推论 ``eq:l1-multi-source-only``）。

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

import math
from itertools import product as itertools_product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum


def _ceil_decimals(x: float, decimals: int = 4) -> float:
    """保留 ``decimals`` 位小数并向上取整，如 3.141512 → 3.1416。"""
    scale = 10**decimals
    return math.ceil(float(x) * scale - 1e-12) / scale


def _default_xi_from_empirical(
    empirical_distributions: Mapping[int, np.ndarray],
    K: int,
    D: int,
    floor_hi: float = 25.0,
    margin: float = 1.25,
) -> Dict[str, List[float]]:
    """当未显式给出 Xi 时，由各源样本逐维上界构造盒约束（与 blood.build_Xi 思路一致，简化版）。"""
    hi = np.zeros(D, dtype=float)
    for k in range(K):
        a = np.asarray(empirical_distributions[k], dtype=float)
        for d in range(D):
            hi[d] = max(hi[d], float(np.max(a[:, d])))
    return {
        str(d + 1): [0.0, float(max(floor_hi, hi[d] * margin + 2.0))] for d in range(D)
    }


def _w1_l1_ot_cost(
    locations_a: np.ndarray,
    mass_a: np.ndarray,
    locations_b: np.ndarray,
    mass_b: np.ndarray,
    output_flag: int,
) -> float:
    """离散分布间 L1 地面度量的 W_1（最优传输费用）。"""
    na, nb = int(locations_a.shape[0]), int(locations_b.shape[0])
    mass_a = np.asarray(mass_a, dtype=float).ravel()
    mass_b = np.asarray(mass_b, dtype=float).ravel()
    if na < 1 or nb < 1:
        raise ValueError("支撑至少各含一个点")
    if mass_a.size != na or mass_b.size != nb:
        raise ValueError("质量向量长度与位置行数不一致")
    if not np.isclose(mass_a.sum(), 1.0, atol=1e-7) or not np.isclose(mass_b.sum(), 1.0, atol=1e-7):
        raise ValueError("两侧质量之和均须为 1")
    C = np.abs(locations_a[:, None, :] - locations_b[None, :, :]).sum(axis=-1)
    mdl = gp.Model("W1_OT_L1")
    mdl.setParam("OutputFlag", output_flag)
    x = mdl.addVars(na, nb, lb=0.0, name="pi")
    mdl.setObjective(
        quicksum(float(C[i, j]) * x[i, j] for i in range(na) for j in range(nb)),
        GRB.MINIMIZE,
    )
    for i in range(na):
        mdl.addConstr(quicksum(x[i, j] for j in range(nb)) == float(mass_a[i]), f"row_{i}")
    for j in range(nb):
        mdl.addConstr(quicksum(x[i, j] for i in range(na)) == float(mass_b[j]), f"col_{j}")
    mdl.optimize()
    if mdl.status != GRB.OPTIMAL:
        raise RuntimeError(f"W_1 OT 子问题未最优，状态码 {mdl.status}")
    return float(mdl.ObjVal)


def _barycenter_from_beta_transport(
    hat_zeta: Mapping[Tuple[Tuple[int, ...], int, int], float],
    beta_sol: Mapping[Tuple[Tuple[int, ...], int, int], float],
    gamma_sol: Mapping[Tuple[int, ...], float],
    K: int,
    I: int,
    D: int,
    N: Sequence[int],
    eps_mass: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    由 choosing 最优 β、γ 与 \\hat{ζ} 恢复 \\mathbb{R}^D 上离散质心。

    对任意固定的参考源 k，β_{α,d,t} 表示从经验点 \\hat{ξ}_{k,α_k} 运往标量
    \\hat{ζ}_{α,d,t} 的质量（``∑_t β_{α,d,t}=γ_α``）。
    场景 α 上到达的 D 维落点取坐标加权平均
    ``z_{α,d}=∑_t β_{α,d,t}\\hat{ζ}_{α,d,t}/γ_α``，质量为 ``γ_α``；
    合并重复支撑后归一化为概率向量。
    """
    Tloc = K + I + 2
    all_alpha = list(itertools_product(*[range(N[k]) for k in range(K)]))
    rows: List[np.ndarray] = []
    masses: List[float] = []
    for alpha in all_alpha:
        g = float(gamma_sol.get(alpha, 0.0))
        if g <= eps_mass:
            continue
        z = np.zeros(D, dtype=float)
        for d in range(1, D + 1):
            num = 0.0
            for t in range(1, Tloc + 1):
                b = float(beta_sol.get((alpha, d, t), 0.0))
                num += b * float(hat_zeta[(alpha, d, t)])
            z[d - 1] = num / g
        rows.append(z)
        masses.append(g)
    if not rows:
        raise RuntimeError("choosing 解未能恢复非空质心支撑（γ 全为 0？）")
    Xmt = np.stack(rows, axis=0)
    pm = np.asarray(masses, dtype=float)
    pm = pm / float(pm.sum())
    return _merge_duplicate_support_rows(Xmt, pm)


def _merge_duplicate_support_rows(
    X: np.ndarray,
    p: np.ndarray,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    agg: Dict[Tuple[float, ...], float] = {}
    for i in range(X.shape[0]):
        key = tuple(np.round(X[i], decimals))
        agg[key] = agg.get(key, 0.0) + float(p[i])
    Xo = np.array([list(k) for k in agg], dtype=float)
    po = np.array([agg[k] for k in agg], dtype=float)
    s = float(po.sum())
    if s <= 0.0:
        raise ValueError("合并后总质量为 0")
    po = po / s
    return Xo, po


def wasserstein1_barycenter_k_discrete(
    empirical_distributions: Mapping[int, np.ndarray],
    weights: Optional[Union[np.ndarray, Mapping[int, float]]] = None,
    eta_k_n: Optional[Mapping[int, np.ndarray]] = None,
    Xi: Optional[Mapping] = None,
    output_flag: int = 0,
) -> Dict[str, Any]:
    """
    通过 ``solve_choosing_theta_epsilon``（I=0、θ̲=0）求解 Wasserstein 预算，并由最优
    ``β_{α,d,t}`` 恢复离散质心。

    **β 的含义**（可任意固定参考源 k）：``β_{α,d,t}`` 表示从经验点
    ``\\hat{ξ}_{k,α_k}`` 上将质量运往目标标量 ``\\hat{ζ}_{α,d,t}`` 的数量
    （``∑_t β_{α,d,t}=γ_α``）。场景 α 的质心落点为
    ``z_{α,d}=∑_t β_{α,d,t}\\hat{ζ}_{α,d,t}/γ_α``，质量 ``γ_α``，
    见 ``_barycenter_from_beta_transport``。

    须给定盒约束 ``Xi``；为 ``None`` 时见 ``_default_xi_from_empirical``。

    返回 ``objective_value`` 为 choosing 最优 θ；``wasserstein1_distances[k]`` 为恢复出的
    离散质心 ``(X,b)`` 与各经验源 ``P_k`` 之间 L1 地面度量下的 OT ``W_1``。
    另含 ``choosing_solution``（β、γ、\\hat{ζ} 等）及 ``wasserstein_linear_contrib``（模型线性项分解）。
    """
    keys = sorted(empirical_distributions.keys())
    K = len(keys)
    if K < 1:
        raise ValueError("至少需要 1 个经验分布")

    emp_solve = {j: np.asarray(empirical_distributions[t], dtype=float) for j, t in enumerate(keys)}
    D = int(next(iter(emp_solve.values())).shape[1])
    N = [int(emp_solve[k].shape[0]) for k in range(K)]

    if weights is None:
        w = np.ones(K, dtype=float) / K
    elif isinstance(weights, Mapping):
        w = np.array([float(weights[t]) for t in keys], dtype=float)
        if w.shape != (K,):
            raise ValueError("weights 字典须对每个经验键 k 给出一个权重")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(f"权重之和须为 1，当前为 {w.sum():.6f}")
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != K:
            raise ValueError(f"weights 长度须为 K={K}（与 sorted(keys) 一致）")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(f"权重之和须为 1，当前为 {w.sum():.6f}")

    eta_ch: Optional[Dict[int, np.ndarray]] = None
    if eta_k_n is not None:
        eta_ch = {}
        for j, t in enumerate(keys):
            if t not in eta_k_n:
                raise ValueError(f"提供 eta_k_n 时须包含每个经验键 k 的质量向量，缺少键 {t!r}")
            eta_ch[j] = np.asarray(eta_k_n[t], dtype=float)

    Xi_use = Xi if Xi is not None else _default_xi_from_empirical(emp_solve, K, D)
    Xi_n = _normalize_xi(Xi_use)

    th_star, _eps, obj_val, _mdl, sol = solve_choosing_theta_epsilon(
        K=K,
        I=0,
        D=D,
        N=N,
        mathcal_D={},
        Xi=Xi_n,
        empirical_distributions=emp_solve,
        eta_k_n=eta_ch,
        w_k=w,
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    if th_star is None or obj_val is None or sol is None:
        raise RuntimeError("solve_choosing_theta_epsilon 未得到最优解，无法构造重心输出")

    hat_z = sol["hat_zeta"]
    beta_sol = sol["beta"]
    gamma_sol = sol["gamma"]
    abs_diff_sol = sol["abs_diff"]
    X, b = _barycenter_from_beta_transport(hat_z, beta_sol, gamma_sol, K, 0, D, N)

    w1_lin: List[float] = []
    for kk in range(K):
        sk = 0.0
        for key, difs in abs_diff_sol.items():
            sk += float(beta_sol.get(key, 0.0)) * float(difs[kk])
        w1_lin.append(sk)
    w1_lin_arr = np.asarray(w1_lin, dtype=float)

    w1_list: List[float] = []
    for kk in range(K):
        Pk = np.asarray(emp_solve[kk], dtype=float)
        if eta_ch is None:
            pk = np.ones(N[kk], dtype=float) / N[kk]
        else:
            pk = np.asarray(eta_ch[kk], dtype=float).ravel()
        w1_list.append(_w1_l1_ot_cost(X, b, Pk, pk, output_flag))
    w1_dists = np.asarray(w1_list, dtype=float)

    return {
        "X": X,
        "barycenter": b.astype(float),
        "objective_value": float(obj_val),
        "wasserstein1_distances": w1_dists,
        "wasserstein_linear_contrib": w1_lin_arr,
        "transport_matrices": (),
        "n_points": int(X.shape[0]),
        "K": K,
        "weights": w.copy(),
        "keys_order": keys,
        "dimension": D,
        "per_dimension_support": [],
        "choosing_solution": sol,
    }


def solve_multi_reference_intersection(
    empirical_distributions: Mapping[int, np.ndarray],
    theta_k: Mapping[int, float],
    Xi: Mapping,
    A_0: Sequence[Sequence[float]],
    b_0: Sequence[float],
    b_0_coef: Mapping[int, float],
    h_0_coef: Mapping[int, float],
    obj_x0_linear: Optional[Mapping[int, float]] = None,
    eta_k_n: Optional[Mapping[int, np.ndarray]] = None,
    output_flag: int = 0,
) -> Tuple[
    Optional[Dict[int, float]],
    Optional[Dict[int, float]],
    Optional[Dict[int, List[float]]],
    float,
    Any,
]:
    """
        inf  ∑_{k,n} η_{k,n} f_{k,n} + ∑_{k} λ_k θ_k
        s.t. λ_k ≥ 0, f_{k,n}, ω_{α,d} ∈ ℝ,
             ∑_d ω_{α,d} ≤ ∑_{k} f_{k,α_k}   ∀α ∈ A = ∏_k [N_k],
             b_{0,d} y_{t,α,d} + h_{0,d} y'_{t,α,d}
                 − ∑_{k} λ_k |\\hat{ζ}_{t,α,d} − \\hat{ξ}_{k,α_k,d}|
                 ≤ ω_{α,d}
             y_{t,α,d} ≥ \\hat{ζ}_{t,α,d} − x_{0,d},
             y'_{t,α,d} ≥ x_{0,d} − \\hat{ζ}_{t,α,d}
             ∀α, d, t ∈ [K+2].

    其中 \\hat{ζ}_{t,α,d} 在 t∈[K] 为 \\hat{ξ}_{t,α_t,d}，t=K+1 为 \\underline{Ξ}_d，t=K+2 为 \\overline{Ξ}_d。

    与 ``inverse_refined_mr_DRO``（I=0）的差异：各源独立 ``λ_k``；Wasserstein 项为
    ``∑_k λ_k |·|``；聚合约束右端为 ``∑_k f_{k,α_k}``（无 ``w_k`` 系数）。

    参数 ``empirical_distributions`` 的键须为 ``0,…,K−1``（与仓库中其它求解器一致）；
    ``theta_k`` 须含每个键 ``k`` 的半径 ``θ_k``。
    """
    keys = sorted(empirical_distributions.keys())
    K = len(keys)
    if K < 1:
        raise ValueError("至少需要 1 个经验分布")
    arrs = [np.asarray(empirical_distributions[t], dtype=float) for t in keys]
    D = int(arrs[0].shape[1])
    if D < 1:
        raise ValueError("维度 D 须至少为 1")
    N: List[int] = []
    for k, a in enumerate(arrs):
        if len(a.shape) != 2 or int(a.shape[1]) != D:
            raise ValueError(f"经验分布 k={keys[k]} 须为 (n_k, D) 且 D={D}")
        N.append(int(a.shape[0]))
        if N[-1] < 1:
            raise ValueError(f"分布 k={keys[k]} 至少含一个样本点")

    for k in keys:
        if k not in theta_k:
            raise ValueError(f"theta_k 须包含键 {k!r}")

    Xi_n = _normalize_xi(Xi)
    Xi_lower = {str(d): float(Xi_n[str(d)][0]) for d in range(1, D + 1)}
    Xi_upper = {str(d): float(Xi_n[str(d)][1]) for d in range(1, D + 1)}

    if eta_k_n is None:
        eta_map = {keys[k]: np.ones(N[k], dtype=float) / N[k] for k in range(K)}
    else:
        eta_map = {}
        for t in keys:
            if t not in eta_k_n:
                raise ValueError(f"提供 eta_k_n 时须包含每个键，缺少 {t!r}")
            eta = np.asarray(eta_k_n[t], dtype=float).ravel()
            if eta.size != int(empirical_distributions[t].shape[0]):
                raise ValueError(f"eta_k_n[{t}] 长度须等于该分布样本数")
            if not np.isclose(eta.sum(), 1.0, atol=1e-6):
                raise ValueError(f"eta_k_n[{t}] 之和须为 1")
            eta_map[t] = eta

    # 重排为 0..K-1 下标以便与 alpha 元组对齐
    emp_by_idx = {i: arrs[i] for i in range(K)}
    eta_by_idx = {i: eta_map[keys[i]] for i in range(K)}
    theta_by_idx = {i: float(theta_k[keys[i]]) for i in range(K)}

    all_alpha = list(itertools_product(*[range(N[k]) for k in range(K)]))
    T = K + 2

    mdl = gp.Model("MultiRefIntersection") # 多源经验交集
    mdl.setParam("OutputFlag", output_flag)

    x_0 = {d: mdl.addVar(lb=0.0, name=f"x_0_{d}") for d in range(1, D + 1)}
    lam = {i: mdl.addVar(lb=0.0, name=f"lambda_{i}") for i in range(K)}
    f: Dict[int, Any] = {
        i: mdl.addVars(N[i], lb=-GRB.INFINITY, name=f"f_{i}") for i in range(K)
    }
    omega: Dict[Tuple[Tuple[int, ...], int], Any] = {}
    for alpha in all_alpha:
        for d in range(1, D + 1):
            omega[(alpha, d)] = mdl.addVar(lb=-GRB.INFINITY, name=f"omega_a{alpha}_d{d}")

    terms = []
    if obj_x0_linear is not None:
        terms.append(
            gp.quicksum(float(obj_x0_linear[d]) * x_0[d] for d in range(1, D + 1))
        )
    terms.append(
        gp.quicksum(
            gp.quicksum(float(eta_by_idx[k][n]) * f[k][n] for n in range(N[k])) for k in range(K)
        )
    )
    terms.append(gp.quicksum(lam[k] * theta_by_idx[k] for k in range(K)))
    mdl.setObjective(gp.quicksum(terms), GRB.MINIMIZE)

    for constraint_idx, (A_row, b_val) in enumerate(zip(A_0, b_0)):
        mdl.addConstr(
            gp.quicksum(A_row[j] * x_0[j + 1] for j in range(D)) <= b_val,
            name=f"retailer_feas_{constraint_idx}",
        )

    hat_zeta: Dict[Tuple[Tuple[int, ...], int, int], float] = {}
    abs_diff: Dict[Tuple[Tuple[int, ...], int, int], List[float]] = {}

    for alpha in all_alpha:
        for t in range(1, T + 1):
            for d in range(1, D + 1):
                d_idx = d - 1
                if t <= K:
                    kk = t - 1
                    n_t = alpha[kk]
                    zv = float(emp_by_idx[kk][n_t, d_idx])
                elif t == K + 1:
                    zv = Xi_lower[str(d)]
                else:
                    zv = Xi_upper[str(d)]
                key = (alpha, d, t)
                hat_zeta[key] = zv
                diffs = []
                for kk in range(K):
                    nk = alpha[kk]
                    xik = float(emp_by_idx[kk][nk, d_idx])
                    diffs.append(abs(zv - xik))
                abs_diff[key] = diffs

    y0: Dict[Tuple[Tuple[int, ...], int, int], Any] = {}
    y0p: Dict[Tuple[Tuple[int, ...], int, int], Any] = {}
    for alpha in all_alpha:
        for t in range(1, T + 1):
            for d in range(1, D + 1):
                key = (alpha, d, t)
                zv = hat_zeta[key]
                y0[key] = mdl.addVar(lb=0.0, name=f"y0_a{alpha}_t{t}_d{d}")
                y0p[key] = mdl.addVar(lb=0.0, name=f"y0p_a{alpha}_t{t}_d{d}")
                mdl.addConstr(y0[key] >= zv - x_0[d], name=f"c_y0_{alpha}_{t}_{d}")
                mdl.addConstr(y0p[key] >= x_0[d] - zv, name=f"c_y0p_{alpha}_{t}_{d}")

    mdl.update()

    for aidx, alpha in enumerate(all_alpha):
        lhs = gp.quicksum(omega[(alpha, d)] for d in range(1, D + 1))
        rhs = gp.quicksum(f[k][alpha[k]] for k in range(K))
        mdl.addConstr(lhs <= rhs, name=f"omega_agg_{aidx}")

    for aidx, alpha in enumerate(all_alpha):
        for t in range(1, T + 1):
            for d in range(1, D + 1):
                key = (alpha, d, t)
                b0 = float(b_0_coef[d])
                h0 = float(h_0_coef[d])
                lhs = (
                    b0 * y0[key]
                    + h0 * y0p[key]
                    - gp.quicksum(lam[k] * abs_diff[key][k] for k in range(K))
                )
                mdl.addConstr(lhs <= omega[(alpha, d)], name=f"main_a{aidx}_t{t}_d{d}")

    mdl.optimize()

    if mdl.status == GRB.OPTIMAL:
        ox0 = {d: x_0[d].X for d in range(1, D + 1)}
        olam = {keys[k]: lam[k].X for k in range(K)}
        of = {keys[k]: [f[k][n].X for n in range(N[k])] for k in range(K)}
        return ox0, olam, of, float(mdl.ObjVal), mdl

    print(f"警告：solve_multi_reference_intersection 未求得最优解，状态码：{mdl.status}") 
    return None, None, None, float("nan"), mdl


def _normalize_xi(Xi: Mapping) -> Dict[str, Sequence[float]]:
    out = {}
    for k, v in Xi.items():
        out[str(k)] = v
    return out


def inverse_refined_mr_DRO(
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


def solve_choosing_theta_epsilon(
    K: int,
    I: int,
    D: int,
    N: Sequence[int],
    mathcal_D: Mapping[str, Any],
    Xi: Mapping,
    empirical_distributions: Optional[Mapping[int, np.ndarray]],
    b_i_coef: Optional[Mapping[int, Mapping[int, float]]] = None,
    h_i_coef: Optional[Mapping[int, Mapping[int, float]]] = None,
    A_I: Optional[Mapping[str, Any]] = None,
    b_I: Optional[Mapping[str, Any]] = None,
    order_data: Optional[Mapping[int, Mapping[int, float]]] = None,
    eta_k_n: Optional[Mapping[int, np.ndarray]] = None,
    w_k: Optional[np.ndarray] = None,
    theta_bar: float = 0.0,
    tau: float = 1.0,
    epsilon_norm: str = "l1",
    k0_beta_mass: float = 1.0,
    epsilon_i_fixed: Optional[Mapping[int, float]] = None,
    output_flag: int = 0,
) -> Tuple[
    Optional[float],
    Optional[Dict[int, float]],
    Optional[float],
    Any,
    Optional[Dict[str, Any]],
]:
    """
    根据文中「选择参数」模型的对偶 LP（式 choosing parameter）求解

        min  θ + τ ‖ε‖

    其中 ε_i ≥ 0，θ ≥ 0；‖·‖ 由 epsilon_norm 指定：'l1' 为 ∑_i ε_i，'linf' 为 max_i ε_i。

    (κ_i, ρ_i) ∈ \\bar{X}_i^* 对应原锥 {(x_i, μ_i): x_i ≥ 0, A_i x_i ≤ b_i μ_i} 的对偶外表示：
    存在 ν_{i,r} ≥ 0 使得 κ_{i,d} + ∑_r A_{i,r,j(d)} ν_{i,r} ≥ 0（每个 d ∈ plist），
    且 ρ_i − ∑_r b_{i,r} ν_{i,r} ≥ 0；j(d) 为 d 在 plist 中的列下标。

    特殊情形：
      - I = 0：无 ψ, ψ', κ, ρ, ν, ε；仅 γ, β, θ 与 Wasserstein / 质量守恒约束。
      - I = 0 且 θ̲ = 0：最小 θ 为加权 Wasserstein 型对偶预算（K≥2 时典型为正；K=1 时
        可将质量置于 t∈[K] 使 |\\hat{ζ}-\\hat{ξ}| 项为 0，常出现 θ^*=0 的退化）。
      - K = 0：无经验分布与 γ = η；不设 Wasserstein 项（左端为 0）；用约束
        ∑_{d,t} β_{(),d,t} = k0_beta_mass 固定质量标度（默认 1），否则尺度不定。

    参数 k0_beta_mass 仅在 K = 0 时使用。

    参数 ``epsilon_i_fixed``：若给定，则将对应 ``ε_i`` 固定为该常数（仅优化其余变量与 θ）。

    返回:
      ``(theta*, ε 或 None, objective, gurobi_model, choosing_snapshot 或 None)``。
      ``theta*`` 与各 ``ε_i`` 在返回前保留 4 位小数并向上取整。
      求得最优解时第五项为字典，含 ``beta``、``gamma``、``hat_zeta``（数值化）、以及 ``K,I,D,N`` 等元数据；
      未最优时为 ``None``。
    """
    if order_data is None:
        order_data = {}
    if epsilon_i_fixed is not None:
        epsilon_i_fixed = {int(i): float(v) for i, v in epsilon_i_fixed.items()}

    if K < 0 or I < 0:
        raise ValueError("K,I 必须非负")
    if K > 0 and empirical_distributions is None:
        raise ValueError("K>0 时必须提供 empirical_distributions")
    if I > 0 and (not order_data or b_i_coef is None or h_i_coef is None):
        raise ValueError("I>0 时必须提供 order_data、b_i_coef、h_i_coef")
    if I > 0 and (A_I is None or b_I is None or not A_I or not b_I):
        raise ValueError("I>0 时必须提供 A_I、b_I（用于 \\bar{X}_i^*）")
    if epsilon_norm not in ("l1", "linf"):
        raise ValueError("epsilon_norm 须为 'l1' 或 'linf'")

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

    T = K + I + 2
    if K == 0:
        all_alpha: List[Tuple[int, ...]] = [()]
    else:
        all_alpha = list(itertools_product(*[range(N[k]) for k in range(K)]))

    b_i_coef = b_i_coef or {}
    h_i_coef = h_i_coef or {}

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
                    for kk in range(K):
                        nk = alpha[kk]
                        xik = float(empirical_distributions[kk][nk, d_idx])
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

    mdl = gp.Model("ChoosingThetaEpsilon")
    mdl.setParam("OutputFlag", output_flag)

    theta = mdl.addVar(lb=0.0, name="theta")
    eps_vars: Dict[int, Any] = {}
    if I > 0:
        for i in range(1, I + 1):
            if epsilon_i_fixed is not None and i in epsilon_i_fixed:
                ev = float(epsilon_i_fixed[i])
                eps_vars[i] = mdl.addVar(lb=ev, ub=ev, name=f"eps_{i}")
            else:
                eps_vars[i] = mdl.addVar(lb=0.0, name=f"eps_{i}")

    if epsilon_norm == "l1":
        if I > 0:
            obj_expr = theta + tau * gp.quicksum(eps_vars[i] for i in range(1, I + 1))
        else:
            obj_expr = theta
    else:
        if I > 0:
            z_linf = mdl.addVar(lb=0.0, name="eps_linf")
            for i in range(1, I + 1):
                mdl.addConstr(z_linf >= eps_vars[i], name=f"eps_linf_dom_{i}")
            obj_expr = theta + tau * z_linf
        else:
            obj_expr = theta

    mdl.setObjective(obj_expr, GRB.MINIMIZE)

    gamma: Dict[Tuple[int, ...], Any] = {}
    for alpha in all_alpha:
        gamma[alpha] = mdl.addVar(lb=0.0, name=f"gamma_a{alpha}")

    beta: Dict[Tuple[Tuple[int, ...], int, int], Any] = {}
    for alpha in all_alpha:
        for d in range(1, D + 1):
            for t in range(1, T + 1):
                beta[(alpha, d, t)] = mdl.addVar(lb=0.0, name=f"beta_a{alpha}_d{d}_t{t}")

    psi: Dict[Tuple[Tuple[int, ...], int, int, int], Any] = {}
    psi_p: Dict[Tuple[Tuple[int, ...], int, int, int], Any] = {}
    if I > 0:
        for alpha in all_alpha:
            for i in range(1, I + 1):
                sk = str(i)
                for d in sorted(mathcal_D[sk]):
                    for t in range(1, T + 1):
                        keyp = (alpha, i, d, t)
                        psi[keyp] = mdl.addVar(lb=0.0, name=f"psi_a{alpha}_i{i}_d{d}_t{t}")
                        psi_p[keyp] = mdl.addVar(lb=0.0, name=f"psip_a{alpha}_i{i}_d{d}_t{t}")

    kappa: Dict[Tuple[int, int], Any] = {}
    rho: Dict[int, Any] = {}
    nu: Dict[Tuple[int, int], Any] = {}
    if I > 0:
        for i in range(1, I + 1):
            sk = str(i)
            plist = sorted(mathcal_D[sk])
            rho[i] = mdl.addVar(lb=-GRB.INFINITY, name=f"rho_{i}")
            for d in plist:
                kappa[(i, d)] = mdl.addVar(lb=-GRB.INFINITY, name=f"kappa_{i}_d{d}")
            A_i = A_I[sk]
            b_i_row = b_I[sk]
            for r in range(len(b_i_row)):
                nu[(i, r)] = mdl.addVar(lb=0.0, name=f"nu_{i}_r{r}")

    if K > 0:
        wasserstein_lhs = gp.quicksum(
            w_k[k]
            * gp.quicksum(
                beta[(alpha, d, t)] * abs_diff[(alpha, d, t)][k]
                for alpha in all_alpha
                for d in range(1, D + 1)
                for t in range(1, T + 1)
            )
            for k in range(K)
        )
        mdl.addConstr(wasserstein_lhs <= theta_bar + theta, name="wasserstein_budget")
    else:
        mdl.addConstr(0.0 <= theta_bar + theta, name="wasserstein_budget_k0")

    if K > 0:
        for k in range(K):
            for n in range(N[k]):
                mdl.addConstr(
                    gp.quicksum(gamma[alpha] for alpha in all_alpha if alpha[k] == n)
                    == float(eta_k_n[k][n]),
                    name=f"gamma_mass_k{k}_n{n}",
                )
    else:
        mdl.addConstr(
            gp.quicksum(
                beta[(alpha, d, t)]
                for alpha in all_alpha
                for d in range(1, D + 1)
                for t in range(1, T + 1)
            )
            == float(k0_beta_mass),
            name="k0_beta_total_mass",
        )

    for aidx, alpha in enumerate(all_alpha):
        for d in range(1, D + 1):
            mdl.addConstr(
                gp.quicksum(beta[(alpha, d, t)] for t in range(1, T + 1)) == gamma[alpha],
                name=f"beta_sum_gamma_a{aidx}_d{d}",
            )

    if I > 0:
        for alpha in all_alpha:
            for i in range(1, I + 1):
                sk = str(i)
                for d in sorted(mathcal_D[sk]):
                    for t in range(1, T + 1):
                        keyp = (alpha, i, d, t)
                        bi = float(b_i_coef[i][d])
                        hi = float(h_i_coef[i][d])
                        mdl.addConstr(
                            -bi * beta[(alpha, d, t)] + psi[keyp] <= 0,
                            name=f"psi_bd_a{alpha}_i{i}_d{d}_t{t}",
                        )
                        mdl.addConstr(
                            -hi * beta[(alpha, d, t)] + psi_p[keyp] <= 0,
                            name=f"psip_hd_a{alpha}_i{i}_d{d}_t{t}",
                        )

        for i in range(1, I + 1):
            sk = str(i)
            plist = sorted(mathcal_D[sk])
            A_i = A_I[sk]
            b_i_row = b_I[sk]
            for d in plist:
                mdl.addConstr(
                    gp.quicksum(
                        psi[(alpha, i, d, t)] - psi_p[(alpha, i, d, t)]
                        for alpha in all_alpha
                        for t in range(1, T + 1)
                    )
                    + kappa[(i, d)]
                    == 0.0,
                    name=f"kappa_balance_i{i}_d{d}",
                )
            mdl.addConstr(
                rho[i]
                - gp.quicksum(float(b_i_row[r]) * nu[(i, r)] for r in range(len(b_i_row)))
                >= 0.0,
                name=f"dualcone_rho_i{i}",
            )
            for j, d in enumerate(plist):
                mdl.addConstr(
                    kappa[(i, d)]
                    + gp.quicksum(
                        float(A_i[r][j]) * nu[(i, r)] for r in range(len(b_i_row))
                    )
                    >= 0.0,
                    name=f"dualcone_kappa_i{i}_d{d}",
                )

        for i in range(1, I + 1):
            sk = str(i)
            plist = sorted(mathcal_D[sk])
            eps_sum = gp.LinExpr()
            for alpha in all_alpha:
                for d in plist:
                    for t in range(1, T + 1):
                        keyp = (alpha, i, d, t)
                        hy, hyp = hat_y[(i, d, alpha, t)]
                        bi = float(b_i_coef[i][d])
                        hi = float(h_i_coef[i][d])
                        zv = hat_zeta[(alpha, d, t)]
                        eps_sum += (
                            beta[(alpha, d, t)] * (bi * hy + hi * hyp)
                            - zv * (psi[keyp] - psi_p[keyp])
                        )
            mdl.addConstr(eps_sum + rho[i] == eps_vars[i], name=f"epsilon_def_i{i}")

    mdl.optimize()

    if mdl.status == GRB.OPTIMAL:
        th_out = _ceil_decimals(theta.X)
        o_eps = (
            {i: _ceil_decimals(eps_vars[i].X) for i in range(1, I + 1)} if I > 0 else {}
        )
        beta_sol: Dict[Tuple[Tuple[int, ...], int, int], float] = {
            key: float(beta[key].X) for key in beta
        }
        gamma_sol: Dict[Tuple[int, ...], float] = {
            alpha: float(gamma[alpha].X) for alpha in all_alpha
        }
        hat_z_sol = {k: float(v) for k, v in hat_zeta.items()}
        abs_diff_sol: Dict[Tuple[Tuple[int, ...], int, int], List[float]] = {
            k: [float(vv) for vv in v] for k, v in abs_diff.items()
        }
        choosing_sol: Dict[str, Any] = {
            "beta": beta_sol,
            "gamma": gamma_sol,
            "hat_zeta": hat_z_sol,
            "abs_diff": abs_diff_sol,
            "K": K,
            "I": I,
            "D": D,
            "N": list(N),
            "w_k": np.asarray(w_k, dtype=float).copy() if K > 0 else np.array([]),
            "theta_bar": float(theta_bar),
            "tau": float(tau),
        }
        return (
            th_out,
            o_eps if I > 0 else None,
            float(mdl.ObjVal),
            mdl,
            choosing_sol,
        )

    print(f"警告：solve_choosing_theta_epsilon 未求得最优解，状态码：{mdl.status}")
    return None, None, None, mdl, None


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
    res = inverse_refined_mr_DRO(
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

    th0, _, obj0, _, _ = solve_choosing_theta_epsilon(
        K=K_,
        I=0,
        D=D_,
        N=N_,
        mathcal_D={},
        Xi=Xi_,
        empirical_distributions=emp,
        theta_bar=2.0,
        tau=1.0,
        output_flag=0,
    )
    print("choosing theta I=0: theta*=", th0, "obj=", obj0)

    th_b, _, obj_b, _, _ = solve_choosing_theta_epsilon(
        K=2,
        I=0,
        D=D_,
        N=[3, 3],
        mathcal_D={},
        Xi=Xi_,
        empirical_distributions={
            0: rng.uniform(10, 50, size=(3, D_)),
            1: rng.uniform(10, 50, size=(3, D_)),
        },
        theta_bar=0.0,
        tau=1.0,
        output_flag=0,
    )
    print("choosing theta I=0 barycenter K=2 (theta_bar=0): theta*=", th_b, "obj=", obj_b)

    th_k0, eps_k0, obj_k0, _, _ = solve_choosing_theta_epsilon(
        K=0,
        I=1,
        D=D_,
        N=[],
        mathcal_D={"1": {1, 2}},
        Xi=Xi_,
        empirical_distributions=None,
        order_data={1: {1: 10.0, 2: 20.0}},
        b_i_coef={1: {1: 1.0, 2: 1.0}},
        h_i_coef={1: {1: 0.0, 2: 0.0}},
        A_I={"1": [[1.0, 0.0], [0.0, 1.0]]},
        b_I={"1": [100.0, 100.0]},
        k0_beta_mass=1.0,
        output_flag=0,
    )
    print("choosing theta K=0,I=1: theta*=", th_k0, "eps=", eps_k0, "obj=", obj_k0)

    rng3 = np.random.default_rng(7)
    D3 = 2
    emp3 = {
        0: rng3.uniform(8, 16, size=(3, D3)),
        1: rng3.uniform(9, 17, size=(3, D3)),
        2: rng3.uniform(7, 15, size=(3, D3)),
    }

