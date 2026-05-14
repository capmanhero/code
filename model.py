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

from itertools import product as itertools_product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum


def _wasserstein1_barycenter_lp(
    C: np.ndarray,
    dists: List[np.ndarray],
    w: np.ndarray,
    output_flag: int,
) -> Dict[str, Any]:
    """在已知代价矩阵 C 上解加权 W_1 离散重心 LP（C[i,j]=c(i,j)）。"""
    n = int(C.shape[0])
    K = len(dists)
    for k, p in enumerate(dists):
        if p.shape != (n,):
            raise ValueError(f"第 {k} 个概率向量长度须为 n={n}")
        if not np.isclose(p.sum(), 1.0, atol=1e-6):
            raise ValueError(f"第 {k} 个概率向量之和须为 1")
        if (p < -1e-12).any():
            raise ValueError(f"第 {k} 个概率向量须非负")

    model = gp.Model("W1_Barycenter_K_Discrete")
    model.setParam("OutputFlag", output_flag)

    b = model.addVars(n, lb=0.0, name="barycenter")
    Ts: List[Any] = []
    for k in range(K):
        Ts.append(model.addVars(n, n, lb=0.0, name=f"T_{k}"))

    obj = quicksum(
        float(w[k]) * quicksum(float(C[i, j]) * Ts[k][i, j] for i in range(n) for j in range(n))
        for k in range(K)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstr(quicksum(b[i] for i in range(n)) == 1, "barycenter_sum")

    for i in range(n):
        for k in range(K):
            model.addConstr(quicksum(Ts[k][i, j] for j in range(n)) == b[i], f"T{k}_row_{i}")

    for j in range(n):
        for k in range(K):
            model.addConstr(
                quicksum(Ts[k][i, j] for i in range(n)) == float(dists[k][j]),
                f"T{k}_col_{j}",
            )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"W1 重心求解未得到最优解，Gurobi 状态码 {model.Status}")

    bary = np.array([b[i].X for i in range(n)], dtype=float)
    T_mats = tuple(
        np.array([[Ts[k][i, j].X for j in range(n)] for i in range(n)], dtype=float)
        for k in range(K)
    )
    w1_dists = np.array([float(np.sum(C * T_mats[k])) for k in range(K)], dtype=float)
    return {
        "barycenter": bary,
        "objective_value": float(model.ObjVal),
        "transport_matrices": T_mats,
        "wasserstein1_distances": w1_dists,
        "n_points": n,
        "K": K,
        "weights": w.copy(),
    }


def _grid_row_key(row: np.ndarray, decimals: int = 10) -> Tuple[float, ...]:
    return tuple(np.round(row, decimals))


def _cartesian_product_grid_from_samples(
    sample_arrays: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    对每个维度 d，取所有样本在该维出现过的取值之并（唯一化），再作笛卡尔积作为网格行。
    """
    if not sample_arrays:
        raise ValueError("至少需要一组经验样本")
    D = int(sample_arrays[0].shape[1])
    if D < 1:
        raise ValueError("维度 D 须至少为 1")
    for a in sample_arrays:
        if len(a.shape) != 2 or int(a.shape[1]) != D:
            raise ValueError("各经验分布须为二维数组 (n_k, D) 且 D 一致")
        if a.shape[0] < 1:
            raise ValueError("每组经验分布至少含一个样本点")

    per_dim: List[np.ndarray] = []
    for d in range(D):
        cols = [a[:, d] for a in sample_arrays]
        per_dim.append(np.unique(np.concatenate(cols)))

    grid_tuples = list(itertools_product(*[v.tolist() for v in per_dim]))
    X = np.asarray(grid_tuples, dtype=float)
    return X, per_dim


def _embed_empirical_on_grid(
    samples: np.ndarray,
    X: np.ndarray,
    row_masses: Optional[np.ndarray] = None,
    decimals: int = 10,
) -> np.ndarray:
    """
    将离散经验分布嵌入网格 X 上的概率向量：第 i 行样本携带质量 m_i，同网格点质量相加。

    row_masses 为 None 时等价于 inverse_refined_mr_DRO 中 eta_k_n 缺省：m_i = 1/n_k。
    """
    n = X.shape[0]
    idx_map = {_grid_row_key(X[i], decimals): i for i in range(n)}
    p = np.zeros(n, dtype=float)
    nk = int(samples.shape[0])
    if row_masses is None:
        m = np.ones(nk, dtype=float) / nk
    else:
        m = np.asarray(row_masses, dtype=float).ravel()
        if m.size != nk:
            raise ValueError(f"质量向量长度须等于样本行数 n_k={nk}，当前为 {m.size}")
        if (m < -1e-12).any():
            raise ValueError("样本质量须非负")
        if not np.isclose(m.sum(), 1.0, atol=1e-6):
            raise ValueError(f"样本质量之和须为 1，当前为 {m.sum():.6f}")

    for i in range(nk):
        row = samples[i]
        mass = float(m[i])
        key = _grid_row_key(row, decimals)
        if key in idx_map:
            p[idx_map[key]] += mass
            continue
        matched = None
        for j in range(n):
            if np.allclose(X[j], row, rtol=0.0, atol=1e-8):
                matched = j
                break
        if matched is None:
            raise ValueError(
                f"样本行无法匹配到积网格（请检查数值或是否由各维观测值生成）: {row!r}"
            )
        p[matched] += mass
    return p


def wasserstein1_barycenter_k_discrete(
    empirical_distributions: Mapping[int, np.ndarray],
    weights: Optional[Union[np.ndarray, Mapping[int, float]]] = None,
    eta_k_n: Optional[Mapping[int, np.ndarray]] = None,
    output_flag: int = 0,
) -> Dict[str, Any]:
    """
    在「各维支撑之并」的笛卡尔积网格上，求加权 1-Wasserstein（L1 地面度量）离散重心。

    输入为 inverse_refined_mr_DRO 风格的 ``{k: 样本矩阵 (n_k, D)}``；每个 k 上经验律取
    样本行上的离散分布，再嵌入到公共网格：第 d 维取所有经验样本在该维出现过的值的集合
    S_d，网格点为 ``S_1 × … × S_D``（与题设 (1,3),(2,4) → {(1,3),(1,4),(2,3),(2,4)} 一致）。

    **是否「精确」Wasserstein 重心？**
    - 在 **全空间** \\mathbb{R}^D 上不限制支撑时，W_1 的 Fréchet 均值一般 **不是**
      有限点集上的测度，更无法用该笛卡尔积 **精确** 表示；网格上的解是 **限制重心支撑在该积网格上**
      时的 LP **精确最优**（对该离散化问题零对偶间隙意义下的精确）。
    - 若问题本意就是「在所有网格点上选 b 使 sum_k w_k W_1(b, \\hat P_k) 最小」，则本构造给出该问题的精确解。

    参数:
        empirical_distributions: 键为 k、值为 (n_k, D) 的样本矩阵（与 inverse_refined_mr_DRO 一致）。
        weights: 与 ``sorted(keys)`` 同序的长度 K 向量，或 ``{k: w_k}`` 字典（和为 1）；默认等权。
        eta_k_n: 可选，与 inverse_refined_mr_DRO 中 ``eta_k_n`` 含义一致：``eta_k_n[k]`` 为长度 n_k、
            非负且和为 1 的向量，``eta_k_n[k][i]`` 为第 k 个分布第 i 行样本的质量。
            为 ``None`` 时退化为 ``np.ones(n_k) / n_k``（行等权）。
            若传入字典，须对 ``sorted(empirical_distributions.keys())`` 中每个 k 都给出条目。
        output_flag: Gurobi OutputFlag。

    返回:
        dict：含 ``X``（积网格 n×D）、``per_dimension_support``、``keys_order``、
        ``barycenter``（与 X 行对齐）、``objective_value``、``transport_matrices``、
        ``wasserstein1_distances``、``dimension``、``K``、``weights``。
        网格点数 n = prod_d |S_d|，维数 D 大时指数级增长。
    """
    keys = sorted(empirical_distributions.keys())
    K = len(keys)
    if K < 1:
        raise ValueError("至少需要 1 个经验分布")

    arrs = [np.asarray(empirical_distributions[t], dtype=float) for t in keys]
    X, per_dim = _cartesian_product_grid_from_samples(arrs)
    n, D = int(X.shape[0]), int(X.shape[1])

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

    if eta_k_n is None:
        row_weights_per_k: List[Optional[np.ndarray]] = [None] * K
    else:
        row_weights_per_k = []
        for t in keys:
            if t not in eta_k_n:
                raise ValueError(
                    f"提供 eta_k_n 时须包含每个经验键 k 的质量向量，缺少键 {t!r}"
                )
            eta = np.asarray(eta_k_n[t], dtype=float)
            row_weights_per_k.append(eta)

    dist_vecs = [
        _embed_empirical_on_grid(a, X, row_masses=rw)
        for a, rw in zip(arrs, row_weights_per_k)
    ]
    C = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=-1)

    out = _wasserstein1_barycenter_lp(C, dist_vecs, w, output_flag)
    out["X"] = X
    out["per_dimension_support"] = per_dim
    out["keys_order"] = keys
    out["dimension"] = D
    return out


def solve_multi_reference_l1_only(
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
    推论 ``coro:finite convex reduction--multi_references_l1``（仅多源经验、L1 地面度量）：

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

    mdl = gp.Model("MultiRefL1Only")
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

    print(f"警告：solve_multi_reference_l1_only 未求得最优解，状态码：{mdl.status}")
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
    output_flag: int = 0,
) -> Tuple[
    Optional[float],
    Optional[Dict[int, float]],
    Optional[float],
    Any,
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
    """
    if order_data is None:
        order_data = {}

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
        o_eps = {i: eps_vars[i].X for i in range(1, I + 1)} if I > 0 else {}
        return theta.X, o_eps if I > 0 else None, float(mdl.ObjVal), mdl

    print(f"警告：solve_choosing_theta_epsilon 未求得最优解，状态码：{mdl.status}")
    return None, None, None, mdl


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

    th0, _, obj0, _ = solve_choosing_theta_epsilon(
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

    th_b, _, obj_b, _ = solve_choosing_theta_epsilon(
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

    th_k0, eps_k0, obj_k0, _ = solve_choosing_theta_epsilon(
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

