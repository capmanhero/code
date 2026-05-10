# -*- coding: utf-8 -*-
"""
可分离情形的一般凸约化模型（Gurobi），整合：
- K=0：订单数据主导（类比 order_only_model，λ=0，无 f，|A|=1）
- I=0：多源经验分布（类比 multi_references_model）
- K>0 且 I>0：全文模型

原问题（最小化）：
    inf  ∑_{k} w_k ∑_{n} η_{k,n} f_{k,n}
         + λ (θ + θ̲)
         + ∑_{i} ε_i μ_i
         + [可选] 零售商线性利润项 -∑_d (p_d - c_d) x_{0,d}（与 reference 中实现一致）

s.t. (x_i, μ_i) ∈ \\bar{X}_i, λ ≥ 0, f, ω_{α,d}，
     ∑_d ω_{α,d} ≤ ∑_k w_k f_{k,α_k}  ∀α ∈ A，
     以及正文中的 b,h 与 y,y' 线性化约束。

其中
    \\hat{ζ}_{t,α,d} =
        \\hat{ξ}_{t,α_t,d},           t = 1,…,K
        \\hat{x}_{t-K,d},             t = K+1,…,K+I
        \\underline{Ξ}_d,             t = K+I+1
        \\overline{Ξ}_d,             t = K+I+2

    \\hat{y}_{i,t,α,d}  = (\\hat{ζ} - \\hat{x}_{i,d})^+,
    \\hat{y}'_{i,t,α,d} = (\\hat{x}_{i,d} - \\hat{ζ})^+.
"""

from __future__ import annotations

import importlib.util
import os
from itertools import product as itertools_product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _normalize_xi(Xi: Mapping) -> Dict[str, Sequence[float]]:
    out = {}
    for k, v in Xi.items():
        out[str(k)] = v
    return out


def _default_b_h_from_prices(
    D: int,
    I: int,
    p: Mapping[str, float],
    c: Mapping[str, float],
    p_I: Mapping[str, float],
    c_I: Mapping[str, float],
    mathcal_D: Mapping[str, Any],
    use_margin_in_loss: bool = False,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    """
    报童正部惩罚系数默认值。
    - use_margin_in_loss=False（默认）：与 data_fusion_model / multi_references / order_only 一致，
      主约束中用标价 p、p_I 作 shortage 系数（目标中仍用 p-c 作 x 的线性利润项）。
    - use_margin_in_loss=True：b = p-c, h = 0（对应边际形式）。
    """
    if use_margin_in_loss:
        b_0 = {d: float(p[str(d)] - c[str(d)]) for d in range(1, D + 1)}
        b_i_fn = lambda sd: float(p_I[sd] - c_I[sd])
    else:
        b_0 = {d: float(p[str(d)]) for d in range(1, D + 1)}
        b_i_fn = lambda sd: float(p_I[sd])
    h_0 = {d: 0.0 for d in range(1, D + 1)}
    b_I: Dict[int, Dict[int, float]] = {}
    h_I: Dict[int, Dict[int, float]] = {}
    for i in range(1, I + 1):
        sk = str(i)
        b_I[i] = {}
        h_I[i] = {}
        for d in sorted(mathcal_D[sk]):
            b_I[i][d] = b_i_fn(str(d))
            h_I[i][d] = 0.0
    return b_0, h_0, b_I, h_I


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
    w_k: Optional[np.ndarray] = None,
    eta_k_n: Optional[Mapping[int, np.ndarray]] = None,
    theta: float = 0.0,
    theta_bar: float = 0.0,
    epsilon_i: Optional[Mapping[int, float]] = None,
    b_0_coef: Optional[Mapping[int, float]] = None,
    h_0_coef: Optional[Mapping[int, float]] = None,
    b_i_coef: Optional[Mapping[int, Mapping[int, float]]] = None,
    h_i_coef: Optional[Mapping[int, Mapping[int, float]]] = None,
    p: Optional[Mapping[str, float]] = None,
    c: Optional[Mapping[str, float]] = None,
    p_I: Optional[Mapping[str, float]] = None,
    c_I: Optional[Mapping[str, float]] = None,
    include_retailer_profit: bool = True,
    use_margin_in_loss: bool = False,
    formulation: str = "separable",
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
    求解上述可分离一般模型。

    当 K=0：不创建 f，固定 λ=0，场景集 A = {()}，约束 ∑_d ω_d ≤ 0。
    当 I=0：不创建 x_i, μ_i，目标中无 ε_i μ_i，且 t 仅取 1,…,K 与 K+1,K+2（支撑界）。

    若未给定 b_0,h_0,b_i,h_i，需传入 p,c,p_I,c_I 以构造默认系数（默认与 reference 一致为价格 p，见 use_margin_in_loss）。

    formulation:
        - \"separable\"：按文中 \\hat\\zeta_{t,\\alpha,d}（全局 t，长度 K+I+2）与可分离主约束写法。
        - \"data_fusion\"：与 reference/models/data_fusion_model.py 中 solve_data_fusion_newsvendor 相同
          （T=K+3，\\hat\\zeta 依赖 (i,t)，逐 (\\alpha,d,i,t) 约束；ω 聚合按各店 D_i 求和）。
    """
    if formulation not in ("separable", "data_fusion"):
        raise ValueError('formulation 须为 "separable" 或 "data_fusion"')

    if formulation == "data_fusion":
        if empirical_distributions is None:
            raise ValueError("data_fusion 形式需要 K>0 及 empirical_distributions")
        _root = os.path.dirname(os.path.abspath(__file__))
        _df_path = os.path.join(_root, "reference", "models", "data_fusion_model.py")
        _spec = importlib.util.spec_from_file_location("data_fusion_model", _df_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"无法加载 data_fusion_model: {_df_path}")
        _df = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_df)

        return _df.solve_data_fusion_newsvendor(
            empirical_distributions=empirical_distributions,
            order_data=order_data,
            K=K,
            I=I,
            D=D,
            N=list(N),
            mathcal_D=mathcal_D,
            p=p,  # type: ignore[arg-type]
            c=c,  # type: ignore[arg-type]
            p_I=p_I,  # type: ignore[arg-type]
            c_I=c_I,  # type: ignore[arg-type]
            A_0=A_0,
            b_0=b_0,
            A_I=A_I,
            b_I=b_I,
            Xi=Xi,
            w_k=w_k,
            eta_k_n=eta_k_n,
            theta=theta + theta_bar,
            epsilon_i=epsilon_i,
            output_flag=output_flag,
        )

    if K < 0 or I < 0:
        raise ValueError("K,I 必须非负")
    if K > 0 and empirical_distributions is None:
        raise ValueError("K>0 时必须提供 empirical_distributions")
    if I > 0 and not order_data:
        raise ValueError("I>0 时必须提供 order_data")

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

    if b_0_coef is None or h_0_coef is None or b_i_coef is None or h_i_coef is None:
        if p is None or c is None or p_I is None or c_I is None:
            raise ValueError("未提供 b/h 系数时，必须提供 p, c, p_I, c_I 以构造默认值")
        b_0_coef_d, h_0_coef_d, b_i_coef_d, h_i_coef_d = _default_b_h_from_prices(
            D, I, p, c, p_I, c_I, mathcal_D, use_margin_in_loss=use_margin_in_loss
        )
        b_0_coef = b_0_coef or b_0_coef_d
        h_0_coef = h_0_coef or h_0_coef_d
        b_i_coef = b_i_coef or b_i_coef_d
        h_i_coef = h_i_coef or h_i_coef_d

    T = K + I + 2
    if K == 0:
        all_alpha: List[Tuple[int, ...]] = [()]
    else:
        all_alpha = list(itertools_product(*[range(N[k]) for k in range(K)]))

    model = gp.Model("SeparableGeneral")
    model.setParam("OutputFlag", output_flag)

    # —— 决策变量 ——
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

    # —— 目标 ——
    terms = []
    if include_retailer_profit:
        if p is None or c is None:
            raise ValueError("include_retailer_profit=True 时需要 p, c")
        terms.append(-gp.quicksum((p[str(d)] - c[str(d)]) * x_0[d] for d in range(1, D + 1)))

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

    # 零售商与门店可行域
    for constraint_idx, (A_row, b_val) in enumerate(zip(A_0, b_0)):
        model.addConstr(
            gp.quicksum(A_row[j] * x_0[j + 1] for j in range(D)) <= b_val,
            name=f"retailer_feas_{constraint_idx}",
        )
    if I > 0:
        for i in range(1, I + 1):
            sk = str(i)
            A_i = A_I[sk]
            b_i = b_I[sk]
            plist = sorted(mathcal_D[sk])
            for constraint_idx, (A_row, b_val) in enumerate(zip(A_i, b_i)):
                model.addConstr(
                    gp.quicksum(A_row[idx] * x_i[i][plist[idx]] for idx in range(len(plist)))
                    <= b_val * mu[i],
                    name=f"store_{i}_feas_{constraint_idx}",
                )

    # hat_zeta[alpha,d,t] 与 |zeta - xi_k|、hat_y 常数
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

    # 辅助变量 y0, y0', yi, yi'
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

    # ∑_d ω_{α,d} ≤ ∑_k w_k f_{k,α_k}
    for aidx, alpha in enumerate(all_alpha):
        lhs = gp.quicksum(omega[(alpha, d)] for d in range(1, D + 1))
        if K > 0:
            rhs = gp.quicksum(w_k[k] * f[k][alpha[k]] for k in range(K))
        else:
            rhs = 0.0
        model.addConstr(lhs <= rhs, name=f"omega_agg_{aidx}")

    # 主约束
    for aidx, alpha in enumerate(all_alpha):
        for t in range(1, T + 1):
            for d in range(1, D + 1):
                key = (alpha, d, t)
                zv = hat_zeta[key]
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


def solve_newsvendor_separable(
    empirical_distributions: Optional[Mapping[int, np.ndarray]],
    order_data: Mapping[int, Mapping[int, float]],
    K: int,
    I: int,
    D: int,
    N: Sequence[int],
    mathcal_D: Mapping[str, Any],
    p: Mapping[str, float],
    c: Mapping[str, float],
    p_I: Mapping[str, float],
    c_I: Mapping[str, float],
    A_0: Sequence[Sequence[float]],
    b_0: Sequence[float],
    A_I: Mapping[str, Any],
    b_I: Mapping[str, Any],
    Xi: Mapping,
    w_k: Optional[np.ndarray] = None,
    eta_k_n: Optional[Mapping[int, np.ndarray]] = None,
    theta: float = 0.0,
    theta_bar: float = 0.0,
    epsilon_i: Optional[Mapping[int, float]] = None,
    output_flag: int = 0,
):
    """报童情形快捷接口：b,h 由 p,c,p_I,c_I 默认构造。"""
    return solve_separable_general(
        empirical_distributions=empirical_distributions,
        order_data=order_data,
        K=K,
        I=I,
        D=D,
        N=N,
        mathcal_D=mathcal_D,
        Xi=Xi,
        A_0=A_0,
        b_0=b_0,
        A_I=A_I,
        b_I=b_I,
        w_k=w_k,
        eta_k_n=eta_k_n,
        theta=theta,
        theta_bar=theta_bar,
        epsilon_i=epsilon_i,
        p=p,
        c=c,
        p_I=p_I,
        c_I=c_I,
        include_retailer_profit=True,
        use_margin_in_loss=False,
        formulation="separable",
        output_flag=output_flag,
    )


if __name__ == "__main__":
    # 小规模自检：I=0,K=1 与 multi_references 同型；K=0,I=1 与 order-only 结构一致（索引略异）
    rng = np.random.default_rng(0)
    D_ = 2
    N_ = [3]
    K_ = 1
    I_ = 0
    emp = {0: rng.uniform(10, 50, size=(N_[0], D_))}
    Xi_ = {str(d): [0.0, 100.0] for d in range(1, D_ + 1)}
    p_ = {str(d): 5.0 for d in range(1, D_ + 1)}
    c_ = {str(d): 2.0 for d in range(1, D_ + 1)}
    res = solve_newsvendor_separable(
        empirical_distributions=emp,
        order_data={1: {1: 0.0, 2: 0.0}},
        K=K_,
        I=I_,
        D=D_,
        N=N_,
        mathcal_D={"1": {1, 2}},
        p=p_,
        c=c_,
        p_I=p_,
        c_I=c_,
        A_0=[[1.0, 1.0]],
        b_0=[200.0],
        A_I={"1": [[1.0, 0.0], [0.0, 1.0]]},
        b_I={"1": [100.0, 100.0]},
        Xi=Xi_,
        theta=1.0,
        output_flag=0,
    )
    print("smoke I=0,K=1 obj:", res[5], "lambda:", res[3])
