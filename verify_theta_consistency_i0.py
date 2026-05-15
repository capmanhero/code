# -*- coding: utf-8 -*-
"""
I=0 时验证：solve_choosing_theta_epsilon 在 theta_bar_input=0 下得到的最优 theta*
与 inverse_refined_mr_DRO 在固定 theta=0 时「使模型求得最优解」的最小 theta_bar 一致。

记号说明（与 model.py 一致）：
  - choosing 问题：min theta，约束 W <= theta_bar_input + theta（theta 为决策变量）。
  - inverse：目标中含 lambda * (theta + theta_bar)，二者均为传入常数。

当 theta_bar_input = 0 且最优时通常 W_min = theta*（紧），故 theta* 应等于 inverse 在 theta=0 时所需的最小 theta_bar。
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

import model
from gurobipy import GRB


def _inverse_optimal(
    *,
    theta: float,
    theta_bar: float,
    D: int,
    K: int,
    N: list,
    emp: dict,
    Xi: dict,
    w_k: np.ndarray | None,
    eta: dict,
    output_flag: int,
) -> bool:
    """I=0、无零售商目标线性项时，inverse 是否求得 OPTIMAL。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = model.inverse_refined_mr_DRO(
            empirical_distributions=emp,
            order_data={},
            K=K,
            I=0,
            D=D,
            N=N,
            mathcal_D={},
            Xi=Xi,
            A_0=[[1.0] * D],
            b_0=[1e9],
            A_I={},
            b_I={},
            b_0_coef={d: 1.0 for d in range(1, D + 1)},
            h_0_coef={d: 0.0 for d in range(1, D + 1)},
            obj_x0_linear=None,
            w_k=w_k,
            eta_k_n=eta,
            theta=theta,
            theta_bar=theta_bar,
            output_flag=output_flag,
        )
    _x0, _xi, _mu, _lam, _f, obj, m = r
    if _x0 is None or m is None:
        return False
    return m.status == GRB.OPTIMAL


def min_feasible_theta_bar_inverse(
    *,
    D: int,
    K: int,
    N: list,
    emp: dict,
    Xi: dict,
    w_k: np.ndarray | None,
    eta: dict,
    hi: float = 500.0,
    tol: float = 1e-5,
    output_flag: int = 0,
) -> float:
    """在 theta=0 下对 theta_bar 做二分，找使 inverse 为 OPTIMAL 的最小值（上界 hi 须足够大）。"""
    if not _inverse_optimal(
        theta=0.0,
        theta_bar=hi,
        D=D,
        K=K,
        N=N,
        emp=emp,
        Xi=Xi,
        w_k=w_k,
        eta=eta,
        output_flag=output_flag,
    ):
        raise RuntimeError(f"theta_bar={hi} 仍不可行，请增大 hi")

    lo = 0.0
    if _inverse_optimal(
        theta=0.0,
        theta_bar=lo,
        D=D,
        K=K,
        N=N,
        emp=emp,
        Xi=Xi,
        w_k=w_k,
        eta=eta,
        output_flag=output_flag,
    ):
        return lo

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _inverse_optimal(
            theta=0.0,
            theta_bar=mid,
            D=D,
            K=K,
            N=N,
            emp=emp,
            Xi=Xi,
            w_k=w_k,
            eta=eta,
            output_flag=output_flag,
        ):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    rng = np.random.default_rng(42)
    D = 2
    K = 2
    N = [3, 3]
    emp = {
        0: rng.uniform(10, 50, size=(N[0], D)),
        1: rng.uniform(10, 50, size=(N[1], D)),
    }
    Xi = {str(d): [0.0, 100.0] for d in range(1, D + 1)}
    w_k = np.array([0.4, 0.6])
    eta = {k: np.ones(N[k]) / N[k] for k in range(K)}

    th_star, _eps, obj_ch, _m, _snap = model.solve_choosing_theta_epsilon(
        K=K,
        I=0,
        D=D,
        N=N,
        mathcal_D={},
        Xi=Xi,
        empirical_distributions=emp,
        w_k=w_k,
        eta_k_n=eta,
        theta_bar=0.0,
        tau=1.0,
        output_flag=0,
    )
    assert th_star is not None and obj_ch is not None

    th_min_inv = min_feasible_theta_bar_inverse(
        D=D,
        K=K,
        N=N,
        emp=emp,
        Xi=Xi,
        w_k=w_k,
        eta=eta,
        hi=500.0,
        tol=1e-4,
        output_flag=0,
    )

    diff = abs(float(th_star) - float(th_min_inv))
    print("solve_choosing_theta_epsilon (theta_bar_input=0) 最优 theta* =", th_star)
    print("inverse_refined_mr_DRO (theta=0) 最小可行 theta_bar ≈", th_min_inv)
    print("|差| =", diff)
    assert diff < 0.02, "二者应数值接近（同一 Wasserstein 预算阈值）"

    # 额外：给定 choosing 里固定的 underline_theta，应有 theta* ≈ max(0, W_min - underline)
    ubar = 3.0
    th2, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=K,
        I=0,
        D=D,
        N=N,
        mathcal_D={},
        Xi=Xi,
        empirical_distributions=emp,
        w_k=w_k,
        eta_k_n=eta,
        theta_bar=ubar,
        tau=1.0,
        output_flag=0,
    )
    assert th2 is not None
    th_min_inv2 = min_feasible_theta_bar_inverse(
        D=D,
        K=K,
        N=N,
        emp=emp,
        Xi=Xi,
        w_k=w_k,
        eta=eta,
        hi=500.0,
        tol=1e-4,
        output_flag=0,
    )
    # 最小总预算 W_min；choosing: th2 + ubar ≈ W_min；inverse theta=0 时最小 theta_bar ≈ W_min
    print("\n固定 underline_theta =", ubar, "时 choosing 的 theta* =", th2)
    print("仍有 th* + ubar ≈ min_inverse_theta_bar(θ=0)：", th2 + ubar, "vs", th_min_inv2)
    assert abs((th2 + ubar) - th_min_inv2) < 0.02

    print("\n验证通过。")


if __name__ == "__main__":
    main()
