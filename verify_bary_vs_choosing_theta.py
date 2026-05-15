# -*- coding: utf-8 -*-
"""
数值检查（I=0）：
  - ``solve_choosing_theta_epsilon(..., theta_bar=0)`` 的最优 θ*；
  - ``wasserstein1_barycenter_k_discrete`` 的 ``objective_value``（同一 choosing 路径）；
  - 由 β 运输解释恢复的离散质心 ``(X,b)`` 满足
    ``∑_k w_k W_1^{OT}(b, P_k) ≈ θ*``（L1 地面度量 OT）。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import model


def run_pair(
    *,
    emp: Dict[int, np.ndarray],
    Xi: Dict[str, list[float]],
    K: int,
    D: int,
    N: List[int],
    w_k: np.ndarray,
    eta_k_n: Dict[int, np.ndarray] | None,
    output_flag: int = 0,
) -> tuple[float, float, float, float]:
    th_star, _eps, obj_ch, _m, sol = model.solve_choosing_theta_epsilon(
        K=K,
        I=0,
        D=D,
        N=N,
        mathcal_D={},
        Xi=Xi,
        empirical_distributions=emp,
        w_k=w_k,
        eta_k_n=eta_k_n,
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    assert th_star is not None and obj_ch is not None and sol is not None
    bary = model.wasserstein1_barycenter_k_discrete(
        emp, weights=w_k, eta_k_n=eta_k_n, Xi=Xi, output_flag=output_flag
    )
    b_obj = float(bary["objective_value"])
    w1d = np.asarray(bary["wasserstein1_distances"], dtype=float)
    w1_weighted = float(np.dot(w_k, w1d))
    assert abs(b_obj - float(th_star)) < 1e-9 * max(1.0, abs(float(th_star)))
    return float(th_star), w1_weighted, float(th_star) - w1_weighted, b_obj - float(th_star)


def main() -> None:
    max_ot_gap = 0.0
    cases: List[Dict[str, Any]] = []

    for seed in range(15):
        rng = np.random.default_rng(seed)
        K, D = 3, 2
        N = [3, 3, 3]
        emp = {k: rng.uniform(2.0, 8.0, size=(N[k], D)) for k in range(K)}
        Xi = {str(d + 1): [0.0, 25.0] for d in range(D)}
        w_k = np.ones(K) / K
        th, wot, og, _ = run_pair(emp=emp, Xi=Xi, K=K, D=D, N=N, w_k=w_k, eta_k_n=None)
        max_ot_gap = max(max_ot_gap, abs(og))
        cases.append({"case": f"rand K=3 D=2 seed={seed}", "ot_gap": og})

    rng = np.random.default_rng(100)
    K, D, N = 3, 2, [3, 3, 3]
    emp = {k: rng.uniform(1.0, 9.0, size=(N[k], D)) for k in range(K)}
    Xi = {str(d + 1): [0.0, 25.0] for d in range(D)}
    for w in ([0.6, 0.25, 0.15], [0.9, 0.05, 0.05]):
        w_k = np.asarray(w, dtype=float)
        w_k = w_k / w_k.sum()
        th, wot, og, _ = run_pair(emp=emp, Xi=Xi, K=K, D=D, N=N, w_k=w_k, eta_k_n=None)
        max_ot_gap = max(max_ot_gap, abs(og))
        cases.append({"case": f"nonuniform w={w_k.tolist()}", "ot_gap": og})

    rng = np.random.default_rng(0)
    emp = {k: rng.uniform(1.0, 9.0, size=(3, 2)) for k in range(K)}
    eta = {k: np.array([0.7, 0.2, 0.1], dtype=float) for k in range(K)}
    for k in range(K):
        eta[k] = eta[k] / eta[k].sum()
    th, wot, og, _ = run_pair(
        emp=emp, Xi=Xi, K=3, D=2, N=[3, 3, 3], w_k=np.ones(3) / 3.0, eta_k_n=eta
    )
    max_ot_gap = max(max_ot_gap, abs(og))
    cases.append({"case": "custom eta_k_n", "ot_gap": og})

    rng = np.random.default_rng(200)
    emp = {0: rng.uniform(0, 5, (5, 2)), 1: rng.uniform(1, 6, (4, 2))}
    Xi = {"1": [0.0, 12.0], "2": [0.0, 12.0]}
    th, wot, og, _ = run_pair(
        emp=emp, Xi=Xi, K=2, D=2, N=[5, 4], w_k=np.array([0.4, 0.6]), eta_k_n=None
    )
    max_ot_gap = max(max_ot_gap, abs(og))
    cases.append({"case": "K=2", "ot_gap": og})

    rng = np.random.default_rng(42)
    K, D, N = 3, 4, [2, 2, 2]
    emp = {k: rng.integers(0, 3, size=(N[k], D)).astype(float) for k in range(K)}
    Xi = {str(d + 1): [0.0, 10.0] for d in range(D)}
    th, wot, og, _ = run_pair(
        emp=emp, Xi=Xi, K=K, D=D, N=N, w_k=np.ones(K) / K, eta_k_n=None
    )
    max_ot_gap = max(max_ot_gap, abs(og))
    cases.append({"case": "D=4 integers N=2 each", "ot_gap": og})

    emp = {
        0: np.array([[0.0, 0.0]]),
        1: np.array([[10.0, 0.0]]),
        2: np.array([[0.0, 10.0]]),
    }
    Xi = {str(d + 1): [0.0, 15.0] for d in range(2)}
    th, wot, og, _ = run_pair(
        emp=emp, Xi=Xi, K=3, D=2, N=[1, 1, 1], w_k=np.ones(3) / 3.0, eta_k_n=None
    )
    max_ot_gap = max(max_ot_gap, abs(og))
    cases.append({"case": "triangle 1-sample/ref", "ot_gap": og})

    print("theta* vs sum_k w_k W_1^OT(b, P_k) from beta-transport barycenter\n")
    for c in cases[-6:]:
        print(f"  {c['case']}: |theta* - sum w_k W1_OT| = {abs(c['ot_gap']):.3e}")
    print(f"\nMax |theta* - sum_k w_k W1_OT| over {len(cases)} sub-cases: {max_ot_gap:.3e}")
    assert max_ot_gap < 1e-6, max_ot_gap
    print("\nPASS.")


if __name__ == "__main__":
    main()
