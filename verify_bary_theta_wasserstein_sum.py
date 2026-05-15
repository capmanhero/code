# -*- coding: utf-8 -*-
"""
验证：由 β 运输构造的离散质心 b 与各经验分布 P_k 的 OT-Wasserstein 距离，
其加权和是否等于 ``solve_choosing_theta_epsilon``（I=0, θ̲=0）的最优 θ*。

检验量（与 choosing 模型一致）：
    ∑_{k=0}^{K-1} w_k · W_1^{OT}(b, P_k)  ?=  θ*

其中 W_1^{OT} 为 L1 地面度量下的离散最优传输距离；
b 由 ``wasserstein1_barycenter_k_discrete`` 从最优 β, γ 恢复。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import model


def check_one(
    *,
    label: str,
    emp: Dict[int, np.ndarray],
    Xi: Dict[str, list[float]],
    K: int,
    D: int,
    N: List[int],
    w_k: np.ndarray,
    eta_k_n: Dict[int, np.ndarray] | None = None,
    output_flag: int = 0,
) -> Dict[str, Any]:
    th_star, _eps, obj_ch, _mdl, _sol = model.solve_choosing_theta_epsilon(
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
    assert th_star is not None and obj_ch is not None

    bary = model.wasserstein1_barycenter_k_discrete(
        emp, weights=w_k, eta_k_n=eta_k_n, Xi=Xi, output_flag=output_flag
    )
    w1_ot = np.asarray(bary["wasserstein1_distances"], dtype=float)
    w1_lin = np.asarray(bary.get("wasserstein_linear_contrib", []), dtype=float)
    weighted_ot = float(np.dot(w_k, w1_ot))
    weighted_lin = float(np.dot(w_k, w1_lin)) if w1_lin.size else float("nan")

    gap_ot = weighted_ot - float(th_star)
    gap_lin = weighted_lin - float(th_star) if w1_lin.size else float("nan")

    return {
        "label": label,
        "theta_star": float(th_star),
        "obj_choosing": float(obj_ch),
        "w_k": w_k.copy(),
        "W1_OT": w1_ot,
        "W1_linear": w1_lin,
        "sum_w_W1_OT": weighted_ot,
        "sum_w_W1_linear": weighted_lin,
        "gap_OT": gap_ot,
        "gap_linear": gap_lin,
        "bary_support_n": int(bary["X"].shape[0]),
    }


def main() -> None:
    cases: List[Dict[str, Any]] = []

    rng = np.random.default_rng(7)
    K, D, N = 3, 2, [3, 3, 3]
    emp = {k: rng.uniform(2.0, 8.0, size=(N[k], D)) for k in range(K)}
    Xi = {str(d + 1): [0.0, 25.0] for d in range(D)}
    w_eq = np.ones(K) / K
    cases.append(
        check_one(
            label="示例 K=3 D=2 等权",
            emp=emp,
            Xi=Xi,
            K=K,
            D=D,
            N=N,
            w_k=w_eq,
        )
    )

    for seed in range(20):
        rng = np.random.default_rng(seed)
        emp_s = {k: rng.uniform(1.0, 9.0, size=(3, 2)) for k in range(3)}
        w = rng.uniform(0.1, 1.0, size=3)
        w = w / w.sum()
        cases.append(
            check_one(
                label=f"随机 seed={seed}",
                emp=emp_s,
                Xi={str(d + 1): [0.0, 25.0] for d in range(2)},
                K=3,
                D=2,
                N=[3, 3, 3],
                w_k=w,
            )
        )

    rng = np.random.default_rng(0)
    eta = {k: np.array([0.7, 0.2, 0.1], dtype=float) for k in range(3)}
    for k in range(3):
        eta[k] /= eta[k].sum()
    cases.append(
        check_one(
            label="非均匀 eta_k_n",
            emp={k: rng.uniform(1.0, 9.0, size=(3, 2)) for k in range(3)},
            Xi={str(d + 1): [0.0, 25.0] for d in range(2)},
            K=3,
            D=2,
            N=[3, 3, 3],
            w_k=np.ones(3) / 3.0,
            eta_k_n=eta,
        )
    )

    print("=" * 72)
    print("验证：∑_k w_k · W_1^OT(质心 b, P_k)  ?=  模型最优 θ*")
    print("=" * 72)

    ex = cases[0]
    print(f"\n【{ex['label']}】")
    print(f"  θ* (solve_choosing_theta_epsilon)     = {ex['theta_star']:.12f}")
    print(f"  质心支撑点数                           = {ex['bary_support_n']}")
    for k in range(len(ex["W1_OT"])):
        print(
            f"  W_1^OT(b, P_{k})                      = {ex['W1_OT'][k]:.12f}  "
            f"(w_{k}={ex['w_k'][k]:.4f})"
        )
    print(f"  ∑_k w_k · W_1^OT(b, P_k)              = {ex['sum_w_W1_OT']:.12f}")
    print(f"  |sum w_k W_1^OT - theta*|              = {abs(ex['gap_OT']):.3e}")
    if ex["W1_linear"].size:
        print(f"  (ref) sum_k w_k * linear term         = {ex['sum_w_W1_linear']:.12f}")
        print(f"  |sum w_k linear - theta*|              = {abs(ex['gap_linear']):.3e}")

    max_gap = 0.0
    for c in cases[1:]:
        max_gap = max(max_gap, abs(c["gap_OT"]))

    print(f"\n【批量】共 {len(cases)} 组实例")
    print(f"  max |sum_k w_k W_1^OT(b,P_k) - theta*| = {max_gap:.3e}")
    worst = max(cases[1:], key=lambda c: abs(c["gap_OT"]))
    print(f"  最大误差实例: {worst['label']}, gap = {worst['gap_OT']:.3e}")

    tol = 1e-6
    if max_gap < tol:
        print(f"\n结论：在 {len(cases)} 组测试下，加权和与 theta* 一致（容差 {tol}）。")
    else:
        print(f"\n结论：存在超出容差 {tol} 的实例，需检查构造。")
        raise AssertionError(f"max gap {max_gap} >= {tol}")

    print("\nPASS.")


if __name__ == "__main__":
    main()
