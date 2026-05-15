# -*- coding: utf-8 -*-
"""
血小板供血决策：真实数据（data.xlsx）上对比多种鲁棒/多参考模型。

实验设计（血站为 CL 或 ET 决策；SX、PJ 为参考支持）：
- 训练：前 10 个月需求的经验分布；每家医院 bootstrap 生成 10 条场景；
  对 CL（或 ET）实验，三源为 {目标, SX, PJ}。
- 验证：目标医院第 11 个月逐日需求，用于超参数网格选优（样本外损失仅血站侧）。
- 测试：第 12 个月逐日需求，同上。
- 医院订单：提前若干天给出，仅作为 ``mr_DRO_with_order``（I=1）鲁棒模型中的参考信息；
  **不计入**验证/测试的样本外损失（样本外损失只含血站对实现需求的缺货/过剩）。
- ``mr_DRO_with_order`` 在验证集上对 ``(theta, epsilon_i[1])`` 联合网格选优，其中 ``epsilon_i[1] ∈ {0,1,3}``。
- 成本：血站 b0=3,h0=1；医院 bI=4,hI=1；约束 sum(x0)<=15，医院 sum(x_i)<=15*mu（I=1 时）。
- 10 次独立 bootstrap 重复；输出 xlsx（明细 + 汇总均值/方差）。
- ``solve_multi_reference_l1_only``：各源 ``theta_k`` 基准为重心到该源的 ``W_1`` 距离，
  仅加上 ``THETA_DELTA_OFFSETS`` 中的 ``δ``（``{0,0.05,…,5}``），**不**再叠 ``max(th_\star, bary\_obj)``。
- 离散 Wasserstein 重心：对训练阶段得到的**完整**三源经验分布（每源 N_SCEN 条）调用
  ``wasserstein1_barycenter_k_discrete``（内部通过 ``solve_choosing_theta_epsilon``，与主流程 ``Xi`` 一致）。

依赖：pandas、numpy、openpyxl、Gurobi（model.py）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import model  # noqa: E402

from gurobipy import GRB  # noqa: E402

COLS = ["A", "AB", "B", "O"]
D = len(COLS)
MONTHS_TRAIN = tuple(range(1, 11))  # 1..10
MONTH_VAL = 11
MONTH_TEST = 12
N_BOOT = int(os.environ.get("BLOOD_N_BOOT", "10"))
N_SCEN = 5  # 每源训练场景数
B0, H0 = 3.0, 1.0
BI, HI = 4.0, 1.0
CAP = 15.0
# ``mr_DRO_with_order``（I=1）中 ``epsilon_i[1]`` 的验证/测试选参网格
EPSILON_I_GRID = (0.0, 1.0, 3.0)
# 单标量 θ：``theta = max(th_ch, bary_obj) + δ``；multi-ref：``theta_k = W_1(重心,P_k) + δ``（δ 仅此列）
THETA_DELTA_OFFSETS = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 5.0], dtype=float)
RNG_SEEDS = tuple(range(100, 100 + max(N_BOOT, 1)))


def _data_path() -> pathlib.Path:
    return _ROOT / "data.xlsx"


def load_demand(hospital: str) -> pd.DataFrame:
    df = pd.read_excel(_data_path(), sheet_name=f"demand_{hospital}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_orders(hospital: str) -> pd.DataFrame:
    df = pd.read_excel(_data_path(), sheet_name=f"orders_{hospital}")
    df["data"] = pd.to_datetime(df["data"])
    return df.rename(columns={"data": "date"})


def pool_train(hospital: str) -> np.ndarray:
    df = load_demand(hospital)
    m = df["date"].dt.month
    return df.loc[m.isin(MONTHS_TRAIN), COLS].to_numpy(dtype=float)


def month_matrix_demand(hospital: str, month: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (需求矩阵 (T,4), 对应日期数组)。"""
    df = load_demand(hospital)
    sub = df[df["date"].dt.month == month]
    return sub[COLS].to_numpy(dtype=float), sub["date"].to_numpy()


def month_matrix_orders(hospital: str, month: int) -> Tuple[np.ndarray, np.ndarray]:
    df = load_orders(hospital)
    sub = df[df["date"].dt.month == month]
    return sub[COLS].to_numpy(dtype=float), sub["date"].to_numpy()


def wasserstein_bary_exact(
    empirical_distributions: Mapping[int, np.ndarray],
    Xi: Optional[Mapping[str, List[float]]] = None,
    output_flag: int = 0,
) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
    """
    在给定离散经验分布上求 ``model.wasserstein1_barycenter_k_discrete``（choosing 对偶路径）：
    各源样本矩阵原样传入（不减少行数），等权 ``w_k=1/K`` 为函数默认值。
    若提供 ``Xi``，须与同一 bootstrap 下 ``solve_choosing_theta_epsilon`` / inverse 所用盒约束一致。

    返回 (barycenter 输出字典, 与传入键一致、内容为 float 副本的经验字典)。
    """
    emp_in: Dict[int, np.ndarray] = {
        int(k): np.asarray(v, dtype=float).copy() for k, v in sorted(empirical_distributions.items())
    }
    out = model.wasserstein1_barycenter_k_discrete(
        emp_in, Xi=Xi, output_flag=output_flag
    )
    return out, emp_in


def theta_k_base_from_barycenter(bary_out: Mapping[str, Any]) -> Dict[int, float]:
    """
    由 ``wasserstein1_barycenter_k_discrete`` 的输出得到各源到恢复质心的 OT ``W_1`` 距离，
    作为 ``solve_multi_reference_l1_only`` 中各 ``theta_k`` 的基准。
    """
    keys_ord = list(bary_out["keys_order"])
    dists = np.asarray(bary_out["wasserstein1_distances"], dtype=float).ravel()
    if len(keys_ord) != int(dists.size):
        raise ValueError(
            f"keys_order 长度 ({len(keys_ord)}) 与 wasserstein1_distances ({dists.size}) 不一致"
        )
    return {int(k): float(dists[i]) for i, k in enumerate(keys_ord)}


def prune_bary_to_inverse(
    bary_out: Mapping[str, Any],
    eps: float = 1e-15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    供 K=1 的 ``inverse_refined_mr_DRO`` 使用：保留 ``barycenter`` 向量中概率严格为正的支撑点
    （K=1 时模型规模可接受，不再按质量/点数截断）。
    """
    X = np.asarray(bary_out["X"], dtype=float)
    p = np.asarray(bary_out["barycenter"], dtype=float).ravel()
    if X.shape[0] != p.size:
        raise ValueError("barycenter 长度须与 X 行数一致")
    mask = p > eps
    idx = np.flatnonzero(mask)
    Xs = X[idx]
    ps = p[idx].astype(float)
    s = float(ps.sum())
    if s <= 0.0:
        raise ValueError("重心概率在 eps 阈值下全为 0，请检查 barycenter 输出")
    ps = ps / s
    return Xs, ps


def build_Xi(
    pools: List[np.ndarray],
    val_z: np.ndarray,
    test_z: np.ndarray,
    margin: float = 1.25,
) -> Dict[str, List[float]]:
    mx = 0.0
    for p in pools:
        mx = max(mx, float(np.max(p)) if p.size else 0.0)
    mx = max(mx, float(np.max(val_z)) if val_z.size else 0.0)
    mx = max(mx, float(np.max(test_z)) if test_z.size else 0.0)
    hi = max(25.0, CAP, mx * margin + 2.0)
    return {str(d): [0.0, hi] for d in range(1, D + 1)}


def bootstrap_emp(
    pools: Dict[str, np.ndarray],
    target: str,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    """三源：0=目标, 1=SX, 2=PJ。"""
    p0, p1, p2 = pools[target], pools["SX"], pools["PJ"]
    return {
        0: p0[rng.choice(p0.shape[0], size=N_SCEN, replace=True)],
        1: p1[rng.choice(p1.shape[0], size=N_SCEN, replace=True)],
        2: p2[rng.choice(p2.shape[0], size=N_SCEN, replace=True)],
    }


def oos_cost_station(z: np.ndarray, x0: Mapping[int, float]) -> float:
    x = np.array([x0[d] for d in range(1, D + 1)], dtype=float)
    return float(
        np.sum(B0 * np.maximum(z - x, 0.0) + H0 * np.maximum(x - z, 0.0))
    )


def mean_oos_days(x0: Mapping[int, float], Z: np.ndarray) -> float:
    """各日样本外日均成本：仅血站 newsvendor（实现需求 z 与决策 x0）。"""
    if Z.size == 0:
        return float("nan")
    return float(np.mean([oos_cost_station(Z[t], x0) for t in range(Z.shape[0])]))


def solve_inverse_i0(
    emp: Mapping[int, np.ndarray],
    Xi: Mapping,
    theta: float,
    theta_bar: float,
    w_k: Optional[np.ndarray],
    output_flag: int = 0,
) -> Tuple[Optional[Dict[int, float]], float]:
    b0c = {d: B0 for d in range(1, D + 1)}
    h0c = {d: H0 for d in range(1, D + 1)}
    r = model.inverse_refined_mr_DRO(
        empirical_distributions=emp,
        order_data={},
        K=3,
        I=0,
        D=D,
        N=[N_SCEN, N_SCEN, N_SCEN],
        mathcal_D={},
        Xi=Xi,
        A_0=[np.ones(D).tolist()],
        b_0=[CAP],
        A_I={},
        b_I={},
        b_0_coef=b0c,
        h_0_coef=h0c,
        w_k=w_k,
        eta_k_n=None,
        theta=float(theta),
        theta_bar=float(theta_bar),
        output_flag=output_flag,
    )
    x0, _, _, _, _, obj, mdl = r
    if mdl is None or mdl.status != GRB.OPTIMAL or x0 is None:
        return None, float("nan")
    return x0, float(obj)


def solve_inverse_i1(
    emp: Mapping[int, np.ndarray],
    Xi: Mapping,
    order_data: Mapping[int, Mapping[int, float]],
    theta: float,
    theta_bar: float,
    w_k: Optional[np.ndarray],
    epsilon_1: float = 0.0,
    output_flag: int = 0,
) -> Tuple[Optional[Dict[int, float]], float]:
    b0c = {d: B0 for d in range(1, D + 1)}
    h0c = {d: H0 for d in range(1, D + 1)}
    bi = {1: {d: BI for d in range(1, D + 1)}}
    hi = {1: {d: HI for d in range(1, D + 1)}}
    r = model.inverse_refined_mr_DRO(
        empirical_distributions=emp,
        order_data=order_data,
        K=3,
        I=1,
        D=D,
        N=[N_SCEN, N_SCEN, N_SCEN],
        mathcal_D={"1": set(range(1, D + 1))},
        Xi=Xi,
        A_0=[np.ones(D).tolist()],
        b_0=[CAP],
        A_I={"1": [np.ones(D).tolist()]},
        b_I={"1": [CAP]},
        b_0_coef=b0c,
        h_0_coef=h0c,
        b_i_coef=bi,
        h_i_coef=hi,
        epsilon_i={1: float(epsilon_1)},
        w_k=w_k,
        eta_k_n=None,
        theta=float(theta),
        theta_bar=float(theta_bar),
        output_flag=output_flag,
    )
    x0, _, _, _, _, obj, mdl = r
    if mdl is None or mdl.status != GRB.OPTIMAL or x0 is None:
        return None, float("nan")
    return x0, float(obj)


def solve_inverse_k1(
    emp1: np.ndarray,
    Xi: Mapping,
    theta: float,
    theta_bar: float,
    eta: Optional[np.ndarray],
    output_flag: int = 0,
) -> Tuple[Optional[Dict[int, float]], float]:
    n = int(emp1.shape[0])
    b0c = {d: B0 for d in range(1, D + 1)}
    h0c = {d: H0 for d in range(1, D + 1)}
    eta_map = None
    if eta is not None:
        eta_map = {0: np.asarray(eta, dtype=float).ravel()}
    r = model.inverse_refined_mr_DRO(
        empirical_distributions={0: np.asarray(emp1, dtype=float)},
        order_data={},
        K=1,
        I=0,
        D=D,
        N=[n],
        mathcal_D={},
        Xi=Xi,
        A_0=[np.ones(D).tolist()],
        b_0=[CAP],
        A_I={},
        b_I={},
        b_0_coef=b0c,
        h_0_coef=h0c,
        w_k=None,
        eta_k_n=eta_map,
        theta=float(theta),
        theta_bar=float(theta_bar),
        output_flag=output_flag,
    )
    x0, _, _, _, _, obj, mdl = r
    if mdl is None or mdl.status != GRB.OPTIMAL or x0 is None:
        return None, float("nan")
    return x0, float(obj)


def solve_multi_l1(
    emp: Mapping[int, np.ndarray],
    Xi: Mapping,
    theta_offset: float,
    base_theta_k: Mapping[int, float],
    output_flag: int = 0,
) -> Tuple[Optional[Dict[int, float]], float]:
    """``theta_k[k] = base_theta_k[k] + theta_offset``，其中 ``base_theta_k`` 来自重心输出各源 ``W_1`` 距离。"""
    b0c = {d: B0 for d in range(1, D + 1)}
    h0c = {d: H0 for d in range(1, D + 1)}
    tk = {
        int(k): float(base_theta_k[int(k)]) + float(theta_offset)
        for k in sorted(base_theta_k.keys())
    }
    ox0, _, _, obj, mdl = model.solve_multi_reference_l1_only(
        empirical_distributions=emp,
        theta_k=tk,
        Xi=Xi,
        A_0=[np.ones(D).tolist()],
        b_0=[CAP],
        b_0_coef=b0c,
        h_0_coef=h0c,
        eta_k_n=None,
        output_flag=output_flag,
    )
    if mdl is None or mdl.status != GRB.OPTIMAL or ox0 is None:
        return None, float("nan")
    return ox0, float(obj)


def theta_grid_from_anchors(th_ch: float, bary_obj: float) -> np.ndarray:
    """
    与 ``solve_choosing_theta_epsilon`` 及重心目标共同标定尺度。
    """
    base = max(float(th_ch), float(bary_obj), 1e-6)
    scaled = base + THETA_DELTA_OFFSETS
    g = np.unique(np.round(np.concatenate([scaled]), 6))
    return g[g >= 0]


def order_dict_from_row(row: np.ndarray) -> Dict[int, float]:
    return {d + 1: float(row[d]) for d in range(D)}


def mean_order_vec(Z: np.ndarray, H: np.ndarray) -> np.ndarray:
    """用于 I=1 鲁棒模型中的固定订单场景（验证/测试阶段分别用对应月份均值）。"""
    if H.size == 0:
        return np.zeros(D, dtype=float)
    return np.clip(np.mean(H, axis=0), 0.0, None)


def grid_search_val(
    solve_fn,
    val_Z: np.ndarray,
    thetas: np.ndarray,
) -> Tuple[Optional[Dict[int, float]], float, float]:
    """在验证集上选最优 theta；返回 (best_x0, best_val_mean, best_theta)。"""
    best_x0 = None
    best_m = float("inf")
    best_t = float("nan")
    for t in thetas:
        x0, _ = solve_fn(float(t))
        if x0 is None:
            continue
        m = mean_oos_days(x0, val_Z)
        if m < best_m:
            best_m, best_x0, best_t = m, x0, float(t)
    return best_x0, best_m, best_t


def run_one_target(
    target: str,
    rng: np.random.Generator,
    output_flag: int = 0,
) -> List[Dict[str, Any]]:
    pools_arr = {h: pool_train(h) for h in ("CL", "ET", "SX", "PJ")}
    emp = bootstrap_emp(pools_arr, target, rng)

    val_Z, _ = month_matrix_demand(target, MONTH_VAL)
    val_H, _ = month_matrix_orders(target, MONTH_VAL)
    test_Z, _ = month_matrix_demand(target, MONTH_TEST)

    Xi = build_Xi(
        [pools_arr[target], pools_arr["SX"], pools_arr["PJ"], emp[0], emp[1], emp[2]],
        val_Z,
        test_Z,
    )

    w_eq = np.ones(3, dtype=float) / 3.0

    th_star, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=3,
        I=0,
        D=D,
        N=[N_SCEN, N_SCEN, N_SCEN],
        mathcal_D={},
        Xi=Xi,
        empirical_distributions=emp,
        w_k=w_eq,
        eta_k_n=None,
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    if th_star is None:
        th_star = 1.0

    bary_out, _emp_used = wasserstein_bary_exact(emp, Xi=Xi, output_flag=output_flag)
    print(
        f"    barycenter ok, support_n={bary_out['X'].shape[0]}, objective={bary_out.get('objective_value')}",
        flush=True,
    )
    bary_B = float(bary_out.get("objective_value", 0.0) or 0.0)
    base_theta_multi = theta_k_base_from_barycenter(bary_out)

    thetas = theta_grid_from_anchors(float(th_star), bary_B)

    rows: List[Dict[str, Any]] = []

    def record(method: str, x0: Optional[Mapping[int, float]], val_m: float, test_m: float, hp: Any):
        rows.append(
            {
                "hospital": target,
                "method": method,
                "val_mean_daily": val_m,
                "test_mean_daily": test_m,
                "hyperparam": hp,
                "x0_sum": float(sum(x0[d] for d in range(1, D + 1))) if x0 else float("nan"),
            }
        )

    # ----- mr_DRO I=0 -----
    def inv0(t: float):
        return solve_inverse_i0(emp, Xi, t, 0.0, w_eq, output_flag)

    x0, vm, th = grid_search_val(inv0, val_Z, thetas)
    record("mr_DRO_no_order", x0, vm, mean_oos_days(x0, test_Z) if x0 else float("nan"), th)

    # ----- mr_DRO I=1（订单：该月需求对应月份的日均订单向量） -----
    ord_val = {1: order_dict_from_row(mean_order_vec(val_Z, val_H))}

    best_x0_i1: Optional[Dict[int, float]] = None
    best_vm_i1 = float("inf")
    best_th_i1 = float("nan")
    best_eps_i1 = float("nan")
    for eps in EPSILON_I_GRID:
        for t in thetas:
            x0, _ = solve_inverse_i1(
                emp, Xi, ord_val, float(t), 0.0, w_eq, float(eps), output_flag
            )
            if x0 is None:
                continue
            m = mean_oos_days(x0, val_Z)
            if m < best_vm_i1:
                best_vm_i1, best_x0_i1, best_th_i1, best_eps_i1 = m, x0, float(t), float(eps)
    if best_x0_i1 is None:
        best_vm_i1 = float("nan")
    record(
        "mr_DRO_with_order",
        best_x0_i1,
        best_vm_i1,
        mean_oos_days(best_x0_i1, test_Z) if best_x0_i1 else float("nan"),
        {
            "theta": best_th_i1,
            "epsilon_1": best_eps_i1,
            "order_mode": "month_mean_for_solve",
        },
    )

    # ----- 交集模型 multi-ref L1 only -----
    def ml1_wrap(t: float):
        return solve_multi_l1(emp, Xi, float(t), base_theta_multi, output_flag)

    x0, vm, delta_sel = grid_search_val(ml1_wrap, val_Z, THETA_DELTA_OFFSETS)
    tk_sel = (
        {int(k): float(base_theta_multi[k]) + float(delta_sel) for k in sorted(base_theta_multi)}
        if x0 is not None
        else {}
    )
    record(
        "multi_ref_l1_only",
        x0,
        vm,
        mean_oos_days(x0, test_Z) if x0 else float("nan"),
        {
            "theta_k_delta": delta_sel,
            "theta_k_base": {int(k): float(base_theta_multi[k]) for k in sorted(base_theta_multi)},
            "theta_k": tk_sel,
        },
    )

    # ----- 单参考：仅验证月目标医院经验 -----
    val_emp = val_Z.copy()
    n_val = val_emp.shape[0]
    th1_star, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=1,
        I=0,
        D=D,
        N=[n_val],
        mathcal_D={},
        Xi=Xi,
        empirical_distributions={0: val_emp},
        w_k=None,
        eta_k_n=None,
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    if th1_star is None:
        th1_star = 1.0
    thetas1 = theta_grid_from_anchors(float(th1_star), float(th1_star))

    def inv_val(t: float):
        return solve_inverse_k1(val_emp, Xi, t, 0.0, None, output_flag)

    x0, vm, th = grid_search_val(inv_val, val_Z, thetas1)
    record("single_ref_validation_only", x0, vm, mean_oos_days(x0, test_Z) if x0 else float("nan"), th)

    # ----- 混合（等权）：拼成 K=1 的 30 样本 -----
    mix = np.vstack([emp[0], emp[1], emp[2]])
    n_mix = mix.shape[0]
    thm, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=1,
        I=0,
        D=D,
        N=[n_mix],
        mathcal_D={},
        Xi=Xi,
        empirical_distributions={0: mix},
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    if thm is None:
        thm = 1.0
    thetas_m = theta_grid_from_anchors(float(thm), float(bary_B))

    def inv_mix(t: float):
        return solve_inverse_k1(mix, Xi, t, 0.0, None, output_flag)

    x0, vm, th = grid_search_val(inv_mix, val_Z, thetas_m)
    record("mixture_equal_weights_K1", x0, vm, mean_oos_days(x0, test_Z) if x0 else float("nan"), th)

    # ----- Barycenter：等权离散重心 + 在 theta 上网格 -----
    Xb, pb = prune_bary_to_inverse(bary_out)
    thb, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=1,
        I=0,
        D=D,
        N=[Xb.shape[0]],
        mathcal_D={},
        Xi=Xi,
        empirical_distributions={0: Xb},
        w_k=None,
        eta_k_n={0: pb},
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    if thb is None:
        thb = 1.0
    thetas_b = theta_grid_from_anchors(float(thb), bary_B)

    def inv_bary(t: float):
        return solve_inverse_k1(Xb, Xi, t, 0.0, pb, output_flag)

    x0, vm, th = grid_search_val(inv_bary, val_Z, thetas_b)
    record(
        "barycenter_equal_weights_K1",
        x0,
        vm,
        mean_oos_days(x0, test_Z) if x0 else float("nan"),
        {"theta": th, "bary_objective": bary_B, "bary_support": Xb.shape[0]},
    )

    return rows


def main() -> None:
    out_path = pathlib.Path(__file__).resolve().parent / "blood_experiment_results.xlsx"
    all_detail: List[Dict[str, Any]] = []
    for target in ("CL", "ET"):
        for rep, seed in enumerate(RNG_SEEDS):
            rng = np.random.default_rng(seed)
            print(f"[{target}] repeat {rep + 1}/{len(RNG_SEEDS)} seed={seed} ...")
            rows = run_one_target(target, rng, output_flag=0)
            for r in rows:
                r["repeat"] = rep
                r["seed"] = seed
            all_detail.extend(rows)

    df_d = pd.DataFrame(all_detail)

    def _fmt_hp(x: Any) -> Any:
        if isinstance(x, dict):
            return json.dumps(x, ensure_ascii=False)
        return x

    if not df_d.empty and "hyperparam" in df_d.columns:
        df_d = df_d.copy()
        df_d["hyperparam"] = df_d["hyperparam"].map(_fmt_hp)
    # 汇总：各医院-方法 在 10 次重复上 test_mean_daily 的均值与方差
    if not df_d.empty:

        def _var_across_repeats(s: pd.Series) -> float:
            return float(s.var(ddof=1)) if len(s) > 1 else 0.0

        df_s = (
            df_d.groupby(["hospital", "method"], as_index=False)
            .agg(
                test_mean_of_repeats=("test_mean_daily", "mean"),
                test_var_of_repeats=("test_mean_daily", _var_across_repeats),
            )
        )
    else:
        df_s = pd.DataFrame()

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_d.to_excel(w, sheet_name="detail", index=False)
        df_s.to_excel(w, sheet_name="summary", index=False)
    print("已写入:", out_path)


if __name__ == "__main__":
    main()
