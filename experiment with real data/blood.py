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
- ``mr_DRO_with_order``：超参网格为 ``(w, epsilon_1, theta)``（``theta`` 取 ``THETA_DELTA_OFFSETS``）。
  逐日：在固定 ``w, ε`` 与**当日订单**下 ``solve_choosing_theta_epsilon`` 得 **θ 锚点**，逆问题用
  ``θ_solve = θ_anchor + theta``（``theta`` 为验证集选定的超参偏移）。**仅在验证集**上网格选 ``(w,ε,θ)``；
  测试集固定已选三元组，逐日算锚点并求解，**不再**用测试需求选超参。需求与订单按日期对齐。
- 成本：血站 b0=3,h0=1；医院 bI=4,hI=1；约束 sum(x0)<=15，医院 sum(x_i)<=15*mu（I=1 时）。
- 10 次独立 bootstrap 重复；输出 xlsx（明细 + 汇总均值/方差）。
- 三源权重 ``w_k``：源 0=决策医院 ``w_self``，SX、PJ 各 ``(1-w_self)/2``。对依赖 ``w_k`` 的方法输出两种结果：
  **equal**——固定 ``w_self=1/3``（等权）；**val_w_selected**——在 ``w_self ∈ {1/3,1/2,2/3}`` 上按验证集日均损失
  各自选最优（``mr_DRO_with_order`` 在验证上对 ``(w,ε,θ)`` 联合选优）。明细列 ``w_mode`` 区分；
  汇总按 ``hospital, method, w_mode`` 分组。不涉三源 ``w_k`` 的方法（单参考、混合）``w_mode`` 为 ``-``。
- ``solve_multi_reference_l1_only``：各源 ``theta_k`` 基准为重心到该源的 ``W_1`` 距离，
  仅加上 ``THETA_DELTA_OFFSETS`` 中的 ``δ``（``{0,0.05,…,5}``），**不**再叠 ``max(theta_*, bary_obj)``。
- 离散 Wasserstein 重心：对训练阶段得到的**完整**三源经验分布（每源 N_SCEN 条）调用
  ``wasserstein1_barycenter_k_discrete``（内部通过 ``solve_choosing_theta_epsilon``，与主流程 ``Xi`` 一致）。
- ``single_ref_validation_only``：K=1，单参考经验为与同次 bootstrap 下 ``mr_DRO`` 相同的**目标医院源**
  ``emp[0]``（训练月 1–10 池中抽 ``N_SCEN`` 条），**不用** 11 月逐日需求作训练律；仅在 11 月 ``val_Z`` 上选 θ、在 12 月测试。

依赖：pandas、numpy、openpyxl、Gurobi（model.py）。
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

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
N_BOOT = 10
N_SCEN = 5  # 每源训练场景数
B0, H0 = 4.0, 1.0
BI, HI = 1.0, 4.0
CAP = 20.0
# ``mr_DRO_with_order``（I=1）超参网格：``epsilon_1`` 与加在逐日 θ 锚点上的 ``theta`` 偏移
EPSILON_I_GRID = (0.0, 1.0, 3.0, 5.0, 8.0)
THETA_HP_GRID = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0], dtype=float)
# 其它方法：``theta = max(th_ch, bary_obj) + δ``；multi-ref：``theta_k = W_1(重心,P_k) + δ``
THETA_DELTA_OFFSETS = THETA_HP_GRID
# 三源 Wasserstein / inverse 权重：源 0=决策医院，源 1=SX，源 2=PJ
THREE_SOURCE_W_SELF_GRID = np.array([1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0], dtype=float)
W_MODE_EQUAL = "equal"
W_MODE_VAL_W_SELECTED = "val_w_selected"
W_MODE_NA = "-"
seed_ = 100
RNG_SEEDS = tuple(range(seed_, seed_ + max(N_BOOT, 1)))


def _w_pack_cache_key(w_self: float) -> float:
    return round(float(w_self), 9)


def w_k_three_source(w_self: float) -> np.ndarray:
    """返回 ``[w_0, w_1, w_2]``，其中 ``w_0=w_self``，``w_1=w_2=(1-w_self)/2``。"""
    w0 = float(w_self)
    r = 0.5 * (1.0 - w0)
    return np.array([w0, r, r], dtype=float)


def w_key_from_w_self(w_self: float) -> str:
    if abs(w_self - 1.0 / 3.0) < 1e-9:
        return "w0_1/3"
    if abs(w_self - 1.0 / 2.0) < 1e-9:
        return "w0_1/2"
    if abs(w_self - 2.0 / 3.0) < 1e-9:
        return "w0_2/3"
    return f"w0_{w_self:.6g}"


def w_weights_dict(w_k: np.ndarray) -> Dict[int, float]:
    wk = np.asarray(w_k, dtype=float).ravel()
    return {0: float(wk[0]), 1: float(wk[1]), 2: float(wk[2])}


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


def align_month_demand_orders(hospital: str, month: int) -> Tuple[np.ndarray, np.ndarray]:
    """按 ``date`` 内连接对齐某月需求与订单，返回 ``(Z, H)``，形状均为 ``(T, D)``。"""
    dz = load_demand(hospital)
    dh = load_orders(hospital)
    mz = dz[dz["date"].dt.month == month][["date"] + list(COLS)].copy()
    mh = dh[dh["date"].dt.month == month][["date"] + list(COLS)].copy()
    ren = {c: f"z_{c}" for c in COLS}
    mz = mz.rename(columns=ren)
    ren = {c: f"h_{c}" for c in COLS}
    mh = mh.rename(columns=ren)
    m = pd.merge(mz, mh, on="date", how="inner").sort_values("date")
    if m.empty:
        return np.zeros((0, D), dtype=float), np.zeros((0, D), dtype=float)
    zcols = [f"z_{c}" for c in COLS]
    hcols = [f"h_{c}" for c in COLS]
    return m[zcols].to_numpy(dtype=float), m[hcols].to_numpy(dtype=float)


def wasserstein_bary_exact(
    empirical_distributions: Mapping[int, np.ndarray],
    Xi: Optional[Mapping[str, List[float]]] = None,
    weights: Optional[Union[np.ndarray, Mapping[int, float]]] = None,
    output_flag: int = 0,
) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
    """
    在给定离散经验分布上求 ``model.wasserstein1_barycenter_k_discrete``（choosing 对偶路径）：
    各源样本矩阵原样传入（不减少行数）；``weights`` 为 ``None`` 时用等权 ``1/K``。
    若提供 ``Xi``，须与同一 bootstrap 下 ``solve_choosing_theta_epsilon`` / inverse 所用盒约束一致。

    返回 (barycenter 输出字典, 与传入键一致、内容为 float 副本的经验字典)。
    """
    emp_in: Dict[int, np.ndarray] = {
        int(k): np.asarray(v, dtype=float).copy() for k, v in sorted(empirical_distributions.items())
    }
    out = model.wasserstein1_barycenter_k_discrete(
        emp_in, weights=weights, Xi=Xi, output_flag=output_flag
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


def solve_intersection(
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
    ox0, _, _, obj, mdl = model.solve_multi_reference_intersection(
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


def choosing_theta_i1_anchor(
    emp: Mapping[int, np.ndarray],
    Xi: Mapping,
    w_k: np.ndarray,
    order_data: Mapping[int, Mapping[int, float]],
    epsilon_1: float,
    bi_i1: Mapping[int, Mapping[int, float]],
    hi_i1: Mapping[int, Mapping[int, float]],
    A_I_i1: Mapping[str, Any],
    b_I_i1: Mapping[str, Any],
    output_flag: int,
) -> float:
    """I=1 choosing 在固定 ``epsilon_1`` 与给定 ``order_data`` 下的 ``θ^*``（4 位向上取整）。"""
    th, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=3,
        I=1,
        D=D,
        N=[N_SCEN, N_SCEN, N_SCEN],
        mathcal_D={"1": set(range(1, D + 1))},
        Xi=Xi,
        empirical_distributions=emp,
        b_i_coef=bi_i1,
        h_i_coef=hi_i1,
        A_I=A_I_i1,
        b_I=b_I_i1,
        order_data=order_data,
        w_k=w_k,
        eta_k_n=None,
        theta_bar=0.0,
        tau=1.0,
        epsilon_i_fixed={1: float(epsilon_1)},
        output_flag=output_flag,
    )
    return float(th)


def eval_mr_dro_i1_fixed_hyperparams(
    emp: Mapping[int, np.ndarray],
    Xi: Mapping,
    Z: np.ndarray,
    H: np.ndarray,
    w_k: np.ndarray,
    epsilon_1: float,
    theta_hp: float,
    bi_i1: Mapping[int, Mapping[int, float]],
    hi_i1: Mapping[int, Mapping[int, float]],
    A_I_i1: Mapping[str, Any],
    b_I_i1: Mapping[str, Any],
    output_flag: int,
) -> Tuple[float, int, float]:
    """
    固定超参 ``(w_k, epsilon_1, theta_hp)``：逐日用当日订单求 θ 锚点，``inverse`` 用 ``θ_anchor + theta_hp``。
    返回 ``(日均 OOS, 成功天数, 日均 sum(x0))``。
    """
    if Z.size == 0 or H.size == 0 or Z.shape[0] != H.shape[0]:
        return float("nan"), 0, float("nan")
    costs: List[float] = []
    x0_sums: List[float] = []
    th_off = float(theta_hp)
    eps = float(epsilon_1)
    for t in range(Z.shape[0]):
        ord_t = {1: order_dict_from_row(H[t])}
        anchor = choosing_theta_i1_anchor(
            emp, Xi, w_k, ord_t, eps, bi_i1, hi_i1, A_I_i1, b_I_i1, output_flag
        )
        th_solve = float(anchor) + th_off
        x0, _ = solve_inverse_i1(
            emp, Xi, ord_t, th_solve, 0.0, w_k, eps, output_flag
        )
        # if x0 is None:
        #     continue
        costs.append(oos_cost_station(Z[t], x0))
        x0_sums.append(float(sum(x0[d] for d in range(1, D + 1))))
    if not costs:
        return float("nan"), 0, float("nan")
    return float(np.mean(costs)), len(costs), float(np.mean(x0_sums))


def grid_search_mr_dro_i1_hyperparams(
    emp: Mapping[int, np.ndarray],
    Xi: Mapping,
    val_Z: np.ndarray,
    val_H: np.ndarray,
    w_k: np.ndarray,
    bi_i1: Mapping[int, Mapping[int, float]],
    hi_i1: Mapping[int, Mapping[int, float]],
    A_I_i1: Mapping[str, Any],
    b_I_i1: Mapping[str, Any],
    output_flag: int,
) -> Tuple[float, float, float, float]:
    """
    在验证集上网格选 ``(epsilon_1, theta)``（``theta`` 为加在逐日锚点上的偏移，取自 ``THETA_HP_GRID``）。
    返回 ``(best_val_m, best_eps, best_theta, mean_x0_sum)``。
    """
    best_vm = float("inf")
    best_eps = float("nan")
    best_theta = float("nan")
    best_x0_sum = float("nan")
    for eps in EPSILON_I_GRID:
        for theta_hp in THETA_HP_GRID:
            vm, n_ok, x0s = eval_mr_dro_i1_fixed_hyperparams(
                emp,
                Xi,
                val_Z,
                val_H,
                w_k,
                float(eps),
                float(theta_hp),
                bi_i1,
                hi_i1,
                A_I_i1,
                b_I_i1,
                output_flag,
            )
            if n_ok == 0 or not math.isfinite(vm):
                continue
            if vm < best_vm:
                best_vm, best_eps, best_theta, best_x0_sum = (
                    vm,
                    float(eps),
                    float(theta_hp),
                    x0s,
                )
    if not math.isfinite(best_vm):
        return float("nan"), float("nan"), float("nan"), float("nan")
    return best_vm, best_eps, best_theta, best_x0_sum


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


def _is_better_val_loss(vm_new: float, vm_old: float) -> bool:
    """验证集日均损失：越小越好；新值非有限时不采纳。"""
    if not math.isfinite(vm_new):
        return False
    if not math.isfinite(vm_old):
        return True
    return vm_new < vm_old


def _run_three_source_pack(
    emp: Mapping[int, np.ndarray],
    Xi: Mapping,
    val_Z: np.ndarray,
    val_H: np.ndarray,
    test_Z: np.ndarray,
    test_H: np.ndarray,
    w_self: float,
    output_flag: int,
) -> Dict[str, Any]:
    """
    给定 ``w_self``，完成三源 choosing、加权重心、mr_DRO I=0/I=1、multi-ref、barycenter→K1 inverse。
    返回各方法的字典及 ``bary_B``（供混合模型 θ 网格锚点）。
    """
    w_k = w_k_three_source(float(w_self))
    w_dict = w_weights_dict(w_k)
    wlist = [float(w_k[i]) for i in range(3)]

    th_star, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=3,
        I=0,
        D=D,
        N=[N_SCEN, N_SCEN, N_SCEN],
        mathcal_D={},
        Xi=Xi,
        empirical_distributions=emp,
        w_k=w_k,
        eta_k_n=None,
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    # if th_star is None:
    #     th_star = 1.0

    bary_out, _ = wasserstein_bary_exact(emp, Xi=Xi, weights=w_dict, output_flag=output_flag)
    print(
        f"    barycenter w_self={w_self:.4g} ({w_key_from_w_self(float(w_self))}) "
        f"support_n={bary_out['X'].shape[0]}, objective={bary_out.get('objective_value')}",
        flush=True,
    )
    bary_B = float(bary_out.get("objective_value"))
    base_theta_multi = theta_k_base_from_barycenter(bary_out)
    thetas = theta_grid_from_anchors(float(th_star), 0)

    def inv0(t: float):
        return solve_inverse_i0(emp, Xi, t, 0.0, w_k, output_flag)

    x0_no, vm_no, th_no = grid_search_val(inv0, val_Z, thetas)
    test_no = mean_oos_days(x0_no, test_Z) if x0_no else float("nan")
    hp_no = {"theta": th_no, "w_self": float(w_self), "w_k": list(wlist)}

    bi_i1 = {1: {d: BI for d in range(1, D + 1)}}
    hi_i1 = {1: {d: HI for d in range(1, D + 1)}}
    A_I_i1 = {"1": [np.ones(D).tolist()]}
    b_I_i1 = {"1": [CAP]}

    best_vm_i1, best_eps_i1, best_theta_i1, best_x0_sum_i1 = grid_search_mr_dro_i1_hyperparams(
        emp,
        Xi,
        val_Z,
        val_H,
        w_k,
        bi_i1,
        hi_i1,
        A_I_i1,
        b_I_i1,
        output_flag,
    )
    test_i1, n_test_ok, test_x0_sum = eval_mr_dro_i1_fixed_hyperparams(
        emp,
        Xi,
        test_Z,
        test_H,
        w_k,
        best_eps_i1,
        best_theta_i1,
        bi_i1,
        hi_i1,
        A_I_i1,
        b_I_i1,
        output_flag,
    )
    hp_i1 = {
        "epsilon_1": best_eps_i1,
        "theta": best_theta_i1,
        "theta_solve_rule": "choosing_anchor_daily(w,eps,order_t) + theta",
        "order_mode": "daily_orders_per_solve",
        "n_val_days": int(val_Z.shape[0]),
        "n_test_days_ok": int(n_test_ok),
        "mean_daily_x0_sum": best_x0_sum_i1,
        "mean_daily_x0_sum_test": test_x0_sum,
        "w_self": float(w_self),
        "w_k": list(wlist),
    }

    def intersection_wrap(t: float):
        return solve_intersection(emp, Xi, float(t), base_theta_multi, output_flag)

    x0_m, vm_m, delta_sel = grid_search_val(intersection_wrap, val_Z, THETA_DELTA_OFFSETS)
    tk_sel = (
        {int(k): float(base_theta_multi[k]) + float(delta_sel) for k in sorted(base_theta_multi)}
        if x0_m is not None
        else {}
    )
    test_m = mean_oos_days(x0_m, test_Z) if x0_m else float("nan")
    hp_m = {
        "theta_k_delta": delta_sel,
        "theta_k_base": {int(k): float(base_theta_multi[k]) for k in sorted(base_theta_multi)},
        "theta_k": tk_sel,
        "w_self": float(w_self),
        "w_k": list(wlist),
    }

    Xb, pb = prune_bary_to_inverse(bary_out)
    thb = 0
    # if thb is None:
    #     thb = 1.0
    thetas_b = theta_grid_from_anchors(float(thb), 0)

    def inv_bary(t: float):
        return solve_inverse_k1(Xb, Xi, t, 0.0, pb, output_flag)

    x0_b, vm_b, th_b = grid_search_val(inv_bary, val_Z, thetas_b)
    test_b = mean_oos_days(x0_b, test_Z) if x0_b else float("nan")
    hp_b = {
        "theta": th_b,
        "bary_objective": bary_B,
        "bary_support": Xb.shape[0],
        "w_self": float(w_self),
        "w_k_three_source": list(wlist),
    }

    return {
        "bary_B": bary_B,
        "mr_DRO_no_order": {"x0": x0_no, "val_m": vm_no, "test_m": test_no, "hp": hp_no},
        "mr_DRO_with_order": {
            "x0": None,
            "x0_sum": best_x0_sum_i1,
            "val_m": best_vm_i1,
            "test_m": test_i1,
            "hp": hp_i1,
        },
        "multi_ref_l1_only": {"x0": x0_m, "val_m": vm_m, "test_m": test_m, "hp": hp_m},
        "barycenter_equal_weights_K1": {"x0": x0_b, "val_m": vm_b, "test_m": test_b, "hp": hp_b},
    }


def run_one_target(
    target: str,
    rng: np.random.Generator,
    output_flag: int = 0,
) -> List[Dict[str, Any]]:
    pools_arr = {h: pool_train(h) for h in ("CL", "ET", "SX", "PJ")}
    emp = bootstrap_emp(pools_arr, target, rng)

    val_Z, val_H = align_month_demand_orders(target, MONTH_VAL)
    test_Z, test_H = align_month_demand_orders(target, MONTH_TEST)

    Xi = build_Xi(
        [pools_arr[target], pools_arr["SX"], pools_arr["PJ"], emp[0], emp[1], emp[2]],
        val_Z,
        test_Z,
    )

    rows: List[Dict[str, Any]] = []

    def record(
        method: str,
        x0: Optional[Mapping[int, float]],
        val_m: float,
        test_m: float,
        hp: Any,
        w_mode: str = W_MODE_NA,
        x0_sum: Optional[float] = None,
    ) -> None:
        if x0_sum is None:
            x0_sum = (
                float(sum(x0[d] for d in range(1, D + 1))) if x0 is not None else float("nan")
            )
        rows.append(
            {
                "hospital": target,
                "method": method,
                "w_mode": w_mode,
                "val_mean_daily": val_m,
                "test_mean_daily": test_m,
                "hyperparam": hp,
                "x0_sum": x0_sum,
            }
        )

    pack_cache: Dict[float, Dict[str, Any]] = {}

    def get_pack(ws: float) -> Dict[str, Any]:
        key = _w_pack_cache_key(ws)
        if key not in pack_cache:
            pack_cache[key] = _run_three_source_pack(
                emp, Xi, val_Z, val_H, test_Z, test_H, key, output_flag
            )
        return pack_cache[key]

    three_names = (
        "mr_DRO_no_order",
        "mr_DRO_with_order",
        "multi_ref_l1_only",
        "barycenter_equal_weights_K1",
    )

    # ----- 等权：w_self = 1/3（与三源候选之一相同，缓存复用） -----
    pack_eq = get_pack(_w_pack_cache_key(1.0 / 3.0))
    bary_B_mix_anchor = float(pack_eq["bary_B"])
    for name in three_names:
        r = pack_eq[name]
        record(
            name,
            r["x0"],
            r["val_m"],
            r["test_m"],
            r["hp"],
            w_mode=W_MODE_EQUAL,
            x0_sum=r.get("x0_sum"),
        )

    # ----- 验证选 w：各方法在 w_self 网格上独立取验证损失最小者 -----
    best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for w_self in THREE_SOURCE_W_SELF_GRID:
        p = get_pack(float(w_self))
        key_sel = _w_pack_cache_key(float(w_self))
        for name in three_names:
            vm = float(p[name]["val_m"])
            cur = best.get(name)
            if cur is None:
                best[name] = (key_sel, p[name])
            elif _is_better_val_loss(vm, float(cur[1]["val_m"])):
                best[name] = (key_sel, p[name])

    for name in three_names:
        if name not in best:
            continue
        w_sel, sub = best[name]
        hp = dict(sub["hp"])
        hp["w_self_selected"] = float(w_sel)
        hp["w_self_label"] = w_key_from_w_self(float(w_sel))
        record(
            name,
            sub["x0"],
            sub["val_m"],
            sub["test_m"],
            hp,
            w_mode=W_MODE_VAL_W_SELECTED,
            x0_sum=sub.get("x0_sum"),
        )

    # ----- 单参考：目标医院训练池 bootstrap（与同次 emp[0] 一致），在 11 月验证选 θ -----
    ref0 = np.asarray(emp[0], dtype=float).copy()
    th1_star, _, _, _, _ = model.solve_choosing_theta_epsilon(
        K=1,
        I=0,
        D=D,
        N=[N_SCEN],
        mathcal_D={},
        Xi=Xi,
        empirical_distributions={0: ref0},
        w_k=None,
        eta_k_n=None,
        theta_bar=0.0,
        tau=1.0,
        output_flag=output_flag,
    )
    # if th1_star is None:
    #     th1_star = 1.0
    thetas1 = theta_grid_from_anchors(float(th1_star), float(th1_star))

    def inv_val(t: float):
        return solve_inverse_k1(ref0, Xi, t, 0.0, None, output_flag)

    x0, vm, th = grid_search_val(inv_val, val_Z, thetas1)
    record(
        "single_ref_validation_only",
        x0,
        vm,
        mean_oos_days(x0, test_Z) if x0 else float("nan"),
        {
            "theta": th,
            "train_emp": "target_bootstrap_N_SCEN_same_as_emp0",
        },
        w_mode=W_MODE_NA,
    )

    # ----- 混合（等权）：拼成 K=1 的 30 样本；θ 网格第二锚点用三源等权重心目标（w0=1/3） -----
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
    b_mix_anchor = bary_B_mix_anchor
    thetas_m = theta_grid_from_anchors(float(thm), b_mix_anchor)

    def inv_mix(t: float):
        return solve_inverse_k1(mix, Xi, t, 0.0, None, output_flag)

    x0, vm, th = grid_search_val(inv_mix, val_Z, thetas_m)
    record(
        "mixture_equal_weights_K1",
        x0,
        vm,
        mean_oos_days(x0, test_Z) if x0 else float("nan"),
        {"theta": th, "bary_objective_anchor_w0_1/3": b_mix_anchor},
        w_mode=W_MODE_NA,
    )

    return rows


def main() -> None:
    out_path = pathlib.Path(__file__).resolve().parent / f"blood_experiment_results_seed={seed_}.xlsx"
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
            df_d.groupby(["hospital", "method", "w_mode"], as_index=False)
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
