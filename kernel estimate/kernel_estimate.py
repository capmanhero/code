"""
Kernel density estimates of hospital blood demand by blood type.
Reads demand_CL, demand_ET, demand_SX, demand_PJ from data.xlsx; one figure per blood type.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data.xlsx"
# FIGURES_DIR = ROOT / "figures"

SHEETS = {
    "CL": "demand_CL",
    # "ET": "demand_ET",
    "SX": "demand_SX",
    "PJ": "demand_PJ",
}
BLOOD_TYPES = ["A", "AB", "B", "O"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _load_hospital_series() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {h: {} for h in SHEETS}
    for hosp, sheet in SHEETS.items():
        df = pd.read_excel(DATA_PATH, sheet_name=sheet)
        for bt in BLOOD_TYPES:
            s = df[bt].to_numpy(dtype=float)
            s = s[np.isfinite(s)]
            out[hosp][bt] = s
    return out


def _kde_on_positive(x: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    x = x[x > 0]
    if x.size < 2:
        return None
    if np.unique(x).size < 2:
        return None
    kde = gaussian_kde(x, bw_method="scott")
    y = kde(grid)
    return np.clip(y, 1e-300, None)


def plot_blood_type(ax, data_by_hosp: dict[str, np.ndarray], title: str) -> None:
    global_min = np.inf
    global_max = -np.inf
    for arr in data_by_hosp.values():
        pos = arr[arr > 0]
        if pos.size:
            global_min = min(global_min, float(pos.min()))
            global_max = max(global_max, float(pos.max()))
    if not np.isfinite(global_min):
        ax.set_title(title)
        ax.text(0.5, 0.5, "无正值样本", ha="center", va="center", transform=ax.transAxes)
        return

    lo = max(global_min * 0.6, 1e-3)
    hi = global_max * 1.2
    grid = np.logspace(np.log10(lo), np.log10(hi), 512)

    for (hosp, color) in zip(SHEETS.keys(), COLORS):
        arr = data_by_hosp[hosp]
        y = _kde_on_positive(arr, grid)
        if y is None:
            continue
        ax.plot(grid, y, color=color, linewidth=1.8, label=hosp)
        ax.fill_between(grid, y, alpha=0.25, color=color)

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(False)


def main() -> None:
    # FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_data = _load_hospital_series()

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
        }
    )

    for bt in BLOOD_TYPES:
        fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained")
        data_bt = {h: all_data[h][bt] for h in SHEETS}
        plot_blood_type(ax, data_bt, title=bt)
        out = f"kernel estimate/figures/kde_demand_{bt}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
