"""
Ground truth vs Model 散布圖：橫軸為實測值 (ground truth)、縱軸為模型預測值，
點以「局部點密度」上色（彩虹色階），左上角標示 r、R2、RMSE、MD、MAPE、N，並畫 1:1 紅色虛線。
  - plot_obs_vs_model：單一資料集一張圖（訓練集、測試集整體）
  - plot_by_hour：依小時（Hr）各畫一張，所有小時共用同一組座標範圍，方便比較
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import config
from metrics import calculate_evaluation_metrics


def local_point_density(x, y, lim, bins=config.SCATTER_DENSITY_BINS,
                        smooth=config.SCATTER_DENSITY_SMOOTH):
    """
    每個點所在位置的局部密度：先在 (x, y) 平面統計 2D 直方圖，
    經高斯平滑後取每個點所在格子的值，再取 ln(1 + 密度) 壓縮動態範圍。
    比 gaussian_kde 快很多，樣本數很大時也能用。
    """
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[lim, lim])
    if smooth:
        h = gaussian_filter(h, sigma=smooth)
    ix = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    return np.log1p(h[ix, iy])


def axis_limits(*arrays):
    """兩軸共用的座標範圍：下限取 0（或資料最小值，若有負值），上限為最大值再多 2%。"""
    lo = min(0.0, min(float(np.nanmin(a)) for a in arrays))
    hi = max(float(np.nanmax(a)) for a in arrays) * 1.02
    return (lo, hi)


def plot_obs_vs_model(y_true, y_pred, output_dir, name="test", title_suffix="Test",
                      unit=config.SCATTER_UNIT, lim=None):
    y_true = np.asarray(y_true, dtype='float64')
    y_pred = np.asarray(y_pred, dtype='float64')

    # 指標一律用全部樣本計算
    m = {k.split(' (')[0]: v for k, v in calculate_evaluation_metrics(y_true, y_pred).items()}
    n = len(y_true)

    # 點數過多時，只抽樣繪圖（指標與密度的統計仍使用全部樣本）
    if lim is None:
        lim = axis_limits(y_true, y_pred)
    dens_all = local_point_density(y_true, y_pred, lim)
    if n > config.SCATTER_MAX_PLOT_POINTS:
        idx = np.random.RandomState(config.SHAP_RANDOM_STATE).choice(
            n, config.SCATTER_MAX_PLOT_POINTS, replace=False)
        x, y, d = y_true[idx], y_pred[idx], dens_all[idx]
    else:
        x, y, d = y_true, y_pred, dens_all

    order = np.argsort(d)              # 密度高的點畫在最上層
    x, y, d = x[order], y[order], d[order]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=config.FIG_DPI)
    sc = ax.scatter(x, y, c=d, cmap='Spectral_r', s=2, linewidths=0, vmin=0, rasterized=True)
    ax.plot(lim, lim, color='red', linestyle='--', linewidth=1.5)   # 1:1 線

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.5)
    ax.set_xlabel(f'{config.SCATTER_XLABEL} (unit : {unit})', fontsize=13)
    ax.set_ylabel(f'MODEL (unit : {unit})', fontsize=13)
    ax.set_title(f'{config.SCATTER_TITLE}\n({title_suffix})', fontsize=13)
    ax.tick_params(labelsize=11)

    text = (f"r = {m['r']:.3f}\n"
            f"R² = {m['R2']:.3f}\n"
            f"RMSE = {m['RMSE']:.3f}\n"
            f"MD = {m['MD']:.3f}\n"
            f"MAPE = {m['MAPE']:.1f}%\n"
            f"N = {n}")
    ax.text(0.04, 0.96, text, transform=ax.transAxes, fontsize=9.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
            zorder=10)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Local point density', fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"ground_truth_vs_model_{name}.png")
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    print(f"✅ 散布圖已儲存至: {path}")
    return path


def _hour_tag(h):
    return f"hr{int(h):02d}" if float(h).is_integer() else f"hr{h:g}"


def plot_by_hour(y_true, y_pred, hours, output_dir, name="test", title_label="Test",
                 min_points=config.SCATTER_MIN_POINTS_PER_HOUR):
    """
    依小時各畫一張散布圖，存到 output_dir/Scatter_by_hour/，並輸出各小時指標的 CSV。
    所有小時共用同一組座標範圍（由全部資料決定），方便比較不同小時的表現。
    """
    y_true = np.asarray(y_true, dtype='float64')
    y_pred = np.asarray(y_pred, dtype='float64')
    hours = np.asarray(hours, dtype='float64')
    out_dir = os.path.join(output_dir, "Scatter_by_hour")
    lim = axis_limits(y_true, y_pred)

    rows = []
    for h in np.unique(hours[np.isfinite(hours)]):
        sel = hours == h
        n = int(sel.sum())
        if n < min_points:
            print(f"   - Hr={h:g}: 只有 {n} 筆（少於 {min_points}），略過")
            continue
        plot_obs_vs_model(y_true[sel], y_pred[sel], out_dir, name=f"{name}_{_hour_tag(h)}",
                          title_suffix=f"{title_label}, Hr = {h:g}", lim=lim)
        m = {k.split(' (')[0]: v for k, v in calculate_evaluation_metrics(y_true[sel], y_pred[sel]).items()}
        rows.append({'Hr': h, 'N': n, **m})

    if rows:
        csv_path = os.path.join(out_dir, f"metrics_by_hour_{name}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 逐小時指標已儲存至: {csv_path}")
    return rows