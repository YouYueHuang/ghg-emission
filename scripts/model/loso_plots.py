"""
LOSO 圖表：每個測站一張時間序列圖（實測 vs 移除該測站後的預測）、各測站指標長條圖、測站空間分佈圖。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from features import first_existing

plt.rcParams['font.sans-serif'] = list(config.PLOT_CJK_FONTS) + ['DejaVu Sans']   # 測站名稱含中文時才顯示得出來
plt.rcParams['axes.unicode_minus'] = False


def _timestamps(sub):
    """日期 + 小時 → 完整時間；無法解析時回傳 None。"""
    try:
        ts = pd.to_datetime(sub[config.LOSO_DATE_COL]).dt.normalize()
        hour_col = first_existing(sub.columns, config.HOUR_COL_CANDIDATES)
        if hour_col is not None:
            ts = ts + pd.to_timedelta(sub[hour_col].astype(float), unit='h')
        return ts
    except Exception:
        return None


def _best_window(hourly, days):
    """資料最多的連續 days 天（以有實測且有預測的小時數計）。回傳 (起, 迄)。"""
    valid = hourly.dropna().groupby(hourly.dropna().index.normalize()).size()
    full = valid.reindex(pd.date_range(hourly.index.min().normalize(), hourly.index.max().normalize(), freq='D'),
                         fill_value=0)
    if len(full) <= days:
        return full.index[0], full.index[-1] + pd.Timedelta(days=1)
    start = full.rolling(days).sum().shift(-(days - 1)).idxmax()
    return start, start + pd.Timedelta(days=days)


def plot_station_timeseries(sub, row, out_path):
    """上圖：每日平均（整段期間）；下圖：逐小時（資料最多的 LOSO_ZOOM_DAYS 天，缺資料處斷線）。"""
    ts = _timestamps(sub)
    obs, pred = sub['obs'].to_numpy(), sub['pred'].to_numpy()
    zoom_note = ''
    if ts is None:
        s = pd.DataFrame({'obs': obs, 'pred': pred}, index=pd.RangeIndex(len(sub)))
        hourly = daily = s
        xlabel = 'sample order'
    else:
        s = pd.DataFrame({'obs': obs, 'pred': pred}, index=pd.DatetimeIndex(ts.to_numpy()).floor('h'))
        s = s.groupby(level=0).mean().sort_index()
        daily = s.resample('D').mean()                                              # 沒資料的日子是 NaN，折線自動斷開
        hourly = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))    # 缺的小時補 NaN
        if config.LOSO_ZOOM_DAYS and config.LOSO_ZOOM_DAYS > 0:
            t0, t1 = _best_window(hourly, int(config.LOSO_ZOOM_DAYS))
            hourly = hourly[(hourly.index >= t0) & (hourly.index < t1)]
            zoom_note = f" (zoom: {t0.date()} ~ {(t1 - pd.Timedelta(days=1)).date()})"
        xlabel = 'time'

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=config.FIG_DPI)
    for ax, data, title, ms in ((axes[0], daily, 'Daily mean', 5), (axes[1], hourly, 'Hourly' + zoom_note, 4)):
        ax.plot(data.index, data['obs'], color='black', marker='o', markersize=ms, linewidth=0.9, label='Observed')
        ax.plot(data.index, data['pred'], color='tab:red', marker='o', markersize=ms, linewidth=0.9,
                label='Predicted (station left out)')
        ax.set_ylabel(f'NO2 ({config.SCATTER_UNIT})')
        ax.set_title(title, fontsize=11, loc='left')
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[1].set_xlabel(xlabel)

    txt = (f"r = {row['r']:.3f}\nR² = {row['R2']:.3f}\nRMSE = {row['RMSE']:.3f}\n"
           f"MD = {row['MD']:.3f}\nMAPE = {row['MAPE']:.1f}%\nN = {int(row['n_test'])} (hourly, all)")
    axes[0].text(0.01, 0.97, txt, transform=axes[0].transAxes, va='top', ha='left', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9), zorder=10)
    fig.suptitle(f"Leave-one-station-out: {row['name']} ({row['station']})", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)


def plot_all_timeseries(preds, metrics_df, output_dir, station_col):
    from loso import safe_name
    out_dir = os.path.join(output_dir, "timeseries")
    ids = preds[station_col].astype(str)
    n = 0
    for _, row in metrics_df.iterrows():
        sub = preds[ids == row['station']]
        plot_station_timeseries(sub, row, os.path.join(out_dir, f"loso_{safe_name(row['station'])}.png"))
        n += 1
    print(f"✅ 已輸出 {n} 張測站時間序列圖至: {out_dir}")


def plot_station_metrics(metrics_df, output_dir):
    """各測站 R² 與 RMSE 長條圖（依 R² 由高到低），虛線為中位數。"""
    m = metrics_df.sort_values('R2', ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.28 * len(m) + 3), 8), dpi=config.FIG_DPI, sharex=True)
    for ax, col, color in ((axes[0], 'R2', 'tab:blue'), (axes[1], 'RMSE', 'tab:orange')):
        ax.bar(m['name'], m[col], color=color)
        ax.axhline(m[col].median(), color='red', linestyle='--', linewidth=1, label=f"median = {m[col].median():.3f}")
        ax.set_ylabel(col if col == 'RMSE' else 'R²')
        ax.grid(alpha=0.3, axis='y')
        ax.legend()
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xticks(range(len(m)))
    axes[1].set_xticklabels(m['name'], rotation=90, fontsize=8)
    fig.suptitle('Leave-one-station-out: per-station skill', fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "loso_station_metrics.png")
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    print(f"✅ 各測站指標圖已儲存至: {path}")


def plot_station_map(metrics_df, output_dir, value='R2'):
    """測站位置，顏色表示 R²（或 RMSE）；有 geopandas 與縣市界檔時疊上縣市邊界。"""
    m = metrics_df.dropna(subset=['lat', 'lon'])
    if m.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 9), dpi=config.FIG_DPI)
    try:
        import geopandas as gpd
        counties = gpd.read_file(config.SHP_PATH)
        counties = (counties.set_crs(epsg=3824) if counties.crs is None else counties).to_crs(epsg=4326)
        counties.boundary.plot(ax=ax, color='gray', linewidth=0.5, zorder=1)
    except Exception as e:
        print(f"[資訊] 略過縣市邊界（{e}）")
    sc = ax.scatter(m['lon'], m['lat'], c=m[value], cmap='RdYlGn' if value == 'R2' else 'RdYlGn_r',
                    s=70, edgecolor='black', linewidth=0.5, zorder=3)
    pad = 0.15
    ax.set_xlim(m['lon'].min() - pad, m['lon'].max() + pad)
    ax.set_ylim(m['lat'].min() - pad, m['lat'].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Leave-one-station-out: {value} by station')
    fig.colorbar(sc, ax=ax, shrink=0.75, label=value if value != 'R2' else 'R²')
    path = os.path.join(output_dir, f"loso_station_map_{value}.png")
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    print(f"✅ 測站空間分佈圖已儲存至: {path}")
    return path