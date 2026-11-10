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
           f"MD = {row['MD']:.3f}\nMAPE = {row['MAPE']:.1f}%\nN = {int(row['n_test'])}\n(hourly, whole period)")
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


_MAP_STYLE = {   # 欄位 -> (色階, 色階標籤)
    'R2': ('RdYlGn', 'R² (hourly, whole period)'),
    'RMSE': ('RdYlGn_r', f'RMSE ({config.SCATTER_UNIT})'),
    'obs_mean': ('Spectral_r', f'Observed seasonal mean ({config.SCATTER_UNIT})'),
    'pred_mean': ('Spectral_r', f'Predicted seasonal mean ({config.SCATTER_UNIT})'),
    'MD': ('RdBu_r', f'Mean deviation, pred - obs ({config.SCATTER_UNIT})'),
}


_MAP_FORMAT = {'R2': '.2f', 'RMSE': '.2f', 'obs_mean': '.1f', 'pred_mean': '.1f', 'MD': '.2f'}   # 標註的小數位數


def plot_station_map(metrics_df, output_dir, value='R2', vlim=None):
    """
    測站位置，顏色表示 value（R2、RMSE、obs_mean、pred_mean、MD），每個測站旁標註數值（見 config.LOSO_MAP_LABELS）。
    指標（R2、RMSE、MD）來自整段期間的逐小時預測；obs_mean / pred_mean 是整段期間的平均。
    vlim=(vmin, vmax) 可讓多張圖共用色階（例如實測與預測的季節平均）。有 geopandas 與縣市界檔時疊上縣市邊界。
    """
    m = metrics_df.dropna(subset=['lat', 'lon'])
    if m.empty:
        return None
    cmap, label = _MAP_STYLE[value]
    vmin, vmax = vlim if vlim else (None, None)
    if value == 'MD' and vlim is None:
        v = float(np.abs(m['MD']).max())
        vmin, vmax = -v, v
    fig, ax = plt.subplots(figsize=(7, 9), dpi=config.FIG_DPI)
    try:
        import geopandas as gpd
        counties = gpd.read_file(config.SHP_PATH)
        counties = (counties.set_crs(epsg=3824) if counties.crs is None else counties).to_crs(epsg=4326)
        counties.boundary.plot(ax=ax, color='gray', linewidth=0.5, zorder=1)
    except Exception as e:
        print(f"[資訊] 略過縣市邊界（{e}）")
    sc = ax.scatter(m['lon'], m['lat'], c=m[value], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=70, edgecolor='black', linewidth=0.5, zorder=3)
    # 在每個測站旁標上數值（可選擇連同測站名稱；測站密集處標籤可能互相靠近）
    if config.LOSO_MAP_LABELS != 'none':
        import matplotlib.patheffects as pe
        fmt = _MAP_FORMAT.get(value, '.2f')
        for _, r_ in m.iterrows():
            txt = format(r_[value], fmt)
            if config.LOSO_MAP_LABELS == 'name_value':
                txt = f"{r_['name']}\n{txt}"
            ax.annotate(txt, (r_['lon'], r_['lat']), xytext=(4, 4), textcoords='offset points',
                        fontsize=config.LOSO_MAP_LABEL_FONTSIZE, zorder=5,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    pad = 0.15
    ax.set_xlim(m['lon'].min() - pad, m['lon'].max() + pad)
    ax.set_ylim(m['lat'].min() - pad, m['lat'].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Leave-one-station-out: {label}', fontsize=11)
    fig.colorbar(sc, ax=ax, shrink=0.75, label=label)
    path = os.path.join(output_dir, f"loso_station_map_{value}.png")
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    print(f"✅ 測站空間分佈圖已儲存至: {path}")
    return path


def plot_seasonal_mean_maps(metrics_df, output_dir):
    """季節平均的測站分佈圖：實測與預測共用色階，另加偏差圖。"""
    m = metrics_df.dropna(subset=['lat', 'lon'])
    if m.empty:
        return
    vlim = (float(min(m['obs_mean'].min(), m['pred_mean'].min())), float(max(m['obs_mean'].max(), m['pred_mean'].max())))
    plot_station_map(metrics_df, output_dir, 'obs_mean', vlim=vlim)
    plot_station_map(metrics_df, output_dir, 'pred_mean', vlim=vlim)
    plot_station_map(metrics_df, output_dir, 'MD')


def plot_all_scatter(preds, metrics_df, output_dir, station_col):
    """
    每個測站一張散布圖（格式同 ground_truth_vs_model_*.png）：
    橫軸實測、縱軸移除該測站後的預測，使用整段期間「所有逐小時」的點，
    r、R²、RMSE、MD、MAPE、N 都由這些逐小時資料計算（不使用每日平均）。所有測站共用同一組座標範圍。
    """
    from loso import safe_name
    from scatter_plot import plot_obs_vs_model, axis_limits
    out_dir = os.path.join(output_dir, "scatter")
    lim = axis_limits(preds['obs'].to_numpy(), preds['pred'].to_numpy())
    ids = preds[station_col].astype(str)
    for _, row in metrics_df.iterrows():
        sub = preds[ids == row['station']]
        plot_obs_vs_model(sub['obs'], sub['pred'], out_dir, name=f"loso_{safe_name(row['station'])}",
                          title_suffix=f"LOSO, {row['name']} ({row['station']}), hourly, whole period", lim=lim)
    print(f"✅ 已輸出 {len(metrics_df)} 張測站散布圖至: {out_dir}")


def plot_station_hour_scatter(preds, metrics_df, output_dir, station_col):
    """
    每個測站、每個小時（Hr）各一張散布圖，例如 4 個測站、8 個小時 = 32 張。
    每張圖的點是「該測站該小時」整個期間的逐日資料，r、R²、RMSE、MD、MAPE、N 都由這些點計算。
    所有圖共用同一組座標範圍。同時輸出 loso_station_hour_metrics.csv。
    """
    from loso import safe_name
    from scatter_plot import plot_obs_vs_model, axis_limits, _hour_tag
    from metrics import calculate_evaluation_metrics
    import warnings
    hour_col = first_existing(preds.columns, config.HOUR_COL_CANDIDATES)
    if hour_col is None:
        print(f"[警告] 找不到小時欄位（候選 {config.HOUR_COL_CANDIDATES}），略過逐測站逐小時散布圖")
        return None
    lim = axis_limits(preds['obs'].to_numpy(), preds['pred'].to_numpy())
    ids = preds[station_col].astype(str)
    rows, n_fig = [], 0
    for _, srow in metrics_df.iterrows():
        sub_st = preds[ids == srow['station']]
        out_dir = os.path.join(output_dir, "scatter_by_hour", safe_name(srow['station']))
        for h in sorted(sub_st[hour_col].dropna().unique()):
            sub = sub_st[sub_st[hour_col] == h]
            if len(sub) < config.LOSO_MIN_POINTS_PER_HOUR:
                print(f"   - {srow['station']} Hr={h:g}: 只有 {len(sub)} 筆（少於 {config.LOSO_MIN_POINTS_PER_HOUR}），略過")
                continue
            plot_obs_vs_model(sub['obs'], sub['pred'], out_dir,
                              name=f"loso_{safe_name(srow['station'])}_{_hour_tag(h)}",
                              title_suffix=f"LOSO, {srow['name']} ({srow['station']}), Hr = {h:g}", lim=lim)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                m = {k.split(' (')[0]: v for k, v in
                     calculate_evaluation_metrics(sub['obs'].to_numpy(), sub['pred'].to_numpy()).items()}
            rows.append({'station': srow['station'], 'name': srow['name'], 'Hr': h, 'N': len(sub), **m})
            n_fig += 1
    df = pd.DataFrame(rows)
    if len(df):
        path = os.path.join(output_dir, "loso_station_hour_metrics.csv")
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"✅ 已輸出 {n_fig} 張逐測站逐小時散布圖至 {os.path.join(output_dir, 'scatter_by_hour')}，指標表: {path}")
    return df


def plot_shap_heatmap(shap_df, output_dir):
    """
    各測站的 SHAP 特徵重要性彙整：列 = 測站、欄 = 特徵，數值為該測站平均 |SHAP| 佔該測站所有特徵總和的比例。
    可看出模型在不同測站依賴的特徵是否一致。
    """
    feats = [c for c in shap_df.columns if c not in ('station', 'name')]
    mat = shap_df[feats].to_numpy(dtype='float64')
    share = mat / mat.sum(axis=1, keepdims=True)
    order = np.argsort(-share.mean(axis=0))
    share = share[:, order]
    feats = [feats[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(feats) + 3), max(4, 0.35 * len(shap_df) + 2)), dpi=config.FIG_DPI)
    im = ax.imshow(share, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels([f"{n} ({s})" for n, s in zip(shap_df['name'], shap_df['station'])], fontsize=8)
    if share.size <= 600:
        for i in range(share.shape[0]):
            for j in range(share.shape[1]):
                ax.text(j, i, f"{share[i, j] * 100:.0f}", ha='center', va='center', fontsize=6,
                        color='white' if share[i, j] > 0.5 * share.max() else 'black')
    fig.colorbar(im, ax=ax, label='Share of mean |SHAP| (%)', format=lambda v, _: f"{v * 100:.0f}")
    ax.set_title('Leave-one-station-out: SHAP importance by station', fontsize=11)
    path = os.path.join(output_dir, "loso_shap_importance_heatmap.png")
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    print(f"✅ SHAP 彙整熱圖已儲存至: {path}")
    return path