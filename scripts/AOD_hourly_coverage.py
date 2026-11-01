"""
分析 AOD 該月「每一日、每一個時次」有值的 pixel coverage
- 沿用 AOD.py 的 day 座標修復；time 座標讀不出真實時間，因此以索引 0 ~ N-1 表示
- 涵蓋率 = 該 (day, time) 有效像素數 / 總像素數 (lat x lon) x 100
- 輸出：CSV（有效像素數、涵蓋率、各時次/各日摘要）與涵蓋率熱圖
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ===== 設定中文字型與負號正常顯示 =====
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 路徑與參數 =====
input_dir = r"../data/raw/nc_datas/data_2023/AOD"
output_dir = r"../results/tables/AOD"
filename = "202303.nc"
var_name = 'GK2_D_AOD'

# 若知道每個時次對應的實際時間（例如 UTC 小時），可填在這裡，長度需等於時次數；否則維持 None 用索引
TIME_LABELS = None     # 例如 ['00', '01', '02', ...]

os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(input_dir, filename)
tag = os.path.splitext(filename)[0]

# ===== 2. 色階：優先使用 color_palette（20 階，每階 5%），找不到則用預設 =====
try:
    from color_palette import boundaries as cov_bounds
    from color_palette import hex_colors as cov_colors
    assert len(cov_colors) == len(cov_bounds) - 1
except Exception:
    cov_bounds = np.arange(0, 100.1, 5)
    cov_colors = [mcolors.to_hex(plt.get_cmap('viridis')(i / 19)) for i in range(20)]
cov_cmap = mcolors.ListedColormap(cov_colors)
cov_norm = mcolors.BoundaryNorm(cov_bounds, cov_cmap.N)

# ===== 3. 讀取檔案與修復座標 =====
ds = xr.open_dataset(file_path, decode_times=False)

day_dim, time_dim = 'day', 'time'
if day_dim in ds.coords:
    dv = ds[day_dim].values
    if np.issubdtype(dv.dtype, np.number) and (dv > 1e30).any():
        ds = ds.assign_coords({day_dim: np.arange(1, len(dv) + 1)})

da = ds[var_name].drop_vars(time_dim, errors='ignore')   # 移除 2D 的 time 字串座標
da = da.transpose(day_dim, time_dim, 'latitude', 'longitude')

days = ds[day_dim].values.astype(int)
nd, nt = da.sizes[day_dim], da.sizes[time_dim]
npix = da.sizes['latitude'] * da.sizes['longitude']

time_idx = np.arange(nt)
if TIME_LABELS is not None:
    assert len(TIME_LABELS) == nt, "TIME_LABELS 長度必須等於時次數"
    time_cols = list(TIME_LABELS)
else:
    time_cols = [f"t{i}" for i in time_idx]

# ===== 4. 計算每個 (day, time) 的有效像素數與涵蓋率 =====
arr = da.values.astype('float64')                        # (day, time, lat, lon)
with np.errstate(invalid='ignore'):
    is_valid = np.isfinite(arr) & (arr > 0) & (arr < 1e30)

valid_count = is_valid.sum(axis=(2, 3))                  # (day, time)
coverage = valid_count / npix * 100

count_df = pd.DataFrame(valid_count, index=days, columns=time_cols)
cov_df = pd.DataFrame(coverage, index=days, columns=time_cols).round(2)
for df in (count_df, cov_df):
    df.index.name = 'day'
    df.columns.name = 'time'

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)

print("=" * 60)
print(f"{tag}：每日每時次有效像素涵蓋率 (%)；總像素數 = {npix}")
print("=" * 60)
print(cov_df)

# ===== 5. 摘要 =====
# 各時次：整月統計
hour_summary = pd.DataFrame({
    'time': time_cols,
    'days_with_data': (valid_count > 0).sum(axis=0),
    'mean_coverage_pct': coverage.mean(axis=0).round(2),
    'max_coverage_pct': coverage.max(axis=0).round(2),
    'day_of_max': days[coverage.argmax(axis=0)],
})
# 各日：全部時次統計
day_summary = pd.DataFrame({
    'day': days,
    'times_with_data': (valid_count > 0).sum(axis=1),
    'mean_coverage_pct': coverage.mean(axis=1).round(2),
    'max_coverage_pct': coverage.max(axis=1).round(2),
    'time_of_max': [time_cols[j] for j in coverage.argmax(axis=1)],
})

print("\n各時次摘要（整月）:")
print(hour_summary.to_string(index=False))
print("\n各日摘要:")
print(day_summary.to_string(index=False))

empty_slices = int((valid_count == 0).sum())
print(f"\n完全沒有資料的 (day, time) 組合: {empty_slices}/{nd * nt}")
print(f"完全沒有資料的時次: {[time_cols[j] for j in range(nt) if valid_count[:, j].sum() == 0]}")
print(f"完全沒有資料的日期: {[int(days[i]) for i in range(nd) if valid_count[i].sum() == 0]}")

gi, gj = np.unravel_index(coverage.argmax(), coverage.shape)
print(f"涵蓋率最高的 (day, time): day={days[gi]}, time={time_cols[gj]} ({coverage[gi, gj]:.2f}%)")

# ===== 6. 輸出 CSV =====
count_df.to_csv(os.path.join(output_dir, f'AOD_{tag}_hourly_valid_count.csv'))
cov_df.to_csv(os.path.join(output_dir, f'AOD_{tag}_hourly_coverage_pct.csv'))
hour_summary.to_csv(os.path.join(output_dir, f'AOD_{tag}_hourly_summary_by_time.csv'), index=False)
day_summary.to_csv(os.path.join(output_dir, f'AOD_{tag}_hourly_summary_by_day.csv'), index=False)

# ===== 7. 熱圖：每格標示涵蓋率 =====
cov_values = cov_df.values
fig, ax = plt.subplots(figsize=(max(8, 1.0 * nt + 3), max(6, 0.4 * nd + 2)))
im = ax.imshow(cov_values, aspect='auto', cmap=cov_cmap, norm=cov_norm)

for r in range(nd):
    for c in range(nt):
        val = cov_values[r, c]
        idx = int(np.clip(np.digitize(val, cov_bounds) - 1, 0, len(cov_colors) - 1))
        rgb = mcolors.to_rgb(cov_colors[idx])
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        ax.text(c, r, "0" if val == 0 else f"{val:.1f}", ha='center', va='center',
                fontsize=7, color='white' if lum < 0.5 else 'black')

ax.set_xticks(range(nt))
ax.set_xticklabels(time_cols)
ax.set_yticks(range(nd))
ax.set_yticklabels([str(d) for d in days])
ax.set_xlabel('time index' if TIME_LABELS is None else 'time')
ax.set_ylabel('day')
ax.set_title(f'AOD 有效像素涵蓋率 (%) - {tag}')

ax.set_xticks(np.arange(-0.5, nt, 1), minor=True)
ax.set_yticks(np.arange(-0.5, nd, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.5)
ax.tick_params(which='minor', length=0)

cbar = fig.colorbar(im, ax=ax, ticks=cov_bounds, shrink=0.8)
cbar.set_label('coverage (%)')

png = os.path.join(output_dir, f'AOD_{tag}_hourly_coverage_heatmap.png')
fig.savefig(png, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\n已輸出 CSV 與熱圖至: {output_dir}")
ds.close()