"""
AOD (Aerosol Optical Depth) 月資料探索。

- 自動偵測主要變數與 day / hour / 經度 / 緯度 的名稱（偵測不到可在下方設定區手動指定）
- 檢查缺值、各 (day, hour) 的有效像素涵蓋率、哪些 hour / day 完全沒資料
- 輸出 CSV 與 20 階色階的涵蓋率熱圖
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from color_palette import boundaries as cov_bounds
from color_palette import hex_colors as cov_colors

# ===== 路徑設定 =====
input_dir = r"../data/raw/nc_datas/data_2023/AOD"
output_dir = r"../results/tables/AOD"
filename = "202301.nc"

# ===== 名稱設定：None = 自動偵測；偵測錯誤時可手動指定 =====
VAR_NAME = None        # 例如 'aod'
DAY_DIM = None         # 例如 'day'
HOUR_DIM = None        # 例如 'hour'
LON_NAME = None        # 例如 'longitude'
LAT_NAME = None        # 例如 'latitude'
EXTRA_ISEL = {}        # 變數若還有其他維度（如波長），在此選一個，例如 {'wavelength': 0}

# ===== 缺值判斷 =====
EXTRA_FILL_VALUES = (-999, -9999)   # 除了 NaN 與 <= -1e30 之外，也算作 fill value 的數值
SUSPECT_MAX = 5.0                   # AOD 超過此值通常不合理，僅統計、不排除

os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(input_dir, filename)
tag = os.path.splitext(filename)[0]


# ===== 名稱偵測 =====
def pick_variable(ds, var_name=None):
    if var_name:
        return var_name
    cands = [n for n, v in ds.data_vars.items() if v.ndim >= 3]
    if not cands:
        raise SystemExit("找不到維度 >= 3 的資料變數，請在設定區手動指定 VAR_NAME。")
    named = [n for n in cands if 'aod' in n.lower()]
    pool = named or cands
    best = max(pool, key=lambda n: ds[n].size)
    if len(cands) > 1:
        print(f"[資訊] 候選變數 {cands}，選用 '{best}'（可用 VAR_NAME 指定）")
    return best


def pick_dim(dims, key, given=None):
    if given:
        return given
    for d in dims:
        if d.lower() == key:
            return d
    for d in dims:
        if key in d.lower():
            return d
    return None


def pick_coord(ds, candidates, given=None):
    if given:
        return given
    names = list(ds.variables)
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


# ===== 1. 基本結構 =====

ds = xr.open_dataset(file_path, decode_times=False)
print("=" * 60)
print("1. 檔案結構")
print("=" * 60)
print(ds)
print("\n全域屬性:", dict(ds.attrs))
print("\n各資料變數:")
for n, v in ds.data_vars.items():
    print(f"  {n}: dims={v.dims}, shape={v.shape}, dtype={v.dtype}")
    print(f"     attrs={dict(v.attrs)}")

var = pick_variable(ds, VAR_NAME)
da = ds[var]
if EXTRA_ISEL:
    da = da.isel(**EXTRA_ISEL)
print(f"\n使用變數: '{var}'，維度 {dict(da.sizes)}")

lon_name = pick_coord(ds, ['longitude', 'lon', 'long', 'x'], LON_NAME)
lat_name = pick_coord(ds, ['latitude', 'lat', 'y'], LAT_NAME)
day_dim = pick_dim(da.dims, 'day', DAY_DIM)
hour_dim = pick_dim(da.dims, 'hour', HOUR_DIM)
print(f"偵測結果: lon={lon_name}, lat={lat_name}, day={day_dim}, hour={hour_dim}")
if lon_name is None or lat_name is None:
    raise SystemExit("找不到經緯度變數，請設定 LON_NAME / LAT_NAME。")

# 缺少 day 或 hour 維度時，補上長度 1 的維度，後續流程不變
if day_dim is None:
    print("[資訊] 沒有 day 維度，補上長度 1 的 day 維度")
    da = da.expand_dims(day=[1]); day_dim = 'day'
if hour_dim is None:
    print("[資訊] 沒有 hour 維度，補上長度 1 的 hour 維度")
    da = da.expand_dims(hour=[1]); hour_dim = 'hour'

# ===== 2. 經緯度資訊 =====
lon = ds[lon_name].values
lat = ds[lat_name].values
print("\n" + "=" * 60)
print("2. 經緯度")
print("=" * 60)
print(f"{lon_name} shape={lon.shape}, 範圍 {np.nanmin(lon):.4f} ~ {np.nanmax(lon):.4f}")
print(f"{lat_name} shape={lat.shape}, 範圍 {np.nanmin(lat):.4f} ~ {np.nanmax(lat):.4f}")
if lon.ndim == 2:
    print(f"經度方向間距(中位數): {np.nanmedian(np.diff(lon, axis=1)):.4f}")
    print(f"緯度方向間距(中位數): {np.nanmedian(np.diff(lat, axis=0)):.4f}")
else:
    print(f"經度間距(中位數): {np.nanmedian(np.diff(lon)):.4f}")
    print(f"緯度間距(中位數): {np.nanmedian(np.diff(lat)):.4f}")

# ===== 3. 整理成 (day, hour, 空間像素) =====
da = da.transpose(day_dim, hour_dim, ...)
day_vals = da[day_dim].values
hour_vals = da[hour_dim].values
nd, nh = len(day_vals), len(hour_vals)
arr = da.values.reshape(nd, nh, -1).astype('float64')
npix = arr.shape[2]

with np.errstate(invalid='ignore'):
    finite = np.isfinite(arr)
    is_nan = ~finite
    is_fill = finite & (arr <= -1e30)
    for fv in EXTRA_FILL_VALUES:
        is_fill |= finite & (arr == fv)
    is_nonpos = finite & ~is_fill & (arr <= 0)
    is_valid = finite & (arr > 0)

print("\n" + "=" * 60)
print("3. 缺值分類統計（全部像素）")
print("=" * 60)
total = arr.size
print(f"總像素數      : {total}")
print(f"NaN           : {is_nan.sum()} ({is_nan.mean()*100:.2f}%)")
print(f"fill value    : {is_fill.sum()} ({is_fill.mean()*100:.2f}%)")
print(f"0 或負值      : {is_nonpos.sum()} ({is_nonpos.mean()*100:.2f}%)")
print(f"有效值(>0)    : {is_valid.sum()} ({is_valid.mean()*100:.2f}%)")

# ===== 4. day x hour 有效像素矩陣 =====
valid_count = is_valid.sum(axis=2)                      # (day, hour)
coverage = valid_count / npix * 100                     # 涵蓋率 %

count_df = pd.DataFrame(valid_count, index=day_vals, columns=hour_vals)
count_df.index.name = 'day'
count_df.columns.name = 'hour'

cov_df = pd.DataFrame(coverage, index=day_vals, columns=hour_vals).round(1)
cov_df.index.name = 'day'
cov_df.columns.name = 'hour'

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

print("\n" + "=" * 60)
print("4. 各 (day, hour) 有效像素涵蓋率 % （列=day, 欄=hour）")
print("=" * 60)
print(cov_df)

# ===== 5. 哪些 hour / day 完全沒有 value =====
hour_total = valid_count.sum(axis=0)    # 各 hour 在整個月的有效像素
day_total = valid_count.sum(axis=1)     # 各 day 在所有 hour 的有效像素

empty_hours = hour_vals[hour_total == 0]
empty_days = day_vals[day_total == 0]
hours_with_data = hour_vals[hour_total > 0]
days_with_data = day_vals[day_total > 0]

print("\n" + "=" * 60)
print("5. 完全沒有 value 的 hour / day")
print("=" * 60)
print(f"【整個月都沒有任何資料的 hour】({len(empty_hours)}/{nh}): {list(empty_hours)}")
print(f"【有資料的 hour】({len(hours_with_data)}/{nh}): {list(hours_with_data)}")
print(f"【整天都沒有任何資料的 day】({len(empty_days)}/{nd}): {list(empty_days)}")
print(f"【有資料的 day】({len(days_with_data)}/{nd}): {list(days_with_data)}")

print("\n各 hour 在整月的有效天數（該 hour 至少有 1 個有效像素的天數）:")
hour_days = (valid_count > 0).sum(axis=0)
print(pd.Series(hour_days, index=hour_vals, name='days_with_data').to_string())

print("\n各 day 有資料的 hour 數:")
day_hours = (valid_count > 0).sum(axis=1)
print(pd.Series(day_hours, index=day_vals, name='hours_with_data').to_string())

# ===== 6. 數值統計 =====
print("\n" + "=" * 60)
print("6. 有效值統計（AOD 無單位）")
print("=" * 60)
vals = arr[is_valid]
if vals.size > 0:
    q = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
    print(f"筆數 {vals.size}, min {vals.min():.4f}, max {vals.max():.4f}, mean {vals.mean():.4f}")
    print("百分位 1/5/25/50/75/95/99: " + ", ".join(f"{x:.4f}" for x in q))
    print(f"超過 1.0 的比例: {(vals > 1.0).mean() * 100:.2f}%；"
          f"超過 {SUSPECT_MAX} (可能異常) 的筆數: {int((vals > SUSPECT_MAX).sum())}")

    hour_mean = []
    for j in range(nh):
        v = arr[:, j, :][is_valid[:, j, :]]
        hour_mean.append(v.mean() if v.size else np.nan)
    print("\n各 hour 有效值平均:")
    print(pd.Series(hour_mean, index=hour_vals, name='mean').round(4).to_string())
else:
    print("沒有任何有效值！")

# ===== 7. 建議用來畫圖的 day / hour =====
print("\n" + "=" * 60)
print("7. 建議挑選")
print("=" * 60)
if valid_count.max() > 0:
    best_day_i = int(np.argmax(day_hours))               # 有資料 hour 最多的一天
    best_day = day_vals[best_day_i]
    best_hour_j = int(np.argmax(valid_count[best_day_i]))
    best_hour = hour_vals[best_hour_j]
    print(f"有資料的 hour 最多的一天: day={best_day} ({day_hours[best_day_i]} 個 hour)")
    print(f"該天涵蓋率最高的 hour   : hour={best_hour} "
          f"(涵蓋率 {coverage[best_day_i, best_hour_j]:.1f}%)")
    gi, gj = np.unravel_index(valid_count.argmax(), valid_count.shape)
    print(f"整月涵蓋率最高的 (day, hour): day={day_vals[gi]}, hour={hour_vals[gj]} "
          f"(涵蓋率 {coverage[gi, gj]:.1f}%)")

# ===== 8. 輸出檔案 =====
count_df.to_csv(os.path.join(output_dir, f'explore_{tag}_valid_count.csv'))
cov_df.to_csv(os.path.join(output_dir, f'explore_{tag}_coverage_pct.csv'))

summary = pd.DataFrame({
    'type': ['hour'] * nh + ['day'] * nd,
    'value': list(hour_vals) + list(day_vals),
    'valid_pixels_total': list(hour_total) + list(day_total),
    'n_slices_with_data': list(hour_days) + list(day_hours),
})
summary['has_data'] = summary['valid_pixels_total'] > 0
summary.to_csv(os.path.join(output_dir, f'explore_{tag}_summary.csv'), index=False)

# ===== 9. 涵蓋率熱圖：20 階離散色階 (0 ~ 100%，每階 5%)，每格標示百分比 =====


assert len(cov_colors) == len(cov_bounds) - 1, "顏色數必須等於區間數"
cov_cmap = mcolors.ListedColormap(cov_colors)
cov_norm = mcolors.BoundaryNorm(cov_bounds, cov_cmap.N)

cov_values = cov_df.values
fig, ax = plt.subplots(figsize=(max(8, 0.9 * nh), max(6, 0.45 * nd + 2)))
im = ax.imshow(cov_values, aspect='auto', cmap=cov_cmap, norm=cov_norm)

# 每一格標示百分比；依底色深淺自動選擇白字或黑字
for r in range(nd):
    for c in range(nh):
        val = cov_values[r, c]
        idx = int(np.clip(np.digitize(val, cov_bounds) - 1, 0, len(cov_colors) - 1))
        rgb = mcolors.to_rgb(cov_colors[idx])
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt = "0" if val == 0 else f"{val:.1f}"
        ax.text(c, r, txt, ha='center', va='center', fontsize=7,
                color='white' if lum < 0.5 else 'black')

ax.set_xticks(range(nh))
ax.set_xticklabels([f"{h:g}" for h in hour_vals])
ax.set_yticks(range(nd))
ax.set_yticklabels([f"{d:g}" for d in day_vals])
ax.set_xlabel('hour')
ax.set_ylabel('day')
ax.set_title(f'AOD valid pixel coverage (%) - {tag}')

ax.set_xticks(np.arange(-0.5, nh, 1), minor=True)
ax.set_yticks(np.arange(-0.5, nd, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.5)
ax.tick_params(which='minor', length=0)

cbar = fig.colorbar(im, ax=ax, ticks=cov_bounds, shrink=0.8)
cbar.set_label('coverage (%)')

fig.savefig(os.path.join(output_dir, f'explore_{tag}_coverage_heatmap.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\n已輸出 CSV 與熱圖至: {output_dir}")
ds.close()