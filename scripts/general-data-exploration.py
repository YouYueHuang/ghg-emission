"""
AOD (Aerosol Optical Depth) NetCDF 資料探索與涵蓋率分析腳本
- 支援自動修正無效 day 座標 (9.969e+36 fill values)
- 關閉自動時間解碼 (decode_times=False) 以避免 time 轉型報錯
- 統計無效值、缺值與涵蓋率
- 輸出 CSV 統計表與 20 階離散色階涵蓋率熱圖
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ===== 1. 路徑與參數設定 =====
input_dir = r"../data/raw/nc_datas/data_2023/AOD"
output_dir = r"../results/tables/AOD"
filename = "202301.nc"

VAR_NAME = None        # 例如 'GK2_D_AOD'（None 代表自動偵測）
DAY_DIM = None         # 例如 'day'
HOUR_DIM = None        # 例如 'time' 或 'hour'
LON_NAME = None        # 例如 'longitude'
LAT_NAME = None        # 例如 'latitude'
EXTRA_ISEL = {}        # 其他多餘維度（如波長）：{'wavelength': 0}

EXTRA_FILL_VALUES = (-999, -9999)   # 額外視為缺值的數值
SUSPECT_MAX = 5.0                   # AOD 異常高值門檻

# 色彩設定 (20 階離散色階 0~100%，每階 5%)
cov_bounds = np.linspace(0, 100, 21)
cov_colors = [
    "#440154", "#481b6d", "#462f7c", "#414487", "#39568c",
    "#31688e", "#2a788e", "#23888e", "#1f988b", "#22a884",
    "#35b779", "#54c568", "#7ad151", "#a2da37", "#c7e020",
    "#e3e418", "#fde725", "#fee838", "#feba37", "#f89540"
]

os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(input_dir, filename)
tag = os.path.splitext(filename)[0]


# ===== 2. 自動偵測輔助函數 =====
def pick_variable(ds, var_name=None):
    if var_name:
        return var_name
    cands = [n for n, v in ds.data_vars.items() if v.ndim >= 3]
    if not cands:
        raise SystemExit("找不到維度 >= 3 的資料變數，請手動指定 VAR_NAME。")
    named = [n for n in cands if 'aod' in n.lower()]
    pool = named or cands
    best = max(pool, key=lambda n: ds[n].size)
    if len(cands) > 1:
        print(f"[資訊] 候選變數 {cands}，自動選用 '{best}'")
    return best


def pick_dim(dims, key, given=None):
    if given:
        return given
    for d in dims:
        if d.lower() == key:
            return d
    for d in dims:
        if key in d.lower() or d.lower() in key:
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


# ===== 3. 讀取檔案與處理座標 =====
# 設定 decode_times=False 避開時間格式不相容問題
ds = xr.open_dataset(file_path, decode_times=False)

print("=" * 60)
print("1. 檔案基本結構")
print("=" * 60)
print(ds)
print("\n全域屬性:", dict(ds.attrs))

# 自動偵測變數與維度
var = pick_variable(ds, VAR_NAME)
da = ds[var]
if EXTRA_ISEL:
    da = da.isel(**EXTRA_ISEL)

lon_name = pick_coord(ds, ['longitude', 'lon', 'long', 'x'], LON_NAME)
lat_name = pick_coord(ds, ['latitude', 'lat', 'y'], LAT_NAME)
day_dim = pick_dim(da.dims, 'day', DAY_DIM)
hour_dim = pick_dim(da.dims, 'hour', HOUR_DIM) or pick_dim(da.dims, 'time', HOUR_DIM)

# 自動修復壞掉的 day 座標（處理解碼出 9.969e+36 的狀況）
if day_dim and day_dim in ds.coords:
    day_vals = ds[day_dim].values
    if np.issubdtype(day_vals.dtype, np.number) and (day_vals > 1e30).any():
        print("\n[警告] 偵測到 day 座標包含無效填補值 (9.969e+36)，已自動修正重設為 1 ~ N 日。")
        ds = ds.assign_coords({day_dim: np.arange(1, len(day_vals) + 1)})
        da = ds[var]  # 更新 DataArray

print(f"\n偵測結果: 變數='{var}', lon={lon_name}, lat={lat_name}, day={day_dim}, hour={hour_dim}")

if lon_name is None or lat_name is None:
    raise SystemExit("找不到經緯度變數，請手動設定 LON_NAME / LAT_NAME。")

# 維度補全（若缺少 day 或 hour 維度）
if day_dim is None:
    print("[資訊] 沒有 day 維度，自動擴充長度 1 的 day 維度")
    da = da.expand_dims(day=[1]); day_dim = 'day'
if hour_dim is None:
    print("[資訊] 沒有 hour/time 維度，自動擴充長度 1 的 hour 維度")
    da = da.expand_dims(hour=[1]); hour_dim = 'hour'


# ===== 4. 經緯度檢查 =====
lon = ds[lon_name].values
lat = ds[lat_name].values
print("\n" + "=" * 60)
print("2. 經緯度範圍")
print("=" * 60)
print(f"{lon_name}: shape={lon.shape}, 範圍 {np.nanmin(lon):.4f} ~ {np.nanmax(lon):.4f}")
print(f"{lat_name}: shape={lat.shape}, 範圍 {np.nanmin(lat):.4f} ~ {np.nanmax(lat):.4f}")


# ===== 5. 缺值與有效像素統計 =====
da = da.transpose(day_dim, hour_dim, ...)

# 直接從 da.sizes 取得正確的維度大小 (nd=31, nh=8)
nd = da.sizes[day_dim]
nh = da.sizes[hour_dim]

day_vals = ds[day_dim].values

# 因為 time 座標是 2D (day, time)，避免直接取長度錯誤，改用維度大小建立 1 ~ 8 小時序列
if ds[hour_dim].ndim > 1:
    hour_vals = np.arange(1, nh + 1)
else:
    hour_vals = ds[hour_dim].values

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
print("3. 缺值分類統計（全月份總像素）")
print("=" * 60)
total = arr.size
print(f"總像素數      : {total}")
print(f"NaN           : {is_nan.sum()} ({is_nan.mean()*100:.2f}%)")
print(f"Fill Value    : {is_fill.sum()} ({is_fill.mean()*100:.2f}%)")
print(f"0 或負值      : {is_nonpos.sum()} ({is_nonpos.mean()*100:.2f}%)")
print(f"有效值 (>0)   : {is_valid.sum()} ({is_valid.mean()*100:.2f}%)")


# ===== 6. 整理 (day, hour) 有效涵蓋率 =====
valid_count = is_valid.sum(axis=2)
coverage = valid_count / npix * 100

count_df = pd.DataFrame(valid_count, index=day_vals, columns=hour_vals)
count_df.index.name = 'day'
count_df.columns.name = 'hour'

cov_df = pd.DataFrame(coverage, index=day_vals, columns=hour_vals).round(1)
cov_df.index.name = 'day'
cov_df.columns.name = 'hour'

print("\n" + "=" * 60)
print("4. 各 (day, hour) 有效像素涵蓋率 % （列=day, 欄=hour）")
print("=" * 60)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
print(cov_df)


# ===== 7. 數值分佈與異常值分析 =====
print("\n" + "=" * 60)
print("5. AOD 有效值統計")
print("=" * 60)
vals = arr[is_valid]
if vals.size > 0:
    q = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
    print(f"有效筆數: {vals.size}")
    print(f"極值: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")
    print("百分位數 (1/5/25/50/75/95/99%): " + ", ".join(f"{x:.4f}" for x in q))
    print(f"超過 1.0 的比例: {(vals > 1.0).mean() * 100:.2f}%")
    print(f"超過 {SUSPECT_MAX} (極端值) 筆數: {int((vals > SUSPECT_MAX).sum())}")
else:
    print("警告：全檔案無任何有效資料！")


# ===== 8. 輸出 CSV 與繪製熱圖 =====
count_df.to_csv(os.path.join(output_dir, f'explore_{tag}_valid_count.csv'))
cov_df.to_csv(os.path.join(output_dir, f'explore_{tag}_coverage_pct.csv'))

# 熱圖繪製
cov_cmap = mcolors.ListedColormap(cov_colors)
cov_norm = mcolors.BoundaryNorm(cov_bounds, cov_cmap.N)

cov_values = cov_df.values
fig, ax = plt.subplots(figsize=(max(8, 0.9 * nh), max(6, 0.45 * nd + 2)))
im = ax.imshow(cov_values, aspect='auto', cmap=cov_cmap, norm=cov_norm)

# 標註文字（自動依顏色調配黑/白字）
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
ax.set_xticklabels([f"{h}" for h in hour_vals])
ax.set_yticks(range(nd))
ax.set_yticklabels([f"{d}" for d in day_vals])
ax.set_xlabel('hour / time')
ax.set_ylabel('day')
ax.set_title(f'AOD Valid Pixel Coverage (%) - {tag}')

cbar = fig.colorbar(im, ax=ax, ticks=cov_bounds, shrink=0.8)
cbar.set_label('Coverage (%)')

fig.savefig(os.path.join(output_dir, f'explore_{tag}_coverage_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

ds.close()
print(f"\n[成功] 探索完成，結果已存至: {output_dir}")