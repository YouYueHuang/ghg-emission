"""
繪製 AOD 日平均 (每 8 天一張圖，8 個子圖) 與月平均空間分佈圖 (Latitude vs Longitude)
- 沿用 AOD.py 的讀檔與 day 座標修復邏輯
- 色階以同資料夾的 color_palette.py 為準 (0 ~ 1，每階 0.05，共 20 階)；超過 1.0 的值以最後一色表示
- 疊上縣市 shapefile 邊界
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 確保能匯入與本檔同資料夾的 color_palette.py（不論從哪個工作目錄執行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_palette import boundaries as cov_bounds
from color_palette import hex_colors as cov_colors

# ===== 設定中文字型與負號正常顯示 =====
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows 使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 路徑與參數設定 =====
input_dir = r"../data/raw/nc_datas/data_2023/AOD"
output_dir = r"../results/tables/AOD"
filename = "202303.nc"
var_name = 'GK2_D_AOD'

shp_path = r"../data/raw/nc_datas/ground-info/COUNTY_MOI_1140318.shp"
SHP_EDGE_COLOR = 'black'
SHP_LINEWIDTH = 0.6

os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(input_dir, filename)
tag = os.path.splitext(filename)[0]

# ===== 讀取縣市 shapefile，統一轉成經緯度 (EPSG:4326) 才能和 AOD 的 lon/lat 對齊 =====
counties = gpd.read_file(shp_path)
if counties.crs is None:
    print("[警告] shapefile 沒有 CRS，假設為 EPSG:3824 (TWD97 經緯度)；若位置不對請修改")
    counties = counties.set_crs(epsg=3824)
counties = counties.to_crs(epsg=4326)
print(f"shapefile: {len(counties)} 個圖徵，座標系已轉為 EPSG:4326")


def add_counties(ax, xlim, ylim, linewidth=SHP_LINEWIDTH):
    """把縣市邊界疊在 ax 上（只畫外框、不填色），並把範圍鎖回 AOD 資料範圍。"""
    counties.boundary.plot(ax=ax, color=SHP_EDGE_COLOR, linewidth=linewidth, zorder=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# ===== 色階：完全依 color_palette.py（離散色階）=====
assert len(cov_colors) == len(cov_bounds) - 1, "顏色數必須等於區間數"
cmap = mcolors.ListedColormap(cov_colors)
cmap.set_over(cov_colors[-1])                       # > 最大邊界 (1.0) 的值用最後一色
norm = mcolors.BoundaryNorm(cov_bounds, cmap.N)

PER_FIG = 8            # 每張圖放幾天的日平均（8 天 = 一張圖，2 列 x 4 欄）
FIG_COLS = 4
MIN_OBS = 1            # 月平均時，像素至少要有幾筆有效觀測才計算（避免只靠 1 筆的像素）
# 月平均方式：
#   'all_obs'    = 該像素整月所有有效觀測 (day x time) 直接平均（每筆觀測權重相同）
#   'daily_mean' = 先算每日平均，再對有資料的天數平均（每天權重相同）
MONTHLY_METHOD = 'all_obs'

# ===== 2. 讀取檔案與修復座標 =====
ds = xr.open_dataset(file_path, decode_times=False)

day_dim = 'day'
if day_dim in ds.coords:
    day_vals = ds[day_dim].values
    if np.issubdtype(day_vals.dtype, np.number) and (day_vals > 1e30).any():
        ds = ds.assign_coords({day_dim: np.arange(1, len(day_vals) + 1)})

da = ds[var_name]
# time 是 (day, time) 的 2D 字串座標，對平均沒有用，先移除避免干擾
da = da.drop_vars('time', errors='ignore')

# 只保留有效值：NaN、fill value、<=0 的都視為缺值
da = da.where(np.isfinite(da) & (da > 0) & (da < 1e30))

days = da['day'].values
nd = len(days)

# AOD 資料的經緯度範圍，疊 shapefile 後用來限制顯示範圍（排除金門、馬祖等範圍外的島嶼）
lon_vals = da['longitude'].values
lat_vals = da['latitude'].values
XLIM = (float(lon_vals.min()), float(lon_vals.max()))
YLIM = (float(lat_vals.min()), float(lat_vals.max()))

# ===== 3. 計算日平均與月平均 =====
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)

    # 日平均：對每一天的 8 個時次平均（略過 NaN）→ (day, lat, lon)
    daily_mean = da.mean(dim='time', skipna=True)

    # 各像素整月有效觀測筆數 (day x time)
    n_obs = da.notnull().sum(dim=['day', 'time'])

    if MONTHLY_METHOD == 'all_obs':
        monthly_mean = da.mean(dim=['day', 'time'], skipna=True)
    elif MONTHLY_METHOD == 'daily_mean':
        monthly_mean = daily_mean.mean(dim='day', skipna=True)
    else:
        raise ValueError("MONTHLY_METHOD 只能是 'all_obs' 或 'daily_mean'")

monthly_mean = monthly_mean.where(n_obs >= MIN_OBS)

# 各天統計（全區域平均與有效像素涵蓋率）
npix = da.sizes['latitude'] * da.sizes['longitude']
day_cov = (daily_mean.notnull().sum(dim=['latitude', 'longitude']) / npix * 100).values
day_domain_mean = daily_mean.mean(dim=['latitude', 'longitude'], skipna=True).values

stat_df = pd.DataFrame({
    'day': days,
    'domain_mean_AOD': np.round(day_domain_mean, 4),
    'coverage_pct': np.round(day_cov, 1),
})
stat_df.to_csv(os.path.join(output_dir, f'AOD_{tag}_daily_stats.csv'), index=False)

print(f"月平均方式: {MONTHLY_METHOD}，MIN_OBS={MIN_OBS}")
print(f"整月有資料的天數: {int((day_cov > 0).sum())}/{nd}")
print(f"月平均全區域平均: {float(monthly_mean.mean(skipna=True)):.4f}")
print(stat_df.to_string(index=False))

# ===== 4. 色階說明 =====
n_over = int((daily_mean.values > cov_bounds[-1]).sum())
print(f"\n色階: {cov_bounds[0]} ~ {cov_bounds[-1]}（{len(cov_colors)} 階）；"
      f"日平均中超過 {cov_bounds[-1]} 的像素數: {n_over}")

# ===== 5. 日平均圖：每 PER_FIG 天輸出一張圖，共用 color_palette 色階 =====
FIG_ROWS = int(np.ceil(PER_FIG / FIG_COLS))
daily_pngs = []

for part, start in enumerate(range(0, nd, PER_FIG), start=1):
    idxs = list(range(start, min(start + PER_FIG, nd)))
    fig, axes = plt.subplots(FIG_ROWS, FIG_COLS, figsize=(FIG_COLS * 3.2 + 1.5, FIG_ROWS * 4.4 + 1.2),
                             sharex=True, sharey=True, squeeze=False)
    im = None
    for k in range(FIG_ROWS * FIG_COLS):
        ax = axes[k // FIG_COLS, k % FIG_COLS]
        if k >= len(idxs):
            ax.axis('off')
            continue
        i = idxs[k]
        im = daily_mean.isel(day=i).plot(
            ax=ax, cmap=cmap, norm=norm, add_colorbar=False, add_labels=False,
        )
        add_counties(ax, XLIM, YLIM, linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Day {int(days[i])}  ({day_cov[i]:.0f}%)", fontsize=10)
        ax.tick_params(labelsize=8)

    d0, d1 = int(days[idxs[0]]), int(days[idxs[-1]])
    fig.suptitle(f'AOD 日平均空間分佈 - {tag}  Day {d0} ~ {d1}（括號內為有效像素涵蓋率）', fontsize=14)
    fig.supxlabel('Longitude', fontsize=10)
    fig.supylabel('Latitude', fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 0.91, 0.96))

    cax = fig.add_axes((0.93, 0.15, 0.015, 0.7))
    cbar = fig.colorbar(im, cax=cax, ticks=cov_bounds, extend='max')
    cbar.set_label('AOD (無單位)')

    png = os.path.join(output_dir, f'AOD_{tag}_daily_mean_part{part}_day{d0:02d}-{d1:02d}.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    daily_pngs.append(png)

# ===== 6. 月平均圖 + 有效觀測筆數圖 =====
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

monthly_mean.plot(
    ax=axes[0], cmap=cmap, norm=norm, extend='max',
    cbar_kwargs={'label': 'AOD (無單位)', 'ticks': cov_bounds},
)
axes[0].set_title(f'AOD 月平均 - {tag}\n（{MONTHLY_METHOD}，至少 {MIN_OBS} 筆觀測）', fontsize=12)

n_obs.where(n_obs > 0).plot(
    ax=axes[1], cmap='magma_r',
    cbar_kwargs={'label': '有效觀測筆數 (day x time)'},
)
axes[1].set_title(f'各像素整月有效觀測筆數 - {tag}', fontsize=12)

for ax in axes:
    add_counties(ax, XLIM, YLIM)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)

fig.tight_layout()
monthly_png = os.path.join(output_dir, f'AOD_{tag}_monthly_mean_map.png')
fig.savefig(monthly_png, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print("\n已輸出:")
for p_ in daily_pngs:
    print(f"  {p_}")
print(f"  {monthly_png}\n  AOD_{tag}_daily_stats.csv（位於 {output_dir}）")
ds.close()