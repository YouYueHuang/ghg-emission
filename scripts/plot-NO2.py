import os
from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from color_palette import boundaries, hex_colors

# ===== 路徑與選項 =====
input_dir = r"../data/raw/nc_datas/data_2023/GEMS_NO2"
output_dir = r"../results/figures/GEMS_NO2"
filename = "202301.nc"
os.makedirs(output_dir, exist_ok=True)
tag = os.path.splitext(filename)[0]

# 依探索結果：每天只有 hour 10~14 有資料，涵蓋率約 97%
DAY = 5      # 要畫小時圖 / 日平均圖的日期 (day 座標值)
HOUR = 12    # 要畫小時圖的 hour (hour 座標值)

# 缺值像素顏色（設為 None 則不特別標示）
MISSING_COLOR = '#FFFFFF'

# 視覺化尺度：固定使用 10^16 (molecules/cm²)，數值會除以 10^SCALE_EXPONENT
SCALE_EXPONENT = 16


# ===== 讀取並清理資料 =====
ds = xr.open_dataset(os.path.join(input_dir, filename))
ds = ds.set_coords(['longitude', 'latitude'])
no2 = ds['no2'].transpose('day', 'hour', ...)

no2 = no2.where(np.isfinite(no2))
no2 = no2.where(no2 > -1e30)      # fill value
no2 = no2.where(no2 > 0)          # 0 或負值視為缺值

# 以固定尺度縮放 (三張圖共用同一尺度，才能互相比較)
exponent = SCALE_EXPONENT
no2 = no2 / (10.0 ** exponent)
cbar_label = f'NO2 Concentration (x $10^{{{exponent}}}$ molecules/cm²)'
print(f"視覺化尺度: 10^{exponent}")

day_vals = no2['day'].values
hour_vals = no2['hour'].values
i = int(np.where(day_vals == DAY)[0][0])
j = int(np.where(hour_vals == HOUR)[0][0])

# ===== 三種資料 =====
# 1) 小時圖：指定的 day + hour
hourly = no2.isel(day=i, hour=j)

# 2) 日平均圖：指定那一天，有資料的各 hour 取平均
day_slice = no2.isel(day=i)
daily_mean = day_slice.mean(dim='hour', skipna=True)
hours_used = [int(h) for h in hour_vals
              if int(day_slice.sel(hour=h).notnull().sum()) > 0]

# 3) 月平均圖：先算每天的日平均，再對所有天取平均
monthly_mean = no2.mean(dim='hour', skipna=True).mean(dim='day', skipna=True)
n_days_used = int((no2.notnull().any(dim=[d for d in no2.dims if d not in ('day',)])).sum())

maps = [
    ('hourly', hourly, f'GEMS NO2 Hourly: {tag} Day {int(DAY)}, Hour {int(HOUR)}',
     f'no2_{tag}_hourly_d{int(DAY):02d}_h{int(HOUR):02d}.png'),
    ('daily_mean', daily_mean,
     f'GEMS NO2 Daily Mean: {tag} Day {int(DAY)} (hours {hours_used[0]}-{hours_used[-1]}, n={len(hours_used)})',
     f'no2_{tag}_daily_mean_d{int(DAY):02d}.png'),
    ('monthly_mean', monthly_mean,
     f'GEMS NO2 Monthly Mean: {tag} ({n_days_used} days)',
     f'no2_{tag}_monthly_mean.png'),
]

for name, da, _, _ in maps:
    print(f"{name:13s} 有效像素 {int(da.notnull().sum())}/{da.size}, "
          f"min {float(da.min()):.3f}, median {float(da.median()):.3f}, max {float(da.max()):.3f} "
          f"(x1e{exponent})")


assert len(hex_colors) == len(boundaries) - 1, "顏色數必須等於區間數"

cmap = mcolors.ListedColormap(hex_colors)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)
upper = boundaries[-1]
# extend='max'：超過上限的值以最深色顯示，色階條上端會出現延伸三角形
plot_kwargs = dict(cmap=cmap, norm=norm, extend='max')

# 檢查有多少有效像素超出色階上限，方便判斷色階範圍是否合適
for name, da, _, _ in maps:
    v = da.values[np.isfinite(da.values)]
    print(f"{name:13s} 超過 {upper} 的像素比例: {(v > upper).mean() * 100:.1f}%")

if MISSING_COLOR:
    cmap.set_bad(MISSING_COLOR)

# ===== 儲存圖片的輔助函式 =====
def safe_savefig(fig, path):
    """存檔；若原檔名寫入失敗，印出診斷資訊並改用加時間戳記的檔名重試。"""
    path = os.path.abspath(path)
    try:
        fig.savefig(path, dpi=300, bbox_inches='tight')
        return path
    except OSError as e:
        print(f"[警告] 無法寫入: {path!r}")
        print(f"       錯誤: {e}")
        print(f"       檔案已存在: {os.path.exists(path)}")
        root, ext = os.path.splitext(path)
        alt = f"{root}_{datetime.now():%H%M%S}{ext}"
        fig.savefig(alt, dpi=300, bbox_inches='tight')   # 仍失敗就直接丟出錯誤
        print(f"       已改存為: {alt}")
        return alt


# ===== 繪圖 =====
for name, da, title, outname in maps:
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 8))

    da.plot(
        ax=ax,
        x='longitude',
        y='latitude',
        transform=ccrs.PlateCarree(),
        cbar_kwargs={'label': cbar_label},
        **plot_kwargs
    )

    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    if MISSING_COLOR:
        ax.legend(handles=[Patch(facecolor=MISSING_COLOR, edgecolor='black',
                                 label='Missing value')], loc='lower right')
    plt.title(title, fontsize=13)

    out_path = safe_savefig(fig, os.path.join(output_dir, outname))
    plt.close(fig)
    print(f"已儲存: {out_path}")

ds.close()