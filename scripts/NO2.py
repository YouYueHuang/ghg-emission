import os
import glob
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from color_palette import boundaries, hex_colors

# 1. 定義路徑並確保輸出資料夾存在
input_dir = r"../data/raw/nc_datas/data_2023/GEMS_NO2"
output_dir = r"../results/figures/GEMS_NO2"
os.makedirs(output_dir, exist_ok=True)

# 選擇要畫的資料: 'best_slice' = 每個檔案中有效像素最多的 (day, hour)
#                'monthly_mean' = 整月平均
MODE = 'best_slice'

# 缺值像素的顏色（設為 None 則不特別標示）
MISSING_COLOR = '#ffffff'

# 視覺化尺度：固定使用 10^16 (molecules/cm²)，數值會除以 10^SCALE_EXPONENT
SCALE_EXPONENT = 16

# 2. 搜尋所有 .nc 檔案
nc_files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
if not nc_files:
    raise SystemExit(f"找不到任何 .nc 檔案於路徑: {input_dir}")
print(f"找到 {len(nc_files)} 個 .nc 檔案，開始進行批次處理...")

assert len(hex_colors) == len(boundaries) - 1, "顏色數必須等於區間數"

cmap = mcolors.ListedColormap(hex_colors)
if MISSING_COLOR:
    cmap.set_bad(MISSING_COLOR)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)
upper = boundaries[-1]


# 4. 批次處理
for filepath in nc_files:
    name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n正在處理檔案: {os.path.basename(filepath)}")

    ds = xr.open_dataset(filepath)
    ds = ds.set_coords(['longitude', 'latitude'])
    no2 = ds['no2']

    # 清理無效值：NaN、fill value、<=0 都視為缺值
    no2 = no2.where(np.isfinite(no2))
    no2 = no2.where(no2 > -1e30)
    no2 = no2.where(no2 > 0)

    # 每個 (day, hour) 的有效像素數
    other_dims = [d for d in no2.dims if d not in ('day', 'hour')]
    valid = no2.notnull().sum(dim=other_dims).transpose('day', 'hour')
    v = valid.values
    print(f"  有資料的 (day, hour) 數量: {int((v > 0).sum())} / {v.size}")

    if v.max() == 0:
        print("  [略過] 此檔案所有時段都沒有有效資料")
        ds.close()
        continue

    # 以固定尺度縮放
    exponent = SCALE_EXPONENT
    no2 = no2 / (10.0 ** exponent)
    cbar_label = f'NO2 Concentration (x $10^{{{exponent}}}$ molecules/cm²)'

    if MODE == 'monthly_mean':
        target = no2.mean(dim=['day', 'hour'], skipna=True)
        title = f'GEMS NO2 Monthly Mean: {name}'
        tag = 'mean'
    else:
        i, j = np.unravel_index(v.argmax(), v.shape)
        d_val = no2['day'].values[i]
        h_val = no2['hour'].values[j]
        target = no2.isel(day=i, hour=j)
        title = f'GEMS NO2: {name} (Day {d_val}, Hour {h_val})'
        tag = f'd{d_val}_h{h_val}'
        print(f"  選用 Day {d_val}, Hour {h_val}，有效像素 {int(v[i, j])}")

    arr = target.values[np.isfinite(target.values)]
    print(f"  數值範圍 (x1e{exponent}): {arr.min():.3f} ~ {arr.max():.3f}, "
          f"中位數 {np.median(arr):.3f}, 超過 {upper} 的像素比例 {(arr > upper).mean() * 100:.1f}%")

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))

    target.plot(
        ax=ax,
        x='longitude',
        y='latitude',
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        extend='max',   # 超過上限的值以最深色顯示，色階條上端出現延伸三角形
        cbar_kwargs={'label': cbar_label}
    )

    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    if MISSING_COLOR:
        ax.legend(handles=[Patch(facecolor=MISSING_COLOR, edgecolor='black',
                                 label='Missing value')], loc='lower right')
    plt.title(title, fontsize=14)

    output_path = os.path.join(output_dir, f'no2_{name}_{tag}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    ds.close()
    print(f"  已儲存圖片至: {output_path}")

print("\n所有檔案視覺化處理完畢！")