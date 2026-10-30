"""
繪製 AOD 特定天、特定時間的空間分佈圖 (Latitude vs Longitude)
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ===== 1. 路徑與目標設定 =====
input_dir = r"../data/raw/nc_datas/data_2023/AOD"
filename = "202301.nc"
file_path = os.path.join(input_dir, filename)

# 【自訂參數】：你想畫哪一天、哪一個小時？
# 註：索引從 0 開始算。例如 target_day_idx = 0 代表第 1 天；target_hour_idx = 0 代表第 1 個時段
target_day_idx = 30    # 第 1 天 (範圍 0 ~ 30)
target_hour_idx = 5   # 第 1 個時段 (範圍 0 ~ 7)

# ===== 2. 讀取檔案與修復座標 =====
# 設定 decode_times=False 避開時間格式錯誤
ds = xr.open_dataset(file_path, decode_times=False)

# 自動修復壞掉的 day 座標 (9.969e+36)
day_dim = 'day'
if day_dim in ds.coords:
    day_vals = ds[day_dim].values
    if np.issubdtype(day_vals.dtype, np.number) and (day_vals > 1e30).any():
        ds = ds.assign_coords({day_dim: np.arange(1, len(day_vals) + 1)})

var_name = 'GK2_D_AOD'  # 你的 AOD 變數名稱
da = ds[var_name]

# ===== 3. 切片 (Slice) 選取特定時間與空間 =====
# 使用 .isel() 透過矩陣索引來抓取特定 day 和 time
da_slice = da.isel(day=target_day_idx, time=target_hour_idx)

# 取得實際顯示的日期與時間代號
actual_day = ds['day'].values[target_day_idx]
time_coords = ds['time'].values
# 如果 time 是 2D 矩陣 (day, time)，抓出對應的值
current_time_label = time_coords[target_day_idx, target_hour_idx] if time_coords.ndim == 2 else time_coords[target_hour_idx]

# ===== 4. 繪製地圖 =====
plt.figure(figsize=(9, 7))

# 使用 xarray 內建繪圖，會自動對齊經緯度 (longitude, latitude)
# cmap 可以更換色階，例如 'jet', 'viridis', 'plasma', 'Spectral_r' 等
im = da_slice.plot(
    cmap='viridis', 
    cbar_kwargs={'label': 'AOD Concentration (無單位)'},
    robust=True  # 自動排除極端離散值，讓色彩對比更明顯
)

plt.title(f'AOD Spatial Distribution\nDay: {actual_day}, Time Index: {target_hour_idx} (Label: {current_time_label})', fontsize=12)
plt.xlabel('Longitude', fontsize=10)
plt.ylabel('Latitude', fontsize=10)

# 若需要存成圖片，可以取消下面這行註解
# plt.savefig(f'../results/tables/AOD/AOD_map_day{actual_day}_hour{target_hour_idx}.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# 關閉檔案
ds.close()