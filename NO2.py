import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# 1. 讀取 .nc 檔案
ds = xr.open_dataset('../GEMS_NO2/202306.nc') # 請換成您的檔名

# 2. 如果 lat 和 lon 是資料變數而非座標，可以先將其設為 xarray 的座標準備畫圖
# (看您的結構提示，lon/lat 是 data_vars，我們將其指定進 dataset 中)
ds = ds.set_coords(['longitude', 'latitude'])

# 3. 挑選要畫的變數：明確指定 'no2'
# 由於 no2 維度是 (hour, day, lat, lon)，我們取第 0 天、第 0 小時的 2D 平面資料
target_data = ds['no2'].isel(day=0, hour=0)

print(f"正在繪製變數: no2 (Day 1, Hour 1)")

# 4. 繪製 2D 地圖
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))

# 使用 pcolormesh 或 xarray 內建 plot 畫 2D 網格
# 這裡明確指定 x='longitude', y='latitude'
plot_data = target_data.plot(
    ax=ax, 
    x='longitude', 
    y='latitude', 
    transform=ccrs.PlateCarree(), 
    cmap='viridis',
    cbar_kwargs={'label': 'NO2 Concentration'}
)

ax.coastlines()
plt.title('NO2 Spatial Distribution (Day 1, Hour 1)')

# 5. 儲存圖片
plt.savefig('no2_exploration.png', dpi=300, bbox_inches='tight')
plt.close()
print("已成功儲存圖片至: no2_exploration.png")
