import os
import glob
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# 1. 使用 glob 自動搜尋 GEMS_NO2 資料夾底下的所有 .nc 檔案
nc_files = sorted(glob.glob(os.path.join('../GEMS_NO2', '*.nc')))

if not nc_files:
    print("找不到任何 .nc 檔案，請檢查 GEMS_NO2 資料夾路徑是否正確！")
else:
    print(f"找到以下 .nc 檔案，共 {len(nc_files)} 個:")
    for f in nc_files:
        print(f" - {f}")

# 2. 讀取 .shp 邊界檔案（選填，若有則疊加）
shp_path = ' ../../ground-info/COUNTY_MOI_1140318.shp'  # 請換成您的 shp 檔名
if os.path.exists(shp_path):
    gdf = gpd.read_file(shp_path)
    has_shp = True
else:
    print(f"提示: 找不到 {shp_path}，將僅繪製網格與海岸線。")
    has_shp = False

# 3. 迴圈遍歷所有找到的 .nc 檔案
for filepath in nc_files:
    # 取得檔名（不含路徑與副檔名），例如從 "GEMS_NO2/202301.nc" 抓出 "202301"
    filename_with_ext = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(filename_with_ext)[0]
    
    print(f"\n正在處理檔案: {filepath}")
    
    # 讀取 NetCDF 檔案
    ds = xr.open_dataset(filepath)
    
    # 檢查並設定經緯度座標
    if 'longitude' in ds.data_vars and 'latitude' in ds.data_vars:
        ds = ds.set_coords(['longitude', 'latitude'])
        
    # 抓取變數（具備防呆機制，自動找 no2 或第一個變數）
    try:
        target_data = ds['no2'].isel(day=0, hour=0)
    except KeyError:
        var_names = [v for v in ds.data_vars if v not in ['longitude', 'latitude']]
        target_data = ds[var_names[0]].isel(0)

    # 建立畫布
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 8))
    
    # 繪製網格圖
    plot_data = target_data.plot(
        ax=ax, 
        x='longitude', 
        y='latitude', 
        transform=ccrs.PlateCarree(), 
        cmap='YlOrRd',
        cbar_kwargs={'label': 'NO2 Concentration'}
    )
    
    # 疊加 Shapefile 邊界
    if has_shp:
        gdf.to_crs(epsg=4326).plot(
            ax=ax, 
            transform=ccrs.PlateCarree(), 
            facecolor='none', 
            edgecolor='blue', 
            linewidth=1.2
        )
        
    ax.coastlines()
    plt.title(f'GEMS NO2 Concentration: {filename_no_ext}', fontsize=14)
    
    # 儲存圖片（檔名會自動對應原檔名，例如 no2_202301.png）
    output_image = f'no2_{filename_no_ext}.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已成功產出並儲存: {output_image}")

print("\n所有檔案的視覺化處理完畢！")
