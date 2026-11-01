"""
NetCDF Variables Daily Average Spatial Visualization with Shapefile Overlay
--------------------------------------------------------------------------
此腳本計算 NetCDF 變數的日平均，略過 Hr, Lat, Lon, MM，
並將結果與台灣 Shapefile 疊加，使用 color_palette.py 的標準色階儲存圖片。

Requirements:
    pip install xarray numpy matplotlib geopandas shapely


python Merged_maps.py ../data/raw/nc_datas/data_2023/Merged/2023/202303.nc ../data/raw/shape/COUNTY_MOI_1140318.shp
"""

import os
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import geopandas as gpd

# 從外部的 color_palette.py 引入標準色塊（hex_colors）
from color_palette import hex_colors

def visualize_daily_averages(nc_path, shp_path):
    print("=" * 75)
    print(f"開始處理 NetCDF 檔案: {nc_path}")
    print(f"使用 Shapefile 邊界: {shp_path}")
    print("=" * 75)
    
    # 開啟 NetCDF 檔案
    ds = xr.open_dataset(nc_path)
    
    # 讀取 Shapefile
    gdf_shp = gpd.read_file(shp_path)
    
    # 取得網格的經緯度範圍，用來強制鎖定畫布大小
    lon = ds.lon.values
    lat = ds.lat.values
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    
    # 建立 colormap
    cmap = ListedColormap(hex_colors)
    
    base_output_dir = "../results/figures/Merged"
    
    # 不需要繪製的座標/輔助變數清單
    skip_vars = {'Hr', 'Lat', 'Lon', 'MM', 'hr', 'lat', 'lon', 'mm'}
    
    # 逐一走訪所有資料變數
    for var_name in ds.data_vars:
        if var_name in skip_vars:
            print(f"⏩ 略過座標/輔助變數: [{var_name}]")
            continue
            
        da = ds[var_name]
        
        # 檢查是否包含 day 與 hour 維度
        if 'day' not in da.dims or 'hour' not in da.dims:
            print(f"⚠️ 變數 [{var_name}] 缺少 day 或 hour 維度，跳過繪圖。")
            continue
            
        print(f"正在分析變數分佈並處理: {var_name} ...")
        
        # 計算每日平均 (跨 hour 維度取平均)
        da_daily = da.mean(dim='hour')
        
        # 收集整個月的有效數據
        all_vals = da_daily.values.flatten()
        valid_all = all_vals[~np.isnan(all_vals)]
        
        if len(valid_all) == 0:
            print(f"⚠️ 變數 [{var_name}] 無有效數值，跳過。")
            continue
            
        # 判斷是否為 U 或 V 風場變數（不分大小寫）
        is_wind_component = var_name.lower() in ['u', 'v']
        
        if is_wind_component:
            # 針對風場 U, V：以 0 為中心對稱處理（取絕對值的 98% 分位數作為上下限）
            max_abs = float(np.percentile(np.abs(valid_all), 98))
            if max_abs == 0:
                max_abs = float(np.max(np.abs(valid_all)))
            if max_abs == 0:
                max_abs = 1e-5
                
            v_min = -max_abs
            v_max = max_abs
            print(f"   🌬️ 偵測到風場變數 [{var_name}]，已設定對稱色階範圍: [{v_min:.4f} ~ {v_max:.4f}]（0 置中）")
        else:
            # 一般變數：使用 2% 與 98% 百分位數過濾極端值
            v_min = float(np.percentile(valid_all, 2))
            v_max = float(np.percentile(valid_all, 98))
            
            if v_max <= v_min:
                v_min = float(np.min(valid_all))
                v_max = float(np.max(valid_all))
            if v_max <= v_min:
                v_max = v_min + 1e-5
                
        # 根據決定的範圍，切分出對應 color_palette 階數的 boundaries
        boundaries = np.linspace(v_min, v_max, len(hex_colors) + 1)
        norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
        
        # 建立專屬變數資料夾
        var_output_dir = os.path.join(base_output_dir, var_name)
        os.makedirs(var_output_dir, exist_ok=True)
        
        # 迴圈繪製每一天的日平均地圖
        for day_val in da_daily.day.values:
            slice_data = da_daily.sel(day=day_val).values
            
            # 建立畫布
            fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
            
            # 1. 繪製原始數值
            im = ax.pcolormesh(lon, lat, slice_data, cmap=cmap, norm=norm, shading='auto', zorder=1)
            
            # 2. 疊加台灣 Shapefile 邊界
            gdf_shp.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6, zorder=2)
            
            # 3. 強制鎖定畫布範圍為網格經緯度邊界
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            
            # 設定色條與標籤
            cbar = plt.colorbar(im, ax=ax, extend='both', fraction=0.035, pad=0.04)
            cbar.set_label(f"{var_name} (Raw Value)", fontsize=10)
            
            ax.set_title(f"Daily Average: {var_name} (Day {int(day_val)})", fontsize=12)
            ax.set_xlabel("Longitude (°E)", fontsize=10)
            ax.set_ylabel("Latitude (°N)", fontsize=10)
            
            # 儲存圖片
            output_filename = f"day_{int(day_val):02d}.png"
            output_filepath = os.path.join(var_output_dir, output_filename)
            plt.savefig(output_filepath, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
        print(f"✅ 變數 [{var_name}] 完成！色階範圍: [{v_min:.4f} ~ {v_max:.4f}] 已存至: {var_output_dir}")

    print("=" * 75)
    print("🎉 所有變數（包含 U/V 0 置中對稱色階）視覺化作業全部完成！")
    print("=" * 75)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        nc_file = sys.argv[1]
        shp_file = sys.argv[2]
        visualize_daily_averages(nc_file, shp_file)
    else:
        print("使用方式: python plot_daily_averages.py <path_to_nc_file.nc> <path_to_shapefile.shp>")