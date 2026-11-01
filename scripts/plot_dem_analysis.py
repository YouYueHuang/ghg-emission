"""
GeoTIFF (DEM) Visualization & 500m Mask Script
----------------------------------------------
此腳本讀取 DEM 檔與 Shapefile，並繪製兩張圖：
1. 套用 color_palette_2.py 標準色階的原始高程圖。
2. 超過 500 公尺為黑色、其餘為白色的二值化遮罩圖。
兩者皆會疊加鮮紅色的台灣 Shapefile 邊界。

Requirements:
    pip install rasterio matplotlib numpy geopandas

python plot_dem_analysis.py ../data/raw/nc_datas/ground-info/dem_20m.tif ../data/raw/nc_datas/ground-info/COUNTY_MOI_1140318.shp
"""

"""
GeoTIFF (DEM) Visualization & 500m Mask Script (Fixed CRS & Extent)
------------------------------------------------------------------
此腳本修正了坐標系不匹配的問題：
1. 自動讀取 DEM 的 CRS（坐標參考系統）。
2. 將 Shapefile 自動 `.to_crs()` 轉換至與 DEM 相同的坐標系。
3. 依據 DEM 的實際像素範圍（extent）精準對齊繪圖。

Requirements:
    pip install rasterio matplotlib numpy geopandas
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import plotting_extent
from matplotlib.colors import BoundaryNorm, ListedColormap
import geopandas as gpd

# 從 color_palette_2.py 引入自訂色階與邊界
from color_palette import hex_colors

def process_dem_and_mask(tif_path, shp_path):
    print("=" * 75)
    print(f"正在讀取 DEM 檔案: {tif_path}")
    print(f"正在讀取 Shapefile: {shp_path}")
    print("=" * 75)
    
    # 建立指定的輸出資料夾
    output_dir = "../results/figures/Digital_Elevation_Model"
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(tif_path) as src:
        band_data = src.read(1).astype(float)
        dem_crs = src.crs  # 取得 DEM 的坐標系（例如 EPSG:3826 TM2）
        
        # 處理無效值（NoData）
        nodata_val = src.nodatavals[0]
        if nodata_val is not None:
            band_data[band_data == nodata_val] = np.nan
            
        # 取得 DEM 在其坐標系下的實體範圍 (Left, Right, Bottom, Top)
        extent = plotting_extent(src)
        
        # 讀取 Shapefile 並自動轉換坐標系以符合 DEM
        gdf_shp = gpd.read_file(shp_path)
        if gdf_shp.crs != dem_crs:
            print(f"🔄 偵測到 Shapefile 坐標系 ({gdf_shp.crs}) 與 DEM ({dem_crs}) 不符，正在進行自動投影轉換...")
            gdf_shp = gdf_shp.to_crs(dem_crs)
        
        # ==========================================
        # 圖表一：原始高度圖 (套用 color_palette_2.py 色階)
        # ==========================================
        print("正在繪製圖表一：原始高程圖...")
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
        
        valid_mask = ~np.isnan(band_data)
        v_min = float(np.nanmin(band_data[valid_mask]))
        v_max = float(np.nanmax(band_data[valid_mask]))
        
        scaled_boundaries = np.linspace(v_min, v_max, len(hex_colors) + 1)
        cmap = ListedColormap(hex_colors)
        norm = BoundaryNorm(scaled_boundaries, ncolors=cmap.N, clip=True)
        
        # 繪製 DEM 影像
        im = ax.imshow(band_data, cmap=cmap, norm=norm, extent=extent, origin='upper', zorder=1)
        
        # 疊加轉換後的 Shapefile（鮮紅色外框）
        gdf_shp.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.8, zorder=2)
        
        # 鎖定範圍為 DEM 自身的邊界，確保畫面完美包覆 DEM 與地圖
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Elevation (m)", fontsize=10)
        ax.set_title("DEM Original Elevation Map", fontsize=13)
        ax.set_xlabel("Easting (Projected Coordinate)", fontsize=10)
        ax.set_ylabel("Northing (Projected Coordinate)", fontsize=10)
        
        output_fig1 = os.path.join(output_dir, "dem_original_map.png")
        plt.savefig(output_fig1, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"✅ 原始高程圖已成功儲存至: {output_fig1}")
        
        # ==========================================
        # 圖表二：500公尺遮罩圖 (>500m 黑色，其餘白色)
        # ==========================================
        print("正在繪製圖表二：500m 遮罩圖...")
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
        
        mask_img = np.ones((*band_data.shape, 4))
        over_500 = (band_data > 500) & valid_mask
        mask_img[over_500] = [0, 0, 0, 1]
        
        invalid = ~valid_mask
        mask_img[invalid, 3] = 0
        
        # 繪製遮罩影像
        ax.imshow(mask_img, extent=extent, origin='upper', zorder=1)
        
        # 疊加轉換後的 Shapefile（鮮紅色外框）
        gdf_shp.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.8, zorder=2)
        
        # 同樣鎖定範圍為 DEM 邊界
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        ax.set_title("DEM Elevation Mask (> 500m: Black, <= 500m: White)", fontsize=13)
        ax.set_xlabel("Easting (Projected Coordinate)", fontsize=10)
        ax.set_ylabel("Northing (Projected Coordinate)", fontsize=10)
        
        output_fig2 = os.path.join(output_dir, "dem_mask_500m.png")
        plt.savefig(output_fig2, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"✅ 500m 遮罩圖已成功儲存至: {output_fig2}")

    print("=" * 75)
    print(f"🎉 坐標系對齊與視覺化修正完成！圖檔已輸出至: {output_dir}")
    print("=" * 75)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        tif_file = sys.argv[1]
        shp_file = sys.argv[2]
        process_dem_and_mask(tif_file, shp_file)
    else:
        print("使用方式: python plot_dem_analysis.py <path_to_dem.tif> <path_to_shapefile.shp>")