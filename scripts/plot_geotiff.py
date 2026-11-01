"""
GeoTIFF Spatial Visualization Script
------------------------------------
This script reads a GeoTIFF raster file and plots its spatial distribution
using matplotlib and rasterio.

Requirements:
    pip install rasterio matplotlib numpy
"""

import sys
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

def plot_tif(file_path):
    print(f"Opening and plotting: {file_path}")
    
    with rasterio.open(file_path) as src:
        # 讀取第一波段
        band_data = src.read(1)
        
        # 處理 Nodata 值（將極端負值或 Nodata 轉為 NaN 避免影響圖面渲染）
        nodata_val = src.nodatavals[0]
        if nodata_val is not None:
            band_data = band_data.astype(float)
            band_data[band_data == nodata_val] = float('nan')
        
        # 建立畫布
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        
        # 繪製地圖影像 (若為 DEM 建議使用 'terrain' 或 'gist_earth' 色彩；一般影像可使用 'gray')
        # extent 可以讓圖軸顯示真實的投影座標範圍
        im = show(src, ax=ax, cmap='terrain', title="GeoTIFF Spatial Map")
        
        # 設定圖表標籤
        ax.set_xlabel("Easting (TM2 Coordinate)")
        ax.set_ylabel("Northing (TM2 Coordinate)")
        ax.set_title(f"Spatial Visualization: {file_path.split('/')[-1]}", fontsize=14)
        
        # 儲存圖片
        output_fig = "geotiff_plot_output.png"
        plt.savefig(output_fig, bbox_inches='tight', dpi=300)
        print(f"Visualization saved successfully as: {output_fig}")
        
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_tif(sys.argv[1])
    else:
        print("Usage: python plot_geotiff.py <path_to_geotiff_file.tif>")