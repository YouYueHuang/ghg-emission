"""
GeoTIFF (.tif) Geospatial Data Exploration Toolkit
--------------------------------------------------
This script inspects and summarizes GeoTIFF raster metadata, coordinate reference systems,
per-band statistics, missing values (NODATA), and pixel value distributions.

Requirements:
    pip install rasterio numpy pandas
"""

import sys
import numpy as np
import pandas as pd
import rasterio

def explore_geotiff(file_path):
    """Opens a GeoTIFF file and performs comprehensive data exploration."""
    print("=" * 70)
    print(f"INSPECTING GEOTIFF FILE: {file_path}")
    print("=" * 70)
    
    with rasterio.open(file_path) as src:
        # 1. 基本屬性與空間資訊
        print("\n--- 1. Dataset Overview & Metadata ---")
        print(f"Driver: {src.driver}")
        print(f"Width (Cols): {src.width} | Height (Rows): {src.height}")
        print(f"Number of Bands: {src.count}")
        print(f"Coordinate Reference System (CRS): {src.crs}")
        print(f"Bounding Box (Bounds): {src.bounds}")
        print(f"Resolution (Pixel Size): {src.res}")
        print(f"Nodata Value(s): {src.nodatavals}")
        
        # 2. 轉換矩陣與地理對齊
        print("\n--- 2. Affine Transform ---")
        print(src.transform)
        
        # 3. 逐波段統計與缺值分析
        print("\n--- 3. Per-Band Statistics & Missing Value Analysis ---")
        stats_list = []
        
        for i in range(1, src.count + 1):
            band_data = src.read(i)
            nodata_val = src.nodatavals[i - 1]
            
            # 處理 NoData / NaN 遮罩
            if nodata_val is not None:
                valid_mask = (band_data != nodata_val) & (~np.isnan(band_data))
            else:
                valid_mask = ~np.isnan(band_data)
                
            total_pixels = band_data.size
            valid_pixels = np.sum(valid_mask)
            missing_pixels = total_pixels - valid_pixels
            coverage_pct = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            valid_vals = band_data[valid_mask]
            
            if valid_pixels > 0:
                vmin = float(np.min(valid_vals))
                vmax = float(np.max(valid_vals))
                vmean = float(np.mean(valid_vals))
                vstd = float(np.std(valid_vals))
                vmedian = float(np.median(valid_vals))
                p95 = float(np.percentile(valid_vals, 95))
            else:
                vmin = vmax = vmean = vstd = vmedian = p95 = np.nan
                
            stats_list.append({
                "Band": i,
                "Dtype": str(band_data.dtype),
                "Valid %": f"{coverage_pct:.2f}%",
                "Missing": missing_pixels,
                "Min": round(vmin, 4) if not np.isnan(vmin) else "-",
                "Mean": round(vmean, 4) if not np.isnan(vmean) else "-",
                "Median": round(vmedian, 4) if not np.isnan(vmedian) else "-",
                "95%": round(p95, 4) if not np.isnan(p95) else "-",
                "Max": round(vmax, 4) if not np.isnan(vmax) else "-",
                "Std": round(vstd, 4) if not np.isnan(vstd) else "-"
            })
            
        df_stats = pd.DataFrame(stats_list)
        print(df_stats.to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        explore_geotiff(sys.argv[1])
    else:
        print("Usage: python explore_tif_data.py <path_to_geotiff_file.tif>")