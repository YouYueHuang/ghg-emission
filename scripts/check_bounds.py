"""
Spatial Bounding Box Diagnostic Script
--------------------------------------
檢查 GEMS NO2 與 DEM TIF 的經緯度範圍是否一致。
"""

import xarray as xr
import rasterio
import numpy as np

def check_spatial_bounds(no2_nc_path, dem_tif_path):
    print("=" * 75)
    print("🔍 檢查空間範圍與座標系統 (Bounding Box Check)")
    print("=" * 75)
    
    # 1. 檢查 GEMS NO2 座標
    ds_no2 = xr.open_dataset(no2_nc_path)
    print("📦 GEMS 座標變數:", list(ds_no2.coords))
    
    # 嘗試找出 lat / lon 變數
    lat_vars = [v for v in ds_no2.variables if 'lat' in v.lower() or 'y' in v.lower()]
    lon_vars = [v for v in ds_no2.variables if 'lon' in v.lower() or 'x' in v.lower()]
    
    print(found_lat := f"  - 發現緯度變數: {lat_vars}" if lat_vars else "  - 未直接發現 lat 變數")
    print(found_lon := f"  - 發現經度變數: {lon_vars}" if lon_vars else "  - 未直接發現 lon 變數")
    
    # 2. 檢查 DEM TIF 範圍
    with rasterio.open(dem_tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        print(f"\n📦 DEM TIF 資訊:")
        print(f"  - 座標系統 (CRS): {crs}")
        print(f"  - 空間邊界 (Bounds): 左右 (X: {bounds.left} ~ {bounds.right}), 南北 (Y: {bounds.bottom} ~ {bounds.top})")
        
    ds_no2.close()
    print("=" * 75)

if __name__ == "__main__":
    check_spatial_bounds(
        "../data/raw/nc_datas/data_2023/GEMS_NO2/202304.nc",
        "../data/raw/nc_datas/ground-info/dem_20m.tif"
    )