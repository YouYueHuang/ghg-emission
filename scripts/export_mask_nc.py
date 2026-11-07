"""
GEMS & DEM Direct Coordinate Sampling & Filtering Script (Rectilinear Fix)
------------------------------------------------------------------------
修正 1D 經緯度網格的 Meshgrid 展開問題，確保 pyproj 能正確批次轉換坐標。

Requirements:
    pip install xarray numpy rasterio pyproj netcdf4
"""

import os
import numpy as np
import xarray as xr
import rasterio
from pyproj import Transformer

def gems_dem_spatial_filter(no2_nc_path, dem_tif_path, threshold=500):
    print("=" * 75)
    print("🚀 開始執行坐標系轉換與高程採樣過濾流程（Meshgrid 修正版）...")
    print("=" * 75)
    
    # 1. 讀取 GEMS NO2 資料與經緯度網格
    print(f"📦 讀取 NO2 資料: {no2_nc_path}")
    ds_no2 = xr.open_dataset(no2_nc_path)
    
    no2_vars = [v for v in ds_no2.data_vars if 'no2' in v.lower()]
    target_var = no2_vars[0] if no2_vars else list(ds_no2.data_vars)[0]
    da_no2_raw = ds_no2[target_var]
    
    # 提取 1D 經緯度座標
    lats = ds_no2['latitude'].values
    lons = ds_no2['longitude'].values
    print(f"✅ GEMS 原始緯度大小: {lats.shape}, 經度大小: {lons.shape}")
    
    # 如果是 1D 陣列，透過 meshgrid 建立完整的 2D 網格
    if lats.ndim == 1 and lons.ndim == 1:
        lons_2d, lats_2d = np.meshgrid(lons, lats)
    else:
        lons_2d, lats_2d = lons, lats
        
    print(f"✅ 展開後的 2D 網格形狀: {lats_2d.shape}")
    
    # 2. 自動挑選有效資料最多的時間點（取出 2D NO2 畫面）
    time_dims = [d for d in da_no2_raw.dims if 'time' in d.lower() or 'day' in d.lower()]
    if time_dims:
        time_dim = time_dims[0]
        total_times = da_no2_raw.sizes[time_dim]
        best_idx, max_valid = 0, -1
        for i in range(total_times):
            slice_arr = da_no2_raw.isel({time_dim: i}).values
            valid_count = np.sum((slice_arr >= 0) & (~np.isnan(slice_arr)))
            if valid_count > max_valid:
                max_valid = valid_count
                best_idx = i
        da_no2_2d = da_no2_raw.isel({time_dim: best_idx})
        print(f"🎯 鎖定最佳時間點索引 {best_idx}，有效像素: {max_valid:,}")
    else:
        da_no2_2d = da_no2_raw
        
    while da_no2_2d.ndim > 2:
        da_no2_2d = da_no2_2d[0]
        
    no2_data = da_no2_2d.values.astype(float)
    no2_data[no2_data < 0] = np.nan
    no2_data[no2_data > 1e15] = np.nan
    
    # 3. 開啟 DEM TIF 並建立坐標轉換器 (WGS84 -> DEM 投影坐標)
    print(f"📦 讀取 DEM TIF 檔: {dem_tif_path}")
    with rasterio.open(dem_tif_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_data = src.read(1).astype(float)
        
        # 建立 Transformer (從 EPSG:4326 轉到 DEM 的 CRS)
        transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        
        print(f"🔄 正在將 GEMS 的 ({lats_2d.size:,}) 個網格點轉換至 DEM 投影坐標系...")
        flat_lons = lons_2d.flatten()
        flat_lats = lats_2d.flatten()
        
        xs, ys = transformer.transform(flat_lons, flat_lats)
        
        # 將投影坐標轉回 DEM 的矩陣 Row/Col 索引
        rows, cols = rasterio.transform.rowcol(dem_transform, xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)
        
        # 4. 在 DEM 中採樣高程值
        dem_height, dem_width = dem_data.shape
        valid_indices = (rows >= 0) & (rows < dem_height) & (cols >= 0) & (cols < dem_width)
        
        sampled_elevation = np.full(flat_lons.shape, np.nan)
        valid_rows = rows[valid_indices]
        valid_cols = cols[valid_indices]
        sampled_elevation[valid_indices] = dem_data[valid_rows, valid_cols]
        
        # 恢復成與 GEMS 相同的 2D 形狀
        elevation_2d = sampled_elevation.reshape(lats_2d.shape)
        
    # 5. 根據 500m 門檻建立 Mask 並過濾 NO2
    mask_2d = np.where((elevation_2d <= threshold) & (~np.isnan(elevation_2d)), 1, np.nan)
    filtered_no2 = no2_data * mask_2d
    
    print(f"✅ 轉換與採樣完成！")
    print(f"   - 原始有效 NO2 資料點數: {np.count_nonzero(~np.isnan(no2_data)):,}")
    print(f"   - 落在 DEM 範圍內且 <=500m 的有效點數: {np.count_nonzero(~np.isnan(filtered_no2)):,}")
    print("=" * 75)
    
    ds_no2.close()
    return filtered_no2

if __name__ == "__main__":
    no2_path = "../data/raw/nc_datas/data_2023/GEMS_NO2/202304.nc"
    dem_path = "../data/raw/nc_datas/ground-info/dem_20m.tif"
    
    filtered_result = gems_dem_spatial_filter(no2_path, dem_path, threshold=500)