"""
NetCDF Variable Inspector
-------------------------
此腳本讀取指定的 NetCDF (.nc) 檔案，並印出所有變數名稱、維度及屬性。

Requirements:
    pip install xarray netcdf4

python check_nc_vars.py  ../data/raw/nc_datas/data_2023/GEMS_NO2/202303.nc
"""

import sys
import xarray as xr

def inspect_nc_variables(nc_path):
    print("=" * 75)
    print(f"正在檢查 NetCDF 檔案: {nc_path}")
    print("=" * 75)
    
    # 開啟 NetCDF 檔案
    with xr.open_dataset(nc_path) as ds:
        print("\n【1. 座標與維度 (Coordinates & Dimensions)】")
        for coord_name in ds.coords:
            coord = ds[coord_name]
            print(f"   - 座標: {coord_name} (維度: {coord.dims}, 長度: {coord.size})")
            
        print("\n【2. 資料變數 (Data Variables)】")
        # 逐一列出所有變數及其對應的維度
        for var_name, var_data in ds.variables.items():
            # 排除純座標軸，只看資料變數與屬性
            if var_name not in ds.coords:
                print(f"   - 變數名稱: {var_name}")
                print(f"     維度: {var_data.dims}")
                print(f"     形狀 (Shape): {var_data.shape}")
                if "units" in var_data.attrs:
                    print(f"     單位: {var_data.attrs['units']}")
                print("-" * 50)
                
    print("=" * 75)
    print("🎉 變數檢查完畢！")
    print("=" * 75)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        nc_file = sys.argv[1]
        inspect_nc_variables(nc_file)
    else:
        print("使用方式: python check_nc_vars.py <path_to_file.nc>")