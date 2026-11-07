"""
NetCDF Variable Inspector
-------------------------
此腳本讀取指定的 NetCDF (.nc) 檔案，並印出所有變數名稱、維度及屬性。

Requirements:
    pip install xarray netcdf4

python check_nc_vars.py  ../data/raw/nc_datas/data_2023/GEMS_NO2/202303.nc
"""

import xarray as xr

ds = xr.open_dataset('../data/raw/nc_datas/data_2023/GEMS_NO2/202303.nc')

# 檢查檔案的全域屬性，通常會寫明資料來源與單位定義
print(ds.attrs)