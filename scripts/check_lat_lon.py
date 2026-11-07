"""
Check GEMS Latitude and Longitude Matrix Range
"""
import xarray as xr

ds = xr.open_dataset("../data/raw/nc_datas/data_2023/GEMS_NO2/202304.nc")
print("GEMS 變數清單:", list(ds.variables))

# 尋找經緯度變數
lat_var = ds['latitude'] if 'latitude' in ds.variables else ds[list(ds.coords)[0]]
lon_var = ds['longitude'] if 'longitude' in ds.variables else ds[list(ds.coords)[1]]

print(f"GEMS 緯度範圍: {lat_var.values.min()} ~ {lat_var.values.max()}")
print(f"GEMS 經度範圍: {lon_var.values.min()} ~ {lon_var.values.max()}")
ds.close()