"""
NetCDF (.nc) Data Exploration Toolkit
-------------------------------------
This script provides a clean workflow for exploring, inspecting, and 
summarizing NetCDF dataset structures, dimensions, variables, metadata, 
and missing values.

Requirements:
    pip install netcdf4 numpy pandas xarray
"""

import sys
import numpy as np
import pandas as pd

def inspect_with_xarray(file_path):
    """Performs data exploration using xarray and pandas."""
    import xarray as xr
    
    print("=" * 70)
    print(f"INSPECTING NETCDF FILE: {file_path}")
    print("=" * 70)
    
    ds = xr.open_dataset(file_path)
    print("Dataset Overview ---")
    print(ds)
    print ("")

    print ("印出所有變數名稱") 
    for var in ds.data_vars:
        print(var)
    
    print("Dimensions ---")
    for dim, size in ds.sizes.items():
        print(f"  {dim}: {size}")
        
    print("Global Attributes ---")
    for attr, val in ds.attrs.items():
        print(f"  {attr}: {val}")
        
    print("--- Variable Statistics & Missing Value Analysis ---")
    stats = []
    for var in ds.data_vars:
        da = ds[var]
        arr = da.values
        total = arr.size
        nans = np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0
        valid = total - nans
        coverage = (valid / total * 100) if total > 0 else 0
        
        if valid > 0 and np.issubdtype(arr.dtype, np.number):
            vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
            vmean, vmed = float(np.nanmean(arr)), float(np.nanmedian(arr))
        else:
            vmin = vmax = vmean = vmed = np.nan
            
        stats.append({
            "Variable": var,
            "Shape": str(da.shape),
            "Dtype": str(da.dtype),
            "Valid %": f"{coverage:.2f}%",
            "Min": round(vmin, 4) if not np.isnan(vmin) else "-",
            "Mean": round(vmean, 4) if not np.isnan(vmean) else "-",
            "Median": round(vmed, 4) if not np.isnan(vmed) else "-",
            "Max": round(vmax, 4) if not np.isnan(vmax) else "-",
        })
        
    df = pd.DataFrame(stats)
    print(df.to_string(index=False))
    ds.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_with_xarray(sys.argv[1])
    else:
        print("Usage: python explore_nc_data.py <path_to_nc_file>")