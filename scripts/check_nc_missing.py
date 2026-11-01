"""
NetCDF (.nc) Missing Value Inspector & Exporter
-----------------------------------------------
This script inspects all data variables in a NetCDF dataset, 
calculates the count and percentage of missing values (NaN) for each, 
prints the summary, and exports the report to '../results/figures/missing_value/'.

Requirements:
    pip install xarray numpy pandas
"""

import sys
import os
import xarray as xr
import numpy as np
import pandas as pd

def check_missing_values(file_path):
    print("=" * 75)
    print(f"INSPECTING NETCDF FILE FOR MISSING VALUES: {file_path}")
    print("=" * 75)
    
    # 開啟 NetCDF 檔案
    ds = xr.open_dataset(file_path)
    
    # 建立輸出資料夾
    output_dir = "../results/figures/missing_value"
    os.makedirs(output_dir, exist_ok=True)
    
    # 取得檔名（不含副檔名），作為輸出報表的前綴
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    csv_output_path = os.path.join(output_dir, f"{base_name}_missing_report.csv")
    
    print(f"{'Variable Name':<25} | {'Missing Count':<15} | {'Total Count':<15} | {'Missing %'}")
    print("-" * 75)
    
    total_vars = len(ds.data_vars)
    vars_with_missing = 0
    report_data = []
    
    for var_name in ds.data_vars:
        da = ds[var_name]
        
        # 計算總元素數量
        total_count = int(da.size)
        
        # 計算缺值數量 (使用 xarray 的 isnull())
        missing_count = int(da.isnull().sum().values)
        
        if missing_count > 0:
            vars_with_missing += 1
            
        missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
        
        # 格式化終端機輸出
        print(f"{var_name:<25} | {missing_count:<15} | {total_count:<15} | {missing_pct:.2f}%")
        
        # 收集報表數據
        report_data.append({
            "Variable": var_name,
            "Missing_Count": missing_count,
            "Total_Count": total_count,
            "Missing_Percentage": round(missing_pct, 4)
        })
        
    print("=" * 75)
    print(f"總結報告：")
    print(f"- 總資料變數數量 (Total Variables): {total_vars}")
    print(f"- 含有缺值的變數數量 (Variables with Missing Values): {vars_with_missing}")
    print(f"- 完全無缺值的變數數量 (Complete Variables): {total_vars - vars_with_missing}")
    print("=" * 75)
    
    # 轉為 DataFrame 並儲存為 CSV 檔案
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"📄 缺值分析報表已成功儲存至：{csv_output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_missing_values(sys.argv[1])
    else:
        print("使用方式: python check_nc_missing.py <path_to_nc_file.nc>")