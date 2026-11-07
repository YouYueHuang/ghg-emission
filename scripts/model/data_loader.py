"""
資料載入：支援 Parquet、NetCDF (.nc) 與 CSV（檔案、資料夾、萬用字元、清單）。
讀取時會依檔名 (YYYYMM) 加入 year、month 欄位（供時間編碼 / 落後特徵選用）。
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import config
from features import parse_year_month


def _find_files(inputs, ext, label):
    files = []
    if isinstance(inputs, str):
        if os.path.isdir(inputs):
            files = sorted(glob.glob(os.path.join(inputs, f"*{ext}")))
        elif "*" in inputs:
            files = sorted(glob.glob(inputs))
        elif inputs.endswith(ext):
            files = [inputs]
    elif isinstance(inputs, list):
        files = inputs

    if not files:
        raise FileNotFoundError(f"❌ 找不到任何符合條件的 {label} ({ext}) 檔案: {inputs}")
    return files


def find_nc_files(nc_inputs):
    return _find_files(nc_inputs, ".nc", "NetCDF")


def find_parquet_files(inputs):
    return _find_files(inputs, ".parquet", "Parquet")


def detect_format(path):
    """回傳 'parquet' / 'nc' / 'csv'；無法判斷時回傳 None。"""
    fmt = config.DATA_FORMAT
    if fmt != 'auto':
        return fmt
    if isinstance(path, list):
        path = path[0]
    if os.path.isdir(path):
        has_pq = bool(glob.glob(os.path.join(path, "*.parquet")))
        has_nc = bool(glob.glob(os.path.join(path, "*.nc")))
        if has_pq and has_nc:
            raise ValueError(f"❌ 資料夾同時有 .parquet 與 .nc，請在 config.DATA_FORMAT 指定其一: {path}")
        return 'parquet' if has_pq else ('nc' if has_nc else None)
    for ext, name in ((".parquet", 'parquet'), (".nc", 'nc'), (".csv", 'csv')):
        if path.endswith(ext):
            return name
    return None


def tag_year_month(df, path):
    """依檔名 (如 202303.nc / 202303.parquet) 加入 year、month 欄位；檔名沒有年月時只提示。"""
    ym = parse_year_month(path)
    if ym:
        df[config.YEAR_COL], df[config.MONTH_COL] = ym
    else:
        print(f"[警告] 檔名 '{os.path.basename(path)}' 找不到 YYYYMM，無法由檔名取得月份")
    return df


def load_parquet_files(inputs):
    """批次讀取多個 .parquet 檔（依檔名排序）並合併。"""
    files = find_parquet_files(inputs)
    print(f"📦 偵測到 {len(files)} 個 Parquet 檔案，正在逐一讀取並合併...")
    df_list = []
    for f in files:
        try:
            df_temp = pd.read_parquet(f)
        except ImportError as e:
            raise ImportError("❌ 讀取 parquet 需要 pyarrow：請執行 pip install pyarrow") from e
        df_temp = tag_year_month(df_temp.reset_index(drop=True), f)
        print(f"   - 讀取: {f}（{len(df_temp)} 列，{df_temp.shape[1]} 欄）")
        df_list.append(df_temp)
    return pd.concat(df_list, ignore_index=True)


def load_nc_files_to_df(nc_inputs):
    """批次讀取多個 .nc 檔案，逐一轉為 DataFrame（加上年月）後合併（不需安裝 Dask）。"""
    import xarray as xr
    nc_files = find_nc_files(nc_inputs)
    print(f"📦 偵測到 {len(nc_files)} 個 NetCDF 檔案，正在逐一讀取並合併...")

    df_list = []
    for f in nc_files:
        ds = xr.open_dataset(f)
        df_temp = tag_year_month(ds.to_dataframe().reset_index(), f)
        ds.close()
        ym = parse_year_month(f)
        print(f"   - 讀取並轉換: {f}" + (f"  → year={ym[0]}, month={ym[1]}" if ym else ""))
        df_list.append(df_temp)

    return pd.concat(df_list, ignore_index=True)


def load_data(data_input_path):
    """依路徑 / 格式載入資料，回傳 DataFrame；找不到時結束程式。"""
    fmt = detect_format(data_input_path)
    if fmt == 'parquet':
        df = load_parquet_files(data_input_path)
    elif fmt == 'nc':
        df = load_nc_files_to_df(data_input_path)
    elif fmt == 'csv' and os.path.exists(data_input_path):
        df = pd.read_csv(data_input_path)
        print(f"✅ 成功載入 CSV 資料檔案: {data_input_path}")
    else:
        print(f"❌ 找不到指定的資料路徑或無法判斷格式: {data_input_path}")
        sys.exit(1)
    print(f"✅ 資料載入完成：共 {len(df)} 列、{df.shape[1]} 欄")
    return df


def load_nc_slice(nc_path, day=None):
    """
    讀取單一 .nc 供預測用，並加入 year、month。
    落後特徵需要前幾小時的資料，所以保留所有小時；若 day 是維度座標，只保留 day-1 與 day 兩天
    （day-1 供跨日的落後值使用），避免把整個月都展平。
    """
    import xarray as xr
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"❌ 找不到 nc 檔: {nc_path}")
    ds = xr.open_dataset(nc_path)
    if day is not None:
        for n in config.DAY_COL_CANDIDATES:
            if n in ds.dims and n in ds.coords:
                keep = np.flatnonzero(np.isin(ds[n].values, [day - 1, day]))
                if keep.size:
                    ds = ds.isel({n: keep})
                break
    df = tag_year_month(ds.to_dataframe().reset_index(), nc_path)
    ds.close()
    print(f"📦 已讀取 {nc_path}（{len(df)} 列）")
    return df