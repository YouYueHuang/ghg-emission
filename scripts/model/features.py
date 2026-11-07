"""
特徵工程：
  1. 由檔名取得年、月（202303.nc -> year=2023, month=3）
  2. hour / day / month 的 sin、cos 循環編碼
  3. NO2 落後特徵：用前 1、2 小時的 NO2 預測下一小時（依像素 lat/lon 對應）
訓練 (run_no2_xgb.py) 與預測 (predict_map.py) 共用同一份，確保特徵一致。
"""
import os
import re
import numpy as np
import pandas as pd
import config


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


# ---------- 1. 年月 ----------
def parse_year_month(path):
    """從檔名取出 (year, month)；取不到或月份不在 1~12 時回傳 None。"""
    m = re.search(config.FILENAME_YM_REGEX, os.path.basename(path))
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    return (year, month) if 1 <= month <= 12 else None


def ensure_year_month(df):
    """確認 df 有 month（及 year）欄位；month 缺少時嘗試用 MM / mm / Month。"""
    if config.MONTH_COL not in df.columns:
        alt = first_existing(df.columns, ['MM', 'mm', 'Month'])
        if alt is None:
            raise KeyError(f"❌ 找不到月份：檔名需含 YYYYMM（如 202303.nc），"
                           f"或資料中要有 '{config.MONTH_COL}' / MM 欄位")
        df[config.MONTH_COL] = df[alt]
    if config.YEAR_COL not in df.columns:
        print(f"[資訊] 沒有 {config.YEAR_COL} 欄位，使用 DEFAULT_YEAR={config.DEFAULT_YEAR}")
        df[config.YEAR_COL] = config.DEFAULT_YEAR
    return df


# ---------- 2. 循環編碼 ----------
def time_feature_names():
    return [f"{n}_{f}" for n in ('hour', 'day', 'month') for f in ('sin', 'cos')]


def add_time_features(df):
    """加入 hour/day/month 的 sin、cos 特徵（USE_CYCLIC_TIME=False 時不做事）。"""
    if not config.USE_CYCLIC_TIME:
        return df
    df = ensure_year_month(df)
    hour_col = first_existing(df.columns, config.HOUR_COL_CANDIDATES)
    day_col = first_existing(df.columns, config.DAY_COL_CANDIDATES)
    if hour_col is None or day_col is None:
        raise KeyError(f"❌ 找不到 hour / day 欄位（候選 {config.HOUR_COL_CANDIDATES} / "
                       f"{config.DAY_COL_CANDIDATES}；實際欄位: {list(df.columns)}）")

    for name, col, period in (('hour', hour_col, config.HOUR_PERIOD),
                              ('day', day_col, config.DAY_PERIOD),
                              ('month', config.MONTH_COL, config.MONTH_PERIOD)):
        angle = 2 * np.pi * df[col].to_numpy(dtype='float64') / period
        df[f"{name}_sin"] = np.sin(angle)
        df[f"{name}_cos"] = np.cos(angle)
    print(f"🕒 已加入時間循環特徵: {time_feature_names()}"
          f"（hour 週期 {config.HOUR_PERIOD}、day 週期 {config.DAY_PERIOD}、month 週期 {config.MONTH_PERIOD}）")
    return df


# ---------- 3. NO2 落後特徵 ----------
def lag_col_names(target_col, lags=None):
    if lags is None:
        lags = config.NO2_LAG_HOURS if config.USE_NO2_LAGS else []
    return [f"{target_col}_lag{k}" for k in lags]


def _absolute_hour(df, day_col, hour_col):
    """(year, month, day, hour) -> 自 1970-01-01 起算的小時數，跨日的落後也能正確對應。"""
    cols = [config.YEAR_COL, config.MONTH_COL, day_col, hour_col]
    if df[cols].isna().any().any():
        raise ValueError(f"❌ {cols} 含缺值，無法建立時間索引")
    dates = pd.to_datetime(pd.DataFrame({
        'year': df[config.YEAR_COL].astype(int),
        'month': df[config.MONTH_COL].astype(int),
        'day': df[day_col].astype(int),
    }))
    days = (dates - pd.Timestamp('1970-01-01')).dt.days.to_numpy(dtype='int64')
    return days * 24 + np.round(df[hour_col].to_numpy(dtype='float64')).astype('int64')


def add_no2_lags(df, target_col, lags=None):
    """
    為每一列 (像素 lat/lon, 時間 t) 加入 {target}_lag{k} = 同一像素在 t-k 小時的 NO2。
    找不到對應的時間（資料缺漏、該小時沒有觀測）時為 NaN。
    """
    if lags is None:
        lags = config.NO2_LAG_HOURS if config.USE_NO2_LAGS else []
    if not lags:
        return df
    if target_col not in df.columns:
        raise KeyError(f"❌ 資料中沒有目標欄位 '{target_col}'，無法建立落後特徵")

    df = ensure_year_month(df)
    lat_col = first_existing(df.columns, config.MAP_LAT_COLS)
    lon_col = first_existing(df.columns, config.MAP_LON_COLS)
    hour_col = first_existing(df.columns, config.HOUR_COL_CANDIDATES)
    day_col = first_existing(df.columns, config.DAY_COL_CANDIDATES)
    if None in (lat_col, lon_col, hour_col, day_col):
        raise KeyError("❌ 建立落後特徵需要 lat/lon/day/hour 欄位")

    key = pd.DataFrame({
        '_lat': df[lat_col].round(5).to_numpy(),
        '_lon': df[lon_col].round(5).to_numpy(),
        '_t': _absolute_hour(df, day_col, hour_col),
    })
    base = key.assign(_y=df[target_col].to_numpy())
    n_dup = int(base.duplicated(['_lat', '_lon', '_t']).sum())
    if n_dup:
        print(f"[警告] 有 {n_dup} 列的 (像素, 時間) 重複，落後特徵只取第一筆")
        base = base.drop_duplicates(['_lat', '_lon', '_t'])

    for k in lags:
        name = f"{target_col}_lag{k}"
        lag_tbl = base.rename(columns={'_y': name}).assign(_t=lambda d, k=k: d['_t'] + k)
        merged = key.merge(lag_tbl, on=['_lat', '_lon', '_t'], how='left')
        df[name] = merged[name].to_numpy()
        print(f"🔁 已加入 {name}（t-{k} 小時的 {target_col}）：有值比例 {df[name].notna().mean() * 100:.1f}%")
    return df