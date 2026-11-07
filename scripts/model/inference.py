"""
推論：從指定 day / hour 的資料切片，依模型的特徵欄位預測 NO2 濃度。
"""
import numpy as np
import pandas as pd
import config
from features import first_existing


def select_day_hour(df, day, hour):
    """依 day、hour 的值篩選資料列；找不到時印出可用的值並回傳空表。"""
    day_col = first_existing(df.columns, config.DAY_COL_CANDIDATES)
    hour_col = first_existing(df.columns, config.HOUR_COL_CANDIDATES)
    if day_col is None or hour_col is None:
        raise KeyError(f"資料中找不到 day / hour 欄位（day 候選 {config.DAY_COL_CANDIDATES}，"
                       f"hour 候選 {config.HOUR_COL_CANDIDATES}；實際欄位: {list(df.columns)}）")

    sel = np.isclose(df[day_col].to_numpy(dtype='float64'), float(day)) & \
          np.isclose(df[hour_col].to_numpy(dtype='float64'), float(hour))
    out = df[sel]
    if out.empty:
        print(f"❌ 找不到 {day_col}={day}、{hour_col}={hour} 的資料。")
        print(f"   可用的 {day_col}: {sorted(set(df[day_col].dropna().tolist()))[:60]}")
        print(f"   可用的 {hour_col}: {sorted(set(df[hour_col].dropna().tolist()))[:60]}")
    return out


def predict_slice(model, feature_cols, df_slice, optional_cols=()):
    """
    回傳 DataFrame（欄位：lat、lon 座標與 pred_no2）及使用的座標欄位名稱。
    任一特徵缺值的像素不預測（訓練時也排除了缺值列），結果為 NaN；
    optional_cols（訓練時允許缺值的落後特徵）除外，缺值照樣交給模型處理。
    """
    missing = [c for c in feature_cols if c not in df_slice.columns]
    if missing:
        raise KeyError(f"❌ 資料缺少模型需要的特徵欄位: {missing}")

    lat_col = first_existing(df_slice.columns, config.MAP_LAT_COLS)
    lon_col = first_existing(df_slice.columns, config.MAP_LON_COLS)
    if lat_col is None or lon_col is None:
        raise KeyError(f"❌ 找不到經緯度欄位（候選 {config.MAP_LAT_COLS} / {config.MAP_LON_COLS}）")

    X = df_slice[feature_cols]
    required = [c for c in feature_cols if c not in set(optional_cols)]
    ok = np.isfinite(X[required].to_numpy(dtype='float64')).all(axis=1)
    pred = np.full(len(X), np.nan)
    if ok.any():
        pred[ok] = model.predict(X[ok])
    print(f"   - 該時間點共 {len(X)} 個像素，特徵齊全可預測: {int(ok.sum())}")

    out = df_slice[[lat_col, lon_col]].copy()
    out['pred_no2'] = pred
    return out, lat_col, lon_col