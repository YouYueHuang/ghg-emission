"""
訓練完成後，用模型預測「全台灣網格」在某段期間內每一個小時的 NO2，平均後輸出一張濃度分佈圖。

  - 期間預設為訓練資料的第一天第一小時到最後一天最後一小時（config.PERIOD_MAP_PERIOD）
  - 網格特徵來自 nc 檔（config.GRID_NC_PATH，欄位需與模型的特徵一致；檔名要含 YYYYMM 以取得年月）
  - 每個像素的值 = 期間內所有「特徵齊全」的小時預測值的平均（缺特徵的小時不納入，圖上以 n_hours 記錄每個像素用了幾個小時）
  - 為了控制記憶體，nc 檔依「天」逐段讀取、預測，只累加總和與筆數
"""
import os
import numpy as np
import pandas as pd

import config
from features import first_existing, parse_year_month
from map_plot import plot_no2_map


# ---------- 時間 ----------
def frame_timestamps(df):
    """(日期 + 小時) → 時間戳。優先用 monitordate；沒有時用 year / month / day 欄位。"""
    hour_col = first_existing(df.columns, config.HOUR_COL_CANDIDATES)
    if hour_col is None:
        raise KeyError(f"找不到小時欄位（候選 {config.HOUR_COL_CANDIDATES}）")
    if config.LOSO_DATE_COL in df.columns:
        date = pd.to_datetime(df[config.LOSO_DATE_COL]).dt.normalize()
    else:
        day_col = first_existing(df.columns, config.DAY_COL_CANDIDATES)
        if day_col is None or config.YEAR_COL not in df.columns or config.MONTH_COL not in df.columns:
            raise KeyError("找不到日期：需要 monitordate，或 year / month / day 欄位")
        date = pd.to_datetime(pd.DataFrame({'year': df[config.YEAR_COL].astype(int),
                                            'month': df[config.MONTH_COL].astype(int),
                                            'day': df[day_col].astype(int)}))
    return date + pd.to_timedelta(df[hour_col].astype(float), unit='h')


def training_period(df_clean, train_index, test_index=None):
    """訓練資料的第一個小時到最後一個小時；config.PERIOD_MAP_PERIOD='all' 時納入測試資料。"""
    idx = train_index
    if config.PERIOD_MAP_PERIOD == 'all' and test_index is not None:
        idx = train_index.append(test_index)
    ts = frame_timestamps(df_clean.loc[idx])
    return ts.min(), ts.max()


# ---------- 網格特徵 ----------
def prepare_grid_features(df, feature_cols):
    """補上模型需要、但 nc 裡名稱不同的欄位（MM ← month、day / Hr ← 對應的候選欄位），缺少時報錯並列出。"""
    for f in feature_cols:
        if f in df.columns:
            continue
        if f == 'MM' and config.MONTH_COL in df.columns:
            df[f] = df[config.MONTH_COL]
        elif f in config.DAY_COL_CANDIDATES:
            src = first_existing(df.columns, config.DAY_COL_CANDIDATES)
            if src:
                df[f] = df[src]
        elif f in config.HOUR_COL_CANDIDATES:
            src = first_existing(df.columns, config.HOUR_COL_CANDIDATES)
            if src:
                df[f] = df[src]
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        raise KeyError(f"❌ 網格資料缺少模型需要的特徵欄位: {missing}\n   網格資料實際欄位: {list(df.columns)}")
    return df


def iter_grid_chunks(nc_input, start, end):
    """依「天」逐段讀取網格 nc，只產生與 [start, end] 有重疊的資料（DataFrame，已加入 year、month）。"""
    import xarray as xr
    from data_loader import find_nc_files
    for f in find_nc_files(nc_input):
        ym = parse_year_month(f)
        if ym is None:
            print(f"[警告] 檔名 '{os.path.basename(f)}' 找不到 YYYYMM，略過")
            continue
        m0 = pd.Timestamp(ym[0], ym[1], 1)
        m1 = m0 + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
        if m1 < start.normalize() or m0 > end:
            continue
        print(f"   - 讀取 {os.path.basename(f)}")
        ds = xr.open_dataset(f)
        day_name = next((n for n in config.DAY_COL_CANDIDATES if n in ds.dims), None)
        if day_name is None:
            print("     [資訊] 找不到 day 維度，整個檔案一次展平（記憶體用量較大）")
            df = ds.to_dataframe().reset_index()
            df[config.YEAR_COL], df[config.MONTH_COL] = ym
            ds.close()
            yield df
            continue
        vals = ds[day_name].values
        if (not np.issubdtype(vals.dtype, np.number)) or (~np.isfinite(vals)).any() or (np.abs(vals) > 1e30).any():
            print(f"     [警告] {day_name} 座標不是有效數值，改用 1 ~ {len(vals)}")
            ds = ds.assign_coords({day_name: np.arange(1, len(vals) + 1)})
            vals = ds[day_name].values
        for d in vals:
            day_start = m0 + pd.Timedelta(days=int(d) - 1)
            if day_start + pd.Timedelta(hours=23) < start or day_start > end:
                continue
            sub = ds.sel({day_name: [d]}).to_dataframe().reset_index()
            sub[config.YEAR_COL], sub[config.MONTH_COL] = ym
            yield sub
        ds.close()


# ---------- 預測並累加 ----------
def accumulate_period_mean(chunks, model, feature_cols, start, end, optional_cols=()):
    """
    對每一段網格資料預測，累加各像素的總和與筆數。回傳 (各像素平均 DataFrame, 資訊 dict)。
    DataFrame 欄位：lat、lon、sum、n_hours、pred_no2（期間平均）。
    """
    optional = set(optional_cols)
    req_idx = [i for i, c in enumerate(feature_cols) if c not in optional]
    acc, stamps, n_pred = None, set(), 0
    for df in chunks:
        df = prepare_grid_features(df, feature_cols)
        ts = frame_timestamps(df)
        in_period = ((ts >= start) & (ts <= end)).to_numpy()
        if not in_period.any():
            continue
        df, ts = df[in_period], ts[in_period]
        lat_col = first_existing(df.columns, config.MAP_LAT_COLS)
        lon_col = first_existing(df.columns, config.MAP_LON_COLS)
        if lat_col is None or lon_col is None:
            raise KeyError(f"找不到經緯度欄位（候選 {config.MAP_LAT_COLS} / {config.MAP_LON_COLS}）")
        X = df[feature_cols]
        ok = np.isfinite(X.to_numpy(dtype='float64')[:, req_idx]).all(axis=1)
        if not ok.any():
            continue
        pred = model.predict(X[ok])
        g = (pd.DataFrame({'lat': df[lat_col].round(5).to_numpy()[ok], 'lon': df[lon_col].round(5).to_numpy()[ok],
                           'p': np.asarray(pred, dtype='float64')})
             .groupby(['lat', 'lon'])['p'].agg(['sum', 'count']))
        acc = g if acc is None else acc.add(g, fill_value=0)
        stamps.update(pd.DatetimeIndex(ts[ok]).unique())
        n_pred += int(ok.sum())
    if acc is None:
        raise ValueError("期間內沒有任何特徵齊全的網格資料可預測，請確認 nc 檔涵蓋該期間且欄位與模型一致")
    out = acc.reset_index().rename(columns={'count': 'n_hours'})
    out['pred_no2'] = out['sum'] / out['n_hours']
    info = {'n_hours_total': len(stamps), 'n_predictions': n_pred, 'n_pixels': len(out),
            'first': min(stamps), 'last': max(stamps)}
    return out, info


def make_period_mean_map(model, feature_cols, nc_input, start, end, output_dir, optional_cols=(), label=''):
    """預測期間內每個小時的全台網格 NO2、平均並輸出一張濃度分佈圖與網格 CSV。回傳 (圖檔路徑, 資訊)。"""
    print("=" * 75)
    print(f"🗺️  全台 NO2 平均濃度分佈圖：{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}")
    print("=" * 75)
    out, info = accumulate_period_mean(iter_grid_chunks(nc_input, start, end), model, feature_cols,
                                       start, end, optional_cols)
    folder = os.path.join(output_dir, "period_map")
    os.makedirs(folder, exist_ok=True)
    out.to_csv(os.path.join(folder, "period_mean_grid.csv"), index=False, encoding='utf-8-sig')
    title = (f"Mean predicted NO2 {label}\n{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}  "
             f"({info['n_hours_total']} hours, {info['n_pixels']} pixels)")
    png = plot_no2_map(out, 'lat', 'lon', title, os.path.join(folder, "NO2_period_mean_map.png"))
    print(f"   - 預測 {info['n_predictions']} 筆（{info['n_pixels']} 個像素 × 最多 {info['n_hours_total']} 個小時）；"
          f"實際使用的時間: {info['first']:%Y-%m-%d %H:%M} ~ {info['last']:%Y-%m-%d %H:%M}")
    return png, info