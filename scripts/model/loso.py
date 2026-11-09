"""
留一測站交叉驗證 (Leave-One-Station-Out, LOSO)。

做法：對每一個測站，把該測站「所有時間」的資料整個移除，用其餘測站訓練模型，
再預測被移除的測站。因為模型從沒看過這個測站，結果反映的是「推廣到沒有測站的地點」的能力
（與依時間切分的測試不同：那個測試的測站也出現在訓練集裡）。

  - 資料範圍：--mode spring 只用春季月份；full_year 用全部月份（訓練與被預測的測站使用同一段期間）
  - 驗證集：若參數有 early_stopping_rounds，從「其餘測站」抽一部分測站當驗證集；被移除的測站完全不參與訓練或停止判斷
  - 沿用 --params 搜尋到的參數時（固定樹數、無 early stopping），不需要驗證測站
"""
import os
import re
import json
import zlib
import warnings
import numpy as np
import pandas as pd

import config
from features import add_time_features, add_no2_lags, ensure_year_month, first_existing
from preprocessing import sort_by_time, select_feature_cols, clean_data, report_duplicate_features
from metrics import calculate_evaluation_metrics


def prepare_loso_data(df, target_col, mode, station_col, output_dir):
    """排序 → 依組別選月份 → 特徵工程 → 缺值過濾（含過濾報告）。回傳 (清理後資料, 特徵欄位)。"""
    if station_col not in df.columns:
        raise KeyError(f"❌ 資料中找不到測站欄位 '{station_col}'；實際欄位: {list(df.columns)}")
    df = sort_by_time(df)
    df = ensure_year_month(df.copy())
    months_present = sorted(int(m) for m in df[config.MONTH_COL].dropna().unique())
    if mode == 'spring':
        df = df[df[config.MONTH_COL].isin(config.SPRING_MONTHS)]
        if df.empty:
            raise ValueError(f"❌ 資料中沒有春季月份 {config.SPRING_MONTHS} 的資料")
    print(f"🗓️  資料組別: {mode}；使用月份: {sorted(int(m) for m in df[config.MONTH_COL].dropna().unique())}"
          f"（檔案含的月份: {months_present}）")

    d = add_time_features(df.copy())
    d = add_no2_lags(d, target_col)
    feature_cols = select_feature_cols(d, target_col)
    print(f"🧩 最終使用的特徵 ({len(feature_cols)}): {feature_cols}")
    d_clean = clean_data(d, feature_cols, target_col, output_dir=output_dir)
    report_duplicate_features(d_clean, feature_cols)
    return d_clean, feature_cols


def _short(metrics):
    return {k.split(' (')[0]: float(v) for k, v in metrics.items()}


def _fit_fold(X_tr, y_tr, X_va, y_va, params):
    """訓練單一摺；有 early stopping 且有驗證集時用驗證集停止，否則用固定樹數。回傳 (模型, 輪數, 是否用 early stopping)。"""
    import xgboost as xgb
    p = dict(params)
    use_es = bool(p.get('early_stopping_rounds')) and X_va is not None and len(X_va) > 0
    if not use_es:
        p.pop('early_stopping_rounds', None)
    model = xgb.XGBRegressor(**p)
    if use_es:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model, int(model.best_iteration) + 1, True
    model.fit(X_tr, y_tr, verbose=False)
    return model, int(p['n_estimators']), False


def run_loso(d_clean, feature_cols, target_col, params, station_col=None, stations=None,
             min_rows=None, val_frac=None, seed=None):
    """
    對每個測站做留一測站訓練與預測。回傳 (逐筆預測 DataFrame, 各測站指標 DataFrame, 摘要 dict)。
    stations 可指定只做哪幾個測站（以測站欄位的值，字串比對）。
    """
    station_col = station_col or config.LOSO_STATION_COL
    min_rows = config.LOSO_MIN_STATION_ROWS if min_rows is None else min_rows
    val_frac = config.LOSO_VAL_STATION_FRAC if val_frac is None else val_frac
    seed = config.LOSO_SEED if seed is None else seed

    d = d_clean[d_clean[station_col].notna()]
    ids = d[station_col].astype(str).to_numpy()
    X, y = d[feature_cols], d[target_col]
    counts = pd.Series(ids).value_counts()
    eligible = sorted(s for s in counts.index if counts[s] >= min_rows)
    skipped = sorted(set(counts.index) - set(eligible))
    if skipped:
        print(f"[資訊] {len(skipped)} 個測站資料列少於 {min_rows}，不做: {skipped[:20]}{' ...' if len(skipped) > 20 else ''}")
    targets = eligible
    if stations:
        wanted = [str(s) for s in stations]
        unknown = [s for s in wanted if s not in eligible]
        if unknown:
            print(f"[警告] 這些測站不在可用清單內（不存在或資料太少）: {unknown}")
        targets = [s for s in wanted if s in eligible]
    if not targets:
        raise ValueError("沒有可執行的測站")

    name_col = config.LOSO_NAME_COL if config.LOSO_NAME_COL in d.columns else station_col
    lat_col = first_existing(d.columns, config.MAP_LAT_COLS)
    lon_col = first_existing(d.columns, config.MAP_LON_COLS)
    hour_col = first_existing(d.columns, config.HOUR_COL_CANDIDATES)
    id_cols = [c for c in dict.fromkeys([station_col, name_col, config.LOSO_DATE_COL, hour_col, lat_col, lon_col])
               if c and c in d.columns]
    use_es_requested = bool(params.get('early_stopping_rounds'))

    print("=" * 75)
    print(f"🧪 留一測站交叉驗證：{len(targets)} 個測站（以 '{station_col}' 區分），資料 {len(d)} 列，"
          f"{len(counts)} 個測站")
    print("=" * 75)
    all_stations = sorted(counts.index)
    pred_parts, rows = [], []
    for k, st in enumerate(targets, 1):
        is_te = ids == st
        rest = [s for s in all_stations if s != st]
        is_val = np.zeros(len(ids), dtype=bool)
        val_list = []
        if use_es_requested and len(rest) >= 3:
            rng = np.random.RandomState((seed + zlib.crc32(str(st).encode('utf-8'))) % (2 ** 32))   # 種子只取決於測站本身，與 --stations 的內容、順序無關
            val_st = rng.choice(rest, max(1, int(round(val_frac * len(rest)))), replace=False)
            is_val = np.isin(ids, val_st)
            val_list = sorted(str(v) for v in val_st)
        is_tr = ~is_te & ~is_val
        X_va = X[is_val] if is_val.any() else None
        y_va = y[is_val] if is_val.any() else None

        model, rounds, used_es = _fit_fold(X[is_tr], y[is_tr], X_va, y_va, params)
        X_te, y_te = X[is_te], y[is_te]
        pred = model.predict(X_te, iteration_range=(0, rounds)) if used_es else model.predict(X_te)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            m = _short(calculate_evaluation_metrics(y_te.to_numpy(), pred))
        sub = d.loc[is_te, id_cols].copy()
        sub['obs'], sub['pred'] = y_te.to_numpy(), np.asarray(pred, dtype='float64')
        pred_parts.append(sub)

        name = str(sub[name_col].iloc[0])
        rows.append({
            'station': st, 'name': name, 'n_test': int(is_te.sum()), 'n_train': int(is_tr.sum()),
            'n_val_stations': int(len(np.unique(ids[is_val]))) if is_val.any() else 0,
            'val_stations': ','.join(val_list),
            'rounds': rounds, **m,
            'lat': float(sub[lat_col].mean()) if lat_col else np.nan,
            'lon': float(sub[lon_col].mean()) if lon_col else np.nan,
            'obs_mean': float(y_te.mean()),
        })
        print(f"   [{k:>3}/{len(targets)}] {st} {name}: n={int(is_te.sum())}, 訓練 {int(is_tr.sum())} 列, "
              f"R²={m['R2']:.3f}, RMSE={m['RMSE']:.3f}, MD={m['MD']:.3f}")

    preds = pd.concat(pred_parts, ignore_index=True)
    metrics_df = pd.DataFrame(rows)
    pooled = _short(calculate_evaluation_metrics(preds['obs'].to_numpy(), preds['pred'].to_numpy()))
    keys = ['r', 'R2', 'RMSE', 'MD', 'MAPE']
    summary = {
        'n_stations': len(metrics_df), 'n_predictions': len(preds),
        **{f"pooled_{k}": pooled[k] for k in keys},
        **{f"station_mean_{k}": float(metrics_df[k].mean()) for k in keys},
        **{f"station_median_{k}": float(metrics_df[k].median()) for k in keys},
        'stations_R2_above_0': int((metrics_df['R2'] > 0).sum()),
    }
    print("=" * 75)
    print(f"📊 LOSO 摘要（{summary['n_stations']} 個測站、{summary['n_predictions']} 筆預測）")
    print(f"   全部預測合併：r={pooled['r']:.3f}, R²={pooled['R2']:.3f}, RMSE={pooled['RMSE']:.3f}, "
          f"MD={pooled['MD']:.3f}, MAPE={pooled['MAPE']:.1f}%")
    print(f"   各測站中位數：R²={summary['station_median_R2']:.3f}, RMSE={summary['station_median_RMSE']:.3f}；"
          f"R²>0 的測站 {summary['stations_R2_above_0']}/{summary['n_stations']}")
    print("=" * 75)
    return preds, metrics_df, summary


def save_loso_results(preds, metrics_df, summary, output_dir, config_info):
    preds.to_csv(os.path.join(output_dir, "loso_predictions.csv"), index=False, encoding='utf-8-sig')
    metrics_df.sort_values('R2', ascending=False).to_csv(os.path.join(output_dir, "loso_station_metrics.csv"),
                                                          index=False, encoding='utf-8-sig')
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "loso_summary.csv"), index=False, encoding='utf-8-sig')
    with open(os.path.join(output_dir, "loso_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f"💾 LOSO 結果已儲存至: {output_dir}")


def safe_name(s):
    return re.sub(r'[^0-9A-Za-z_-]+', '_', str(s))