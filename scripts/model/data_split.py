"""
訓練 / 測試集的建立，支援兩種訓練組別：

  spring    只用春季（SPRING_MONTHS）的資料：依時間取前 TRAIN_RATIO 訓練、其餘測試
  full_year 用全部月份的資料訓練，但「測試集與 spring 完全相同」：
            先以春季資料定義測試集（春季時間上最後的 1 - TRAIN_RATIO），
            全年模式的測試集就是這批列，訓練集則是全年其餘的列

兩種組別的測試集相同，所以測試指標可以直接比較。測試集的「指紋」(fingerprint) 會印出並存檔，
兩次執行的指紋一致，代表測試集完全相同。
"""
import hashlib
import numpy as np
import pandas as pd

import config
from features import add_time_features, add_no2_lags, ensure_year_month
from preprocessing import (sort_by_time, select_feature_cols, clean_data, split_train_test,
                           report_duplicate_features)

MODES = ('spring', 'full_year')


def _prepare(df_part, target_col, output_dir, verbose):
    """特徵工程 → 選特徵 → 缺值過濾。回傳 (清理後資料, 特徵欄位)。"""
    d = add_time_features(df_part.copy())
    d = add_no2_lags(d, target_col)
    feature_cols = select_feature_cols(d, target_col)
    if verbose:
        print(f"🧩 最終使用的特徵 ({len(feature_cols)}): {feature_cols}")
    d_clean = clean_data(d, feature_cols, target_col,
                         output_dir=output_dir if verbose else None, verbose=verbose)
    return d_clean, feature_cols


def test_set_fingerprint(frame, target_col):
    """以測試列的識別欄位與目標值計算的指紋（與列順序、列編號無關），用來確認兩次執行的測試集相同。"""
    cols = [c for c in config.FILTER_ID_COLS if c in frame.columns] + [target_col]
    h = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy(dtype='uint64')
    return hashlib.md5(np.sort(h).tobytes()).hexdigest()[:12]


def prepare_train_test(df, target_col, mode, output_dir):
    """
    回傳 (X_train, X_test, y_train, y_test, feature_cols, df_clean, info)。
    df_clean 是清理後、尚未切分的資料（供查詢測試列的其他欄位，如 Hr）；
    info 為組別、月份、筆數與測試集指紋，會寫進設定檔與模型中繼資料。
    """
    if mode not in MODES:
        raise ValueError(f"RUN_MODE 只能是 {MODES}，收到: {mode!r}")

    df = sort_by_time(df)
    df = ensure_year_month(df.copy())
    df[config.ROW_ID_COL] = np.arange(len(df))          # 排序後的列編號，用來在不同處理流程間對應同一列

    months_present = sorted(int(m) for m in df[config.MONTH_COL].dropna().unique())
    spring_mask = df[config.MONTH_COL].isin(config.SPRING_MONTHS).to_numpy()
    print(f"🗓️  訓練組別: {mode}；資料含的月份: {months_present}；春季月份定義: {config.SPRING_MONTHS}")
    if not spring_mask.any():
        raise ValueError(f"❌ 資料中沒有春季月份 {config.SPRING_MONTHS} 的資料，無法定義共同的測試集")

    if mode == 'spring':
        d_clean, feature_cols = _prepare(df[spring_mask], target_col, output_dir, verbose=True)
        report_duplicate_features(d_clean, feature_cols)
        X_train, X_test, y_train, y_test = split_train_test(d_clean, feature_cols, target_col)
        df_clean = d_clean

    else:   # full_year
        other = sorted(set(months_present) - set(config.SPRING_MONTHS))
        if not other:
            print(f"[警告] 資料只含春季月份 {months_present}，全年模式與春季模式的訓練資料相同；"
                  f"請確認 .ml-parquet 內已放入其他月份的檔案")

        # 1. 用春季資料定義測試集（與 spring 模式完全相同的流程）
        ref_clean, ref_features = _prepare(df[spring_mask], target_col, None, verbose=False)
        n_train_ref = int(config.TRAIN_RATIO * len(ref_clean))
        test_rids = set(ref_clean[config.ROW_ID_COL].iloc[n_train_ref:].tolist())

        # 2. 全年資料：特徵工程與缺值過濾（報告針對全年資料）
        full_clean, feature_cols = _prepare(df, target_col, output_dir, verbose=True)
        if feature_cols != ref_features:
            raise RuntimeError("全年與春季的特徵欄位不一致，無法保證測試集相同")
        report_duplicate_features(full_clean, feature_cols)

        # 3. 測試集 = 春季測試列；訓練集 = 全年其餘列（可選：只用測試期之前的資料）
        rid = full_clean[config.ROW_ID_COL].to_numpy()
        is_test = np.isin(rid, list(test_rids))
        train_mask = ~is_test
        if config.FULL_YEAR_ONLY_PAST:
            train_mask &= rid < min(test_rids)
            print(f"⏪ FULL_YEAR_ONLY_PAST=True：訓練只用測試期開始之前的資料")
        if int(is_test.sum()) != len(test_rids):
            print(f"[警告] 全年資料中找到 {int(is_test.sum())} 筆測試列，但春季定義為 {len(test_rids)} 筆"
                  f"（可能是落後特徵跨月邊界造成），測試集可能與春季模式不完全相同")
        train, test = full_clean[train_mask], full_clean[is_test]
        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]
        df_clean = full_clean

    fingerprint = test_set_fingerprint(df_clean.loc[X_test.index], target_col)
    train_months = sorted(int(m) for m in df_clean.loc[X_train.index, config.MONTH_COL].dropna().unique())
    info = {
        'run_mode': mode,
        'train_months': train_months,
        'test_months': sorted(int(m) for m in df_clean.loc[X_test.index, config.MONTH_COL].dropna().unique()),
        'spring_months': list(config.SPRING_MONTHS),
        'full_year_only_past': bool(config.FULL_YEAR_ONLY_PAST) if mode == 'full_year' else None,
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'test_set_fingerprint': fingerprint,
    }
    print("=" * 75)
    print(f"📦 訓練組別 {mode}：訓練 {len(X_train)} 筆（月份 {train_months}）、測試 {len(X_test)} 筆")
    print(f"🔑 測試集指紋: {fingerprint}（兩種組別的指紋相同 = 測試集完全相同）")
    print("=" * 75)
    return X_train, X_test, y_train, y_test, feature_cols, df_clean, info


def save_test_set_ids(df_clean, test_index, target_col, output_dir):
    """另存測試集每一列的識別欄位與目標值，方便核對兩次執行的測試集。"""
    cols = [c for c in config.FILTER_ID_COLS + [config.YEAR_COL, config.MONTH_COL, target_col] if c in df_clean.columns]
    path = f"{output_dir}/test_set_ids.csv"
    df_clean.loc[test_index, list(dict.fromkeys(cols))].to_csv(path, index=False, encoding='utf-8-sig')
    print(f"📝 測試集識別資料已儲存至: {path}")
    return path