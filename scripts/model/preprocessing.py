"""
前處理：選擇特徵欄位、檢查重複特徵、清理 NaN、依順序切分訓練/測試集。
"""
import pandas as pd
import config
from features import lag_col_names, time_feature_names
from filter_report import build_filter_report, print_filter_report, save_filter_report


def sort_by_time(df):
    """依 config.SORT_COLS 排序（欄位都存在時才做），讓後續依順序切分等同於時間切分。"""
    cols = list(config.SORT_COLS or [])
    if not cols:
        return df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[資訊] 資料沒有排序欄位 {missing}，保持原始順序")
        return df
    print(f"🔃 依 {cols} 排序（之後前 {config.TRAIN_RATIO:.0%} 為訓練集、其餘為測試集）")
    return df.sort_values(cols, kind='stable').reset_index(drop=True)


def select_feature_cols(df, target_col=config.TARGET_COL):
    """
    FEATURE_COLS 有設定時，直接使用該清單（若啟用時間循環編碼 / 落後特徵，會再附加那些衍生欄位）；
    否則排除 IGNORE_COLS 自動挑選，再把 SPATIAL_COLS 加回，最後套用 EXCLUDE_FEATURES。
    """
    if config.FEATURE_COLS:
        missing = [c for c in config.FEATURE_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"❌ 資料中找不到這些特徵欄位: {missing}\n   資料實際欄位: {list(df.columns)}")
        if target_col not in df.columns:
            raise KeyError(f"❌ 資料中找不到預測目標 '{target_col}'\n   資料實際欄位: {list(df.columns)}")
        feature_cols = list(config.FEATURE_COLS)
        if config.USE_CYCLIC_TIME:
            feature_cols += time_feature_names()
        feature_cols += lag_col_names(target_col)
        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(f"❌ 這些特徵不是數值型別，XGBoost 無法使用: "
                            f"{ {c: str(df[c].dtype) for c in non_numeric} }")
        return feature_cols

    ignore_cols = set(config.IGNORE_COLS) | {target_col, config.YEAR_COL, config.ROW_ID_COL}
    if config.USE_CYCLIC_TIME:      # 已改用 sin/cos，原始的 hour/day/month 不再直接當特徵
        ignore_cols |= set(config.DAY_COL_CANDIDATES) | set(config.HOUR_COL_CANDIDATES) | {config.MONTH_COL}
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    for spatial_col in config.SPATIAL_COLS:
        if spatial_col in df.columns and spatial_col not in feature_cols:
            feature_cols.append(spatial_col)

    excluded = [c for c in feature_cols if c in config.EXCLUDE_FEATURES]
    if excluded:
        print(f"🚫 依 EXCLUDE_FEATURES 排除特徵: {excluded}")
    return [c for c in feature_cols if c not in config.EXCLUDE_FEATURES]


def report_duplicate_features(df, feature_cols, sample=100000,
                              threshold=config.DUPLICATE_CORR_THRESHOLD):
    """
    檢查特徵中是否有「名稱只差大小寫」或「幾乎完全相關」的重複欄位並印出警告
    （例如 lat 與 Lat）。重複特徵會讓樹模型把分裂與 SHAP 貢獻分散在多個欄位上。
    """
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    d = df[num_cols]
    if len(d) > sample:
        d = d.sample(sample, random_state=42)
    corr = d.corr().abs()

    found = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            if a.endswith('_sin') and b == a[:-4] + '_cos':    # 同一個週期量的 sin/cos 成對出現，不算重複
                continue
            same_name = a.lower() == b.lower()
            r = corr.loc[a, b]
            if same_name or (pd.notna(r) and r >= threshold):
                found.append((a, b, r, float((d[a] - d[b]).abs().max()), same_name))

    print("🔎 重複特徵檢查:")
    if not found:
        print("   - 沒有發現名稱只差大小寫或相關係數過高的特徵")
        return found
    for a, b, r, maxdiff, same_name in found:
        note = "（名稱只差大小寫）" if same_name else ""
        print(f"   ⚠️  '{a}' 與 '{b}'{note}: |r|={r:.6f}, 兩者最大絕對差={maxdiff:.6g}")
    print("   → 若兩者其實是同一個量，建議在 config.EXCLUDE_FEATURES 只留下其中一個")
    return found


def clean_data(df, feature_cols, target_col=config.TARGET_COL, output_dir=None, verbose=True):
    """
    移除目標或特徵含 NaN 的列（ALLOW_MISSING_LAGS=True 時，落後特徵允許缺值），
    並印出「哪些資料被過濾、原因」的報告；給定 output_dir 時另存 CSV。verbose=False 時不印也不存報告。
    """
    optional = set(lag_col_names(target_col)) if config.ALLOW_MISSING_LAGS else set()
    required = [c for c in feature_cols if c not in optional]
    checked = [target_col] + required

    na = df[checked].isna()
    drop = na.any(axis=1)
    if verbose:
        report = build_filter_report(df, na, drop, checked, target_col)
        print_filter_report(report)
        if output_dir is not None:
            save_filter_report(report, output_dir)

    return df[~drop].copy()


def split_train_test(df_clean, feature_cols, target_col=config.TARGET_COL,
                     train_ratio=config.TRAIN_RATIO):
    """依資料順序切分（不打亂）：前 train_ratio 為訓練集，其餘為測試集。"""
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    train_size = int(train_ratio * len(df_clean))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    return X_train, X_test, y_train, y_test