"""
切分前的資料過濾報告：哪些資料被濾掉、為什麼。

目前切分前唯一會「丟掉資料列」的步驟是缺值過濾（目標或特徵任一為 NaN）。這個模組把它拆開說明：
  1. 依序過濾表（waterfall）：每一列只算在「第一個」造成它被移除的原因，各原因的列數相加 = 總移除列數
  2. 各欄位缺值表：該欄缺值的總列數（一列可同時缺多欄）、以及「只有這一欄缺值」而被移除的列數
  3. 各月份表：每個月被移除的比例
  4. 被過濾掉的列（識別欄位 + 完整原因），另存 CSV
"""
import os
import numpy as np
import pandas as pd
import config

REPORT_DIRNAME = "data_filtering"


def build_filter_report(df, na, drop, checked_cols, target_col):
    """
    df: 過濾前的資料；na: 各檢查欄位是否缺值的布林 DataFrame（欄位 = checked_cols）；
    drop: 各列是否被移除的布林 Series。回傳 dict，內含各表與被過濾列。
    """
    total = len(df)
    kept = int((~drop).sum())

    # 1. 依序過濾（waterfall）
    alive = np.ones(total, dtype=bool)
    rows = [{'step': 0, 'reason': '原始資料（載入、合併後）', 'rows_removed': 0, 'rows_remaining': total}]
    for i, c in enumerate(checked_cols, 1):
        hit = na[c].to_numpy() & alive
        alive &= ~hit
        label = f"目標 {c} 缺值" if c == target_col else f"特徵 {c} 缺值"
        rows.append({'step': i, 'reason': label, 'rows_removed': int(hit.sum()), 'rows_remaining': int(alive.sum())})
    rows.append({'step': len(checked_cols) + 1, 'reason': '最終保留（進入訓練/測試切分）',
                 'rows_removed': 0, 'rows_remaining': kept})
    summary = pd.DataFrame(rows)

    # 2. 各欄位缺值
    only_this = na[na.sum(axis=1) == 1].sum(axis=0)
    by_col = pd.DataFrame({
        'column': checked_cols,
        'role': ['target' if c == target_col else 'feature' for c in checked_cols],
        'rows_missing': [int(na[c].sum()) for c in checked_cols],
        'pct_missing': [round(float(na[c].mean()) * 100, 2) for c in checked_cols],
        'rows_removed_only_by_this': [int(only_this[c]) for c in checked_cols],
    }).sort_values('rows_missing', ascending=False).reset_index(drop=True)

    # 3. 各月份
    gcols = [c for c in (config.YEAR_COL, config.MONTH_COL) if c in df.columns]
    by_month = None
    if gcols:
        tmp = df[gcols].assign(_drop=drop.to_numpy())
        by_month = tmp.groupby(gcols)['_drop'].agg(rows_total='size', rows_removed='sum').reset_index()
        by_month['rows_kept'] = by_month['rows_total'] - by_month['rows_removed']
        by_month['pct_removed'] = (by_month['rows_removed'] / by_month['rows_total'] * 100).round(2)

    # 4. 被過濾的列（識別欄位 + 完整原因）
    id_cols = [c for c in config.FILTER_ID_COLS + [config.YEAR_COL, config.MONTH_COL] if c in df.columns]
    id_cols = list(dict.fromkeys(id_cols))
    dropped = df.loc[drop, id_cols].copy()
    na_d = na.loc[drop]
    reason = pd.Series('', index=dropped.index, dtype=object)
    for c in checked_cols:
        reason = reason + pd.Series(np.where(na_d[c].to_numpy(), c + '; ', ''), index=dropped.index)
    dropped['reason'] = reason.str[:-2]
    if not id_cols:
        dropped.insert(0, 'row_index', dropped.index)

    # 5. 不會被 dropna 處理、但值得提醒的 inf
    numeric = [c for c in checked_cols if pd.api.types.is_numeric_dtype(df[c])]
    inf_counts = {c: int(n) for c, n in np.isinf(df[numeric]).sum().items() if n}

    return {'total': total, 'kept': kept, 'summary': summary, 'by_col': by_col,
            'by_month': by_month, 'dropped_rows': dropped, 'inf_counts': inf_counts}


def print_filter_report(rep):
    total, kept = rep['total'], rep['kept']
    removed = total - kept
    print("=" * 75)
    print("🧹 切分前資料過濾報告")
    print("=" * 75)
    print(f"原始 {total} 列 → 保留 {kept} 列（移除 {removed} 列，{removed / max(total, 1) * 100:.1f}%）")

    print("\n[依序過濾] 每一列只計入第一個造成它被移除的原因（順序：目標、再依特徵順序）")
    for _, r in rep['summary'].iterrows():
        if r['step'] == 0 or r['rows_removed'] > 0 or r['reason'].startswith('最終'):
            sign = f"-{r['rows_removed']}" if r['rows_removed'] else ""
            print(f"   {r['reason']:<40} {sign:>10}   剩 {r['rows_remaining']}")

    miss = rep['by_col'][rep['by_col']['rows_missing'] > 0]
    print("\n[各欄位缺值]（一列可同時缺多欄，所以各欄的缺值列數加總會大於移除列數）")
    if miss.empty:
        print("   沒有任何欄位有缺值")
    else:
        print(f"   {'欄位':<24}{'缺值列數':>10}{'比例':>9}{'僅此欄缺值而被移除':>20}")
        for _, r in miss.iterrows():
            print(f"   {r['column']:<24}{r['rows_missing']:>10}{r['pct_missing']:>8.1f}%{r['rows_removed_only_by_this']:>20}")

    if rep['by_month'] is not None:
        print("\n[各月份]")
        for _, r in rep['by_month'].iterrows():
            ym = "-".join(str(int(r[c])) for c in (config.YEAR_COL, config.MONTH_COL) if c in rep['by_month'].columns)
            print(f"   {ym:<10} 共 {int(r['rows_total'])} 列，移除 {int(r['rows_removed'])}（{r['pct_removed']:.1f}%），保留 {int(r['rows_kept'])}")

    if rep['inf_counts']:
        print(f"\n[警告] 以下欄位含 inf（不會被缺值過濾移除，XGBoost 可能因此出錯）: {rep['inf_counts']}")
    print("\n※ 切分前只做缺值過濾；不做數值範圍、離群值、品質旗標、重複列、時段/測站篩選。")
    print("=" * 75)


def save_filter_report(rep, output_dir):
    out = os.path.join(output_dir, REPORT_DIRNAME)
    os.makedirs(out, exist_ok=True)
    rep['summary'].to_csv(os.path.join(out, "filter_summary.csv"), index=False, encoding='utf-8-sig')
    rep['by_col'].to_csv(os.path.join(out, "filter_by_column.csv"), index=False, encoding='utf-8-sig')
    if rep['by_month'] is not None:
        rep['by_month'].to_csv(os.path.join(out, "filter_by_month.csv"), index=False, encoding='utf-8-sig')
    if config.SAVE_FILTERED_ROWS:
        rep['dropped_rows'].to_csv(os.path.join(out, "filtered_rows.csv"), index=False, encoding='utf-8-sig')
    print(f"📝 過濾報告已儲存至: {out}")
    return out