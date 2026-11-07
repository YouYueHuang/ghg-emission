"""
用訓練好的模型，對指定 nc 檔的某一天、某一小時預測 NO2，並畫出全台灣濃度分佈圖。

用法（請從本資料夾執行）：
    python predict_map.py --nc ../../data/raw/nc_datas/data_2023/Merged/model/202305.nc --day 22 --hour 12 --model ../../results/models/latest_validated
    python predict_map.py --nc xxx.nc --day 15 --hour 6 --model ../../results/models/latest_validated

nc 檔需包含模型訓練時用的所有特徵，且（因為模型用前 1、2 小時的 NO2 預測下一小時）
需包含該時間點前幾小時的 NO2 觀測；檔名要含 YYYYMM（用來取得月份）。
day / hour 的值要對應資料中 day、Hr（或 hour）欄位的實際數值。
"""
import os
import argparse

import config
from data_loader import load_nc_slice
from features import add_time_features, add_no2_lags, lag_col_names
from inference import select_day_hour, predict_slice
from map_plot import plot_no2_map
from model_io import load_model_bundle


def main():
    ap = argparse.ArgumentParser(description="指定 nc 檔、day、hour，預測並繪製 NO2 濃度圖")
    ap.add_argument('--nc', default=config.PREDICT_NC_PATH, help='含特徵的 nc 檔路徑（檔名需含 YYYYMM）')
    ap.add_argument('--day', type=float, default=config.PREDICT_DAY, help='day 欄位的值')
    ap.add_argument('--hour', type=float, default=config.PREDICT_HOUR, help='hour (Hr) 欄位的值')
    ap.add_argument('--model', default=config.DEFAULT_MODEL_DIR, help='模型資料夾（內含 model.json）')
    ap.add_argument('--out', default=config.MAP_OUTPUT_DIR, help='圖檔輸出資料夾')
    args = ap.parse_args()
    if args.nc is None or args.day is None or args.hour is None:
        ap.error("請提供 --nc、--day、--hour（或在 config.py 設定 PREDICT_NC_PATH / PREDICT_DAY / PREDICT_HOUR）")

    print("=" * 75)
    print(f"🗺️  預測並繪製 NO2 濃度圖：{args.nc}  day={args.day:g}, hour={args.hour:g}")
    print("=" * 75)

    model, meta = load_model_bundle(args.model)
    feature_cols = meta['feature_cols']
    target_col = meta.get('target_col', config.TARGET_COL)
    lag_hours = meta.get('lag_hours', [])
    print(f"📦 已載入模型: {args.model}（{meta.get('model_kind', '?')}，特徵數 {len(feature_cols)}）")

    # 特徵工程設定要與訓練時一致
    if meta.get('use_cyclic_time') != config.USE_CYCLIC_TIME or \
            meta.get('time_periods') not in (None, {'hour': config.HOUR_PERIOD, 'day': config.DAY_PERIOD,
                                                  'month': config.MONTH_PERIOD}):
        print("[警告] config.py 的時間編碼設定與模型訓練時不同，預測結果可能不正確")
    if list(lag_hours) != (list(config.NO2_LAG_HOURS) if config.USE_NO2_LAGS else []):
        print(f"[警告] config 的落後設定與模型不同，改用模型訓練時的 lag_hours={lag_hours}")

    df = load_nc_slice(args.nc, day=args.day)
    df = add_time_features(df)
    if lag_hours and target_col not in df.columns:
        raise SystemExit(f"❌ nc 檔中沒有 '{target_col}'，無法取得前 {lag_hours} 小時的 NO2 來預測")
    df = add_no2_lags(df, target_col, lags=lag_hours)

    df_slice = select_day_hour(df, args.day, args.hour)
    if df_slice.empty:
        raise SystemExit(1)

    optional = lag_col_names(target_col, lag_hours) if meta.get('allow_missing_lags') else ()
    pred_df, lat_col, lon_col = predict_slice(model, feature_cols, df_slice, optional_cols=optional)

    tag = os.path.splitext(os.path.basename(args.nc))[0]
    out_png = os.path.join(args.out, f"NO2_pred_{tag}_day{args.day:g}_hour{args.hour:g}.png")
    title = f"Predicted NO2 - {tag}  day={args.day:g}, hour={args.hour:g}"
    plot_no2_map(pred_df, lat_col, lon_col, title, out_png)


if __name__ == "__main__":
    main()