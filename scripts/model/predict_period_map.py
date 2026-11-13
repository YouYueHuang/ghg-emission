"""
用已訓練好的模型，預測全台灣網格在某段期間內每個小時的 NO2，平均後輸出一張濃度分佈圖。
（訓練完成後會自動執行，見 config.PERIOD_MAP_ENABLED；這支腳本讓你事後用已儲存的模型重畫、或改期間。）

用法（請在 no2_xgb/ 內執行）：
    python predict_period_map.py                                   # 用預設模型、模型訓練時的期間
    python predict_period_map.py --model ../../results/models/full_year/latest_final
    python predict_period_map.py --start "2023-03-01 08:00" --end "2023-05-31 15:00"
    python predict_period_map.py --nc ../../data/raw/nc_datas/data_2023/Merged/model
"""
import os
import argparse
import pandas as pd

import config
from features import lag_col_names
from model_io import load_model_bundle
from period_map import make_period_mean_map


def main():
    ap = argparse.ArgumentParser(description="預測全台網格期間內每個小時的 NO2，平均後畫一張分佈圖")
    ap.add_argument('--model', default=config.DEFAULT_MODEL_DIR, help='模型資料夾（內含 model.json）')
    ap.add_argument('--nc', default=config.GRID_NC_PATH, help='含全台網格特徵的 nc（資料夾 / 檔案 / 萬用字元）')
    ap.add_argument('--start', default=None, help='期間起點，例如 "2023-03-01 08:00"（預設取模型訓練期間）')
    ap.add_argument('--end', default=None, help='期間終點（預設取模型訓練期間）')
    ap.add_argument('--out', default=config.MAP_OUTPUT_DIR, help='輸出資料夾')
    args = ap.parse_args()

    model, meta = load_model_bundle(args.model)
    start = pd.Timestamp(args.start or meta.get('train_period_start') or '')
    end = pd.Timestamp(args.end or meta.get('train_period_end') or '')
    if pd.isna(start) or pd.isna(end):
        ap.error("模型的 meta 沒有訓練期間，請用 --start 與 --end 指定")
    optional = lag_col_names(meta.get('target_col', config.TARGET_COL), meta.get('lag_hours', [])) \
        if meta.get('allow_missing_lags') else ()
    png, _ = make_period_mean_map(model, meta['feature_cols'], args.nc, start, end, args.out,
                                  optional_cols=optional, label=f"({os.path.basename(os.path.normpath(args.model))})")
    print(f"🎉 完成: {png}")


if __name__ == "__main__":
    main()