"""
留一測站交叉驗證 (Leave-One-Station-Out) 主程式。

每次把一個測站（依 siteid 或 sitename）整個移除，用其餘測站訓練，預測被移除的測站，
並以時間序列呈現預測與實測。輸出：各測站時間序列圖、各測站指標、空間分佈圖、逐筆預測 CSV。

用法（請在 no2_xgb/ 內執行）：
    python run_loso.py                                   # 使用 config.RUN_MODE、config.LOSO_STATION_COL
    python run_loso.py --mode full_year                  # 用全年資料
    python run_loso.py --station-col sitename            # 以 sitename 區分測站
    python run_loso.py --stations 1,5,12                 # 只做指定測站（先小規模試跑）
    python run_loso.py --params latest                   # 使用先前 Optuna 搜尋的最佳參數（固定樹數）

    python run_loso.py --stations 15,31,42,53   # 只做指定測站，建議先試跑
"""
import os
import argparse

import config
from data_loader import load_data
from data_split import MODES
from loso import prepare_loso_data, run_loso, save_loso_results
from loso_plots import plot_all_timeseries, plot_station_metrics, plot_station_map
from output_utils import create_run_output_dir
from scatter_plot import plot_obs_vs_model


def main():
    ap = argparse.ArgumentParser(description="留一測站交叉驗證：移除單一測站後預測該測站，並以時間序列呈現")
    ap.add_argument('input_path', nargs='?', default=config.DEFAULT_INPUT_PATH, help='資料路徑（資料夾 / 檔案 / 萬用字元）')
    ap.add_argument('--mode', choices=MODES, default=config.RUN_MODE,
                    help="spring：只用春季月份；full_year：用全部月份（預設取 config.RUN_MODE）")
    ap.add_argument('--station-col', default=config.LOSO_STATION_COL, help="區分測站的欄位：siteid 或 sitename")
    ap.add_argument('--stations', default=None, help="只做這些測站（以測站欄位的值，逗號分隔）；預設全部")
    ap.add_argument('--params', default=None, metavar='PATH|latest',
                    help="使用 Optuna 的最佳參數（best_params.json 路徑或 'latest'）；預設用 config.MODEL_PARAMS")
    ap.add_argument('--min-rows', type=int, default=None, help='測站最少資料列數（預設 config.LOSO_MIN_STATION_ROWS）')
    ap.add_argument('--val-frac', type=float, default=None, help='early stopping 的驗證測站比例（預設 config.LOSO_VAL_STATION_FRAC）')
    args = ap.parse_args()

    output_dir, _ = create_run_output_dir(tag=f"loso_{args.mode}")
    df = load_data(args.input_path)
    d_clean, feature_cols = prepare_loso_data(df, config.TARGET_COL, args.mode, args.station_col, output_dir)

    params = config.MODEL_PARAMS
    if args.params:
        from tuning import load_tuned_params
        params, _ = load_tuned_params(args.params, args.mode, feature_cols)

    stations = [s.strip() for s in args.stations.split(',')] if args.stations else None
    preds, metrics_df, summary = run_loso(d_clean, feature_cols, config.TARGET_COL, params,
                                          station_col=args.station_col, stations=stations,
                                          min_rows=args.min_rows, val_frac=args.val_frac)

    save_loso_results(preds, metrics_df, summary, output_dir, {
        'mode': args.mode, 'station_col': args.station_col, 'input_path': args.input_path,
        'features': feature_cols, 'params': params, 'tuned_params': bool(args.params),
        'min_rows': args.min_rows or config.LOSO_MIN_STATION_ROWS,
        'val_station_frac': args.val_frac if args.val_frac is not None else config.LOSO_VAL_STATION_FRAC,
        'seed': config.LOSO_SEED, 'selected_stations': stations,
    })
    plot_obs_vs_model(preds['obs'], preds['pred'], output_dir, name="loso", title_suffix="Leave-one-station-out")
    plot_station_metrics(metrics_df, output_dir)
    plot_station_map(metrics_df, output_dir, 'R2')
    plot_station_map(metrics_df, output_dir, 'RMSE')
    plot_all_timeseries(preds, metrics_df, output_dir, args.station_col)
    print(f"🎉 LOSO 實驗完成，結果已存入: {output_dir}")


if __name__ == "__main__":
    main()