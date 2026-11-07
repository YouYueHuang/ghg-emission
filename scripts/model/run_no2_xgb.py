"""
XGBoost NO2 Concentration Prediction & SHAP Explanation Model (主程式)
----------------------------------------------------------------------
流程：載入資料 → 依訓練組別（春季 / 全年）建立訓練與測試集（含特徵工程、缺值過濾與報告）→ 訓練 →
      評估 → 儲存模型權重與參數 → 特徵重要性、逐輪趨勢與 SHAP 圖表

兩種訓練組別（測試集完全相同，指標可直接比較）：
    spring     只用春季（config.SPRING_MONTHS）的資料
    full_year  用全年資料訓練，但測試集 = 春季模式的測試集

模組分工：
    config.py          所有可調整設定
    data_loader.py     讀取 .parquet / .nc / CSV（由檔名加入 year、month）
    features.py        月份解析、hour/day/month 的 sin/cos 編碼、落後特徵
    data_split.py      依訓練組別（春季 / 全年）建立訓練與測試集，兩者的測試集相同
    preprocessing.py   排序、特徵選擇、重複特徵檢查、NaN 清理、切分
    filter_report.py   切分前的資料過濾報告（哪些資料被過濾、原因）
    model.py           XGBoost 建立與訓練、最終模型
    model_io.py        模型權重 / 超參數 / 中繼資料的儲存與載入
    metrics.py         評估指標 (r, R2, RMSE, MD, MAPE)
    plotting.py        XGBoost 原生特徵重要性圖
    scatter_plot.py    Ground truth vs Model 散布圖（訓練集、測試集、測試集逐小時；密度上色 + 指標）
    training_curves.py 每一輪訓練後各指標的趨勢圖
    shap_analysis.py   SHAP 圖表
    output_utils.py    輸出資料夾、設定與指標 CSV

預測並畫圖請用 predict_map.py（inference.py、map_plot.py 為其模組）。

Requirements:
    pip install xgboost scikit-learn pandas numpy matplotlib shap pyarrow
    （讀 .nc 才需要 xarray netcdf4）

用法（請勿把檔名取成 xgboost.py，會與 xgboost 套件同名衝突）：
    python run_no2_xgb.py                    # 使用 config.RUN_MODE
    python run_no2_xgb.py --mode spring      # 只用春季資料
    python run_no2_xgb.py --mode full_year   # 用全年資料訓練，測試集與春季相同
"""
import os
import argparse

import pandas as pd

import config
from data_loader import load_data
from features import first_existing
from data_split import prepare_train_test, save_test_set_ids, MODES
from model import train_model, train_final_model, get_best_round
from model_io import save_model_bundle
from metrics import calculate_evaluation_metrics, print_metrics
from plotting import plot_xgb_importance
from scatter_plot import plot_obs_vs_model, plot_by_hour
from training_curves import compute_round_metrics, plot_round_metrics
from shap_analysis import generate_shap_plots
from output_utils import create_run_output_dir, save_run_config, save_metrics


def build_meta(target_col, data_input_path, best_round, metrics, split_info):
    """寫進 model_meta.json 的資訊；預測時用來確認特徵工程設定一致。"""
    return {
        'target_col': target_col,
        'input_data_path': str(data_input_path),
        'best_round': best_round,
        'test_metrics': {k: float(v) for k, v in metrics.items()},
        'use_cyclic_time': config.USE_CYCLIC_TIME,
        'time_periods': {'hour': config.HOUR_PERIOD, 'day': config.DAY_PERIOD, 'month': config.MONTH_PERIOD},
        'lag_hours': list(config.NO2_LAG_HOURS) if config.USE_NO2_LAGS else [],
        'allow_missing_lags': config.ALLOW_MISSING_LAGS,
        **split_info,
    }


def train_no2_xgboost(data_input_path, target_col=config.TARGET_COL, mode=config.RUN_MODE):
    print("=" * 75)
    print(f"🚀 開始執行 NO2 預測模型訓練與評估流程（訓練組別: {mode}）...")
    print("=" * 75)

    # 1. 輸出資料夾（資料夾名稱帶訓練組別）
    output_dir, timestamp = create_run_output_dir(tag=mode)

    # 2. 載入資料 → 依訓練組別建立訓練/測試集（含特徵工程、缺值過濾與報告）
    df = load_data(data_input_path)
    X_train, X_test, y_train, y_test, feature_cols, df_clean, split_info = prepare_train_test(
        df, target_col, mode, output_dir)
    save_test_set_ids(df_clean, X_test.index, target_col, output_dir)

    # 3. 訓練
    model = train_model(X_train, y_train, X_test, y_test)

    # 4. 紀錄設定
    save_run_config(output_dir, timestamp, data_input_path,
                    len(X_train), len(X_test), config.MODEL_PARAMS,
                    extra={**split_info, 'train_months': str(split_info['train_months']),
                           'test_months': str(split_info['test_months']),
                           'spring_months': str(split_info['spring_months'])})

    # 5. 評估
    y_pred = model.predict(X_test)
    metrics = calculate_evaluation_metrics(y_test, y_pred)
    print_metrics(metrics)
    save_metrics(output_dir, metrics)
    # 散布圖：訓練集一張、測試集整體一張、測試集依小時（Hr）各一張
    plot_obs_vs_model(y_train, model.predict(X_train), output_dir, name="train", title_suffix="Train")
    plot_obs_vs_model(y_test, y_pred, output_dir, name="test", title_suffix="Test")
    hour_col = first_existing(df_clean.columns, config.HOUR_COL_CANDIDATES)
    if hour_col is not None:
        plot_by_hour(y_test, y_pred, df_clean.loc[X_test.index, hour_col].to_numpy(), output_dir,
                     name="test", title_label="Test")
    else:
        print(f"[警告] 找不到小時欄位（候選 {config.HOUR_COL_CANDIDATES}），略過逐小時散布圖")

    # 6. 儲存模型權重與參數（先存，避免後面的圖表出錯時模型遺失）
    best_round = get_best_round(model)
    meta = build_meta(target_col, data_input_path, best_round, metrics, split_info)
    validated_meta = {**meta, 'model_kind': 'validated (trained on train split, early stopping)',
                      'train_rows': len(X_train)}
    save_model_bundle(model, feature_cols, os.path.join(output_dir, "model_validated"), validated_meta)
    save_model_bundle(model, feature_cols, os.path.join(config.MODEL_EXPORT_DIR, mode, "latest_validated"),
                      validated_meta)
    if config.TRAIN_FINAL_MODEL:
        final_model = train_final_model(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), best_round)
        final_meta = {**meta, 'model_kind': 'final (trained on all data, fixed rounds)',
                      'train_rows': len(X_train) + len(X_test)}
        save_model_bundle(final_model, feature_cols, os.path.join(output_dir, "model_final"), final_meta)
        save_model_bundle(final_model, feature_cols, os.path.join(config.MODEL_EXPORT_DIR, mode, "latest_final"),
                          final_meta)

    # 7. 圖表（特徵重要性、逐輪指標趨勢、SHAP）
    plot_xgb_importance(model, output_dir)
    round_df = compute_round_metrics(model, X_train, y_train, X_test, y_test)
    plot_round_metrics(round_df, model, output_dir)
    generate_shap_plots(model, X_test, output_dir)
    print(f"🎉 本次執行所有結果與圖表已完整存入: {output_dir}")

    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="XGBoost NO2 訓練（春季或全年），兩者的測試集相同")
    ap.add_argument('input_path', nargs='?', default=config.DEFAULT_INPUT_PATH,
                    help='資料路徑（資料夾 / 檔案 / 萬用字元）')
    ap.add_argument('--mode', choices=MODES, default=config.RUN_MODE,
                    help="spring：只用春季資料；full_year：用全年資料訓練、測試集與 spring 相同（預設取 config.RUN_MODE）")
    args = ap.parse_args()
    train_no2_xgboost(args.input_path, mode=args.mode)