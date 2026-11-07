"""
輸出管理：建立每次執行的獨立資料夾，並儲存設定與評估結果 CSV。
"""
import os
import datetime
import pandas as pd
import config


def create_run_output_dir(root=config.RUN_OUTPUT_ROOT, tag=None):
    """建立 run_YYYYmmdd_HHMMSS[_tag] 資料夾（tag 例如訓練組別），回傳 (output_dir, timestamp)。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(root, f"run_{timestamp}" + (f"_{tag}" if tag else ""))
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 本次執行輸出目錄: {output_dir}")
    return output_dir, timestamp


def save_run_config(output_dir, timestamp, data_input_path, n_train, n_test, model_params, extra=None):
    config_data = {
        'execution_timestamp': timestamp,
        'input_data_path': str(data_input_path),
        'train_samples': n_train,
        'test_samples': n_test,
        **(extra or {}),
        **model_params,
    }
    path = os.path.join(output_dir, "run_config_and_params.csv")
    pd.DataFrame([config_data]).to_csv(path, index=False, encoding='utf-8-sig')
    print(f"✅ 模型參數與輸入檔名已紀錄至: {path}")
    return path


def save_metrics(output_dir, metrics):
    path = os.path.join(output_dir, "evaluation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(path, index=False, encoding='utf-8-sig')
    return path