import os
import numpy as np
import pandas as pd

# 設定檔案目錄與路徑
dir_path = '../results/runs/run_20261111_125508_loso_spring'
pred_file = os.path.join(dir_path, 'loso_predictions.csv')
metrics_file = os.path.join(dir_path, 'loso_station_hour_metrics.csv')

# 1. 讀取 CSV 檔案
df_pred = pd.read_csv(pred_file)
df_metrics = pd.read_csv(metrics_file)

# 2. 計算 PE (Absolute Percentage Error)
# 公式：|(pred - obs) / obs| * 100
df_pred['PE'] = np.where(
    df_pred['obs'] != 0,
    np.abs((df_pred['pred'] - df_pred['obs']) / df_pred['obs']) * 100,
    np.nan,
)

# 3. 處理 metrics 檔案欄位以便進行合併對齊
# 將 station 重新命名為 siteid，保留原本大寫的 Hr 與 MAPE
df_metrics_renamed = df_metrics.rename(columns={'station': 'siteid'})

# 4. 根據 siteid 與 Hr 進行 Left Join
df_merged = pd.merge(
    df_pred,
    df_metrics_renamed[['siteid', 'Hr', 'MAPE']],
    on=['siteid', 'Hr'],
    how='left',
)

# 5. 計算 exceed_mape
# 當 MAPE <= PE 時，exceed_mape = 1，反之則為 0
df_merged['exceed_mape'] = np.where(
    df_merged['MAPE'].isna(),
    np.nan,
    np.where(df_merged['MAPE'] <= df_merged['PE'], 1, 0),
)

# 6. 將更新後的 DataFrame 寫回原本的 loso_predictions.csv 檔案
df_merged.to_csv(pred_file, index=False, encoding='utf-8-sig')

print(f'處理完成！已成功更新並寫回：{pred_file}')