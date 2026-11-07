import pandas as pd

input_filename = '../data/raw/nc_datas/data_2023/.station-raw/空氣品質監測小時值資料(一般污染物,每日更新) (2023-05).csv'
output_filename = '../data/raw/nc_datas/data_2023/.station-raw/no2_monitor_data_202305.csv'

# 讀取完整資料
df = pd.read_csv(input_filename)

# 篩選 NO2 資料（比對 itemname 或 itemengname）
no2_df = df[(df['itemname'].str.upper() == 'NO2') | (df['itemengname'].str.upper() == 'NO2')]

# 匯出為新的 CSV 檔案
no2_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"成功篩選出 {len(no2_df)} 筆 NO2 資料，並已儲存至 {output_filename}")