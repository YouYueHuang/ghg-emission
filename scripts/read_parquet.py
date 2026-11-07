import pandas as pd

# 初始化總計變數
total_rows = 0
total_missing = None

for i in range(3, 6):
    file_path = f'../data/raw/nc_datas/data_2023/.ml-parquet/20230{i}.parquet'
    df = pd.read_parquet(file_path)

    print (df[["troposphere_ozone","ozone"]].head())
    print(f'--- 檔案: {file_path} ---')
    print(f'該檔案長度 (Rows): {len(df)}')
    
    # 計算目前這個檔案每個 column 的 NaN 數量
    missing_per_col = df.isnull().sum()
    print("各欄位 NaN 數量：")
    print(missing_per_col)
    print("-" * 30)

    # 累加總資料筆數
    total_rows += len(df)
    
    # 累加各欄位的 NaN 總數
    if total_missing is None:
        total_missing = missing_per_col
    else:
        total_missing += missing_per_col

print("==============================")
print(f'總計所有檔案的 Data 總筆數: {total_rows}')
print("所有檔案合併後的各欄位 NaN 總數：")
print(total_missing)