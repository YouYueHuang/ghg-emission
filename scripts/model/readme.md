# no2_xgb：地面 NO2 的 XGBoost 預測與 SHAP 解釋

用衛星、氣象與土地利用特徵預測地面測站的 NO2 濃度（`NO2_gnd`，單位 ppb），並輸出評估指標、散布圖、逐輪趨勢圖與 SHAP 圖，同時可指定 nc 檔的某一天、某一小時畫全台灣的預測濃度圖。

---

## 1. 環境

```bash
pip install xgboost scikit-learn shap pandas numpy scipy matplotlib pyarrow
pip install xarray netcdf4 geopandas     # 選用：讀 .nc、預測全台濃度圖（疊縣市邊界）才需要
```

- 讀 `.parquet` 需要 `pyarrow`。
- 圖表若出現中文變成方框，請把相關腳本的 `font.sans-serif` 改成系統有的中文字型。

## 2. 資料夾結構

所有路徑都是相對路徑，**請在 `no2_xgb/` 內執行**。這個資料夾裡的檔案必須放在一起。

```
project/
├─ <上層>/no2_xgb/                       # 程式所在（工作目錄）
├─ data/raw/nc_datas/
│  ├─ data_2023/.ml-parquet/             # 訓練資料：202303.parquet、202304.parquet、202305.parquet
│  └─ ground-info/COUNTY_MOI_1140318.shp # 縣市邊界（預測全台濃度圖時使用）
└─ results/
   ├─ runs/run_YYYYmmdd_HHMMSS/          # 每次訓練的輸出
   ├─ models/latest_validated、latest_final/
   └─ maps/                              # 預測全台濃度圖
```

---

## 3. 模組分工

| 檔案 | 內容 |
|---|---|
| `run_no2_xgb.py` | 主程式，依序呼叫各模組 |
| `config.py` | 所有可調整設定 |
| `data_loader.py` | 讀取 `.parquet`、`.nc`、CSV，並由檔名（如 `202303.parquet`）加入 `year`、`month` 欄位 |
| `features.py` | 月份解析、hour/day/month 的 sin/cos 編碼、落後特徵（訓練與預測共用） |
| `preprocessing.py` | 排序、特徵選擇、重複特徵檢查、缺值清理、訓練/測試切分 |
| `filter_report.py` | 切分前的資料過濾報告（哪些資料被過濾、原因） |
| `model.py` | XGBoost 建立與訓練、最終模型（全部資料重訓） |
| `model_io.py` | 模型權重、超參數、中繼資料的儲存與載入 |
| `metrics.py` | r、R²、RMSE、MD、MAPE |
| `plotting.py` | XGBoost 原生特徵重要性圖 |
| `scatter_plot.py` | Ground truth vs Model 散布圖（訓練集、測試集、測試集逐小時） |
| `training_curves.py` | 每一輪訓練後各指標的趨勢圖 |
| `shap_analysis.py` | SHAP 圖表 |
| `output_utils.py` | 輸出資料夾、設定與指標 CSV |
| `predict_map.py` | 指定 nc 檔、day、hour，預測並畫出全台 NO2 濃度圖 |
| `inference.py`、`map_plot.py` | `predict_map.py` 使用的推論與繪圖模組 |

---

## 4. 訓練

```bash
python run_no2_xgb.py
python run_no2_xgb.py ../../data/raw/nc_datas/data_2023/.ml-parquet     # 或指定輸入
```

輸入可以是含多個 `.parquet`（或 `.nc`）的資料夾（依檔名排序後合併）、萬用字元（如 `"../data/*.parquet"`）、單一檔案或 `.csv`。不帶參數時使用 `config.py` 的 `DEFAULT_INPUT_PATH`。資料夾內 `.parquet` 與 `.nc` 並存時，請在 `DATA_FORMAT` 指定 `'parquet'` 或 `'nc'`。

### 資料與特徵

- 預測目標：`NO2_gnd`（地面測站 NO2，ppb）。
- 輸入特徵由 `FEATURE_COLS` 明確指定：`Hr, U, V, Temp, Hum, PBLH, SWD, troposphere_ozone, ozone, no2, VIIRS, NDVI, Industry, Road total length, Business, Residential, MM, day`。其中 `no2` 是**衛星** NO2（輸入），`NO2_gnd` 才是預測目標。
- 清單中的欄位缺少或不是數值型別時，程式會直接報錯並列出資料的實際欄位。`FEATURE_COLS = None` 則改回自動挑選（依 `IGNORE_COLS`、`SPATIAL_COLS`、`EXCLUDE_FEATURES`）。

### 切分前的資料處理與過濾

依執行順序：

1. **載入**：讀入並合併所有檔案，不篩選任何列。
2. **排序**：依 `SORT_COLS`（預設 `monitordate`、`Hr`）排序，只改順序；設為 `[]` 則保持原始順序。
3. **特徵工程**：時間循環編碼與落後特徵，預設都關閉（見第 6 節）。
4. **缺值過濾**：目標或任一輸入特徵為 NaN 的列會被移除。**這是切分前唯一會丟掉資料列的步驟。**
5. **重複特徵檢查**：只印警告，不改資料。

之後才切分：取排序後的前 80%（`TRAIN_RATIO`）訓練、其餘測試。

**過濾報告**：每次執行會在終端機印出「哪些資料被過濾、原因」，並存到 `data_filtering/`：

| 檔案 | 內容 |
|---|---|
| `filter_summary.csv` | 依序過濾表：每一列只計入第一個造成它被移除的原因（順序為目標、再依特徵順序），各原因的列數相加等於總移除列數 |
| `filter_by_column.csv` | 各欄位缺值：缺值列數與比例（一列可同時缺多欄），以及「只有這一欄缺值而被移除」的列數 |
| `filter_by_month.csv` | 各月份被移除的列數與比例 |
| `filtered_rows.csv` | 被過濾掉的每一列：識別欄位（`FILTER_ID_COLS`，預設 `siteid`、`sitename`、`monitordate`、`Hr`，加上 `year`、`month`）與完整原因（例如 `no2; Temp`）；`SAVE_FILTERED_ROWS = False` 可關閉 |

報告也會提醒資料中的 `inf`（不會被缺值過濾移除）。**目前不做**數值範圍、離群值、品質旗標、重複列、時段或測站篩選。

---

## 5. 設定（`config.py`）

| 參數 | 說明 |
|---|---|
| `DEFAULT_INPUT_PATH`、`DATA_FORMAT` | 預設輸入路徑；`'auto'` 依副檔名判斷，或指定 `'parquet'` / `'nc'` |
| `TARGET_COL`、`FEATURE_COLS` | 預測目標與輸入特徵清單 |
| `IGNORE_COLS`、`SPATIAL_COLS`、`EXCLUDE_FEATURES` | 自動挑選特徵時的排除與加回規則（`FEATURE_COLS` 有設定時不使用） |
| `DUPLICATE_CORR_THRESHOLD` | 特徵相關係數 ≥ 此值（或名稱只差大小寫）時印出重複特徵警告 |
| `SORT_COLS` | 切分前的排序欄位 |
| `SAVE_FILTERED_ROWS`、`FILTER_ID_COLS` | 過濾報告是否另存被過濾的列，以及用哪些欄位識別 |
| `TRAIN_RATIO` | 訓練集比例（預設 0.8） |
| `FILENAME_YM_REGEX`、`YEAR_COL`、`MONTH_COL`、`DEFAULT_YEAR` | 由檔名取得年月的規則與欄位名稱 |
| `USE_CYCLIC_TIME`、`HOUR_PERIOD`、`DAY_PERIOD`、`MONTH_PERIOD` | 是否使用 sin/cos 時間編碼與各週期 |
| `USE_NO2_LAGS`、`NO2_LAG_HOURS`、`ALLOW_MISSING_LAGS` | 落後特徵：是否使用、落後幾小時、落後值缺失時是否保留樣本 |
| `MODEL_PARAMS` | XGBoost 超參數 |
| `TRAIN_FINAL_MODEL`、`MODEL_EXPORT_DIR`、`DEFAULT_MODEL_DIR` | 是否另訓練最終模型、模型匯出位置、預測時預設使用的模型 |
| `CURVE_EVAL_STEP`、`CURVE_MAX_POINTS`、`CURVE_TRAIN_MAX_SAMPLES` | 逐輪趨勢圖的取樣間隔、最多點數、訓練集抽樣上限 |
| `SCATTER_TITLE`、`SCATTER_XLABEL`、`SCATTER_UNIT` | 散布圖標題、橫軸名稱、單位（預設 `ppb`） |
| `SCATTER_MIN_POINTS_PER_HOUR` | 逐小時散布圖要求的最少樣本數（預設 20） |
| `SCATTER_DENSITY_BINS`、`SCATTER_DENSITY_SMOOTH`、`SCATTER_MAX_PLOT_POINTS` | 局部點密度的格數、平滑強度，以及超過此點數時只抽樣繪圖（指標仍用全部樣本） |
| `SHAP_SAMPLE_SIZE`、`SHAP_DEPENDENCE_FEATURE` | SHAP 抽樣筆數與 dependence plot 的特徵 |
| `PREDICT_*`、`MAP_*`、`SHP_PATH` | 預測全台濃度圖的預設輸入、色階、輸出位置與縣市邊界路徑 |
| `RUN_OUTPUT_ROOT` | 輸出根目錄（預設 `../../results/runs`） |

---

## 6. 特徵工程選項（預設關閉）

目前特徵只用 `FEATURE_COLS`。以下兩項可選用：

1. **sin/cos 時間編碼**：`USE_CYCLIC_TIME = True` 會把 hour、day、month 各轉成 `*_sin`、`*_cos`（角度 = 2π × 值 ÷ 週期），共 6 個特徵，附加在 `FEATURE_COLS` 之後。啟用時請同時把 `Hr`、`MM`、`day` 從 `FEATURE_COLS` 拿掉，否則模型會同時看到原始值與 sin/cos。`HOUR_PERIOD` 預設 24，若 `Hr` 不是一天 24 小時的小時數請修改。month 取自檔名（`202303.parquet` 得到 3）。
2. **落後特徵**：`USE_NO2_LAGS = True` 會為每個像素或測站（`lat`、`lon`）加入目標值前 1、2 小時的落後特徵（欄位名為 `<目標欄位>_lag1`、`<目標欄位>_lag2`），依「年月日＋小時」計算絕對時間，所以會處理跨日。落後值缺失的樣本預設會被缺值過濾移除（樣本會明顯減少）；`ALLOW_MISSING_LAGS = True` 則保留，由 XGBoost 處理缺值。

---

## 7. 輸出

每次訓練會建立 `RUN_OUTPUT_ROOT/run_YYYYmmdd_HHMMSS/`，內含：

- `run_config_and_params.csv`：輸入路徑、樣本數與模型參數
- `data_filtering/`：切分前的資料過濾報告（見第 4 節）
- `evaluation_metrics.csv`：測試集的 r、R²、RMSE、MD、MAPE
- `ground_truth_vs_model_train.png`：訓練集散布圖（橫軸實測、縱軸預測，點依局部密度上色，左上角標示 r、R²、RMSE、MD、MAPE、N）
- `ground_truth_vs_model_test.png`：測試集整體散布圖
- `Scatter_by_hour/`：測試集依小時（`Hr`）各一張散布圖（所有小時共用同一組座標範圍）與各小時指標的 `metrics_by_hour_test.csv`；樣本太少的小時會略過
- `xgboost_feature_importance.png`：XGBoost 原生特徵重要性
- `Training_Curves/`：每個指標一張逐輪趨勢圖（`curve_r.png`、`curve_R2.png`、`curve_RMSE.png`、`curve_MD.png`、`curve_MAPE.png`）、總覽圖 `curves_overview.png` 與逐輪數值 `round_metrics.csv`
- `SHAP_Analysis/`：`shap_summary_beeswarm.png`、`shap_feature_importance_bar.png`、`shap_dependence_<特徵>.png`
- `model_validated/`、`model_final/`：模型權重與參數（見下）

### 儲存的模型

**每次訓練完都會儲存模型的權重與參數**。每個模型資料夾包含：

- `model.json`：權重（所有樹的結構與葉節點數值）
- `model_params.json`：超參數（sklearn 介面參數與 XGBoost booster 的完整設定）
- `model_meta.json`：特徵欄位與順序、最佳輪次、測試集指標、時間編碼與落後特徵設定

| 資料夾 | 說明 |
|---|---|
| `model_validated/` | 只用訓練集訓練、以測試集 early stopping 的模型；其測試指標就是 `evaluation_metrics.csv` |
| `model_final/` | 用全部資料（訓練＋測試）、固定在最佳輪次重訓的最終模型（`TRAIN_FINAL_MODEL = False` 可關閉） |

兩者各自存在該次的 `run_時間戳/` 內（不會被覆蓋），同時複製到 `MODEL_EXPORT_DIR`（預設 `../../results/models`）的 `latest_validated/`、`latest_final/`（每次覆蓋，供預測時引用）。最終模型沒有未參與訓練的資料可評估，成效以 validated 模型的指標為準。

---

## 8. 指定時間點預測並畫全台濃度圖

```bash
python predict_map.py --nc <含特徵的 nc 檔> --day 15 --hour 6
```

- `--nc`：含模型特徵的 `.nc` 檔，欄位名稱需與訓練特徵一致（缺少時會列出缺了哪些）；檔名要含 `YYYYMM`
- `--day`、`--hour`：對應資料中 `day`、`Hr`（或 `hour`）欄位的**實際數值**；找不到時會印出可用的值
- `--model`：模型資料夾，預設 `latest_final`；想用 validated 模型可指定 `../../results/models/latest_validated`
- 輸出：`../../results/maps/NO2_pred_<檔名>_day<d>_hour<h>.png`（疊縣市邊界）

任一特徵缺值的像素不預測（訓練時同樣排除了缺值列），會留白。若該時間點沒有任何像素能預測，程式會印出各特徵的缺值數並正常結束，不會當掉。若模型使用了落後特徵，nc 檔還需包含前幾小時的目標變數觀測。

---

## 9. 使用上的注意

1. 不要把主程式命名為 `xgboost.py`，會與 xgboost 套件同名而導致匯入錯誤。
2. `eval_set` 包含測試集，early stopping 是看測試集誤差決定停止輪數，最終指標會略微樂觀。
3. 切分是同一批測站的時間切分：測試集的測站也出現在訓練集裡，成效代表「已知測站的未來時間」，不代表能推廣到沒有測站的地方。
4. 逐輪趨勢圖是用「前 k 棵樹」重新預測得到的，輪數多、資料量大時較慢，可調大 `CURVE_EVAL_STEP` 或調小 `CURVE_TRAIN_MAX_SAMPLES`。

## 10. 常見問題

| 狀況 | 處理方式 |
|---|---|
| 讀 parquet 出現 `ImportError` | 執行 `pip install pyarrow` |
| `AttributeError: module 'config' has no attribute ...` | 你的 `config.py` 缺少新設定，請以最新版 `config.py` 為準，或補上缺少的項目 |
| 「找不到這些特徵欄位」 | 對照訊息中列出的資料實際欄位，修改 `FEATURE_COLS` |
| 保留的樣本很少 | 看 `data_filtering/filter_by_column.csv`，通常是衛星 `no2` 缺值（雲）造成 |
| 預測全台濃度圖時沒有任何預測值 | 看終端機列出的各特徵缺值數；落後特徵缺值時，請選前幾小時有觀測的 hour |
| 縣市邊界和資料錯位 | shapefile 缺少 `.prj` 時假設為 EPSG:3824，請查明正確座標系後修改 `map_plot.py` 的 `set_crs(epsg=...)` |
| 找不到檔案 | 確認是從 `no2_xgb/` 內執行，相對路徑才會對 |