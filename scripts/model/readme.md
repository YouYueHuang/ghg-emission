# no2_xgb：地面 NO2 的 XGBoost 預測與 SHAP 解釋

用衛星、氣象與土地利用特徵預測地面測站的 NO2 濃度（`NO2_gnd`，單位 ppb），並輸出評估指標、散布圖、逐輪趨勢圖與 SHAP 圖，同時可指定 nc 檔的某一天、某一小時畫全台灣的預測濃度圖。

---

## 1. 環境

```bash
pip install xgboost scikit-learn shap pandas numpy scipy matplotlib pyarrow optuna
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
| `data_split.py` | 依訓練組別（春季 / 全年）建立訓練與測試集，兩者的測試集完全相同 |
| `preprocessing.py` | 排序、特徵選擇、重複特徵檢查、缺值清理、訓練/測試切分 |
| `filter_report.py` | 切分前的資料過濾報告（哪些資料被過濾、原因） |
| `model.py` | XGBoost 建立與訓練、最終模型（全部資料重訓） |
| `tuning.py` | Optuna 超參數搜尋（時間序列交叉驗證、TPE、剪枝） |
| `run_loso.py` | 留一測站交叉驗證（Leave-One-Station-Out）的主程式 |
| `loso.py`、`loso_plots.py` | 留一測站的資料準備、逐測站訓練與預測，以及時間序列圖等圖表 |
| `tuning_report.py` | Optuna 搜尋報告（策略、過程、結果與圖表） |
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
python run_no2_xgb.py                    # 使用 config.RUN_MODE（預設 spring）
python run_no2_xgb.py --mode spring      # 只用春季資料訓練
python run_no2_xgb.py --mode full_year   # 用全年資料訓練，測試集與春季完全相同
python run_no2_xgb.py ../../data/raw/nc_datas/data_2023/.ml-parquet --mode full_year   # 也可指定輸入
```

輸入可以是含多個 `.parquet`（或 `.nc`）的資料夾（依檔名排序後合併）、萬用字元（如 `"../data/*.parquet"`）、單一檔案或 `.csv`。不帶參數時使用 `config.py` 的 `DEFAULT_INPUT_PATH`。資料夾內 `.parquet` 與 `.nc` 並存時，請在 `DATA_FORMAT` 指定 `'parquet'` 或 `'nc'`。

### 訓練組別：春季或全年

| 組別 | 訓練資料 | 測試資料 |
|---|---|---|
| `spring` | 春季（`SPRING_MONTHS`，預設 3、4、5 月）的資料，時間上前 80% | 春季時間上最後的 20% |
| `full_year` | 全年所有資料，**扣除測試集** | **與 `spring` 完全相同的那批春季測試列** |

兩種組別的測試集相同，所以測試指標可以直接比較。做法是 `full_year` 也先用春季資料依 `spring` 的方式定義測試集，再從全年資料中取出同一批列當測試集，其餘全部拿去訓練。

- 用 `--mode` 或 `config.RUN_MODE` 選擇組別。兩種組別可以共用同一個資料夾：`.ml-parquet` 放入全年檔案後，`spring` 會自動只取春季月份。
- **核對測試集**：每次執行會印出並記錄「測試集指紋」（`run_config_and_params.csv`、`model_meta.json`），兩次執行的指紋相同，代表測試集完全相同；`test_set_ids.csv` 則列出測試集每一列的識別欄位與目標值。
- `FULL_YEAR_ONLY_PAST`：預設 `False`，訓練用「測試集以外的所有資料」，包含測試期之後的月份（適合有衛星與氣象資料、要補全測站觀測的情境）；設為 `True` 則只用測試期開始之前的資料，不使用未來資料（適合模擬即時預報）。選哪個取決於你的實際用途。
- 資料中只有春季月份時，`full_year` 與 `spring` 等價，程式會提出警告。
- 兩種組別使用相同的特徵與超參數。若要公平比較，請保持 `config.py` 的其他設定不變。

### 資料與特徵

- 預測目標：`NO2_gnd`（地面測站 NO2，ppb）。
- 輸入特徵由 `FEATURE_COLS` 明確指定：`Hr, U, V, Temp, Hum, PBLH, SWD, troposphere_ozone, ozone, no2, VIIRS, NDVI, Industry, Road total length, Business, Residential, MM, day`。其中 `no2` 是**衛星** NO2（輸入），`NO2_gnd` 才是預測目標。
- 清單中的欄位缺少或不是數值型別時，程式會直接報錯並列出資料的實際欄位。`FEATURE_COLS = None` 則改回自動挑選（依 `IGNORE_COLS`、`SPATIAL_COLS`、`EXCLUDE_FEATURES`）。

### 切分前的資料處理與過濾

依執行順序：

1. **載入**：讀入並合併所有檔案，不篩選任何列。
2. **排序**：依 `SORT_COLS`（預設 `monitordate`、`Hr`、`siteid`）排序，只改順序；`siteid` 讓同一時間點的順序固定，兩種訓練組別才能得到相同的測試集。設為 `[]` 則保持原始順序。
3. **特徵工程**：時間循環編碼與落後特徵，預設都關閉（見第 6 節）。
4. **缺值過濾**：目標或任一輸入特徵為 NaN 的列會被移除。**這是切分前唯一會丟掉資料列的步驟。**
5. **重複特徵檢查**：只印警告，不改資料。

之後才切分：`spring` 取排序後春季資料的前 80%（`TRAIN_RATIO`）訓練、其餘測試；`full_year` 的切分方式見上一節。過濾報告針對**實際訓練用的資料**（`spring` 為春季資料，`full_year` 為全年資料）。

**過濾報告**：每次執行會在終端機印出「哪些資料被過濾、原因」，並存到 `data_filtering/`：

| 檔案 | 內容 |
|---|---|
| `filter_summary.csv` | 依序過濾表：每一列只計入第一個造成它被移除的原因（順序為目標、再依特徵順序），各原因的列數相加等於總移除列數 |
| `filter_by_column.csv` | 各欄位缺值：缺值列數與比例（一列可同時缺多欄），以及「只有這一欄缺值而被移除」的列數 |
| `filter_by_month.csv` | 各月份被移除的列數與比例 |
| `filtered_rows.csv` | 被過濾掉的每一列：識別欄位（`FILTER_ID_COLS`，預設 `siteid`、`sitename`、`monitordate`、`Hr`，加上 `year`、`month`）與完整原因（例如 `no2; Temp`）；`SAVE_FILTERED_ROWS = False` 可關閉 |

報告也會提醒資料中的 `inf`（不會被缺值過濾移除）。**目前不做**數值範圍、離群值、品質旗標、重複列、時段或測站篩選。

---

## 4b. 用 Optuna 自動搜尋超參數

```bash
python run_no2_xgb.py --mode spring --tune --n-trials 50      # 先搜尋，再用最佳參數訓練並評估
python run_no2_xgb.py --mode spring --tune-only               # 只搜尋並輸出報告，不訓練
python run_no2_xgb.py --mode spring --params latest           # 沿用該組別最近一次搜尋的最佳參數
python run_no2_xgb.py --mode full_year --params <best_params.json 路徑>
```

`--tune` 與 `--params` 不能同時使用。春季與全年模型請**各自搜尋**（或都不搜尋）再比較，否則比較的是「調過參」與「沒調過參」的差異。

### 搜尋策略

- **只用訓練資料**：測試集完全不參與搜尋，測試指標才不會被調參污染。
- **時間序列交叉驗證**：訓練資料依日期切成「擴展視窗」的幾摺（`OPTUNA_CV_FOLDS`，預設 3）：每摺用較早的資料訓練、之後一段時間驗證，並以整天為單位切分，避免用未來預測過去，也避免同一天同時出現在訓練與驗證。
- **優化目標**：各摺驗證集 RMSE 的平均（`OPTUNA_METRIC` 可改 `mae` 或 `r2`）。
- **取樣器 TPE**（multivariate）：根據已完成試驗的結果，傾向在「表現好」的參數區域取樣；前 10 次先隨機探索。
- **中位數剪枝**：每跑完一摺就回報目前的平均分數，明顯落後的試驗提早結束，節省時間。
- **基準對照**：搜尋前先以 `MODEL_PARAMS` 跑一次相同的交叉驗證，所有改善都以它為比較對象。
- **樹數**：每摺以驗證集 early stopping；最終樹數 = 最佳試驗各摺最佳輪次的平均 × `OPTUNA_FINAL_ROUNDS_SCALE`（預設 1.1）。使用搜尋結果訓練時，**不再用測試集做 early stopping**。
- **搜尋空間**（`OPTUNA_SEARCH_SPACE`，可自行增減）：`learning_rate`、`max_depth`、`min_child_weight`、`subsample`、`colsample_bytree`、`reg_lambda`、`reg_alpha`、`gamma`。

### 搜尋報告與輸出

存在該次執行資料夾的 `optuna/`：

| 檔案 | 內容 |
|---|---|
| `optuna_report.md` | 報告：目的與結論摘要、搜尋策略（驗證方式、演算法、搜尋空間）、搜尋過程、結果（最佳參數、各摺分數、參數重要性、前 5 名試驗）、調參前後的測試集對照、判讀與建議 |
| `optuna_history.png`、`optuna_param_importance.png`、`optuna_param_slices.png`、`optuna_fold_scores.png` | 優化歷程、參數重要性、各參數與分數的關係、各摺分數對照 |
| `trials.csv` | 所有試驗的狀態、分數、各摺分數、各摺最佳輪次與參數 |
| `best_params.json` | 最佳參數（含最終樹數）；同時複製到 `OPTUNA_EXPORT_DIR/<組別>/best_params.json`（每次覆蓋，供 `--params latest` 使用） |

使用 `--tune` 時，報告最後會補上「測試集對照」：調參前與調參後的模型都用整個訓練集、固定樹數訓練（不用測試集 early stopping），所以兩者的測試指標可以公平比較。

報告也會自動提示：最佳解是否出現得太晚（搜尋可能尚未收斂）、最佳值是否貼近搜尋範圍的邊界（可擴大範圍）、改善幅度是否很小，以及各摺分數是否差異過大。

### 中斷後續跑

把 `config.py` 的 `OPTUNA_STORAGE` 設成例如 `"sqlite:///../../results/tuning/optuna_no2.db"`，同一組別重新執行 `--tune` 就會接續先前的試驗（試驗次數是「新增」的次數）。

---

## 4c. 留一測站交叉驗證（Leave-One-Station-Out）

評估模型推廣到「沒有測站的地點」的能力：每次把一個測站的**所有時間**整個移除，用其餘測站訓練，再預測被移除的測站，並以時間序列比較預測與實測。與第 4 節依時間切分的測試不同，那個測試的測站也出現在訓練集裡。

```bash
python run_loso.py                          # 用 config.RUN_MODE，以 config.LOSO_STATION_COL 區分測站
python run_loso.py --mode full_year         # 用全年資料（預設 spring 只用春季月份）
python run_loso.py --station-col sitename   # 以 sitename 區分測站（預設 siteid）
python run_loso.py --stations 1,5,12        # 只做指定測站，建議先小規模試跑
python run_loso.py --params latest          # 使用 Optuna 搜尋的最佳參數（固定樹數）
```

- **測站欄位**：`--station-col` 或 `LOSO_STATION_COL` 指定用 `siteid` 或 `sitename` 區分；圖表顯示用的名稱取自 `LOSO_NAME_COL`（預設 `sitename`）。
- **資料與特徵**：沿用第 4 節的特徵、排序與缺值過濾（過濾報告也會輸出）。訓練與被預測的測站使用同一段期間（`spring` 為春季月份，`full_year` 為全部月份），沒有依時間再切一次。
- **early stopping**：`MODEL_PARAMS` 設有 `early_stopping_rounds` 時，每一摺從「其餘測站」隨機抽 `LOSO_VAL_STATION_FRAC`（預設 15%）的測站當驗證集；**被移除的測站完全不參與訓練或停止判斷**。使用 `--params` 的最佳參數（固定樹數）時不需要驗證測站。
- **只預測特徵齊全的列**：與其他流程一致，任一特徵缺值（例如衛星 `no2` 被雲遮住）的列已在缺值過濾時移除，所以測站的時間序列會有缺口，缺口處折線斷開。
- 資料列少於 `LOSO_MIN_STATION_ROWS`（預設 50）的測站會略過。

### 輸出

存到 `RUN_OUTPUT_ROOT/run_時間戳_loso_<組別>/`：

| 檔案 | 內容 |
|---|---|
| `timeseries/loso_<測站>.png` | **每個測站一張時間序列圖**：上圖為整段期間的每日平均、下圖為逐小時（資料最多的連續 `LOSO_ZOOM_DAYS` 天，預設 14 天；缺資料處斷線），黑線為實測、紅線為移除該測站後的預測，並標示 r、R²、RMSE、MD、MAPE、N |
| `loso_station_metrics.csv`、`loso_station_metrics.png` | 各測站的指標（依 R² 排序）與長條圖 |
| `loso_station_map_R2.png`、`loso_station_map_RMSE.png` | 測站位置，顏色表示 R² 或 RMSE（找得到縣市界檔時疊上邊界） |
| `ground_truth_vs_model_loso.png` | 全部留一預測合併的散布圖 |
| `loso_predictions.csv` | 每一筆被移除測站的預測（測站、日期、`Hr`、經緯度、實測、預測） |
| `loso_summary.csv` | 合併指標、各測站指標的平均與中位數、R² > 0 的測站數 |
| `loso_config.json`、`data_filtering/` | 本次設定、特徵、參數，以及缺值過濾報告 |

### 判讀

- 看各測站的**中位數**與分佈，不要只看合併指標：少數特別難預測的測站（例如偏遠、山區、特殊排放源）會被合併指標掩蓋。
- 與依時間切分的結果比較，差距就是「測站被移除後」損失的表現。
- `Industry`、`Road total length`、`Business`、`Residential`、`NDVI` 等靜態特徵每個測站各有固定值，等於測站的「指紋」，在一般切分中模型可以靠它記住測站；留一測站時模型沒看過這個指紋，更能檢驗這些特徵是否真的有泛化能力。

---

## 5. 設定（`config.py`）

| 參數 | 說明 |
|---|---|
| `DEFAULT_INPUT_PATH`、`DATA_FORMAT` | 預設輸入路徑；`'auto'` 依副檔名判斷，或指定 `'parquet'` / `'nc'` |
| `TARGET_COL`、`FEATURE_COLS` | 預測目標與輸入特徵清單 |
| `IGNORE_COLS`、`SPATIAL_COLS`、`EXCLUDE_FEATURES` | 自動挑選特徵時的排除與加回規則（`FEATURE_COLS` 有設定時不使用） |
| `DUPLICATE_CORR_THRESHOLD` | 特徵相關係數 ≥ 此值（或名稱只差大小寫）時印出重複特徵警告 |
| `RUN_MODE`、`SPRING_MONTHS`、`FULL_YEAR_ONLY_PAST` | 訓練組別（`'spring'` 或 `'full_year'`）、春季月份、全年模式是否只用測試期之前的資料 |
| `SORT_COLS` | 切分前的排序欄位 |
| `SAVE_FILTERED_ROWS`、`FILTER_ID_COLS` | 過濾報告是否另存被過濾的列，以及用哪些欄位識別 |
| `TRAIN_RATIO` | 訓練集比例（預設 0.8） |
| `FILENAME_YM_REGEX`、`YEAR_COL`、`MONTH_COL`、`DEFAULT_YEAR` | 由檔名取得年月的規則與欄位名稱 |
| `USE_CYCLIC_TIME`、`HOUR_PERIOD`、`DAY_PERIOD`、`MONTH_PERIOD` | 是否使用 sin/cos 時間編碼與各週期 |
| `USE_NO2_LAGS`、`NO2_LAG_HOURS`、`ALLOW_MISSING_LAGS` | 落後特徵：是否使用、落後幾小時、落後值缺失時是否保留樣本 |
| `MODEL_PARAMS` | XGBoost 超參數（未搜尋時使用，也是 Optuna 的基準） |
| `LOSO_STATION_COL`、`LOSO_NAME_COL`、`LOSO_MIN_STATION_ROWS`、`LOSO_VAL_STATION_FRAC`、`LOSO_SEED` | 留一測站：區分測站的欄位、顯示名稱欄位、測站最少資料列數、early stopping 的驗證測站比例、隨機種子 |
| `LOSO_DATE_COL`、`LOSO_ZOOM_DAYS`、`LOSO_GAP_HOURS`、`PLOT_CJK_FONTS` | 時間序列圖：日期欄位、逐小時圖顯示的天數、斷線的間隔小時數，以及顯示中文測站名稱的字型 |
| `OPTUNA_N_TRIALS`、`OPTUNA_TIMEOUT` | 搜尋的試驗次數與總時間上限（秒） |
| `OPTUNA_METRIC`、`OPTUNA_CV_FOLDS`、`OPTUNA_DATE_COL` | 優化目標、交叉驗證摺數、切摺用的日期欄位 |
| `OPTUNA_MAX_ROUNDS`、`OPTUNA_EARLY_STOPPING`、`OPTUNA_FINAL_ROUNDS_SCALE` | 每摺樹數上限、early stopping 輪數、最終樹數的放大倍率 |
| `OPTUNA_SEED`、`OPTUNA_STARTUP_TRIALS`、`OPTUNA_PRUNER_STARTUP_TRIALS` | 隨機種子、TPE 初期隨機探索次數、剪枝前的觀察次數 |
| `OPTUNA_SEARCH_SPACE` | 搜尋空間（參數名稱 → 型別、下限、上限、是否取對數） |
| `OPTUNA_STORAGE`、`OPTUNA_EXPORT_DIR` | 搜尋記錄的資料庫（可續跑）與最佳參數的匯出位置 |
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

每次訓練會建立 `RUN_OUTPUT_ROOT/run_YYYYmmdd_HHMMSS_<組別>/`（例如 `run_20260101_120000_full_year`），內含：

- `run_config_and_params.csv`：輸入路徑、訓練組別、訓練與測試的月份與筆數、測試集指紋與模型參數
- `test_set_ids.csv`：測試集每一列的識別欄位與目標值（用來核對兩種組別的測試集相同）
- `data_filtering/`：切分前的資料過濾報告（見第 4 節）
- `optuna/`：使用 `--tune` 時的搜尋報告、圖表、試驗記錄與最佳參數（見第 4b 節）
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
- `model_meta.json`：特徵欄位與順序、最佳輪次、測試集指標、訓練組別與月份、測試集指紋、時間編碼與落後特徵設定

| 資料夾 | 說明 |
|---|---|
| `model_validated/` | 只用訓練集訓練、以測試集 early stopping 的模型；其測試指標就是 `evaluation_metrics.csv` |
| `model_final/` | 用全部資料（訓練＋測試）、固定在最佳輪次重訓的最終模型（`TRAIN_FINAL_MODEL = False` 可關閉） |

兩者各自存在該次的 `run_時間戳/` 內（不會被覆蓋），同時複製到 `MODEL_EXPORT_DIR/<組別>/`（預設 `../../results/models/spring/` 或 `.../full_year/`）下的 `latest_validated/`、`latest_final/`（同一組別每次覆蓋，供預測時引用；兩種組別各有自己的資料夾，不會互相覆蓋）。最終模型沒有未參與訓練的資料可評估，成效以 validated 模型的指標為準。

---

## 8. 指定時間點預測並畫全台濃度圖

```bash
python predict_map.py --nc <含特徵的 nc 檔> --day 15 --hour 6
```

- `--nc`：含模型特徵的 `.nc` 檔，欄位名稱需與訓練特徵一致（缺少時會列出缺了哪些）；檔名要含 `YYYYMM`
- `--day`、`--hour`：對應資料中 `day`、`Hr`（或 `hour`）欄位的**實際數值**；找不到時會印出可用的值
- `--model`：模型資料夾，預設為 `config.DEFAULT_MODEL_DIR`（`../../results/models/<RUN_MODE>/latest_final`）；要用另一組別或 validated 模型時指定，例如 `../../results/models/full_year/latest_final`。春季模型只看過春季月份，不適合預測其他月份
- 輸出：`../../results/maps/NO2_pred_<檔名>_day<d>_hour<h>.png`（疊縣市邊界）

任一特徵缺值的像素不預測（訓練時同樣排除了缺值列），會留白。若該時間點沒有任何像素能預測，程式會印出各特徵的缺值數並正常結束，不會當掉。若模型使用了落後特徵，nc 檔還需包含前幾小時的目標變數觀測。

---

## 9. 使用上的注意

1. 不要把主程式命名為 `xgboost.py`，會與 xgboost 套件同名而導致匯入錯誤。
2. `eval_set` 包含測試集，early stopping 是看測試集誤差決定停止輪數，最終指標會略微樂觀。
3. 比較春季與全年模型時，請確認兩次執行的測試集指紋相同。切分是同一批測站的時間切分：測試集的測站也出現在訓練集裡，成效代表「已知測站的未來時間」，不代表能推廣到沒有測站的地方。
4. 逐輪趨勢圖是用「前 k 棵樹」重新預測得到的，輪數多、資料量大時較慢，可調大 `CURVE_EVAL_STEP` 或調小 `CURVE_TRAIN_MAX_SAMPLES`。
5. Optuna 的最佳 CV 分數是「從多次試驗中挑最好的一個」，會略微樂觀；真正的泛化表現請看測試集。搜尋一次要訓練「試驗次數 × 摺數」個模型，耗時較長，可先用 `--n-trials 10` 試跑。

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
| 時間序列圖的測站名稱變成方框 | `PLOT_CJK_FONTS` 填入系統有的中文字型（Windows 為 `Microsoft JhengHei`） |
| 留一測站要跑很久 | 先用 `--stations` 指定幾個測站試跑；每個測站要訓練一次模型，測站數多時可改用 `--params`（固定樹數、不需驗證測站）或調低 `MODEL_PARAMS['n_estimators']` |
| `ModuleNotFoundError: optuna` | 執行 `pip install optuna`（只有使用 `--tune` 時才需要） |
| 搜尋太久 | 調小 `--n-trials`、設定 `--timeout`，或調低 `OPTUNA_MAX_ROUNDS`、`OPTUNA_CV_FOLDS` |
| `--params latest` 找不到檔案 | 該組別還沒搜尋過，請先執行 `--tune` 或 `--tune-only` |