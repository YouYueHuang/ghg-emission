"""
集中管理所有可調整的設定（欄位名稱、模型超參數、切分比例、輸出路徑等）。
"""

# ===== 資料與欄位 =====
# 資料來源：資料夾（內含 *.parquet 或 *.nc）、單一檔案、萬用字元皆可
DEFAULT_INPUT_PATH = "../../data/raw/nc_datas/data_2023/.ml-parquet"
DATA_FORMAT = 'auto'               # 'auto' 依副檔名判斷；資料夾內 .nc 與 .parquet 並存時，請指定 'nc' 或 'parquet'
TARGET_COL = "NO2_gnd"             # 預測目標：地面測站 NO2 (ppb)

# 不當作一般特徵的欄位（時間、座標、目標變數）
# （只在 FEATURE_COLS = None 時才用來自動挑特徵）
IGNORE_COLS = {
    'Hr', 'Lat', 'Lon', 'MM', 'hr', 'lat', 'lon', 'mm',
    'day', 'hour', 'Time', 'time', 'siteid', 'sitename', 'monitordate', TARGET_COL,
}
# 這些時空欄位即使在 IGNORE_COLS 內，只要資料裡有就會被加回特徵
SPATIAL_COLS = []

# 明確指定的輸入特徵（優先於 IGNORE_COLS / SPATIAL_COLS / EXCLUDE_FEATURES；設為 None 則改用自動挑選）
# 其中 'no2' 是衛星 NO2（輸入特徵），預測目標則是地面測站的 'NO2_gnd'
FEATURE_COLS = [
    'Hr', 'U', 'V', 'Temp', 'Hum', 'PBLH', 'SWD',
    'troposphere_ozone', 'ozone', 'no2', 'VIIRS', 'NDVI',
    'Industry', 'Road total length', 'Business', 'Residential',
    'MM', 'day',
]
# 切分前先依這些欄位排序（欄位都存在時才排序），讓「前 80% 訓練、後 20% 測試」是真正的時間切分；
# 設為 [] 則保持檔案內的原始順序
SORT_COLS = ['monitordate', 'Hr', 'siteid']      # siteid 讓同一時間點的排序固定，兩種訓練組別才能得到完全相同的測試集

# 明確排除的特徵欄位（最後才套用，優先於上面的規則）。例如經緯度有大小寫兩份時：{'Lat', 'Lon'}
EXCLUDE_FEATURES = set()
# 特徵之間相關係數絕對值 >= 此值，或名稱只差大小寫時，會在執行時印出「疑似重複特徵」警告
DUPLICATE_CORR_THRESHOLD = 0.999

# ===== 月份參數（由檔名取得）=====
# 檔名含 YYYYMM（如 202303.nc）時，自動加入 year=2023、month=3 兩個欄位
FILENAME_YM_REGEX = r'(\d{4})(\d{2})'
YEAR_COL = 'year'
MONTH_COL = 'month'
DEFAULT_YEAR = 2023                # 輸入是 CSV 等沒有檔名年份時使用

# ===== 時間循環編碼 (sin / cos) =====
# True：hour、day、month 改用 sin/cos 編碼成 6 個特徵 (hour_sin, hour_cos, day_sin, ...)，
#       原始的 hour / day / month 不再直接當特徵
USE_CYCLIC_TIME = False
HOUR_PERIOD = 24                   # 若 Hr 不是一天 24 小時的小時數（例如 0~7 的時段索引），請修改
DAY_PERIOD = 31                    # day 為月內第幾天
MONTH_PERIOD = 12
DAY_COL_CANDIDATES = ['day', 'Day']
HOUR_COL_CANDIDATES = ['Hr', 'hr', 'hour', 'Hour']

# ===== 落後特徵（用前幾小時的目標值預測下一小時；目前預設關閉，特徵只用 FEATURE_COLS）=====
USE_NO2_LAGS = False
NO2_LAG_HOURS = [1, 2]             # 產生 no2_lag1 (t-1 小時)、no2_lag2 (t-2 小時)，依像素 (lat, lon) 對應
# False：任一落後值缺失的樣本都丟掉（樣本變少）；True：保留，讓 XGBoost 把缺失當作缺值處理
ALLOW_MISSING_LAGS = False

# ===== 訓練組別：春季 / 全年 =====
# 'spring'   ：只用 SPRING_MONTHS 的資料，依時間取前 TRAIN_RATIO 訓練、其餘測試
# 'full_year'：用全部月份訓練，但測試集與 'spring' 完全相同（春季資料時間上最後的 1 - TRAIN_RATIO）
# 也可用命令列指定：python run_no2_xgb.py --mode full_year
RUN_MODE = 'spring'
SPRING_MONTHS = [3, 4, 5]
# True：全年模式只用「測試期開始之前」的資料訓練（不使用未來資料，適合模擬即時預報）；
# False：用除了測試集以外的所有資料（適合有衛星與氣象資料、要補全測站觀測的情境）
FULL_YEAR_ONLY_PAST = False
ROW_ID_COL = '_rid'                # 內部使用的列編號，不會當特徵

# ===== 切分前的資料過濾報告 =====
SAVE_FILTERED_ROWS = True          # 另存被過濾掉的列（識別欄位 + 原因）到 data_filtering/filtered_rows.csv
FILTER_ID_COLS = ['siteid', 'sitename', 'monitordate', 'Hr']   # 被過濾的列用哪些欄位識別（資料中存在的才使用）

# ===== 資料切分 =====
TRAIN_RATIO = 0.8          # 依資料順序切分：前 80% 訓練、後 20% 測試

# ===== XGBoost 超參數 =====
MODEL_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
}
FIT_VERBOSE = 100

# ===== 圖表 =====
FIG_DPI = 300
IMPORTANCE_MAX_FEATURES = 18

# ===== 逐輪指標趨勢圖 =====
CURVE_EVAL_STEP = None             # None = 自動（最多約 CURVE_MAX_POINTS 個取樣點）；1 = 每輪都算
CURVE_MAX_POINTS = 200
CURVE_TRAIN_MAX_SAMPLES = 200000   # 訓練集超過此筆數時，趨勢圖只抽樣這麼多筆計算

# ===== Ground truth vs Model 散布圖 =====
SCATTER_TITLE = 'GROUND TRUTH vs Model'
SCATTER_XLABEL = 'GROUND TRUTH'
SCATTER_MIN_POINTS_PER_HOUR = 20   # 某小時的測試樣本少於此數就不畫（點太少，指標與密度都沒有意義）
# 單位標籤（mathtext 的 $ 必須成對；原本只有一個 $ 會被當成純文字顯示）
SCATTER_UNIT = 'ppb'
SCATTER_DENSITY_BINS = 100         # 計算局部密度的 2D 直方圖格數
SCATTER_DENSITY_SMOOTH = 1.0       # 密度的高斯平滑強度（格）；0 = 不平滑
SCATTER_MAX_PLOT_POINTS = 200000   # 點數超過此值時只抽樣繪圖（指標仍用全部樣本）

# ===== SHAP =====
SHAP_SAMPLE_SIZE = 2000
SHAP_RANDOM_STATE = 42
SHAP_DEPENDENCE_FEATURE = 'PBLH'   # 資料中沒有此欄位時，改用第一個特徵

# ===== 模型匯出（每次訓練完都會存）=====
# 每次執行都存到該次的 run_時間戳/model_validated/、model_final/（不會被覆蓋）；
# 同時複製一份到 MODEL_EXPORT_DIR/latest_*（固定路徑，每次覆蓋，方便預測時引用）
TRAIN_FINAL_MODEL = True                          # 另外用全部資料重訓一個最終模型
MODEL_EXPORT_DIR = "../../results/models"   # 實際存在 MODEL_EXPORT_DIR/<RUN_MODE>/latest_*
DEFAULT_MODEL_DIR = f"../../results/models/{RUN_MODE}/latest_final"   # predict_map.py 預設使用的模型（可用 --model 覆蓋）

# ===== 指定時間點預測並繪圖 (predict_map.py) =====
PREDICT_NC_PATH = None            # 也可用命令列 --nc 指定
PREDICT_DAY = None                # 也可用命令列 --day
PREDICT_HOUR = None               # 也可用命令列 --hour
MAP_LAT_COLS = ['lat', 'Lat', 'latitude']         # 畫圖與像素對應用的座標欄位（依序找第一個存在的）
MAP_LON_COLS = ['lon', 'Lon', 'longitude']
SHP_PATH = "../../data/raw/nc_datas/ground-info/COUNTY_MOI_1140318.shp"
MAP_OUTPUT_DIR = "../../results/maps"
MAP_CMAP = 'Spectral_r'
MAP_VMIN = 0
MAP_VMAX = None                   # None = 取預測值的 99 百分位；也可固定，例如 40

# ===== 輸出 =====
RUN_OUTPUT_ROOT = "../../results/runs"