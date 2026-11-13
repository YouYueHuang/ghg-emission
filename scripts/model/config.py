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
    'early_stopping_rounds': None,
}
FIT_VERBOSE = 100

# ===== Optuna 超參數搜尋 =====
# 用法：python run_no2_xgb.py --mode spring --tune        （搜尋後直接用最佳參數訓練）
#       python run_no2_xgb.py --mode spring --tune-only   （只搜尋、輸出報告）
#       python run_no2_xgb.py --mode spring --params latest（沿用該組別最近一次的最佳參數）
# 搜尋只用「訓練資料」內部的時間序列交叉驗證，測試集完全不參與，確保測試指標沒有被調參污染
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = None              # 總時間上限（秒）；None = 不限，只看 OPTUNA_N_TRIALS
OPTUNA_METRIC = 'rmse'             # 優化目標：'rmse'、'mae'（越小越好）或 'r2'（越大越好）
OPTUNA_CV_FOLDS = 3                # 時間序列交叉驗證的摺數（擴展視窗：每摺用更早的資料訓練、之後一段時間驗證）
OPTUNA_DATE_COL = 'monitordate'    # 以「日期」為單位切摺，避免同一天的資料同時出現在訓練與驗證；欄位不存在時改用列順序
OPTUNA_MAX_ROUNDS = 2000           # 每摺訓練的樹數上限（搭配 early stopping）
OPTUNA_EARLY_STOPPING = 50         # 每摺以驗證集做 early stopping
OPTUNA_FINAL_ROUNDS_SCALE = 1.1    # 最終模型的樹數 = 最佳試驗各摺最佳輪次平均 × 此值（最終用較多資料訓練）
OPTUNA_SEED = 42
OPTUNA_STARTUP_TRIALS = 10         # TPE 取樣器：前幾次試驗先隨機探索
OPTUNA_PRUNER_STARTUP_TRIALS = 5   # 中位數剪枝：前幾次試驗不剪枝
OPTUNA_STORAGE = None              # 例如 "sqlite:///../../results/tuning/optuna_no2.db"，可中斷後續跑；None = 只存在記憶體
OPTUNA_EXPORT_DIR = "../../results/tuning"   # 最佳參數另存於 <此處>/<組別>/best_params.json（每次覆蓋，供 --params latest 使用）
# 搜尋空間：參數名稱 -> (型別 'float' / 'int', 下限, 上限, 是否取對數尺度)
OPTUNA_SEARCH_SPACE = {
    'learning_rate':    ('float', 0.01, 0.2, True),
    'max_depth':        ('int', 3, 10, False),
    'min_child_weight': ('float', 1.0, 20.0, True),
    'subsample':        ('float', 0.5, 1.0, False),
    'colsample_bytree': ('float', 0.5, 1.0, False),
    'reg_lambda':       ('float', 1e-3, 10.0, True),
    'reg_alpha':        ('float', 1e-3, 10.0, True),
    'gamma':            ('float', 0.0, 5.0, False),
}

# ===== 留一測站交叉驗證 (Leave-One-Station-Out, run_loso.py) =====
# 每次把一個測站整個移除（所有時間），用其餘測站訓練，再預測被移除的測站，以時間序列比較預測與實測
LOSO_STATION_COL = 'siteid'        # 以哪個欄位決定測站：'siteid' 或 'sitename'
LOSO_NAME_COL = 'sitename'         # 圖表與表格顯示用的測站名稱（資料沒有此欄位時改用 LOSO_STATION_COL）
LOSO_DATE_COL = 'monitordate'      # 時間序列圖的日期欄位（會再加上 Hr 組成完整時間）
LOSO_MIN_STATION_ROWS = 50         # 清理後資料列少於此數的測站不做（預測點太少，指標沒有意義）
LOSO_VAL_STATION_FRAC = 0.15       # 設定了 early stopping 時，從「其餘測站」中抽這個比例的測站當驗證集（被移除的測站完全不參與）
LOSO_TS_AXIS = 'time'             # 時間序列圖橫軸：'time' = 真實時間（折線連續連接）；'index' = 觀測序號（夜間與缺漏的小時不佔寬度）
LOSO_SEED = 42
LOSO_MIN_POINTS_PER_HOUR = 10       # 逐測站、逐小時散布圖：該測站該小時的資料點少於此數就不畫
LOSO_MAP_LABELS = 'value'           # 測站分佈圖上的標註：'value'（只標數值）、'name_value'（測站名稱 + 數值）、'none'（不標）
LOSO_MAP_LABEL_FONTSIZE = 7
PLOT_CJK_FONTS = ['Microsoft JhengHei', 'Noto Sans CJK TC', 'PingFang TC', 'Arial Unicode MS']   # 測站名稱含中文時使用

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

# ===== 訓練後輸出全台平均濃度分佈圖 (period_map.py) =====
# 訓練完成後，用模型預測全台網格在「訓練第一天第一小時到最後一天最後一小時」每個小時的 NO2，平均後畫一張分佈圖
PERIOD_MAP_ENABLED = True
GRID_NC_PATH = "../../data/raw/nc_datas/data_2023/Merged/model"   # 含全台網格特徵的 .nc（資料夾 / 檔案 / 萬用字元）；欄位需與模型特徵一致、檔名含 YYYYMM
PERIOD_MAP_PERIOD = 'train'        # 'train' = 訓練資料的第一個小時到最後一個小時；'all' = 訓練 + 測試資料
PERIOD_MAP_MODEL = 'final'         # 用哪個模型預測：'final'（全部資料重訓；TRAIN_FINAL_MODEL=False 時自動改用 validated）或 'validated'

# ===== 輸出 =====
RUN_OUTPUT_ROOT = "../../results/runs"