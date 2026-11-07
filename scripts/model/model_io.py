"""
模型的儲存與載入。每個模型資料夾包含：
  model.json         模型權重（所有樹的結構與葉節點數值）
  model_params.json  超參數（sklearn 介面的參數 + XGBoost booster 完整設定）
  model_meta.json    中繼資料：特徵欄位與順序、最佳輪次、測試指標、時間編碼與落後特徵設定等
預測時必須用訓練時的特徵欄位與順序，所以三者一起保存。
"""
import os
import json
import datetime

MODEL_FILE = "model.json"
PARAMS_FILE = "model_params.json"
META_FILE = "model_meta.json"


def _jsonable(o):
    return o.item() if hasattr(o, 'item') else str(o)


def _dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_jsonable)


def save_model_bundle(model, feature_cols, out_dir, meta=None):
    """儲存權重、超參數與中繼資料到 out_dir，回傳 out_dir。"""
    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, MODEL_FILE))
    _dump({
        'sklearn_params': model.get_params(),
        'booster_config': json.loads(model.get_booster().save_config()),
    }, os.path.join(out_dir, PARAMS_FILE))
    _dump({
        'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'feature_cols': list(feature_cols),
        **(meta or {}),
    }, os.path.join(out_dir, META_FILE))
    print(f"💾 模型權重與參數已儲存至: {out_dir}")
    return out_dir


def load_model_bundle(model_dir):
    """載入模型與中繼資料，回傳 (model, meta)。"""
    import xgboost as xgb
    model_path = os.path.join(model_dir, MODEL_FILE)
    meta_path = os.path.join(model_dir, META_FILE)
    for p in (model_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ 找不到模型檔案: {p}（請先執行 run_no2_xgb.py 訓練並匯出模型）")
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model, meta