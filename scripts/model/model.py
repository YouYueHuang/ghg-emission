"""
XGBoost 模型的建立與訓練。
"""
import xgboost as xgb
import config


def build_model(model_params=None):
    params = dict(config.MODEL_PARAMS if model_params is None else model_params)
    return xgb.XGBRegressor(**params)


def train_model(X_train, y_train, X_test, y_test, model_params=None):
    """訓練模型；eval_set 同時監看訓練集與測試集。"""
    model = build_model(model_params)
    print("⏳ 正在訓練 XGBoost 模型...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=config.FIT_VERBOSE,
    )
    return model


def get_best_round(model):
    """early stopping 選出的最佳輪次（從 1 起算）；沒有就用全部樹數。"""
    try:
        return int(model.best_iteration) + 1
    except Exception:
        return int(model.get_booster().num_boosted_rounds())


def train_final_model(X_all, y_all, n_rounds, model_params=None):
    """
    最終模型：用全部資料（訓練+測試）、固定輪數 n_rounds 重新訓練，不再做 early stopping。
    n_rounds 取自驗證階段 early stopping 找到的最佳輪次。
    """
    params = dict(config.MODEL_PARAMS if model_params is None else model_params)
    params.pop('early_stopping_rounds', None)
    params['n_estimators'] = n_rounds
    print(f"⏳ 正在用全部 {len(X_all)} 筆資料訓練最終模型（{n_rounds} 輪）...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_all, y_all, verbose=False)
    return model