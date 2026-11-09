"""
Optuna 超參數搜尋（XGBoost 回歸）。

策略：
  - 驗證方式：在「訓練資料」內做時間序列交叉驗證（擴展視窗，以日期切摺），測試集完全不參與
  - 優化目標：各摺驗證集 RMSE（或 MAE / R2，見 config.OPTUNA_METRIC）的平均
  - 取樣器：TPE（multivariate），前幾次試驗先隨機探索
  - 剪枝：中位數剪枝（每跑完一摺回報目前的平均分數，明顯落後的試驗提早結束）
  - 基準：搜尋前先用 config.MODEL_PARAMS 跑一次相同的交叉驗證，作為「調參前」的對照
  - 最終樹數：最佳試驗各摺 early stopping 輪次的平均 × OPTUNA_FINAL_ROUNDS_SCALE（最終模型不再用測試集做 early stopping）
搜尋完成後輸出報告（見 tuning_report.py）、trials.csv 與 best_params.json。
"""
import os
import json
import time
import datetime
import warnings
import numpy as np
import pandas as pd

import config
from tuning_report import write_report


# ---------- 指標 ----------
def metric_value(y_true, y_pred, name):
    y_true = np.asarray(y_true, dtype='float64')
    y_pred = np.asarray(y_pred, dtype='float64')
    if name == 'rmse':
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if name == 'mae':
        return float(np.mean(np.abs(y_true - y_pred)))
    if name == 'r2':
        return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    raise ValueError(f"OPTUNA_METRIC 只能是 'rmse'、'mae' 或 'r2'，收到: {name!r}")


def direction_of(name):
    return 'maximize' if name == 'r2' else 'minimize'


# ---------- 時間序列交叉驗證 ----------
def _fmt(v):
    try:
        ts = pd.Timestamp(v)
        return str(ts.date())
    except Exception:
        return str(v)


def make_time_folds(group_values, n_splits):
    """
    擴展視窗切摺：把「時間群組（日期）」排序後均分成 n_splits+1 段，
    第 i 摺以前 i+1 段訓練、第 i+2 段驗證。以整個群組為單位，同一天不會同時出現在訓練與驗證。
    回傳 list of dict（含訓練/驗證列位置與期間說明）。
    """
    codes, uniques = pd.factorize(pd.Series(group_values), sort=True)
    n_groups = len(uniques)
    if n_groups < n_splits + 1:
        raise ValueError(f"時間群組只有 {n_groups} 個，不足以切成 {n_splits} 摺")
    chunks = np.array_split(np.arange(n_groups), n_splits + 1)
    folds = []
    for i in range(n_splits):
        tr_last, va = chunks[i][-1], chunks[i + 1]
        train_idx = np.flatnonzero((codes >= 0) & (codes <= tr_last))
        val_idx = np.flatnonzero((codes >= va[0]) & (codes <= va[-1]))
        folds.append({
            'fold': i + 1, 'train_idx': train_idx, 'val_idx': val_idx,
            'n_train': len(train_idx), 'n_val': len(val_idx),
            'train_period': f"{_fmt(uniques[0])} ~ {_fmt(uniques[tr_last])}",
            'val_period': f"{_fmt(uniques[va[0]])} ~ {_fmt(uniques[va[-1]])}",
        })
    return folds


def group_values_for(df_clean, index, date_col=None):
    """取訓練列的日期（時間群組）；日期欄位不存在時，以列順序（已依時間排序）作為群組。"""
    date_col = date_col or config.OPTUNA_DATE_COL
    if date_col in df_clean.columns:
        return df_clean.loc[index, date_col].to_numpy()
    print(f"[資訊] 找不到日期欄位 '{date_col}'，改用列順序切摺")
    return np.arange(len(index))


# ---------- 參數 ----------
def search_space():
    return config.OPTUNA_SEARCH_SPACE


def fixed_params():
    """MODEL_PARAMS 中不在搜尋空間內的參數（如 random_state、n_jobs），搜尋時固定不變。"""
    skip = set(search_space()) | {'n_estimators', 'early_stopping_rounds'}
    return {k: v for k, v in config.MODEL_PARAMS.items() if k not in skip}


def suggest_params(trial):
    p = {}
    for name, (kind, lo, hi, log) in search_space().items():
        if kind == 'int':
            p[name] = trial.suggest_int(name, int(lo), int(hi))
        elif kind == 'float':
            p[name] = trial.suggest_float(name, float(lo), float(hi), log=bool(log))
        else:
            raise ValueError(f"搜尋空間 {name} 的型別只能是 'int' 或 'float'，收到: {kind!r}")
    return p


def baseline_params():
    """調參前的參數（config.MODEL_PARAMS），搭配每摺 early stopping。"""
    p = {k: v for k, v in config.MODEL_PARAMS.items() if k not in ('n_estimators', 'early_stopping_rounds')}
    return {**p, 'n_estimators': config.OPTUNA_MAX_ROUNDS, 'early_stopping_rounds': config.OPTUNA_EARLY_STOPPING}


# ---------- 交叉驗證 ----------
def _fit_predict_fold(params, X_tr, y_tr, X_va, y_va):
    import xgboost as xgb
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    rounds = int(model.best_iteration) + 1
    return model.predict(X_va, iteration_range=(0, rounds)), rounds


def cv_evaluate(params, X, y, folds, metric, trial=None):
    """回傳 (各摺分數, 各摺最佳輪次)。給定 trial 時，每跑完一摺回報並視情況剪枝。"""
    scores, rounds = [], []
    for i, f in enumerate(folds):
        X_tr, y_tr = X.iloc[f['train_idx']], y.iloc[f['train_idx']]
        X_va, y_va = X.iloc[f['val_idx']], y.iloc[f['val_idx']]
        pred, r = _fit_predict_fold(params, X_tr, y_tr, X_va, y_va)
        scores.append(metric_value(y_va, pred, metric))
        rounds.append(r)
        if trial is not None:
            import optuna
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                trial.set_user_attr('fold_scores', scores)
                trial.set_user_attr('best_rounds', rounds)
                raise optuna.TrialPruned()
    return scores, rounds


# ---------- 主流程 ----------
def run_optuna(X_train, y_train, group_values, feature_cols, mode, output_dir, n_trials=None, timeout=None):
    import optuna
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = n_trials or config.OPTUNA_N_TRIALS
    timeout = timeout if timeout is not None else config.OPTUNA_TIMEOUT
    metric = config.OPTUNA_METRIC
    direction = direction_of(metric)
    fixed = fixed_params()
    scale = config.OPTUNA_FINAL_ROUNDS_SCALE

    print("=" * 75)
    print(f"🔧 Optuna 超參數搜尋（訓練組別 {mode}；目標 {metric}，{'越小' if direction == 'minimize' else '越大'}越好）")
    print("=" * 75)
    folds = make_time_folds(group_values, config.OPTUNA_CV_FOLDS)
    print(f"📚 時間序列交叉驗證（{len(folds)} 摺；只使用訓練資料 {len(X_train)} 筆，測試集不參與）:")
    for f in folds:
        print(f"   - 第 {f['fold']} 摺：訓練 {f['n_train']} 筆（{f['train_period']}）→ 驗證 {f['n_val']} 筆（{f['val_period']}）")

    # 1. 基準：調參前的參數
    print("⏳ 先以目前的 MODEL_PARAMS 計算基準分數...")
    b_scores, b_rounds = cv_evaluate(baseline_params(), X_train, y_train, folds, metric)
    b_mean = float(np.mean(b_scores))
    print(f"   基準 {metric} = {b_mean:.4f}（各摺 {[round(s, 4) for s in b_scores]}，最佳輪次 {b_rounds}）")

    # 2. 搜尋
    def objective(trial):
        params = {**fixed, **suggest_params(trial),
                  'n_estimators': config.OPTUNA_MAX_ROUNDS, 'early_stopping_rounds': config.OPTUNA_EARLY_STOPPING}
        scores, rounds = cv_evaluate(params, X_train, y_train, folds, metric, trial=trial)
        trial.set_user_attr('fold_scores', scores)
        trial.set_user_attr('best_rounds', rounds)
        return float(np.mean(scores))

    def progress(study, trial):
        done = [t for t in study.trials if t.state.name in ('COMPLETE', 'PRUNED')]
        val = f"{trial.value:.4f}" if trial.value is not None else "（剪枝）"
        best = f"{study.best_value:.4f}" if any(t.state.name == 'COMPLETE' for t in study.trials) else "-"
        print(f"   trial {trial.number:>3} [{trial.state.name:<8}] {metric}={val}  目前最佳={best}  ({len(done)}/{n_trials})")

    sampler = optuna.samplers.TPESampler(seed=config.OPTUNA_SEED, multivariate=True,
                                         n_startup_trials=config.OPTUNA_STARTUP_TRIALS)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=config.OPTUNA_PRUNER_STARTUP_TRIALS, n_warmup_steps=0)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner,
                                study_name=f"no2_xgb_{mode}", storage=config.OPTUNA_STORAGE,
                                load_if_exists=config.OPTUNA_STORAGE is not None)
    print(f"🚀 開始搜尋：最多 {n_trials} 次試驗" + (f"、時間上限 {timeout} 秒" if timeout else ""))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[progress])
    elapsed = time.time() - t0

    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    if not completed:
        raise RuntimeError("沒有任何完成的試驗，無法取得最佳參數（可能全被剪枝或失敗）")
    best = study.best_trial
    best_rounds = [int(r) for r in best.user_attrs['best_rounds']]
    final_rounds = max(1, int(round(np.mean(best_rounds) * scale)))
    base_final_rounds = max(1, int(round(np.mean(b_rounds) * scale)))
    searched = dict(best.params)
    full_params = {**fixed, **searched, 'n_estimators': final_rounds}

    # 3. 參數重要性（需要 >= 2 個完成的試驗）
    importances = None
    try:
        importances = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=config.OPTUNA_SEED))
    except Exception as e:
        print(f"[資訊] 無法計算參數重要性（{e}）")

    # 4. 整理結果
    rows = []
    for t in study.trials:
        row = {'number': t.number, 'state': t.state.name, 'value': t.value,
               'duration_s': round(t.duration.total_seconds(), 2) if t.duration is not None else None,
               'fold_scores': json.dumps([round(float(s), 5) for s in t.user_attrs.get('fold_scores', [])]),
               'best_rounds': json.dumps([int(r) for r in t.user_attrs.get('best_rounds', [])])}
        row.update({f"p_{k}": v for k, v in t.params.items()})
        rows.append(row)
    trials_df = pd.DataFrame(rows)

    result = {
        'mode': mode, 'metric': metric, 'direction': direction,
        'feature_cols': list(feature_cols), 'n_train_rows': len(X_train),
        'folds': [{k: v for k, v in f.items() if k not in ('train_idx', 'val_idx')} for f in folds],
        'baseline': {'params': {k: v for k, v in baseline_params().items() if k not in ('early_stopping_rounds',)},
                     'score': b_mean, 'fold_scores': [float(s) for s in b_scores],
                     'rounds': [int(r) for r in b_rounds], 'final_rounds': base_final_rounds},
        'best': {'number': best.number, 'score': float(best.value),
                 'fold_scores': [float(s) for s in best.user_attrs['fold_scores']],
                 'rounds': best_rounds, 'final_rounds': final_rounds,
                 'searched_params': searched, 'full_params': full_params},
        'trials_df': trials_df, 'importances': importances,
        'search_space': {k: list(v) for k, v in search_space().items()},
        'fixed_params': fixed,
        'settings': {'n_trials': n_trials, 'timeout': timeout, 'seed': config.OPTUNA_SEED,
                     'startup_trials': config.OPTUNA_STARTUP_TRIALS,
                     'pruner_startup_trials': config.OPTUNA_PRUNER_STARTUP_TRIALS,
                     'max_rounds': config.OPTUNA_MAX_ROUNDS, 'early_stopping': config.OPTUNA_EARLY_STOPPING,
                     'rounds_scale': scale, 'cv_folds': config.OPTUNA_CV_FOLDS,
                     'date_col': config.OPTUNA_DATE_COL, 'storage': config.OPTUNA_STORAGE},
        'elapsed_sec': elapsed,
    }

    # 5. 輸出：trials.csv、best_params.json、報告
    out = os.path.join(output_dir, "optuna")
    os.makedirs(out, exist_ok=True)
    trials_df.to_csv(os.path.join(out, "trials.csv"), index=False, encoding='utf-8-sig')
    save_best_params(result, out)
    export_dir = os.path.join(config.OPTUNA_EXPORT_DIR, mode)
    os.makedirs(export_dir, exist_ok=True)
    save_best_params(result, export_dir)
    write_report(result, output_dir)

    print(f"✅ 搜尋完成：最佳 {metric} = {best.value:.4f}（基準 {b_mean:.4f}），"
          f"第 {best.number} 次試驗；最終樹數 {final_rounds}")
    return result


# ---------- 最佳參數的儲存與載入 ----------
def save_best_params(result, folder):
    info = {
        'mode': result['mode'], 'metric': result['metric'],
        'cv_score': result['best']['score'], 'baseline_cv_score': result['baseline']['score'],
        'params': result['best']['full_params'],
        'searched_params': result['best']['searched_params'],
        'feature_cols': result['feature_cols'],
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    path = os.path.join(folder, "best_params.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    return path


def load_tuned_params(spec, mode, feature_cols=None):
    """
    spec = 'latest'（該組別最近一次搜尋的結果）、best_params.json 檔案，或含該檔案的資料夾。
    回傳 (完整參數 dict, 資訊 dict)。
    """
    path = os.path.join(config.OPTUNA_EXPORT_DIR, mode, "best_params.json") if spec == 'latest' else spec
    if os.path.isdir(path):
        path = os.path.join(path, "best_params.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到最佳參數檔: {path}（請先執行 --tune 或 --tune-only）")
    with open(path, encoding='utf-8') as f:
        info = json.load(f)
    if info.get('mode') != mode:
        print(f"[警告] 這組參數是以 '{info.get('mode')}' 組別搜尋的，現在要用在 '{mode}' 組別")
    if feature_cols is not None and info.get('feature_cols') != list(feature_cols):
        print("[警告] 目前的特徵欄位與搜尋當時不同，這組參數可能不是最佳的")
    print(f"📥 已載入最佳參數: {path}（搜尋時 CV {info.get('metric')} = {info.get('cv_score'):.4f}，"
          f"基準 {info.get('baseline_cv_score'):.4f}）")
    return info['params'], info


def evaluate_baseline_on_test(result, X_train, y_train, X_test, y_test):
    """用調參前的參數（固定樹數、不用測試集 early stopping）訓練並在測試集評估，作為調參前後的對照。"""
    from model import train_model
    from metrics import calculate_evaluation_metrics
    params = {k: v for k, v in result['baseline']['params'].items()}
    params['n_estimators'] = result['baseline']['final_rounds']
    print(f"⏳ 訓練調參前的基準模型（{params['n_estimators']} 輪）供對照...")
    model = train_model(X_train, y_train, X_test, y_test, model_params=params)
    return calculate_evaluation_metrics(y_test, model.predict(X_test))