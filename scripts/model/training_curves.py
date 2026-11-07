"""
每一輪 (boosting round) 的評估指標趨勢：
計算訓練集與測試集在第 1 ~ N 輪的 r、R2、RMSE、MD、MAPE，並每個指標輸出一張圖。
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

import config
from metrics import calculate_evaluation_metrics


def _short_name(key):
    """'RMSE (Root Mean Square Error)' -> 'RMSE'；'MAPE (%)' -> 'MAPE'"""
    return key.split(' (')[0]


def _best_round(model):
    """回傳 early stopping 選出的最佳輪次（從 1 起算）；沒有則回傳 None。"""
    try:
        return int(model.best_iteration) + 1
    except Exception:
        return None


def compute_round_metrics(model, X_train, y_train, X_test, y_test, step=None):
    """
    逐輪計算指標。第 k 輪的預測 = 只使用前 k 棵樹 (iteration_range=(0, k))。
    回傳長表 DataFrame：欄位 round, dataset, r, R2, RMSE, MD, MAPE
    """
    booster = model.get_booster()
    n_rounds = booster.num_boosted_rounds()

    if step is None:
        step = config.CURVE_EVAL_STEP
    if step is None:                                   # 自動：最多約 200 個取樣點
        step = max(1, math.ceil(n_rounds / config.CURVE_MAX_POINTS))
    rounds = list(range(step, n_rounds + 1, step))
    if rounds[-1] != n_rounds:
        rounds.append(n_rounds)

    # 訓練集太大時抽樣，避免逐輪預測過慢（測試集不抽樣）
    if len(X_train) > config.CURVE_TRAIN_MAX_SAMPLES:
        idx = np.random.RandomState(config.SHAP_RANDOM_STATE).choice(
            len(X_train), config.CURVE_TRAIN_MAX_SAMPLES, replace=False)
        X_tr, y_tr = X_train.iloc[idx], y_train.iloc[idx]
        print(f"   - 訓練集共 {len(X_train)} 筆，趨勢圖僅抽樣 {len(X_tr)} 筆計算")
    else:
        X_tr, y_tr = X_train, y_train

    datasets = {
        'train': (xgb.DMatrix(X_tr), y_tr.to_numpy()),
        'test': (xgb.DMatrix(X_test), y_test.to_numpy()),
    }

    print(f"📈 逐輪計算指標：共 {n_rounds} 輪，每 {step} 輪取樣一次（{len(rounds)} 個點）...")
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)   # 前幾輪預測幾乎相同時 r 可能為 NaN
        for k in rounds:
            for name, (dmat, y_true) in datasets.items():
                y_pred = booster.predict(dmat, iteration_range=(0, k))
                m = calculate_evaluation_metrics(y_true, y_pred)
                row = {'round': k, 'dataset': name}
                row.update({_short_name(key): v for key, v in m.items()})
                records.append(row)
    return pd.DataFrame(records)


_METRIC_LABELS = {
    'r': 'r (Correlation Coefficient)',
    'R2': 'R² (Coefficient of Determination)',
    'RMSE': 'RMSE',
    'MD': 'MD (Mean Deviation)',
    'MAPE': 'MAPE (%)',
}


def _draw_metric(ax, df, metric, best_round):
    for name, color in (('train', 'tab:blue'), ('test', 'tab:red')):
        sub = df[df['dataset'] == name]
        ax.plot(sub['round'], sub[metric], label=name, color=color, linewidth=1.6)
    if best_round is not None:
        ax.axvline(best_round, color='gray', linestyle='--', linewidth=1,
                   label=f'best round = {best_round}')
    if metric == 'MD':
        ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Boosting round')
    ax.set_ylabel(_METRIC_LABELS[metric])
    ax.set_title(f'{_METRIC_LABELS[metric]} vs. boosting round')
    ax.grid(alpha=0.3)
    ax.legend()


def plot_round_metrics(df, model, output_dir):
    """每個指標一張圖，另外輸出一張總覽圖與逐輪 CSV。"""
    curve_dir = os.path.join(output_dir, "Training_Curves")
    os.makedirs(curve_dir, exist_ok=True)
    best_round = _best_round(model)
    metrics = list(_METRIC_LABELS)

    df.to_csv(os.path.join(curve_dir, "round_metrics.csv"), index=False, encoding='utf-8-sig')

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=config.FIG_DPI)
        _draw_metric(ax, df, metric, best_round)
        fig.savefig(os.path.join(curve_dir, f"curve_{metric}.png"),
                    bbox_inches='tight', dpi=config.FIG_DPI)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=config.FIG_DPI)
    for ax, metric in zip(axes.ravel(), metrics):
        _draw_metric(ax, df, metric, best_round)
    axes.ravel()[-1].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(curve_dir, "curves_overview.png"),
                bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)

    print(f"✅ 逐輪指標趨勢圖與 CSV 已儲存至: {curve_dir}")
    return curve_dir