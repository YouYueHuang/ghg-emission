"""
模型評估指標：r, R2, RMSE, MD, MAPE。
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_evaluation_metrics(y_true, y_pred):
    """計算評估指標: r, R2, RMSE, MD, MAPE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    md = np.mean(y_pred - y_true)

    valid_mask = y_true != 0
    if np.any(valid_mask):
        mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100
    else:
        mape = np.nan

    return {
        'r (Correlation Coefficient)': r,
        'R2 (Coefficient of Determination)': r2,
        'RMSE (Root Mean Square Error)': rmse,
        'MD (Mean Deviation)': md,
        'MAPE (%)': mape,
    }


def print_metrics(metrics):
    print("=" * 75)
    print("📊 測試集模型預測成效評估 (Evaluation Metrics):")
    for k, v in metrics.items():
        if '%' in k:
            print(f"   - {k:<35}: {v:.2f}%")
        else:
            print(f"   - {k:<35}: {v:.4f}")
    print("=" * 75)