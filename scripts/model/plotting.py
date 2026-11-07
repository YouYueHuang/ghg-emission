"""
XGBoost 原生特徵重要性圖。
"""
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import config


def plot_xgb_importance(model, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=config.FIG_DPI)
    xgb.plot_importance(model, ax=ax, max_num_features=config.IMPORTANCE_MAX_FEATURES,
                        height=0.6, color='darkorange')
    ax.set_title("XGBoost Feature Importance for NO2 Prediction", fontsize=14)
    path = os.path.join(output_dir, "xgboost_feature_importance.png")
    plt.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    return path