"""
SHAP 計算與圖表：Beeswarm、Bar、Dependence。
"""
import os
import matplotlib.pyplot as plt
import shap
import config


def compute_shap_values(model, X_test):
    """對測試集（超過上限時隨機抽樣）計算 SHAP values，回傳 (shap_values, X_sample)。"""
    explainer = shap.TreeExplainer(model)
    n = config.SHAP_SAMPLE_SIZE
    X_sample = X_test.sample(n, random_state=config.SHAP_RANDOM_STATE) if len(X_test) > n else X_test
    return explainer(X_sample), X_sample


def plot_beeswarm(shap_values, X_sample, shap_dir):
    plt.subplots(figsize=(10, 8), dpi=config.FIG_DPI)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot for NO2 Prediction", fontsize=14)
    plt.savefig(os.path.join(shap_dir, "shap_summary_beeswarm.png"),
                bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close()


def plot_bar(shap_values, X_sample, shap_dir):
    plt.subplots(figsize=(10, 8), dpi=config.FIG_DPI)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean Absolute)", fontsize=14)
    plt.savefig(os.path.join(shap_dir, "shap_feature_importance_bar.png"),
                bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close()


def plot_dependence(shap_values, X_sample, shap_dir):
    feature = (config.SHAP_DEPENDENCE_FEATURE
               if config.SHAP_DEPENDENCE_FEATURE in X_sample.columns
               else X_sample.columns[0])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=config.FIG_DPI)
    shap.dependence_plot(feature, shap_values.values, X_sample, ax=ax, show=False)
    plt.title(f"SHAP Dependence Plot: {feature}", fontsize=14)
    plt.savefig(os.path.join(shap_dir, f"shap_dependence_{feature}.png"),
                bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close()


def generate_shap_plots(model, X_test, output_dir):
    """計算並產出全部 SHAP 圖表，存入 output_dir/SHAP_Analysis。"""
    print("=" * 75)
    print("🔬 開始計算 SHAP Values 並繪製解釋性圖表...")
    print("=" * 75)

    shap_dir = os.path.join(output_dir, "SHAP_Analysis")
    os.makedirs(shap_dir, exist_ok=True)

    shap_values, X_sample = compute_shap_values(model, X_test)
    plot_beeswarm(shap_values, X_sample, shap_dir)
    plot_bar(shap_values, X_sample, shap_dir)
    plot_dependence(shap_values, X_sample, shap_dir)
    print(f"✅ 所有 SHAP 圖表已儲存至: {shap_dir}")