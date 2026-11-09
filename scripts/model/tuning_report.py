"""
Optuna 搜尋報告：用 Markdown 描述策略、過程與結果，並輸出圖表。
輸出到 <run 資料夾>/optuna/：optuna_report.md、optuna_history.png、optuna_param_importance.png、
optuna_param_slices.png、optuna_fold_scores.png（trials.csv、best_params.json 由 tuning.py 輸出）。
write_report 可呼叫兩次：搜尋完成時一次，訓練並評估測試集後再補上「測試集對照」一次。
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

XGB_DEFAULTS = {'learning_rate': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 1.0,
                'colsample_bytree': 1.0, 'reg_lambda': 1.0, 'reg_alpha': 0.0, 'gamma': 0.0}


# ---------- 小工具 ----------
def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _duration(sec):
    sec = int(round(sec))
    return f"{sec // 60} 分 {sec % 60} 秒" if sec >= 60 else f"{sec} 秒"


def _improvement_pct(base, best, direction):
    if base == 0:
        return float('nan')
    return (base - best) / abs(base) * 100 if direction == 'minimize' else (best - base) / abs(base) * 100


def _complete(df):
    return df[df['state'] == 'COMPLETE'].sort_values('number')


def _short(k):
    return k.split(' (')[0]


# ---------- 圖 ----------
def plot_history(result, path):
    df = result['trials_df']
    comp, pruned = _complete(df), df[df['state'] == 'PRUNED']
    fig, ax = plt.subplots(figsize=(9, 5), dpi=config.FIG_DPI)
    ax.scatter(comp['number'], comp['value'], s=22, color='tab:blue', label='completed trial')
    best = comp['value'].cummin() if result['direction'] == 'minimize' else comp['value'].cummax()
    ax.step(comp['number'], best, where='post', color='tab:red', linewidth=1.8, label='best so far')
    ax.axhline(result['baseline']['score'], color='gray', linestyle='--', label='baseline (before tuning)')
    if len(pruned):
        y0 = ax.get_ylim()[0]
        ax.plot(pruned['number'], [y0] * len(pruned), '|', color='gray', markersize=9, label=f'pruned ({len(pruned)})')
    ax.set_xlabel('Trial number')
    ax.set_ylabel(f"CV {result['metric'].upper()}")
    ax.set_title('Optimization history')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)


def plot_importance(result, path):
    imp = result.get('importances')
    if not imp:
        return False
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(items) + 1.5), dpi=config.FIG_DPI)
    ax.barh([k for k, _ in items], [v for _, v in items], color='tab:orange')
    ax.set_xlabel('Importance (fANOVA)')
    ax.set_title('Hyperparameter importance')
    ax.grid(alpha=0.3, axis='x')
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    return True


def plot_slices(result, path):
    space = result['search_space']
    comp = _complete(result['trials_df'])
    names = [n for n in space if f"p_{n}" in comp.columns]
    if comp.empty or not names:
        return False
    ncols = min(4, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), dpi=config.FIG_DPI, squeeze=False)
    best_no = result['best']['number']
    for ax, n in zip(axes.ravel(), names):
        ax.scatter(comp[f"p_{n}"], comp['value'], s=14, color='tab:blue', alpha=0.7)
        b = comp[comp['number'] == best_no]
        ax.scatter(b[f"p_{n}"], b['value'], s=55, color='tab:red', zorder=3, label='best')
        if space[n][3]:
            ax.set_xscale('log')
        ax.set_xlabel(n)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    axes[0, 0].set_ylabel(f"CV {result['metric'].upper()}")
    axes[0, 0].legend()
    fig.suptitle('Parameter vs. score (completed trials)')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    return True


def plot_fold_scores(result, path):
    b, t = result['baseline']['fold_scores'], result['best']['fold_scores']
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=config.FIG_DPI)
    ax.bar(x - 0.2, b, 0.4, label='baseline', color='gray')
    ax.bar(x + 0.2, t, 0.4, label='best trial', color='tab:red')
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i + 1}" for i in x])
    ax.set_ylabel(f"{result['metric'].upper()} on validation fold")
    ax.set_title('Per-fold validation score')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    fig.savefig(path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)


# ---------- 報告 ----------
def _boundary_notes(result):
    notes = []
    for name, (kind, lo, hi, log) in result['search_space'].items():
        v = result['best']['searched_params'].get(name)
        if v is None or hi == lo:
            continue
        pos = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log and v > 0 and lo > 0 \
            else (v - lo) / (hi - lo)
        if pos <= 0.05:
            notes.append(f"`{name}` 的最佳值 {_fmt(v)} 貼近搜尋下界 {_fmt(lo)}，可考慮把下界再調低")
        elif pos >= 0.95:
            notes.append(f"`{name}` 的最佳值 {_fmt(v)} 貼近搜尋上界 {_fmt(hi)}，可考慮把上界再調高")
    return notes


def write_report(result, output_dir, test_results=None):
    out = os.path.join(output_dir, "optuna")
    os.makedirs(out, exist_ok=True)
    plot_history(result, os.path.join(out, "optuna_history.png"))
    has_imp = plot_importance(result, os.path.join(out, "optuna_param_importance.png"))
    has_slice = plot_slices(result, os.path.join(out, "optuna_param_slices.png"))
    plot_fold_scores(result, os.path.join(out, "optuna_fold_scores.png"))

    s, m, direction = result['settings'], result['metric'].upper(), result['direction']
    df, base, best = result['trials_df'], result['baseline'], result['best']
    comp = _complete(df)
    n_done = int((df['state'] == 'COMPLETE').sum())
    n_pruned = int((df['state'] == 'PRUNED').sum())
    n_fail = int((df['state'] == 'FAIL').sum())
    gain = _improvement_pct(base['score'], best['score'], direction)
    better = '低' if direction == 'minimize' else '高'
    L = []

    L.append(f"# Optuna 超參數搜尋報告（訓練組別：{result['mode']}）\n")
    L.append("## 1. 目的與結論摘要\n")
    L.append(f"- 目的：自動搜尋 XGBoost 超參數，降低交叉驗證的 {m}。")
    L.append(f"- 資料：{result['n_train_rows']} 筆訓練資料、{len(result['feature_cols'])} 個輸入特徵；**測試集完全不參與搜尋**。")
    L.append(f"- 結果：最佳 CV {m} = **{best['score']:.4f}**（第 {best['number']} 次試驗），調參前基準 = {base['score']:.4f}，"
             f"{'改善' if gain >= 0 else '退步'} **{abs(gain):.1f}%**。")
    L.append(f"- 規模：共 {n_done + n_pruned + n_fail} 次試驗（完成 {n_done}、剪枝 {n_pruned}、失敗 {n_fail}），耗時 {_duration(result['elapsed_sec'])}。")
    L.append(f"- 最終使用的樹數：{best['final_rounds']}（最佳試驗各摺 early stopping 輪次 {best['rounds']} 的平均 × {s['rounds_scale']}）。\n")

    L.append("## 2. 搜尋策略\n")
    L.append("### 2.1 驗證方式：訓練資料內的時間序列交叉驗證\n")
    L.append("- **為什麼不用測試集調參**：測試集若被拿來挑參數，測試指標就不再代表模型對新資料的表現。因此搜尋只用訓練資料，測試集留到最後才評估。")
    L.append(f"- **為什麼按時間、以日期切摺**：資料有時間順序，且同一天、不同測站的資料高度相關。採「擴展視窗」切成 {len(result['folds'])} 摺：每摺用較早的資料訓練、之後一段時間驗證，"
             "並以整天為單位切分（依日期欄位），避免用未來預測過去，也避免同一天同時出現在訓練與驗證。\n")
    L.append(_md_table(['摺', '訓練期間', '訓練筆數', '驗證期間', '驗證筆數'],
                       [[f['fold'], f['train_period'], f['n_train'], f['val_period'], f['n_val']] for f in result['folds']]))
    L.append("")
    L.append(f"- 每摺最多訓練 {s['max_rounds']} 棵樹，並以該摺的驗證集做 early stopping（{s['early_stopping']} 輪沒進步就停），因此樹數不需要列入搜尋。\n")

    L.append("### 2.2 優化目標與搜尋演算法\n")
    L.append(f"- **目標**：各摺驗證集 {m} 的平均（{'越小越好' if direction == 'minimize' else '越大越好'}）。")
    L.append(f"- **取樣器：TPE**（Tree-structured Parzen Estimator，multivariate，seed={s['seed']}）。它根據已完成試驗的結果，把參數分成「表現好」與「表現差」兩群，"
             f"之後傾向在好的那群較常出現的區域取樣；前 {s['startup_trials']} 次先隨機探索以建立初始資訊，multivariate 模式會考慮參數之間的交互作用。")
    L.append(f"- **剪枝：中位數剪枝**。每跑完一摺就回報目前的平均分數，若比同一階段其他試驗的中位數更差，這次試驗就提早結束，把時間留給更有希望的組合"
             f"（前 {s['pruner_startup_trials']} 次試驗不剪枝）。注意：第一摺的訓練資料最少，剪枝主要依據前幾摺的表現。")
    L.append(f"- **基準對照**：搜尋前先用目前 `config.MODEL_PARAMS` 跑一次完全相同的交叉驗證，所有改善都以它為比較對象。")
    L.append(f"- **試驗上限**：{s['n_trials']} 次" + (f"，時間上限 {s['timeout']} 秒" if s['timeout'] else "") + "。\n")

    L.append("### 2.3 搜尋空間\n")
    L.append(_md_table(['參數', '型別', '範圍', '尺度'],
                       [[n, k, f"{_fmt(lo)} ~ {_fmt(hi)}", '對數' if lg else '線性']
                        for n, (k, lo, hi, lg) in result['search_space'].items()]))
    fx = result['fixed_params']
    L.append(f"\n固定不變的參數：{', '.join(f'`{k}={_fmt(v)}`' for k, v in fx.items()) if fx else '無'}。\n")

    L.append("## 3. 搜尋過程\n")
    L.append(f"- 完成 {n_done} 次、剪枝 {n_pruned} 次（{n_pruned / max(n_done + n_pruned, 1) * 100:.0f}%）。")
    if len(comp):
        durs = df['duration_s'].dropna()
        L.append(f"- 單次試驗耗時中位數 {durs.median():.1f} 秒；被剪枝的試驗通常較快結束。")
        comp_best = comp['value'].cummin() if direction == 'minimize' else comp['value'].cummax()
        events, prev = [], None
        for (_, r), bv in zip(comp.iterrows(), comp_best):
            if prev is None or bv != prev:
                events.append([int(r['number']), f"{bv:.4f}", f"{_improvement_pct(base['score'], bv, direction):.1f}%"])
                prev = bv
        L.append("\n最佳分數的進展（只列出刷新紀錄的試驗）：\n")
        L.append(_md_table(['試驗', f'CV {m}', '相對基準改善'], events[:15]))
        if len(events) > 15:
            L.append(f"\n（共 {len(events)} 次刷新，只列前 15 次）")
    L.append("\n![optimization history](optuna_history.png)\n")

    L.append("## 4. 結果\n")
    L.append("### 4.1 最佳參數\n")
    rows = []
    for n in result['search_space']:
        b = base['params'].get(n)
        b_txt = f"{_fmt(b)}" if b is not None else (f"{_fmt(XGB_DEFAULTS[n])}（XGBoost 預設）" if n in XGB_DEFAULTS else '-')
        rows.append([n, b_txt, _fmt(best['searched_params'][n])])
    rows.append(['n_estimators（樹數）', f"{base['final_rounds']}（各摺 early stopping 平均 × {s['rounds_scale']}）",
                 f"{best['final_rounds']}（同上）"])
    L.append(_md_table(['參數', '調參前', '最佳'], rows))
    L.append("")
    L.append("### 4.2 交叉驗證分數\n")
    L.append(_md_table(['', *[f"第 {i + 1} 摺" for i in range(len(best['fold_scores']))], '平均'],
                       [['調參前', *[f"{x:.4f}" for x in base['fold_scores']], f"{base['score']:.4f}"],
                        ['最佳試驗', *[f"{x:.4f}" for x in best['fold_scores']], f"{best['score']:.4f}"]]))
    L.append("\n![fold scores](optuna_fold_scores.png)\n")

    if has_imp:
        L.append("### 4.3 參數重要性（fANOVA）\n")
        L.append(_md_table(['參數', '重要性'], [[k, f"{v:.3f}"] for k, v in
                                               sorted(result['importances'].items(), key=lambda kv: -kv[1])]))
        L.append("\n重要性越高，代表該參數對 CV 分數的影響越大；影響很小的參數可以固定，下一輪把搜尋力氣集中在重要參數上。")
        L.append("\n![importance](optuna_param_importance.png)\n")
    if has_slice:
        L.append("各參數與分數的關係（紅點為最佳試驗）：\n")
        L.append("![slices](optuna_param_slices.png)\n")

    if len(comp):
        top = comp.sort_values('value', ascending=(direction == 'minimize')).head(5)
        pcols = [f"p_{n}" for n in result['search_space'] if f"p_{n}" in top.columns]
        L.append("### 4.4 前 5 名試驗\n")
        L.append(_md_table(['試驗', f'CV {m}', *[c[2:] for c in pcols]],
                           [[int(r['number']), f"{r['value']:.4f}", *[_fmt(r[c]) for c in pcols]] for _, r in top.iterrows()]))
        L.append("\n前幾名的參數越接近，代表搜尋越收斂；差異很大但分數相近，代表有多組參數同樣好。\n")

    if test_results:
        L.append("## 5. 測試集對照（搜尋完成後才評估）\n")
        L.append(f"兩個模型都用**整個訓練集**、固定樹數（調參前 {test_results['baseline_rounds']} 棵、調參後 {test_results['tuned_rounds']} 棵）訓練，"
                 "不使用測試集做 early stopping，因此兩者的測試指標可以公平比較。\n")
        keys = list(test_results['tuned'].keys())
        L.append(_md_table(['指標', '調參前', '調參後'],
                           [[_short(k), *[(f"{test_results[w][k]:.2f}%" if 'MAPE' in k else f"{test_results[w][k]:.4f}")
                                       for w in ('baseline', 'tuned')]] for k in keys]))
        rb, rt = test_results['baseline'], test_results['tuned']
        rk = [k for k in keys if _short(k) == 'RMSE']
        if rk:
            L.append(f"\n測試集 RMSE：{rb[rk[0]]:.4f} → {rt[rk[0]]:.4f}（{_improvement_pct(rb[rk[0]], rt[rk[0]], 'minimize'):+.1f}% 改善）。\n")
    sec = 6 if test_results else 5

    L.append(f"## {sec}. 判讀與建議\n")
    obs = []
    if gain < 1:
        obs.append("相對基準的改善很小（< 1%），代表目前的預設參數已經接近這個搜尋空間內的最佳；進一步提升較可能來自特徵或資料，而不是調參。")
    elif gain < 5:
        obs.append("改善幅度中等，屬於調參常見的範圍。")
    else:
        obs.append("改善幅度明顯，代表原本的參數有調整空間。")
    if best['number'] >= 0.8 * max(s['n_trials'], 1):
        obs.append(f"最佳解出現在第 {best['number']} 次，已接近試驗上限，搜尋可能還沒收斂，可增加 `OPTUNA_N_TRIALS` 後續跑（設定 `OPTUNA_STORAGE` 可接續）。")
    obs += _boundary_notes(result)
    if n_pruned == 0 and n_done > 10:
        obs.append("沒有任何試驗被剪枝，剪枝沒有發揮作用，可考慮調低 `OPTUNA_PRUNER_STARTUP_TRIALS`。")
    if best['fold_scores'] and np.std(best['fold_scores']) > 0.25 * abs(np.mean(best['fold_scores'])):
        obs.append("各摺分數差異很大（標準差超過平均的 25%），代表不同時期的資料表現不穩定，單一組最佳參數的可靠度有限。")
    L += [f"- {o}" for o in obs]
    L.append("\n### 注意事項\n")
    L.append(f"- 交叉驗證的最佳分數是「從 {n_done + n_pruned} 次試驗中挑出最好的一個」，會略微樂觀；真正的泛化表現請看測試集。")
    L.append("- 這組最佳參數是針對**目前的訓練資料、特徵與組別**搜尋的；換組別（春季 / 全年）或改特徵後，建議重新搜尋。")
    L.append("- 比較春季與全年模型時，兩者都應各自搜尋（或都不搜尋），否則比較的是「調過參」與「沒調過參」的差異。")
    L.append(f"\n### 如何使用\n")
    L.append("```bash")
    L.append(f"python run_no2_xgb.py --mode {result['mode']} --params latest      # 沿用這組最佳參數訓練")
    L.append("```")
    L.append(f"\n最佳參數與所有試驗記錄在同一資料夾的 `best_params.json`、`trials.csv`。")

    path = os.path.join(out, "optuna_report.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")
    print(f"📝 Optuna 報告已儲存至: {path}")
    return path