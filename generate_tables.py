#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_tables.py  —  学术表格生成器
参考论文: ARIMA-GARCH 对上证综指的波动率建模
输入 : results/output/*.csv + SH_Index.csv
输出 : results/tables/table_*.csv + results/tables/all_tables.tex
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf  # 仅用于Ljung-Box, 不在表格脚本中
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'results', 'output')
TABLE_DIR = os.path.join(BASE_DIR, 'results', 'tables')
DATA_PATH = os.path.join(BASE_DIR, 'SH_Index.csv')
OUT_TEX = os.path.join(TABLE_DIR, 'all_tables.tex')

os.makedirs(TABLE_DIR, exist_ok=True)

print("=" * 60)
print("  学术表格生成器  —  generate_tables.py")
print("=" * 60)

# ====================== 0. 辅助:  LaTeX表格模板 ======================
LATEX_HEADER = r"""% ============================================================
% 所有表格  —  LaTeX代码
% 由 generate_tables.py 自动生成
% 使用方式: \input{all_tables.tex}  或复制对应表格到论文正文
% ============================================================

\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}

"""


def csv_to_latex(df: pd.DataFrame, caption: str, label: str,
                 placement: str = "htbp",
                 col_fmt: str = None,
                 notes: str = None) -> str:
    """将 DataFrame 转为 LaTeX 表格 (booktabs 风格)."""
    ncols = len(df.columns)
    if col_fmt is None:
        col_fmt = 'l' + 'c' * (ncols - 1)
    lines = []
    lines.append(f"\\begin{{table}}[{placement}]")
    lines.append(f"  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append(f"    \\toprule")
    # 表头
    header = ' & '.join(str(c) for c in df.columns)
    lines.append(f"    {header} \\\\")
    lines.append(f"    \\midrule")
    # 数据行
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append('')
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"    {' & '.join(vals)} \\\\")
    lines.append(f"    \\bottomrule")
    lines.append(f"  \\end{{tabular}}")
    if notes:
        lines.append(f"  \\floatnotes{{{notes}}}")
    lines.append(f"\\end{{table}}")
    return '\n'.join(lines)


# ====================== 1. 表格1: 描述性统计 ======================
def table_descriptive_stats(df: pd.DataFrame):
    """表1: 收益率描述性统计."""
    ret = df['return'].dropna()
    n = len(ret)
    mean_ = ret.mean() * 100  # %
    std_ = ret.std() * 100    # %
    skew_ = ret.skew()
    kurt_ = ret.kurtosis()    # 超额峰度
    
    # 分位数
    q_min = ret.min() * 100
    q25 = ret.quantile(0.25) * 100
    q50 = ret.quantile(0.50) * 100
    q75 = ret.quantile(0.75) * 100
    q_max = ret.max() * 100

    # J-B 检验
    jb_stat, jb_p = stats.jarque_bera(ret)

    rows = [
        ['Statistics', 'Value'],
        ['N', n],
        ['Mean (%)', f'{mean_:.4f}'],
        ['Std. Dev. (%)', f'{std_:.4f}'],
        ['Median (%)', f'{q50:.4f}'],
        ['Min (%)', f'{q_min:.4f}'],
        ['Max (%)', f'{q_max:.4f}'],
        ['Q1 (%)', f'{q25:.4f}'],
        ['Q3 (%)', f'{q75:.4f}'],
        ['Skewness', f'{skew_:.4f}'],
        ['Excess Kurtosis', f'{kurt_:.4f}'],
        ['Jarque-Bera', f'{jb_stat:.4f}'],
        ['J-B p-value', f'{jb_p:.4e}'],
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, 'table_descriptive_stats.csv'),
               index=False, header=False, encoding='utf-8-sig')
    # LaTeX
    tbl = out.copy()
    tbl.columns = ['Statistic', 'Value']
    tbl = tbl.iloc[1:]  # drop header row
    return csv_to_latex(tbl, 
                        'Descriptive statistics of daily logarithmic returns.', 
                        'tab:desc_stats')


# ====================== 2. 表格2: 单位根检验 ======================
def table_unit_root():
    """
    表2: ADF & KPSS 单位根检验。
    格式: 变量 + 检验方法 + 差分阶数 + 统计量 + p值 + AIC + 临界值(1%/5%/10%)。
    使用 multirow 合并"变量"和"检验方法"列。
    """
    df = pd.read_csv(os.path.join(DATA_DIR, 'unit_root_results.csv'), encoding='utf-8-sig')

    # 保存一份CSV供查阅
    df.to_csv(os.path.join(TABLE_DIR, 'table_unit_root.csv'),
              index=False, encoding='utf-8-sig')

    # ---- 构造 LaTeX 代码（符合用户示例格式）----
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Unit root test results (ADF \& KPSS).}")
    lines.append(r"  \label{tab:unit_root}")
    lines.append(r"  \begin{tabular}{lccrrcrrr}")
    lines.append(r"    \toprule")
    # 表头
    lines.append(r"    \multirow{2}{*}{变量} & \multirow{2}{*}{检验方法} & \multirow{2}{*}{差分阶数} "
                 r"& \multirow{2}{*}{$t$ 统计量} & \multirow{2}{*}{$p$ 值} & \multirow{2}{*}{AIC} "
                 r"& \multicolumn{3}{c}{临界值} \\")
    lines.append(r"    \cmidrule(lr){7-9}")
    lines.append(r"    & & & & & & 1\% & 5\% & 10\% \\")
    lines.append(r"    \midrule")

    # 分组: 变量名 + 检验方法 用 multirow
    # 数据按 (变量, 检验方法) 分组
    grouped = df.groupby(['变量', '检验方法'])
    for (var_name, method), grp in grouped:
        n_rows = len(grp)
        var_cell = f"\\multirow{{{n_rows}}}{{*}}{{{var_name}}}"
        method_cell = f"\\multirow{{{n_rows}}}{{*}}{{{method}}}"
        for i, (_, row) in enumerate(grp.iterrows()):
            var_part = var_cell if i == 0 else ""
            mt_part = method_cell if i == 0 else ""
            d = int(row['差分阶数'])
            stat = row['统计量']
            pval = row['p值']
            aic = row['AIC']
            crit1 = row['临界值1%']
            crit5 = row['临界值5%']
            crit10 = row['临界值10%']

            # 格式化数值
            stat_str = f"{stat:.4f}"
            pval_str = f"{pval:.4e}" if not pd.isna(pval) else "--"
            aic_str = f"{aic:.4f}" if not pd.isna(aic) else "--"
            crit1_str = f"{crit1:.4f}" if not pd.isna(crit1) else "--"
            crit5_str = f"{crit5:.4f}" if not pd.isna(crit5) else "--"
            crit10_str = f"{crit10:.4f}" if not pd.isna(crit10) else "--"

            # 显著性标记
            if not pd.isna(pval) and pval < 0.001:
                pval_str += "***"
            elif not pd.isna(pval) and pval < 0.01:
                pval_str += "**"
            elif not pd.isna(pval) and pval < 0.05:
                pval_str += "*"

            lines.append(f"    {var_part} & {mt_part} & {d} & {stat_str} & {pval_str} & {aic_str} "
                         f"& {crit1_str} & {crit5_str} & {crit10_str} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return '\n'.join(lines)


# ====================== 3. 表格3: ARCH效应检验 ======================
def table_arch_test():
    """表3: ARCH-LM 检验."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'arch_test_results.csv'), encoding='utf-8-sig')
    df.columns = ['Test', 'Statistic', 'P-value']
    df = df[['Test', 'Statistic', 'P-value']]
    df.to_csv(os.path.join(TABLE_DIR, 'table_arch_test.csv'),
              index=False, encoding='utf-8-sig')
    return csv_to_latex(df,
                        'ARCH-LM test for conditional heteroscedasticity.',
                        'tab:arch_test')


# ====================== 4. 表格4: ARIMA残差 ARCH-LM 检验 ======================
def table_arch_test_arima_resid():
    """表4: ARIMA残差的 ARCH-LM 检验."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'arch_test_arima_resid.csv'), encoding='utf-8-sig')
    df.columns = ['Test', 'Statistic', 'P-value']
    df = df[['Test', 'Statistic', 'P-value']]
    df.to_csv(os.path.join(TABLE_DIR, 'table_arch_test_arima_resid.csv'),
              index=False, encoding='utf-8-sig')
    return csv_to_latex(df,
                        'ARCH-LM test on ARIMA residuals (lag=10).',
                        'tab:arch_arima_resid')


# ====================== 5. 表格5: ARIMA残差白噪声检验 ======================
def table_arima_residual():
    """表4: ARIMA 残差 Ljung-Box 白噪声检验."""
    arima_df = pd.read_csv(os.path.join(DATA_DIR, 'arima_residuals.csv'), encoding='utf-8-sig')
    resid = arima_df['arima_residual'].dropna().values

    # Ljung-Box 不同滞后阶数
    lags = [6, 12, 18, 24]
    rows = [['Lag', 'LB Statistic', 'P-value']]
    for lag in lags:
        try:
            res = acorr_ljungbox(resid, lags=[lag], return_df=True)
            stat = res['lb_stat'].values[0]
            pval = res['lb_pvalue'].values[0]
            rows.append([lag, f'{stat:.4f}', f'{pval:.4f}'])
        except:
            rows.append([lag, 'N/A', 'N/A'])
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(TABLE_DIR, 'table_arima_residual.csv'),
               index=False, header=False, encoding='utf-8-sig')
    
    # LaTeX
    latex_tbl = tbl.copy()
    latex_tbl.columns = latex_tbl.iloc[0]
    latex_tbl = latex_tbl.iloc[1:]
    return csv_to_latex(latex_tbl,
                        'Ljung-Box test for ARIMA residual white noise.',
                        'tab:arima_resid')


# ====================== 5. 表格5: 模型参数估计 ======================
def table_model_parameters():
    """表5: GARCH族模型参数估计."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'model_parameters.csv'), encoding='utf-8-sig')
    # 重命名列, 使 LaTeX 友好
    rename_map = {
        'mu': r'$\mu$',
        'omega': r'$\omega$',
        'alpha1': r'$\alpha_1$',
        'beta1': r'$\beta_1$',
        'gamma1': r'$\gamma_1$',
        'delta': r'$\delta$',
        'nu': r'$\nu$',
        'persistence': r'$\alpha_1+\beta_1$',
        'AIC': 'AIC',
        'BIC': 'BIC',
        'LogLik': 'LogLik',
    }
    df_show = df[['模型', 'omega', 'alpha1', 'beta1', 'gamma1', 'delta', 'nu', 'persistence', 'AIC', 'BIC', 'LogLik']].copy()
    df_show = df_show.rename(columns=rename_map)
    # 数值四舍五入
    for c in df_show.columns[1:]:
        df_show[c] = df_show[c].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
    
    df_show.to_csv(os.path.join(TABLE_DIR, 'table_model_parameters.csv'),
                   index=False, encoding='utf-8-sig')
    return csv_to_latex(df_show,
                        'Parameter estimation results of GARCH-family models.',
                        'tab:model_params')


# ====================== 6. 表格6: 模型信息准则比较 ======================
def table_model_comparison():
    """表6: 模型信息准则比较."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'model_comparison.csv'), encoding='utf-8-sig')
    df = df.rename(columns={
        'alpha+beta(beta)': r'$\alpha_1+\beta_1$',
    })
    for c in ['AIC', 'BIC', 'LogLik', r'$\alpha_1+\beta_1$']:
        df[c] = df[c].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
    df.to_csv(os.path.join(TABLE_DIR, 'table_model_comparison.csv'),
              index=False, encoding='utf-8-sig')
    return csv_to_latex(df,
                        'Model comparison based on information criteria.',
                        'tab:model_comp')


# ====================== 7. 表格7: 标准化残差统计 ======================
def table_std_residuals():
    """表7: 标准化残差基本统计."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'std_resid_stats.csv'), encoding='utf-8-sig')
    for c in ['均值', '标准差', '偏度', '峰度(超额)']:
        df[c] = df[c].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
    df.to_csv(os.path.join(TABLE_DIR, 'table_std_residuals.csv'),
              index=False, encoding='utf-8-sig')
    return csv_to_latex(df,
                        'Standardized residual statistics across models.',
                        'tab:std_resid')


# ====================== 8. 表格8: 滚动预测评估 ======================
def table_forecast_eval():
    """表8: 滚动预测评估指标."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'forecast_evaluation_rolling.csv'), encoding='utf-8-sig')
    for c in ['RMSE', 'MAE', 'SMAPE(%)', 'QLIKE']:
        df[c] = df[c].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
    df.to_csv(os.path.join(TABLE_DIR, 'table_forecast_eval.csv'),
              index=False, encoding='utf-8-sig')
    return csv_to_latex(df,
                        'Rolling forecast evaluation metrics.',
                        'tab:forecast_eval')


# ====================== 9. 表格9: Diebold-Mariano 检验 ======================
def table_dm_test():
    """表9: DM 检验结果."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'dm_test_rolling.csv'), encoding='utf-8-sig')
    for c in ['DM统计量', 'DM_p值']:
        df[c] = df[c].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
    df.to_csv(os.path.join(TABLE_DIR, 'table_dm_test.csv'),
              index=False, encoding='utf-8-sig')
    return csv_to_latex(df,
                        'Diebold-Mariano test for predictive accuracy comparison.',
                        'tab:dm_test')


# ====================== 10. 表格10: 建模流程摘要 ======================
def table_final_summary():
    """表10: 最终摘要."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'final_summary.csv'), encoding='utf-8-sig')
    df.to_csv(os.path.join(TABLE_DIR, 'table_final_summary.csv'),
              index=False, encoding='utf-8-sig')
    
    # 转换为更紧凑的 LaTeX
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Summary of modeling process.}")
    lines.append(r"  \label{tab:summary}")
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \toprule")
    lines.append(r"    Item & Value \\")
    lines.append(r"    \midrule")
    for _, row in df.iterrows():
        lines.append(f"    {row[0]} & {row[1]} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return '\n'.join(lines)


# ====================== 主函数 ======================
def main():
    # 加载原始数据用于描述性统计
    df_raw = pd.read_csv(DATA_PATH, parse_dates=['日期'])
    df_raw = df_raw.sort_values('日期').reset_index(drop=True)
    df_raw['return'] = np.log(df_raw['收盘']) - np.log(df_raw['收盘'].shift(1))
    
    latex_parts = [LATEX_HEADER]
    
    # 表1
    print("  ✓ 描述性统计  →  table_descriptive_stats.csv")
    latex_parts.append(table_descriptive_stats(df_raw))
    
    # 表2
    print("  ✓ 单位根检验  →  table_unit_root.csv")
    latex_parts.append(table_unit_root())
    
    # 表3
    print("  ✓ ARCH效应检验  →  table_arch_test.csv")
    latex_parts.append(table_arch_test())
    
    # 表4: ARIMA残差 ARCH-LM 检验
    print("  ✓ ARIMA残差ARCH-LM检验  →  table_arch_test_arima_resid.csv")
    latex_parts.append(table_arch_test_arima_resid())

    # 表5: ARIMA残差白噪声检验
    print("  ✓ ARIMA残差白噪声检验  →  table_arima_residual.csv")
    latex_parts.append(table_arima_residual())
    
    # 表6
    print("  ✓ 模型参数估计  →  table_model_parameters.csv")
    latex_parts.append(table_model_parameters())
    
    # 表7
    print("  ✓ 模型信息准则比较  →  table_model_comparison.csv")
    latex_parts.append(table_model_comparison())
    
    # 表8
    print("  ✓ 标准化残差统计  →  table_std_residuals.csv")
    latex_parts.append(table_std_residuals())
    
    # 表9
    print("  ✓ 滚动预测评估  →  table_forecast_eval.csv")
    latex_parts.append(table_forecast_eval())
    
    # 表10
    print("  ✓ Diebold-Mariano检验  →  table_dm_test.csv")
    latex_parts.append(table_dm_test())
    
    # 表11
    print("  ✓ 建模流程摘要  →  table_final_summary.csv")
    latex_parts.append(table_final_summary())
    
    # 写出 LaTeX
    with open(OUT_TEX, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(latex_parts))
    
    print(f"\n  ✓ LaTeX 代码 → all_tables.tex")
    print(f"\n{'=' * 60}")
    print(f"  所有表格输出至: {TABLE_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()