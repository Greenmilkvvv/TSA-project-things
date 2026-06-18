#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA 模型诊断表格生成器
计算 ARIMA(3,0,3) 在训练集（全样本 6399 obs）上的检验统计量
输出 LaTeX 表格代码
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 1. 加载数据 =====
df = pd.read_csv(os.path.join(BASE_DIR, 'SH_Index.csv'), encoding='utf-8')
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values('日期').reset_index(drop=True)
df['log_return'] = np.log(df['收盘'] / df['收盘'].shift(1)) * 100.0
df = df.dropna(subset=['log_return']).reset_index(drop=True)
df = df[df['日期'] >= '2000-01-01'].reset_index(drop=True)
r = df['log_return'].values
n = len(r)
print(f"样本量 N = {n}")

# ===== 2. 拟合 ARIMA(3,0,3) =====
print("拟合 ARIMA(3,0,3)...")
model = ARIMA(r, order=(3, 0, 3))
fit = model.fit(method_kwargs={'maxiter': 500})
print(f"AIC = {fit.aic:.4f}")
print(f"BIC = {fit.bic:.4f}")
print(f"LogLik = {fit.llf:.4f}")
print(f"df_model = {fit.df_model}, df_resid = {fit.df_resid}")

# ===== 3. 残差与 Ljung-Box Q 统计量 =====
resid = fit.resid
if hasattr(resid, 'dropna'):
    resid = resid.dropna().values
else:
    resid = np.array(resid)
resid = resid[~np.isnan(resid)]
n_resid = len(resid)

lags_list = [6, 12, 18, 24, 30]
print(f"\nLjung-Box 检验 (残差长度={n_resid}):")
lb_results = acorr_ljungbox(resid, lags=lags_list, return_df=True)
print(lb_results)

# 格式化 Q 统计量输出
q_rows = []
for lag in lags_list:
    q_val = lb_results.loc[lag, 'lb_stat']
    p_val = lb_results.loc[lag, 'lb_pvalue']
    sig = ''
    if p_val < 0.01:
        sig = '***'
    elif p_val < 0.05:
        sig = '**'
    elif p_val < 0.10:
        sig = '*'
    q_rows.append((lag, q_val, p_val, sig))
    print(f"  Q({lag}) = {q_val:.4f}, p = {p_val:.6f} {sig}")

# ===== 4. 计算 R² =====
ss_res = np.sum(resid ** 2)
ss_tot = np.sum((r - np.mean(r)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"\nR² = {r_squared:.6f}")

# ===== 5. 残差自由度 =====
df_residuals = n_resid - fit.df_model - 1  # N - p - q - 1 (mean)
print(f"残差自由度 = {df_residuals:.0f}")

# ===== 6. 生成 LaTeX 表格 =====
# 按照用户提供的示例格式
latex_table = r"""\begin{table}[htbp]
\centering
\caption{ARIMA(3,0,3) 模型检验统计量}
\label{tab:arima_diagnostics}
\begin{tabular}{l c c}
\toprule
项目 & 符号 & 数值 \\
\midrule
"""

latex_table += f"残差自由度 & Df Residuals & {df_residuals:.0f} \\\\\n"
latex_table += f"样本量 & $N$ & {n} \\\\\n"

# Q 统计量行
first_q = True
for i, (lag, q_val, p_val, sig) in enumerate(q_rows):
    # p 值格式: 3 位小数 + 星号 (与示例一致)
    p_str = f"{p_val:.3f}"
    stars = ''
    if p_val < 0.01:
        stars = '***'
    elif p_val < 0.05:
        stars = '**'
    elif p_val < 0.10:
        stars = '*'
    p_formatted = f"{p_str}{stars}"
    
    if i == 0:
        # 第一行: \multirow{5}{*}{$Q$ 统计量}
        latex_table += f"\\multirow{{5}}{{*}}{{$Q$ 统计量}} & $Q_{{{lag}}}(P)$ & {q_val:.3f} ({p_formatted}) \\\\\n"
    else:
        latex_table += f" & $Q_{{{lag}}}(P)$ & {q_val:.3f} ({p_formatted}) \\\\\n"

# 信息准则
latex_table += f"\\multirow{{2}}{{*}}{{信息准则}} & AIC & {fit.aic:.3f} \\\\\n"
latex_table += f" & BIC & {fit.bic:.3f} \\\\\n"

# 拟合优度
latex_table += f"拟合优度 & $R^2$ & {r_squared:.3f} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""

print("\n" + "=" * 60)
print("生成的 LaTeX 表格代码：")
print("=" * 60)
print(latex_table)

# 保存到文件
out_path = os.path.join(BASE_DIR, 'results', 'tables', 'table_arima_diagnostics.tex')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(latex_table)
print(f"\n表格已保存至: {out_path}")

# 同时保存 CSV 方便查阅
labels = ['Q6', 'Q6_p', 'Q12', 'Q12_p', 'Q18', 'Q18_p', 'Q24', 'Q24_p', 'Q30', 'Q30_p']
values = []
for i in range(5):
    values.append(f"{q_rows[i][1]:.4f}")
    values.append(f"{q_rows[i][2]:.6f}")
csv_data = {
    '项目': ['残差自由度', '样本量N'] + labels + ['AIC', 'BIC', 'R2'],
    '数值': [df_residuals, n] + values +
            [f"{fit.aic:.4f}", f"{fit.bic:.4f}", f"{r_squared:.6f}"]
}
csv_df = pd.DataFrame(csv_data)
csv_path = os.path.join(BASE_DIR, 'results', 'tables', 'table_arima_diagnostics.csv')
csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"CSV 已保存至: {csv_path}")