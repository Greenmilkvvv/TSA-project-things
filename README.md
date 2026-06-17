# TSA-project-things

上证综指波动率建模与预测——基于 ARIMA-GARCH 族模型的实证分析

## 项目概述

本项目对上海证券交易所综合指数（上证综指）的日度收益率进行时间序列建模，系统地构建并比较了四种 ARIMA-GARCH 族模型：
- **GARCH(1,1)-Normal**
- **GARCH(1,1)-t**
- **EGARCH(1,1)-t**
- **APARCH(1,1)-t**

涵盖描述性统计、单位根检验、ARCH 效应检验、ARIMA 均值建模、波动率建模、滚动窗口样本外预测、模型评估与比较等完整流程。

## 项目结构

```
TSA-project-things/
├── SH_Index.csv                     # 原始数据：上证综指日度行情（CSMAR 数据库）
├── model_core.py                    # 核心建模模块 (ARIMA + GARCH 族模型拟合)
├── plot_figures.py                  # 独立制图模块 (读取中间结果 CSV 绘制所有图表)
├── generate_tables.py               # 学术表格生成器 (输出 CSV + LaTeX)
├── README.md                        # 项目说明
│
├── results/
│   ├── analysis_log.txt             # 建模过程日志
│   ├── output/                      # 中间结果 (供 plot_figures / generate_tables 读取)
│   │   ├── descriptive_stats.csv     # 描述性统计
│   │   ├── unit_root_results.csv     # ADF / KPSS 单位根检验
│   │   ├── arch_test_results.csv     # ARCH-LM 效应检验
│   │   ├── arima_residuals.csv       # ARIMA 建模残差
│   │   ├── model_parameters.csv      # GARCH 族模型参数估计
│   │   ├── model_comparison.csv      # 模型信息准则比较
│   │   ├── standardized_residuals.csv# 标准化残差
│   │   ├── std_resid_stats.csv       # 标准化残差基本统计
│   │   ├── conditional_volatility.csv# 条件波动率序列
│   │   ├── rolling_forecast_predictions.csv  # 滚动预测值
│   │   ├── forecast_evaluation_rolling.csv   # 滚动预测评估指标
│   │   ├── dm_test_rolling.csv       # Diebold-Mariano 检验
│   │   └── final_summary.csv         # 建模流程摘要
│   │
│   ├── figures/                     # 图表 (PDF 矢量格式)
│   │   ├── fig_price_series.pdf      # 价格序列图
│   │   ├── fig_return_series.pdf     # 收益率序列图
│   │   ├── fig_arima_acf.pdf         # ARIMA残差 ACF
│   │   ├── fig_arima_pacf.pdf        # ARIMA残差 PACF
│   │   ├── fig_arima_acf_abs.pdf     # 残差绝对值 ACF
│   │   ├── fig_arima_acf_sq.pdf      # 残差平方 ACF
│   │   ├── fig_qq.pdf               # Q-Q 正态性图
│   │   ├── fig_std_resid_ts.pdf      # 标准化残差时序图
│   │   ├── fig_std_resid_hist.pdf    # 标准化残差直方图
│   │   ├── fig_cond_vol.pdf          # 条件波动率图
│   │   ├── fig_forecast_rolling.pdf  # 滚动预测图
│   │   └── fig_eval_bar.pdf          # 预测评估柱状图
│   │
│   └── tables/                      # 学术表格
│       ├── table_descriptive_stats.csv
│       ├── table_unit_root.csv
│       ├── table_arch_test.csv
│       ├── table_arima_residual.csv
│       ├── table_model_parameters.csv
│       ├── table_model_comparison.csv
│       ├── table_std_residuals.csv
│       ├── table_forecast_eval.csv
│       ├── table_dm_test.csv
│       ├── table_final_summary.csv
│       └── all_tables.tex            # 所有表格的 LaTeX 代码合集
```

## 依赖

- Python 3.10
- pandas, numpy, scipy
- statsmodels
- arch (ARCH modeling library)
- matplotlib

## Skill

- 可视化 skill 参考 [ChenLiu-1996/figures4papers](https://github.com/ChenLiu-1996/figures4papers)
- 写作 skill 参考 [yanlin-cheng/skill-thesis-writer](https://github.com/yanlin-cheng/skill-thesis-writer)

## 运行说明

### 第一步：核心建模
```bash
py -3.10 model_core.py
```
加载 `SH_Index.csv`，执行全部分析流程：
1. 数据预处理与描述性统计
2. 单位根检验（ADF / KPSS）
3. ARIMA 均值方程建模
4. ARCH 效应检验（LM 检验）
5. GARCH 族模型拟合（四种模型）
6. 标准化残差诊断
7. 滚动窗口样本外预测
8. 预测评估与 Diebold-Mariano 检验

中间结果一律写入 `results/output/` 目录。

### 第二步：生成图表
```bash
py -3.10 plot_figures.py
```
从 `results/output/` 读取中间结果 CSV，绘制 11 张图表，输出至 `results/figures/`。

### 第三步：生成学术表格
```bash
py -3.10 generate_tables.py
```
从 `results/output/` 和原始数据读取，生成 10 张学术表格：
- 每张表格同时输出 `.csv` 文件（UTF-8 BOM 编码）
- 合并所有表格为一份 `all_tables.tex`，可直接 `\input{all_tables.tex}` 嵌入论文

## 表格说明

| 编号 | 表格名称 | LaTeX Label | 说明 |
|------|---------|-------------|------|
| 1 | 描述性统计 | `tab:desc_stats` | 日对数收益率的均值、标准差、偏度、峰度、Jarque-Bera 检验 |
| 2 | 单位根检验 | `tab:unit_root` | ADF 和 KPSS 单位根检验结果 |
| 3 | ARCH 效应检验 | `tab:arch_test` | 残差序列的 ARCH-LM 条件异方差检验 |
| 4 | ARIMA 残差白噪声检验 | `tab:arima_resid` | ARIMA 拟合残差的 Ljung-Box 白噪声检验 |
| 5 | 模型参数估计 | `tab:model_params` | 四种 GARCH 族模型参数估计值与显著性 |
| 6 | 模型信息准则比较 | `tab:model_comp` | AIC / BIC / LogLik 多模型比较 |
| 7 | 标准化残差统计 | `tab:std_resid` | 标准化残差均值、标准差、偏度、峰度 |
| 8 | 滚动预测评估 | `tab:forecast_eval` | RMSE、MAE、SMAPE、QLIKE 预测评估指标 |
| 9 | Diebold-Mariano 检验 | `tab:dm_test` | 模型预测精度差异显著性检验 |
| 10 | 建模流程摘要 | `tab:summary` | 建模全流程关键参数汇总 |

## 参考文献

- Zhang, J. (2026). An empirical analysis of China's Shenzhen Composite Index based on ARIMA-ARCH model. *International Journal of World Economic Research*, 1(1), 91-98.
- Xu, Y., Xia, Z., Wang, C., Gong, W., Liu, X., & Su, X. (2021). An empirical analysis of the price volatility characteristics of China's soybean futures market based on ARIMA-GJR-GARCH model. *Journal of Mathematics*, 2021, 7765325.
- Yao, Q. (2025). ARIMA-GARCH and ARIMA-EGARCH-t models in fitting and forecasting volatility of the Shanghai Composite Index. *Proceedings of the 2nd International Conference on Business, Management and Sustainability (ICBMS 2025)*, 10, 383-390.