# -*- coding: utf-8 -*-
"""
上证综指波动率分析 — 独立制图模块
==================================
本模块从 results/output/ 读取 model_core.py 生成的中间结果CSV，
绘制所有分析图表，输出到 results/figures/。

此模块完全不依赖 model_core.py 的运行时对象，可离线批量重新绘图。
"""

import os, sys, warnings, logging
import numpy as np
import pandas as pd
from scipy import stats as sc_stats
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings('ignore')

# ===== 字体与数学公式 =====
# matplotlib 3.7 中 font.family 列表回退在 mathtext 混合中文时不生效，
# 改用 SimSun 统一字体（宋体，兼容拉丁/CJK，学术风格）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 英文字体优先，中文回退到宋体
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体，与Times风格匹配
plt.rcParams['axes.unicode_minus'] = False              # 正常显示负号 (重要！) 

# ===== 配色方案：黑白学术风 =====
COLOR_BLACK = '#000000'
COLOR_WHITE = '#FFFFFF'
COLOR_DARK  = '#333333'   # 深灰（CI线、网格）
COLOR_GRID  = '#CCCCCC'   # 极浅灰网格
COLOR_REF   = '#666666'   # 参考线灰色

# 四模型线型区分（全黑色，黑白打印友好）
MODEL_STYLES = {
    'GARCH(1,1)-N':   {'ls': ':',  'lw': 1.2, 'label': 'GARCH(1,1)-N'},
    'GARCH(1,1)-t':   {'ls': '-',  'lw': 1.2, 'label': 'GARCH(1,1)-t'},
    'EGARCH(1,1)-t':  {'ls': '--', 'lw': 1.2, 'label': 'EGARCH(1,1)-t'},
    'APARCH(1,1)-t':  {'ls': '-.', 'lw': 1.2, 'label': 'APARCH(1,1)-t'},
}

MODEL_HATCHES = {
    'GARCH(1,1)-N':  '///',
    'GARCH(1,1)-t':  '\\\\\\',
    'EGARCH(1,1)-t': 'xxx',
    'APARCH(1,1)-t': '...',
}

# ===== 输出目录 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'output')
FIG_DIR  = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
os.chdir(SCRIPT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('plot_figures')

# ===== A4 兼容的图形尺寸 (宽 6.3 英寸 ≈ 16cm，适合A4留边距；文字≈小四号约12pt) =====
SINGLE_FIG = (6.3, 6.3 * 0.618)        # 单图: 黄金比例高
SINGLE_SQ  = (6.3, 6.0)                 # 单图接近方形
TWO_BY_TWO = (8.0, 7.5)                 # 2x2 子图
FOUR_ROW   = (6.3, 8.0)                 # 4行子图

# 全局字体大小 (相对于小尺寸图片)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12


def _parse_dates_from_df(df):
    """从DataFrame的日期列解析datetime"""
    if '日期' in df.columns:
        return pd.to_datetime(df['日期'])
    return np.arange(len(df))


def _savefig(fig, name):
    """统一保存PDF"""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLOR_WHITE, edgecolor='none')
    plt.close(fig)
    logger.info(f"  图表: {name} 已保存")


# ====================== 数据加载 ======================
def load_data_from_output():
    """从 results/output/ 加载所有中间结果"""
    logger.info("加载中间结果CSV...")

    # 原始数据
    df_raw = pd.read_csv(os.path.join(SCRIPT_DIR, 'SH_Index.csv'), encoding='utf-8-sig')
    df_raw['日期'] = pd.to_datetime(df_raw['日期'])
    df_raw = df_raw.sort_values('日期').reset_index(drop=True)
    df_raw['log_return'] = np.log(df_raw['收盘'] / df_raw['收盘'].shift(1)) * 100.0
    df_raw = df_raw.dropna(subset=['log_return']).reset_index(drop=True)
    df = df_raw[df_raw['日期'] >= '2000-01-01'].reset_index(drop=True)

    # ARIMA 残差
    arima_df = pd.read_csv(os.path.join(DATA_DIR, 'arima_residuals.csv'), encoding='utf-8-sig')
    arima_resid = arima_df['arima_residual'].values
    arima_resid = arima_resid[~np.isnan(arima_resid)]

    # 标准化残差
    sr_df = pd.read_csv(os.path.join(DATA_DIR, 'standardized_residuals.csv'), encoding='utf-8-sig')
    std_resid = {}
    model_names = ['GARCH(1,1)-N', 'GARCH(1,1)-t', 'EGARCH(1,1)-t', 'APARCH(1,1)-t']
    for name in model_names:
        col = name + '_std_resid'
        if col in sr_df.columns:
            vals = sr_df[col].values
            std_resid[name] = vals[~np.isnan(vals)]
        else:
            std_resid[name] = np.array([])

    # 条件波动率
    cv_df = pd.read_csv(os.path.join(DATA_DIR, 'conditional_volatility.csv'), encoding='utf-8-sig')
    cond_vol = {}
    for name in model_names:
        col = name + '_cond_vol'
        if col in cv_df.columns:
            vals = cv_df[col].values
            cond_vol[name] = vals[~np.isnan(vals)]
        else:
            cond_vol[name] = np.array([])

    # 滚动预测
    roll_df = pd.read_csv(os.path.join(DATA_DIR, 'rolling_forecast_predictions.csv'), encoding='utf-8-sig')
    actual_vol = roll_df['actual_proxy_vol'].values if 'actual_proxy_vol' in roll_df.columns else None
    roll_preds = {}
    for name in model_names:
        col = name + '_pred_vol'
        if col in roll_df.columns:
            roll_preds[name] = roll_df[col].values
        else:
            roll_preds[name] = np.array([])

    # 评估指标
    eval_df = pd.read_csv(os.path.join(DATA_DIR, 'forecast_evaluation_rolling.csv'), encoding='utf-8-sig')
    eval_dict = {}
    for _, row in eval_df.iterrows():
        name = row['模型']
        eval_dict[name] = {
            'RMSE': row['RMSE'], 'MAE': row['MAE'],
            'SMAPE(%)': row['SMAPE(%)'], 'QLIKE': row['QLIKE']
        }

    return df, arima_resid, std_resid, cond_vol, roll_preds, actual_vol, eval_dict, model_names


# ====================== 图1: 收盘价时序 ======================
def plot_price_series(df):
    logger.info("[1] 收盘价时序图")
    fig, ax = plt.subplots(figsize=SINGLE_FIG)
    ax.plot(df['日期'], df['收盘'], color=COLOR_BLACK, linewidth=0.6)
    ax.set_xlabel('日期')
    ax.set_ylabel('收盘价')
    ax.set_title('上证综指日收盘价 (2000–2026)')
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, 'fig_price_series.pdf')


# ====================== 图2: 收益率时序 ======================
def plot_return_series(df):
    logger.info("[2] 收益率时序图")
    fig, ax = plt.subplots(figsize=SINGLE_FIG)
    ax.plot(df['日期'], df['log_return'], color=COLOR_BLACK, linewidth=0.25, alpha=0.85)
    ax.axhline(y=0, color=COLOR_REF, linestyle='--', linewidth=0.8)
    ax.set_xlabel('日期')
    ax.set_ylabel('对数收益率 (%)')
    ax.set_title('上证综指日对数收益率 (2000–2026)')
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, 'fig_return_series.pdf')


# ====================== 图3–6: ACF/PACF 四张单图 ======================
def plot_acf_pacf_individual(arima_resid):
    """绘制四张独立的 ACF/PACF 图 (lag从1开始, CI线加粗加暗)"""
    logger.info("[3-6] ACF/PACF 图 (四张单图)")

    nlags = 40
    n = len(arima_resid)
    ci95 = 1.96 / np.sqrt(n)  # Bartlett 95% CI

    ls_values = {
        'acf':      ('ARIMA残差 ACF',               acf(arima_resid, nlags=nlags)),
        'pacf':     ('ARIMA残差 PACF',              pacf(arima_resid, nlags=nlags)),
        'acf_abs':  ('|ARIMA残差| ACF (波动聚集)',   acf(np.abs(arima_resid), nlags=nlags)),
        'acf_sq':   ('ARIMA残差² ACF',              acf(arima_resid**2, nlags=nlags)),
    }

    filenames = {
        'acf': 'fig_arima_acf.pdf',
        'pacf': 'fig_arima_pacf.pdf',
        'acf_abs': 'fig_arima_acf_abs.pdf',
        'acf_sq': 'fig_arima_acf_sq.pdf',
    }

    for key, (title, vals) in ls_values.items():
        fig, ax = plt.subplots(figsize=SINGLE_SQ)
        lags = np.arange(1, len(vals))  # 从 lag=1 开始

        # 竖线
        ax.vlines(lags, 0, vals[1:], colors=COLOR_BLACK, linewidths=0.7)
        # 小圆点
        ax.plot(lags, vals[1:], 'o', markersize=3,
                color=COLOR_BLACK, markerfacecolor=COLOR_WHITE,
                markeredgewidth=0.8)

        # 零线
        ax.axhline(y=0, color=COLOR_REF, linestyle='-', linewidth=0.6)

        # 95% CI 线 — 加粗加暗，清晰可见
        ax.axhline(y=ci95, color=COLOR_DARK, linestyle='--', linewidth=1.5, alpha=0.9)
        ax.axhline(y=-ci95, color=COLOR_DARK, linestyle='--', linewidth=1.5, alpha=0.9)

        ax.set_xlabel('滞后阶数')
        ax.set_ylabel('值')
        ax.set_title(title)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
        ax.set_xlim(0.5, nlags + 0.5)

        # # 标注CI
        # ax.annotate('95% CI', xy=(nlags * 0.85, ci95 * 1.15), fontsize=8,
        #             color=COLOR_DARK, ha='center')

        fig.tight_layout()
        _savefig(fig, filenames[key])


# ====================== 图7: 条件波动率对比 ======================
def plot_conditional_volatility(cond_vol, model_names):
    logger.info("[7] 条件波动率对比图")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in model_names:
        cv = cond_vol.get(name, np.array([]))
        if len(cv) == 0:
            continue
        sty = MODEL_STYLES.get(name, {'ls': '-', 'lw': 1.0})
        ax.plot(np.arange(len(cv)), cv, color=COLOR_BLACK,
                linestyle=sty['ls'], linewidth=sty['lw'],
                alpha=0.85, label=sty['label'])
    ax.set_xlabel('观测序号')
    ax.set_ylabel('条件波动率')
    ax.set_title('四模型条件波动率估计对比')
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=COLOR_BLACK)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, 'fig_cond_vol.pdf')


# ====================== 图8: 标准化残差时序 (四模型, 4行子图) ======================
def plot_std_resid_timeseries(std_resid, model_names):
    logger.info("[8] 标准化残差时序图")
    fig, axes = plt.subplots(len(model_names), 1, figsize=FOUR_ROW, sharex=True)
    for idx, name in enumerate(model_names):
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            continue
        ax = axes[idx]
        ax.plot(np.arange(len(sr)), sr, color=COLOR_BLACK, linewidth=0.4)
        ax.axhline(y=0, color=COLOR_REF, linestyle='-', linewidth=0.6)
        ax.axhline(y=3, color=COLOR_DARK, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(y=-3, color=COLOR_DARK, linestyle='--', linewidth=0.8, alpha=0.7)
        mu_sr = np.mean(sr); sg_sr = np.std(sr)
        # ax.set_title(f'{name}  标准化残差 ($\\bar{{z}}$={mu_sr:.3f}, $\\hat{{\\sigma}}$={sg_sr:.3f})',
        ax.set_title(f'{name}  标准化残差 ($\\bar{{\\mu}}$={mu_sr:.3f}, $\\hat{{\\sigma}}$={sg_sr:.3f})',
                     fontsize=10)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    axes[-1].set_xlabel('观测序号')
    fig.tight_layout()
    _savefig(fig, 'fig_std_resid_ts.pdf')


# ====================== 图9: Q-Q 图 (四模型 2x2) ======================
def plot_qq(std_resid, model_names):
    logger.info("[9] Q-Q 图 (2×2)")
    fig, axes = plt.subplots(2, 2, figsize=TWO_BY_TWO)

    # 汇总 Q-Q 数据以确定统一坐标范围
    all_quantiles = []
    for name in model_names:
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            continue
        (_, _), (slope, intercept, r) = sc_stats.probplot(sr, dist="norm")
        all_quantiles.append((sr, slope, intercept))

    for idx, name in enumerate(model_names):
        ax = axes[idx // 2, idx % 2]
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            ax.set_title(f'{name} — 数据缺失', fontsize=10)
            continue

        (osm, osr), (slope, intercept, r) = sc_stats.probplot(sr, dist="norm")

        # 散点：空心圆，细边
        ax.plot(osm, osr, 'o', markersize=2.5,
                markerfacecolor=COLOR_WHITE, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.5, alpha=0.7)
        # 参考线
        ax.plot(osm, slope * osm + intercept, color=COLOR_BLACK, linewidth=1.2)

        ax.set_xlabel('理论分位数')
        ax.set_ylabel('样本分位数')
        ax.set_title(f'{name}', fontsize=10)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, 'fig_qq.pdf')


# ====================== 图10: 标准化残差直方图 (四模型 2x2) ======================
def plot_std_resid_histogram(std_resid, model_names):
    """标准化残差直方图 + N(0,1) 参考曲线"""
    logger.info("[10] 标准化残差直方图 (2×2)")
    fig, axes = plt.subplots(2, 2, figsize=TWO_BY_TWO)

    from scipy.stats import norm as norm_dist

    for idx, name in enumerate(model_names):
        ax = axes[idx // 2, idx % 2]
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            continue

        ax.hist(sr, bins=80, density=True,
                facecolor=COLOR_WHITE, edgecolor=COLOR_BLACK, linewidth=0.6)

        x = np.linspace(sr.min(), sr.max(), 500)
        ax.plot(x, norm_dist.pdf(x, 0, 1), color=COLOR_BLACK, linewidth=1.2,
                linestyle='--', label=r'$\mathcal{N}(0,1)$')

        # 标注统计量
        sk = sc_stats.skew(sr)
        ku = sc_stats.kurtosis(sr, fisher=True)
        ax.text(0.97, 0.93, f'偏度={sk:.3f}\n超额峰度={ku:.3f}',
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_WHITE,
                          edgecolor=COLOR_BLACK, linewidth=0.5))

        ax.set_xlabel('标准化残差')
        ax.set_ylabel('密度')
        ax.set_title(name, fontsize=10)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _savefig(fig, 'fig_std_resid_hist.pdf')


# ====================== 图11: 滚动预测对比 ======================
def plot_rolling_forecast(roll_preds, actual_vol, eval_dict, model_names):
    logger.info("[11] 滚动预测对比图")
    fig, ax = plt.subplots(figsize=(8, 5))

    show_n = min(len(actual_vol) if actual_vol is not None else 0, 250)
    if actual_vol is not None and show_n > 0:
        ax.plot(range(show_n), actual_vol[:show_n], color=COLOR_REF,
                linewidth=1.0, linestyle='-', label=r'$|a_t|$ (代理)', alpha=0.7)

    for name in model_names:
        pred = roll_preds.get(name, np.array([]))
        if len(pred) == 0:
            continue
        sty = MODEL_STYLES.get(name, {'ls': '-', 'lw': 1.0})
        ax.plot(range(min(show_n, len(pred))), pred[:show_n],
                color=COLOR_BLACK, linestyle=sty['ls'], linewidth=sty['lw'],
                alpha=0.85, label=sty['label'])

    ax.set_xlabel('预测步数 (前250步)')
    ax.set_ylabel('波动率')
    ax.set_title('滚动窗口样本外波动率预测对比')
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=COLOR_BLACK)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, 'fig_forecast_rolling.pdf')


# ====================== 图12: 评估指标柱状图 (四指标, 使用 LaTeX) ======================
def plot_evaluation_bar(eval_dict, model_names):
    logger.info("[12] 评估指标柱状图 (4指标 x 4子图)")
    fig, axes = plt.subplots(2, 2, figsize=TWO_BY_TWO)

    metrics = ['RMSE', 'MAE', 'SMAPE(%)', 'QLIKE']
    metric_labels = {
        'RMSE': 'RMSE',
        'MAE': 'MAE',
        'SMAPE(%)': 'SMAPE (%)',
        'QLIKE': 'QLIKE',
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [eval_dict.get(n, {}).get(metric, np.nan) for n in model_names]
        bars = ax.bar(range(len(model_names)), values,
                      facecolor=COLOR_WHITE, edgecolor=COLOR_BLACK, linewidth=1.0)

        # hatch 图案区分
        for bar, name in zip(bars, model_names):
            bar.set_hatch(MODEL_HATCHES.get(name, '///'))

        # 最优加粗边框
        if not all(np.isnan(v) for v in values):
            best_idx = np.nanargmin(values)
            bars[best_idx].set_edgecolor(COLOR_BLACK)
            bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace('(1,1)', '') for n in model_names], fontsize=7, rotation=10)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric), fontsize=10)

        # 数值标注
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4, axis='y')

    fig.tight_layout()
    _savefig(fig, 'fig_eval_bar.pdf')


# ====================== 主函数 ======================
def main():
    logger.info("=" * 60)
    logger.info("独立制图模块 — 从 results/output/ 读取数据")
    logger.info("=" * 60)

    # 加载中间结果
    df, arima_resid, std_resid, cond_vol, roll_preds, actual_vol, eval_dict, model_names = \
        load_data_from_output()

    logger.info(f"  模型列表: {model_names}")
    logger.info(f"  ARIMA残差长度: {len(arima_resid)}")
    logger.info(f"  条件波动率长度: {len(list(cond_vol.values())[0]) if cond_vol else 0}")

    # 图1: 收盘价时序
    plot_price_series(df)

    # 图2: 收益率时序
    plot_return_series(df)

    # 图3–6: ACF/PACF 四张
    plot_acf_pacf_individual(arima_resid)

    # 图7: 条件波动率
    plot_conditional_volatility(cond_vol, model_names)

    # 图8: 标准化残差时序
    plot_std_resid_timeseries(std_resid, model_names)

    # 图9: Q-Q 图
    plot_qq(std_resid, model_names)

    # 图10: 标准化残差直方图
    plot_std_resid_histogram(std_resid, model_names)

    # 图11: 滚动预测对比
    plot_rolling_forecast(roll_preds, actual_vol, eval_dict, model_names)

    # 图12: 评估指标柱状图
    plot_evaluation_bar(eval_dict, model_names)

    logger.info("\n" + "=" * 60)
    logger.info(f"全部图表已保存至 {FIG_DIR}/")
    logger.info(f"共 12 张 PDF 图表")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()