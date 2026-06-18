# -*- coding: utf-8 -*-
"""
上证综指波动率分析 — 独立制图模块（遵循科学图表制作规范）
========================================================
本模块从 results/output/ 读取 model_core.py 生成的中间结果CSV，
绘制所有分析图表，输出到 results/figures/。

此模块完全不依赖 model_core.py 的运行时对象，可离线批量重新绘图。

图表规范遵循 .clinerules/scientific-figures.md 科学图表制作规范：
  - apply_publication_style()     配置出版级 rcParams
  - create_subplots()             统一子图创建，返回展平 axes
  - finalize_figure()             多格式保存 + tight_layout
  - make_grouped_bar()            分组条形图
  - annotate_bars()               柱体数值标注
  - make_trend()                  多线趋势图
"""

import os, sys, warnings, logging
import numpy as np
import pandas as pd
from scipy import stats as sc_stats
from scipy.stats import norm as norm_dist
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings('ignore')

# ===== 字体与数学公式（保持不变）=====
# SimSun（宋体）+ Times New Roman（英文）+ stix 数学公式
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']  # 英文字体优先，中文回退到宋体
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体，与Times风格匹配
plt.rcParams['axes.unicode_minus'] = False              # 正常显示负号 (重要！)

# ===== 配色方案：黑白学术风（保持不变）=====
COLOR_BLACK = '#000000'
COLOR_WHITE = '#FFFFFF'
COLOR_DARK  = '#000000'   # 深线（CI线、参考线）
COLOR_GRID  = '#666666'   # 深灰网格
COLOR_REF   = '#222222'   # 参考线深灰色

# 四模型线型区分（全黑色，黑白打印友好）
MODEL_STYLES = {
    'GARCH(1,1)-N':   {'ls': ':',  'lw': 1.2, 'label': 'GARCH(1,1)-N'},
    'GARCH(1,1)-t':   {'ls': '-',  'lw': 1.2, 'label': 'GARCH(1,1)-t'},
    'EGARCH(1,1)-t':  {'ls': '--', 'lw': 1.2, 'label': 'EGARCH(1,1)-t'},
    'APARCH(1,1)-t':  {'ls': '-.', 'lw': 1.2, 'label': 'APARCH(1,1)-t'},
}

# 四模型填充纹理（黑白打印可区分）
MODEL_HATCHES = {
    'GARCH(1,1)-N':  '///',
    'GARCH(1,1)-t':  '\\\\\\',
    'EGARCH(1,1)-t': 'xxx',
    'APARCH(1,1)-t': '...',
}

# 四模型颜色（用于彩色区分线图）
# 模型配色 — 遵循 .clinerules/scientific-figures.md 调色板
# 语义：红=基线, 蓝=关键结果, 青/紫=变体
MODEL_COLORS = {
    'GARCH(1,1)-N':   '#B64342',   # red_strong — 基线
    'GARCH(1,1)-t':   '#0F4D92',   # blue_main — t分布改进
    'EGARCH(1,1)-t':  '#42949E',   # teal — EGARCH变体
    'APARCH(1,1)-t':  '#9A4D8E',   # violet — APARCH变体
}

# 评估指标配色
# 指标配色 — 遵循 .clinerules/scientific-figures.md 调色板
METRIC_COLORS = {
    'RMSE': '#0F4D92',      # blue_main
    'MAE': '#8BCF8B',       # green_3
    'SMAPE(%)': '#B64342',  # red_strong
    'QLIKE': '#42949E',     # teal
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

# 全局字体大小 (相对于小尺寸图片，保持不变)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12.5
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11


# =====================================================================
#  科学图表规范：核心辅助函数
#  参考 .clinerules/scientific-figures.md
# =====================================================================

def apply_publication_style(style=None):
    """
    配置 matplotlib rcParams 用于论文级出版质量。
    在创建任何图表之前调用一次。

    注意：不修改字体相关配置（字体、字号、数学公式字体），
    这些配置已在模块顶部按用户要求设置。
    """
    pub_rcparams = {
        # 关闭顶部和右侧边框线（出版标准）
        "axes.spines.right": False,
        "axes.spines.top": False,
        # 坐标轴线宽加粗
        "axes.linewidth": 2.5,
        # 图例无边框
        "legend.frameon": False,
        # SVG 导出时不嵌入字体子集（避免中文字体嵌入问题）
        "svg.fonttype": "none",
    }
    plt.rcParams.update(pub_rcparams)
    logger.info("已应用出版级 rcParams（spine 关闭、无框图例、SVG 字体优化）")


def create_subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    创建子图，返回 (fig, axes)。

    axes 始终为展平的一维 numpy 数组，方便用 for idx, ax in enumerate(axes) 统一遍历。
    即使单子图也返回长度为 1 的数组。

    Parameters
    ----------
    nrows, ncols : int
        子图行列数
    figsize : tuple
        图形尺寸 (width, height) 英寸
    **kwargs
        传递给 plt.subplots() 的其他参数
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    return fig, axes.flatten()


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=2, **kwargs):
    """
    保存图表到指定路径，自动调用 tight_layout。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_path : str
        输出文件路径（含扩展名，如 'fig_xxx.pdf'）
    formats : list or None
        输出格式列表，默认 ['pdf']。
        支持: pdf, svg, eps, png, jpg, jpeg, tif, tiff
    dpi : int
        标准导出 300；密集条图可用 600
    close : bool
        保存后是否关闭图形
    pad : float
        tight_layout 的 pad 参数。标准 2，紧凑图用 1
    """
    if formats is None:
        formats = ['pdf']

    fig.tight_layout(pad=pad)

    base = os.path.splitext(out_path)[0]
    os.makedirs(os.path.dirname(base) if os.path.dirname(base) else FIG_DIR, exist_ok=True)

    for fmt in formats:
        path = f"{base}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor=COLOR_WHITE, edgecolor='none')

    # 日志输出
    fname = os.path.basename(out_path)
    if len(formats) == 1:
        logger.info(f"  图表: {fname} 已保存")
    else:
        fmt_str = ', '.join(formats)
        logger.info(f"  图表: {fname} → [{fmt_str}] 已保存")

    if close:
        plt.close(fig)


def make_grouped_bar(ax, categories, series, labels, ylabel='Value',
                     colors=None, bar_width=0.18, edgecolor='black',
                     linewidth=0.8, annotate=False):
    """
    分组条形图（遵循科学图表规范）。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标坐标轴
    categories : list of str
        x 轴类别标签，len(categories) 必须等于 series 中每个数组的长度
    series : list of array-like
        每组数据，series[i] 对应 labels[i]
    labels : list of str
        图例标签
    ylabel : str
        y 轴标签
    colors : list or None
        每个系列的颜色。None 则使用默认调色板
    bar_width : float
        每组中单根条形的宽度
    edgecolor : str
        条形边线颜色（规范要求 'black' 以保证打印可区分）
    linewidth : float
        条形边线宽度（规范建议 0.8-3）
    annotate : bool
        是否调用 annotate_bars 添加数值标注

    Returns
    -------
    all_bars : list of BarContainer
        每个系列对应的 BarContainer 列表
    """
    n_groups = len(categories)
    n_series = len(series)

    if colors is None:
        # 规范默认调色板
        default_colors = ['#0F4D92', '#8BCF8B', '#B64342', '#42949E', '#9A4D8E', '#CFCECE']
        colors = default_colors[:n_series]

    x = np.arange(n_groups)
    all_bars = []

    for i, (ser, label) in enumerate(zip(series, labels)):
        offset = (i - n_series / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, ser, bar_width,
                      color=colors[i], edgecolor=edgecolor,
                      linewidth=linewidth, label=label)
        all_bars.append(bars)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)

    return all_bars


def annotate_bars(ax, bars, fmt='{:.2f}', fontsize=10, padding=3,
                  inside=False, text_color='black', fontweight='normal'):
    """
    在条形上方（或内部）添加数值标注。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    bars : BarContainer
        单个系列的条形容器
    fmt : str
        数值格式化字符串
    fontsize : int
        标注字体大小
    padding : float
        标注与柱体的间距
    inside : bool
        True 时标注在柱体内部顶端，False 时标注在柱体上方
    text_color : str
        文字颜色
    fontweight : str
        文字粗细
    """
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        if inside:
            y_pos = height - padding
            va = 'top'
        else:
            y_pos = height + padding
            va = 'bottom'
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                fmt.format(height), ha='center', va=va,
                fontsize=fontsize, color=text_color, fontweight=fontweight)


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               show_shadow=False, linewidth=2, alpha=1.0, linestyles=None):
    """
    多线趋势图（遵循科学图表规范）。

    规范建议：每轴限制 2-4 条主曲线；线宽 2-3；尽量不用网格线，依赖轴刻度和图例。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : array-like
        x 轴数据
    y_series : list of array-like
        每条曲线的 y 数据，每个元素与 x 等长
    labels : list of str
        每条曲线的图例标签
    colors : list or None
        每条曲线的颜色
    ylabel, xlabel : str or None
        轴标签
    show_shadow : bool
        是否绘制 fill_between 不确定带（需在调用前自行准备 CI 数据）
    linewidth : float
        线宽，规范推荐 2-3
    alpha : float
        透明度
    linestyles : list or None
        每条曲线的线型，None 则默认实线

    Returns
    -------
    lines : list
        绘制的 Line2D 对象列表
    """
    if colors is None:
        default_colors = ['#0F4D92', '#8BCF8B', '#B64342', '#42949E', '#9A4D8E', '#CFCECE']
        colors = default_colors[:len(y_series)]

    if linestyles is None:
        linestyles = ['-'] * len(y_series)

    lines = []
    for i, (y, label) in enumerate(zip(y_series, labels)):
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        line = ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha,
                       linestyle=ls, label=label)
        lines.extend(line)

        if show_shadow and len(y) > 1:
            # 简单不确定带：使用数据的 10% 标准差作为示意
            std_est = np.nanstd(y) * 0.1
            ax.fill_between(x, y - std_est, y + std_est,
                            color=color, alpha=0.15, linewidth=0)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return lines


# =====================================================================
#  数据加载
# =====================================================================

def _parse_dates_from_df(df):
    """从DataFrame的日期列解析datetime（工具函数，保留备用）"""
    if '日期' in df.columns:
        return pd.to_datetime(df['日期'])
    return np.arange(len(df))


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


# =====================================================================
#  图1: 收盘价时序
# =====================================================================

def plot_price_series(df):
    """上证综指日收盘价时序图"""
    logger.info("[1] 收盘价时序图")
    fig, ax = plt.subplots(figsize=SINGLE_FIG)
    ax.plot(df['日期'], df['收盘'], color=COLOR_BLACK, linewidth=0.6)
    ax.set_xlabel('日期')
    ax.set_ylabel('收盘价')
    ax.set_title('上证综指日收盘价 (2000–2026)')
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_price_series.pdf'))


# =====================================================================
#  图2: 收益率时序
# =====================================================================

def plot_return_series(df):
    """上证综指日对数收益率时序图"""
    logger.info("[2] 收益率时序图")
    fig, ax = plt.subplots(figsize=SINGLE_FIG)
    ax.plot(df['日期'], df['log_return'], color=COLOR_BLACK, linewidth=0.25, alpha=0.85)
    ax.axhline(y=0, color=COLOR_REF, linestyle='--', linewidth=0.8)
    ax.set_xlabel('日期')
    ax.set_ylabel('对数收益率 (%)')
    ax.set_title('上证综指日对数收益率 (2000–2026)')
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_return_series.pdf'))


# =====================================================================
#  图3–6: ACF/PACF 四张单图
# =====================================================================

def plot_acf_pacf_individual(arima_resid):
    """绘制四张独立的 ACF/PACF 图 (lag从1开始, CI线加粗加暗)"""
    logger.info("[3-6] ACF/PACF 图 (四张单图)")

    nlags = 40
    n = len(arima_resid)
    ci95 = 1.96 / np.sqrt(n)  # Bartlett 95% CI

    plot_specs = {
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

    for key, (title, vals) in plot_specs.items():
        fig, ax = plt.subplots(figsize=SINGLE_SQ)
        lags = np.arange(1, len(vals))  # 从 lag=1 开始

        # 竖线
        ax.vlines(lags, 0, vals[1:], colors=COLOR_BLACK, linewidths=0.7)
        # 小圆点（空心）
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

        finalize_figure(fig, os.path.join(FIG_DIR, filenames[key]))


# =====================================================================
#  图7: 条件波动率对比（双面板：主图 + 差异子图）
# =====================================================================

def _prepare_cond_vol_data(cond_vol, model_names):
    """条件波动率数据准备（供两个独立图复用）"""
    ref_name = 'GARCH(1,1)-N'
    model_data = []
    ref_cv = None
    for name in model_names:
        cv = cond_vol.get(name, np.array([]))
        if len(cv) == 0:
            continue
        sty = MODEL_STYLES.get(name, {'ls': '-', 'lw': 1.0})
        clr = MODEL_COLORS.get(name, COLOR_BLACK)
        model_data.append((name, cv, sty, clr))
        if name == ref_name:
            ref_cv = cv
    return model_data, ref_name, ref_cv


def plot_conditional_volatility(cond_vol, model_names):
    """图7: 条件波动率叠加 — 四模型条件波动率估计对比"""
    logger.info("[7] 条件波动率叠加图")

    model_data, _, _ = _prepare_cond_vol_data(cond_vol, model_names)

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, cv, sty, clr in model_data:
        ax.plot(np.arange(len(cv)), cv, color=clr,
                linestyle=sty['ls'], linewidth=sty['lw'],
                alpha=0.55, label=sty['label'])

    ax.set_xlabel('观测序号')
    ax.set_ylabel('条件波动率')
    ax.set_title('四模型条件波动率估计对比')
    ax.legend(fontsize=11, framealpha=0.9, edgecolor=COLOR_BLACK)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_cond_vol.pdf'))


def plot_cond_vol_diff(cond_vol, model_names):
    """图8: 条件波动率差异 — 各模型与 GARCH-N 的差值"""
    logger.info("[8] 条件波动率差异图 (vs GARCH-N)")

    model_data, ref_name, ref_cv = _prepare_cond_vol_data(cond_vol, model_names)

    fig, ax = plt.subplots(figsize=(8, 5))

    if ref_cv is not None:
        for name, cv, sty, clr in model_data:
            if name == ref_name:
                continue
            n_common = min(len(cv), len(ref_cv))
            diff = cv[:n_common] - ref_cv[:n_common]
            ax.plot(np.arange(n_common), diff, color=clr,
                    linestyle=sty['ls'], linewidth=sty['lw'],
                    alpha=0.85, label=f'{name} − {ref_name}')

    ax.axhline(y=0, color=COLOR_REF, linestyle='--', linewidth=0.8)
    ax.set_xlabel('观测序号')
    ax.set_ylabel(f'Δ 波动率 (vs {ref_name})')
    ax.set_title(f'各模型与 {ref_name} 的条件波动率差异')
    ax.legend(fontsize=11, framealpha=0.9, edgecolor=COLOR_BLACK)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_cond_vol_diff.pdf'))


# =====================================================================
#  图9: 标准化残差时序 (四模型, 4行子图)
# =====================================================================

def plot_std_resid_timeseries(std_resid, model_names):
    """标准化残差时序图 — 四模型各一行，共享 x 轴"""
    logger.info("[9] 标准化残差时序图")

    fig, axes = create_subplots(len(model_names), 1, figsize=FOUR_ROW, sharex=True)

    for idx, name in enumerate(model_names):
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            continue
        ax = axes[idx]
        ax.plot(np.arange(len(sr)), sr, color=COLOR_BLACK, linewidth=0.4)
        ax.axhline(y=0, color=COLOR_REF, linestyle='-', linewidth=0.6)
        ax.axhline(y=3, color=COLOR_DARK, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(y=-3, color=COLOR_DARK, linestyle='--', linewidth=0.8, alpha=0.7)
        mu_sr = np.mean(sr)
        sg_sr = np.std(sr)
        ax.set_title(f'{name}  ' + '标准化残差'
                     + f' ($\\bar{{z}}$={mu_sr:.3f}, $\\hat{{\\sigma}}$={sg_sr:.3f})',
                     fontsize=10)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    axes[-1].set_xlabel('观测序号')
    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_std_resid_ts.pdf'))


# =====================================================================
#  图10: Q-Q 图 (四模型 2×2)
# =====================================================================

def plot_qq(std_resid, model_names):
    """标准化残差 Q-Q 图 — 四模型 2×2 布局"""
    logger.info("[10] Q-Q 图 (2×2)")

    fig, axes = create_subplots(2, 2, figsize=TWO_BY_TWO)

    for idx, name in enumerate(model_names):
        ax = axes[idx]
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

    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_qq.pdf'))


# =====================================================================
#  图11: 标准化残差直方图 (四模型 2×2)
# =====================================================================

def plot_std_resid_histogram(std_resid, model_names):
    """标准化残差直方图 + N(0,1) 参考曲线 — 四模型 2×2 布局"""
    logger.info("[11] 标准化残差直方图 (2×2)")

    fig, axes = create_subplots(2, 2, figsize=TWO_BY_TWO)

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            continue

        ax.hist(sr, bins=80, density=True,
                facecolor=COLOR_WHITE, edgecolor=COLOR_BLACK, linewidth=0.6)

        x = np.linspace(sr.min(), sr.max(), 500)
        ax.plot(x, norm_dist.pdf(x, 0, 1), color=COLOR_BLACK, linewidth=1.2,
                linestyle='--', label='N(0,1)')

        sk = sc_stats.skew(sr)
        ku = sc_stats.kurtosis(sr, fisher=True)
        info_text = f'偏度={sk:.3f}\n超额峰度={ku:.3f}'
        ax.text(0.03, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLOR_BLACK, alpha=0.85))

        ax.set_xlabel('标准化残差')
        ax.set_ylabel('密度')
        ax.set_title(name, fontsize=13)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
        ax.legend(fontsize=11, framealpha=0.9, loc='upper right')

    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_std_resid_hist.pdf'))


# =====================================================================
#  图12: 滚动预测对比
# =====================================================================

def plot_rolling_forecast(roll_preds, actual_vol, eval_dict, model_names):
    """滚动窗口样本外波动率预测对比"""
    logger.info("[12] 滚动预测对比图")

    fig, ax = plt.subplots(figsize=(8, 5))

    show_n = min(len(actual_vol) if actual_vol is not None else 0, 250)

    # 实际波动率代理（参考线）
    if actual_vol is not None and show_n > 0:
        ax.plot(range(show_n), actual_vol[:show_n], color=COLOR_REF,
                linewidth=1.0, linestyle='-', label='|a_t| (代理)', alpha=0.7)

    # 各模型预测 — 使用不同颜色区分（学术调色板），不用线型
    for name in model_names:
        pred = roll_preds.get(name, np.array([]))
        if len(pred) == 0:
            continue
        clr = MODEL_COLORS.get(name, '#000000')
        ax.plot(range(min(show_n, len(pred))), pred[:show_n],
                color=clr, linestyle='-', linewidth=1.2,
                alpha=0.85, label=name)

    ax.set_xlabel('预测步数 (前250步)')
    ax.set_ylabel('波动率')
    ax.set_title('滚动窗口样本外波动率预测对比')
    ax.legend(fontsize=11, framealpha=0.9, edgecolor=COLOR_BLACK)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_forecast_rolling.pdf'))


# =====================================================================
#  图13: 评估指标柱状图 (四指标分组柱状图)
# =====================================================================

def plot_evaluation_bar(eval_dict, model_names):
    """滚动预测评估指标分组柱状图 — 使用 make_grouped_bar + annotate_bars"""
    logger.info("[13] 评估指标柱状图 (簇状分组)")

    metrics = ['RMSE', 'MAE', 'SMAPE(%)', 'QLIKE']
    metric_labels = {
        'RMSE': 'RMSE',
        'MAE': 'MAE',
        'SMAPE(%)': 'SMAPE',
        'QLIKE': 'QLIKE',
    }

    # 准备数据 — SMAPE 原值是百分比（如82.95表示82.95%），还原为小数
    data = {}
    for metric in metrics:
        raw_vals = [eval_dict.get(n, {}).get(metric, np.nan) for n in model_names]
        if metric == 'SMAPE(%)':
            data[metric] = [v / 100.0 if not np.isnan(v) else v for v in raw_vals]
        else:
            data[metric] = raw_vals

    # 确定 Y 轴范围（紧贴数据放大差异）
    all_vals = [v for metric in metrics for v in data[metric] if not np.isnan(v)]
    y_min = min(all_vals) * 0.98 if all_vals else 0
    y_max = max(all_vals) * 1.02 if all_vals else 1

    n_models = len(model_names)
    n_metrics = len(metrics)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(8, 5))

    # 使用 make_grouped_bar 创建分组条形图
    series_list = [data[m] for m in metrics]
    labels_list = [metric_labels.get(m, m) for m in metrics]
    colors_list = [METRIC_COLORS[m] for m in metrics]

    # 缩短模型名用于 x 轴标签
    short_names = [n.replace('(1,1)', '') for n in model_names]

    all_bar_groups = make_grouped_bar(
        ax, short_names, series_list, labels_list,
        ylabel='值', colors=colors_list, bar_width=bar_width,
        edgecolor=COLOR_BLACK, linewidth=0.8
    )

    # 标注最优值（加粗边框）+ 数值标注（柱体内部白色文字）
    for metric_idx, (metric, bars) in enumerate(zip(metrics, all_bar_groups)):
        vals = data[metric]
        # 找出最优值（最小值）的索引
        if not all(np.isnan(v) for v in vals):
            best_idx = np.nanargmin(vals)
            bars[best_idx].set_edgecolor(COLOR_BLACK)
            bars[best_idx].set_linewidth(2.5)

        # 柱体内部标注
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() - (y_max - y_min) * 0.03,
                        f'{val:.4f}', ha='center', va='top', fontsize=7.5,
                        color='white', fontweight='bold')

    ax.set_ylim(y_min, y_max)
    ax.set_title('滚动窗口样本外预测评估指标对比')
    ax.legend(fontsize=11, framealpha=0.9, edgecolor=COLOR_BLACK, ncol=4)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4, axis='y')

    finalize_figure(fig, os.path.join(FIG_DIR, 'fig_eval_bar.pdf'))


# =====================================================================
#  图14: 最优差分序列 ACF/PACF 诊断图
# =====================================================================

def plot_optimal_diff_acf_pacf():
    """绘制最优差分阶数序列的 ACF 和 PACF 诊断图（两张单图）"""
    logger.info("[14] 最优差分序列 ACF/PACF 诊断图")

    df = pd.read_csv(os.path.join(DATA_DIR, 'optimal_diff_series.csv'), encoding='utf-8-sig')
    series = df['optimal_diff_series'].dropna().values
    nlags = 40
    n = len(series)
    ci95 = 1.96 / np.sqrt(n)

    plot_specs = {
        'acf':  ('最优差分序列 ACF',  acf(series, nlags=nlags)),
        'pacf': ('最优差分序列 PACF', pacf(series, nlags=nlags)),
    }

    filenames = {
        'acf':  'fig_optimal_diff_acf.pdf',
        'pacf': 'fig_optimal_diff_pacf.pdf',
    }

    for key, (title, vals) in plot_specs.items():
        fig, ax = plt.subplots(figsize=SINGLE_SQ)
        lags = np.arange(1, len(vals))

        ax.vlines(lags, 0, vals[1:], colors=COLOR_BLACK, linewidths=0.7)
        ax.plot(lags, vals[1:], 'o', markersize=3,
                color=COLOR_BLACK, markerfacecolor=COLOR_WHITE,
                markeredgewidth=0.8)

        ax.axhline(y=0, color=COLOR_REF, linestyle='-', linewidth=0.6)
        ax.axhline(y=ci95, color=COLOR_DARK, linestyle='--', linewidth=1.5, alpha=0.9)
        ax.axhline(y=-ci95, color=COLOR_DARK, linestyle='--', linewidth=1.5, alpha=0.9)

        ax.set_xlabel('滞后阶数')
        ax.set_ylabel('值')
        ax.set_title(title)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
        ax.set_xlim(0.5, nlags + 0.5)

        finalize_figure(fig, os.path.join(FIG_DIR, filenames[key]))


# =====================================================================
#  主函数
# =====================================================================

def main():
    logger.info("=" * 60)
    logger.info("独立制图模块 — 从 results/output/ 读取数据")
    logger.info("图表规范：.clinerules/scientific-figures.md")
    logger.info("=" * 60)

    # 应用出版级 rcParams（spine、图例、SVG 等，不修改字体和配色）
    apply_publication_style()

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

    # 图7: 条件波动率叠加
    plot_conditional_volatility(cond_vol, model_names)

    # 图8: 条件波动率差异 (vs GARCH-N)
    plot_cond_vol_diff(cond_vol, model_names)

    # 图9: 标准化残差时序
    plot_std_resid_timeseries(std_resid, model_names)

    # 图10: Q-Q 图
    plot_qq(std_resid, model_names)

    # 图11: 标准化残差直方图
    plot_std_resid_histogram(std_resid, model_names)

    # 图12: 滚动预测对比
    plot_rolling_forecast(roll_preds, actual_vol, eval_dict, model_names)

    # 图13: 评估指标柱状图
    plot_evaluation_bar(eval_dict, model_names)

    # 图14: 最优差分序列 ACF/PACF 诊断图
    plot_optimal_diff_acf_pacf()

    logger.info("\n" + "=" * 60)
    logger.info(f"全部图表已保存至 {FIG_DIR}/")
    logger.info(f"共 15 张 PDF 图表")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()