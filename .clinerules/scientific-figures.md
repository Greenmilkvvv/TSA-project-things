# Scientific Figure Making（科学图表制作规范）

本规则用于使用 matplotlib 生成**论文级/汇报级**科学图表。所有 matplotlib 图表均应遵循此规范。

---

## 何时遵循此规范

**必须遵循：**
- 生成用于论文、PPT、报告中的 matplotlib 图表
- 需要 PDF/SVG/高 DPI 输出的科学图表
- 分组条形图、趋势线、热力图、多面板布局

**不适用：**
- Plotly、Altair、Bokeh 等交互式/web 图表
- 纯 EDA 探索（用 seaborn 或 pandas 默认即可）
- 3D 渲染、地理制图等非 matplotlib 场景

---

## 调色板 PALETTE

```python
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
```

**语义规则：**
- **蓝色(blue_main / blue_secondary)**：提出的方法 / 关键结果
- **绿色(green_*)**: 改进 / 正向变体
- **红色(red_*)**: 基线 / 对比 / 替代方法
- **中性色(neutral)**: 背景 / 参考类别
- **高亮(highlight)**: 仅用于单一强调

**默认颜色顺序:** `[blue_main, green_3, red_strong, teal, violet, neutral]`

---

## 样式系统 FigureStyle

```python
@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16       # 大图用24, 紧凑图用15-16
    axes_linewidth: float = 2.5  # 大图用3, 紧凑用2
    use_tex: bool = False     # 仅当 LaTeX 已安装且需要数学标签时设为 True
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")
```

**推荐 rcParams 预设：**

```python
PUBLICATION_RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 16,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
}
```

---

## 核心函数规范（自行实现或参考签名）

### apply_publication_style(style=None)
配置 matplotlib rcParams：spines（关闭 top/right）、无框图例、字体、矢量导出选项。**在创建任何图表之前调用一次。**

### create_subplots(nrows=1, ncols=1, figsize=None, **kwargs)
返回 `(fig, axes)`，axes 为展平的一维数组。

### finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kwargs)
保存图表。`formats` 支持: pdf, svg, eps, png, jpg, jpeg, tif, tiff。
默认用 `tight_layout(pad=2)` 收尾，密集条图可以用 `dpi=600`。

### make_grouped_bar(ax, categories, series, labels, ylabel='Value', colors=None, annotate=False)
分组条形图。`categories`: x轴标签列表；`series`: 每组数据的列表；`labels`: 图例标签。
- `len(categories)` 必须等于 series 中每个数组的长度。

### annotate_bars(ax, bars, fmt='{:.2f}', fontsize=10, padding=3)
在条形上方添加数值标注。

### make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None, show_shadow=True)
多线趋势图。`y_series`: 与 x 等长的一维数组列表。`show_shadow=True` 时用 `fill_between` 绘制不确定带。

### make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap='magma', cbar_label=None, annotate=False)
二维热力图。

### make_scatter(ax, x, y, label=None, color=None, size=50, alpha=0.7)
单系列散点图。x, y 必须等长。

---

## 常见布局模式

### 超宽画布（多指标对比）
对于 3-4 个指标或多个类别的单行对比，使用 `figsize=(28, 6)` 或 `(45, 12)`。宽度通常是高度的 3-4 倍。

### 专属图例面板
当图例太大时，额外创建一个子图专用于图例：
```python
ax_legend.set_axis_off()
ax_legend.legend(handles, labels, loc="center")
```

### 不显示 x 轴刻度标签
当 x 轴类别已由图例标识时，使用 `ax.set_xticks([])`。

### 动态 y 轴范围
不要固定 0-100，而是用 `data.min() - margin` 到 `data.max() + margin`，让差异可见。

### 条形边线和纹理（打印安全）
- 所有条形使用 `edgecolor='black'`，`linewidth=1.5-3`
- 可选不同纹理（`'/'`, `'\\'`, `'.'`）用于消融研究子组，确保灰度打印可区分
- 消融研究：同一颜色不同 alpha 层级（0.2 到 1.0）表示方法完整度

### 趋势线策略
- 每轴限制 2-4 条主曲线
- 线宽 2-3，alpha 可控
- 尽量不用网格线，依赖轴刻度和图例

---

## 导出规范

| 参数 | 默认值 | 说明 |
|------|--------|------|
| dpi | 300 | 标准导出；密集条图用 600 |
| 格式 | pdf/svg/eps | 矢量优先，同时输出 png |
| tight_layout | pad=2 | 标准；紧凑图用 pad=1 |

---

## 前置操作（非交互环境）

在 `import matplotlib.pyplot` 之前设置非交互后端：
```python
import matplotlib
matplotlib.use("Agg")
```

## 文件组织

- 图表输出到项目 `figures/` 目录（或用户指定路径）
- `finalize_figure` 会自动创建父目录
- 使用稳定的文件基名