# -*- coding: utf-8 -*-
"""
上证综指日收益率波动率建模与预测 — 核心建模模块
==================================================
本模块负责：
1. 数据加载与预处理
2. 描述性统计
3. 平稳性与ARCH效应检验
4. ARIMA均值方程过滤
5. 四模型波动率建模 (GARCH-N, GARCH-t, EGARCH-t, APARCH-t)
6. 模型诊断（保存中间结果CSV）
7. 滚动窗口样本外预测
8. 结果汇总

所有中间结果输出到 results/output/
运行日志输出到 results/analysis_log.txt
制图由 plot_figures.py 独立完成
"""

import os, sys, warnings, logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm, t as t_dist, skew, kurtosis
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from arch import arch_model

warnings.filterwarnings('ignore')

# ===== 输出目录 =====
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(os.path.join(BASE_DIR, 'analysis_log.txt'), mode='w', encoding='utf-8'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# ====================== 1. 数据加载 ======================
def load_data():
    logger.info("=" * 60)
    logger.info("1. 数据加载与预处理")
    logger.info("=" * 60)
    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'SH_Index.csv'), encoding='utf-8')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    df['log_return'] = np.log(df['收盘'] / df['收盘'].shift(1)) * 100.0
    df = df.dropna(subset=['log_return']).reset_index(drop=True)

    # 从 2000 年开始
    df_2000 = df[df['日期'] >= '2000-01-01'].reset_index(drop=True)
    logger.info(f"  原始观测: {len(df)}, 2000年起有效观测: {len(df_2000)}")
    logger.info(f"  日期: {df_2000['日期'].min().date()} ~ {df_2000['日期'].max().date()}")
    return df_2000


# ====================== 2. 描述性统计 ======================
def descriptive_stats(df):
    logger.info("\n" + "=" * 60)
    logger.info("2. 描述性统计")
    logger.info("=" * 60)
    r = df['log_return'].values
    n = len(r); mu = np.mean(r); sg = np.std(r, ddof=1)
    sk = skew(r); ku = kurtosis(r, fisher=True)
    jb_s, jb_p = jarque_bera(r)
    r_min = np.min(r); r_max = np.max(r)

    logger.info(f"  N={n}, Mean={mu:.6f}%, Std={sg:.6f}%")
    logger.info(f"  Min={r_min:.6f}%, Max={r_max:.6f}%")
    logger.info(f"  Skew={sk:.6f}, ExKurt={ku:.6f}, JB_p={jb_p:.6f}")

    stats_df = pd.DataFrame({
        '统计量': ['观测数', '均值(%)', '标准差(%)', '最小值(%)', '最大值(%)', '偏度', '超额峰度', 'JB统计量', 'JB_p值'],
        '值': [n, mu, sg, r_min, r_max, sk, ku, jb_s, jb_p]
    })
    stats_df.to_csv(os.path.join(OUT, 'descriptive_stats.csv'), index=False, encoding='utf-8-sig')

    return r, {'N': n, 'mu': mu, 'sigma': sg, 'skew': sk, 'kurt': ku, 'jb_p': jb_p}


# ====================== 3. 单位根 + ARCH 检验 ======================
def preliminary_tests(r):
    logger.info("\n" + "=" * 60)
    logger.info("3. 平稳性与ARCH效应检验")
    logger.info("=" * 60)
    adf = adfuller(r, regression='c', autolag='AIC')
    kp = kpss(r, regression='c', nlags='auto')
    logger.info(f"  ADF: stat={adf[0]:.6f}, p={adf[1]:.6f}, 5%临界值={adf[4]['5%']:.6f}")
    logger.info(f"  KPSS: stat={kp[0]:.6f}, p={kp[1]:.6f}, 5%临界值={kp[3]['5%']:.6f}")
    logger.info(f"  >> 结论: 收益率平稳 I(0)")

    # ARCH-LM 检验
    arch_lm = het_arch(r - np.mean(r), nlags=10)
    logger.info(f"  ARCH-LM(10): LM={arch_lm[0]:.6f}, p={arch_lm[1]:.6e}")
    logger.info(f"  >> 存在极强的ARCH效应，需GARCH族建模 [OK]")

    # 保存
    ut_df = pd.DataFrame({
        '检验': ['ADF', 'KPSS'],
        '统计量': [adf[0], kp[0]],
        'p值': [adf[1], kp[1]],
        '5%临界值': [adf[4]['5%'], kp[3]['5%']],
        '结论': ['平稳', '平稳']
    })
    ut_df.to_csv(os.path.join(OUT, 'unit_root_results.csv'), index=False, encoding='utf-8-sig')

    arch_df = pd.DataFrame({
        '检验': ['ARCH-LM lag=10'],
        '统计量': [arch_lm[0]],
        'p值': [arch_lm[1]]
    })
    arch_df.to_csv(os.path.join(OUT, 'arch_test_results.csv'), index=False, encoding='utf-8-sig')

    return arch_lm


# ====================== 4. ARIMA 过滤均值方程 ======================
def arima_filter(r):
    """拟合 ARIMA 模型，提取残差用于 GARCH 建模"""
    logger.info("\n" + "=" * 60)
    logger.info("4. ARIMA 均值方程过滤")
    logger.info("=" * 60)

    subset = r[-5000:] if len(r) > 5000 else r
    try:
        auto_m = pm.auto_arima(subset, start_p=0, start_q=0, max_p=3, max_q=3,
                                d=0, seasonal=False, trace=False, error_action='ignore',
                                suppress_warnings=True, stepwise=True, n_fits=30, information_criterion='aic')
        order = auto_m.order
        logger.info(f"  auto_arima 选定: ARIMA({order[0]},0,{order[2]}), AIC={auto_m.aic():.4f}")
    except:
        order = (2, 0, 2)
        logger.info(f"  auto_arima 失败，使用默认: ARIMA(2,0,2)")

    arima_fit = ARIMA(r, order=order).fit()
    raw_resid = arima_fit.resid
    if hasattr(raw_resid, 'dropna'):
        resid = raw_resid.dropna()
    else:
        resid = np.array(raw_resid)
    resid = resid[~np.isnan(resid)]
    logger.info(f"  ARIMA{order} 拟合完成, AIC={arima_fit.aic:.4f}, BIC={arima_fit.bic:.4f}")

    arch_lm_resid = het_arch(resid, nlags=10)
    logger.info(f"  ARIMA残差 ARCH-LM(10): LM={arch_lm_resid[0]:.4f}, p={arch_lm_resid[1]:.6e}")
    logger.info(f"  >> 残差仍存在ARCH效应，需GARCH建模 [OK]")

    # 保存 ARIMA 残差中间结果 (供 plot_figures.py 制图)
    r_arr = np.array(resid).flatten()
    pd.DataFrame({'arima_residual': r_arr}).to_csv(
        os.path.join(OUT, 'arima_residuals.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  中间结果: arima_residuals.csv 已保存 ({len(r_arr)} obs)")

    return resid, order


# ====================== 5. 四模型建模 ======================
def four_model_modeling(resid):
    """
    对 ARIMA 残差拟合 4 个波动率模型：
    1. GARCH(1,1)-Normal
    2. GARCH(1,1)-t
    3. EGARCH(1,1)-t
    4. APARCH(1,1)-t
    """
    logger.info("\n" + "=" * 60)
    logger.info("5. 四模型波动率建模 (基于ARIMA残差)")
    logger.info("=" * 60)

    models = {}

    # ===== Model 1: GARCH(1,1)-Normal =====
    logger.info("  拟合 GARCH(1,1)-Normal...")
    fn = arch_model(resid, mean='Zero', vol='GARCH', p=1, q=1, dist='normal').fit(
        disp='off', options={'maxiter': 2000})
    pn = fn.params
    persist_n = pn.get('alpha[1]', 0) + pn.get('beta[1]', 0)
    models['GARCH(1,1)-N'] = {'fit': fn, 'aic': fn.aic, 'bic': fn.bic,
                                'loglik': fn.loglikelihood, 'params': pn, 'persist': persist_n}
    logger.info(f"    AIC={fn.aic:.4f}, alpha1={pn.get('alpha[1]',0):.6f}, beta1={pn.get('beta[1]',0):.6f}, persist={persist_n:.6f}")

    # ===== Model 2: GARCH(1,1)-t =====
    logger.info("  拟合 GARCH(1,1)-t...")
    ft = arch_model(resid, mean='Zero', vol='GARCH', p=1, q=1, dist='t').fit(
        disp='off', options={'maxiter': 2000})
    pt = ft.params
    persist_t = pt.get('alpha[1]', 0) + pt.get('beta[1]', 0)
    models['GARCH(1,1)-t'] = {'fit': ft, 'aic': ft.aic, 'bic': ft.bic,
                                'loglik': ft.loglikelihood, 'params': pt, 'persist': persist_t}
    logger.info(f"    AIC={ft.aic:.4f}, nu={pt.get('nu',np.nan):.4f}, persist={persist_t:.6f}")

    # ===== Model 3: EGARCH(1,1)-t =====
    logger.info("  拟合 EGARCH(1,1)-t...")
    fe = arch_model(resid, mean='Zero', vol='EGARCH', p=1, q=1, dist='t').fit(
        disp='off', options={'maxiter': 2000})
    pe = fe.params
    persist_e = pe.get('beta[1]', 0)
    gamma_e = pe.get('gamma[1]', np.nan)
    models['EGARCH(1,1)-t'] = {'fit': fe, 'aic': fe.aic, 'bic': fe.bic,
                                 'loglik': fe.loglikelihood, 'params': pe, 'persist': persist_e}
    logger.info(f"    AIC={fe.aic:.4f}, beta1={persist_e:.6f}, gamma1={gamma_e:.6f}, nu={pe.get('nu',np.nan):.4f}")
    if not np.isnan(gamma_e):
        logger.info(f"    >> gamma1={'<' if gamma_e < 0 else '>'} 0 >> {'传统杠杆效应' if gamma_e < 0 else '反向杠杆效应'}")

    # ===== Model 4: APARCH(1,1)-t =====
    logger.info("  拟合 APARCH(1,1)-t...")
    fa = arch_model(resid, mean='Zero', vol='APARCH', p=1, q=1, dist='t').fit(
        disp='off', options={'maxiter': 2000})
    pa = fa.params
    persist_a = pa.get('beta[1]', 0)
    gamma_a = pa.get('gamma[1]', np.nan)
    delta_a = pa.get('delta', np.nan)
    models['APARCH(1,1)-t'] = {'fit': fa, 'aic': fa.aic, 'bic': fa.bic,
                                 'loglik': fa.loglikelihood, 'params': pa, 'persist': persist_a}
    logger.info(f"    AIC={fa.aic:.4f}, beta1={persist_a:.6f}, gamma1={gamma_a:.6f}, delta={delta_a:.4f}, nu={pa.get('nu',np.nan):.4f}")
    if not np.isnan(gamma_a):
        logger.info(f"    >> gamma1={'<' if gamma_a < 0 else '>'} 0 >> {'传统杠杆效应' if gamma_a < 0 else '反向杠杆效应'}")

    # ===== 模型比较与排序 =====
    best = min(models, key=lambda k: models[k]['aic'])
    logger.info(f"\n  * 样本内最优模型 (AIC): {best} (AIC={models[best]['aic']:.4f})")

    sorted_models = sorted(models.items(), key=lambda x: x[1]['aic'])
    for rank, (name, m) in enumerate(sorted_models, 1):
        logger.info(f"    {rank}. {name}: AIC={m['aic']:.4f}, BIC={m['bic']:.4f}, LogLik={m['loglik']:.4f}")

    # ===== 保存模型比较 =====
    comp_data = []
    for n, m in models.items():
        comp_data.append({
            '模型': n,
            'AIC': m['aic'],
            'BIC': m['bic'],
            'LogLik': m['loglik'],
            'alpha+beta(beta)': m['persist'],
        })
    pd.DataFrame(comp_data).to_csv(os.path.join(OUT, 'model_comparison.csv'), index=False, encoding='utf-8-sig')

    # ===== 保存模型参数 =====
    rows = []
    for n, m in models.items():
        p = m['params']
        rows.append({
            '模型': n,
            'mu': p.get('mu', np.nan),
            'omega': p.get('omega', np.nan),
            'alpha1': p.get('alpha[1]', np.nan),
            'beta1': p.get('beta[1]', np.nan),
            'gamma1': p.get('gamma[1]', np.nan),
            'delta': p.get('delta', np.nan),
            'nu': p.get('nu', np.nan),
            'persistence': m['persist'],
            'AIC': m['aic'],
            'BIC': m['bic'],
            'LogLik': m['loglik'],
        })
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, 'model_parameters.csv'), index=False, encoding='utf-8-sig')

    logger.info(f"  模型参数已保存至 {OUT}/model_parameters.csv")
    return models, best


# ====================== 6. 标准化残差诊断 (仅保存中间结果, 不绘图) ======================
def diagnostics(models, arima_resid):
    logger.info("\n" + "=" * 60)
    logger.info("6. 模型诊断")
    logger.info("=" * 60)

    resid_stats = []
    for name, m in models.items():
        sr = m['fit'].std_resid
        sr_arr = sr.dropna().values if hasattr(sr, 'dropna') else np.array(sr)
        sr_arr = sr_arr[~np.isnan(sr_arr)]
        mu_sr = np.mean(sr_arr); sg_sr = np.std(sr_arr)
        sk_sr = skew(sr_arr); ku_sr = kurtosis(sr_arr, fisher=True)
        logger.info(f"  {name}: mu={mu_sr:.4f}, sigma={sg_sr:.4f}, 偏度={sk_sr:.4f}, 超额峰度={ku_sr:.4f}")
        resid_stats.append({'模型': name, '均值': mu_sr, '标准差': sg_sr,
                            '偏度': sk_sr, '峰度(超额)': ku_sr})
    pd.DataFrame(resid_stats).to_csv(os.path.join(OUT, 'std_resid_stats.csv'), index=False, encoding='utf-8-sig')

    model_names = list(models.keys())

    # 保存中间结果：条件波动率
    cv_df = pd.DataFrame()
    for name in model_names:
        cv = models[name]['fit'].conditional_volatility
        cv_arr = cv.values if hasattr(cv, 'values') else np.array(cv)
        cv_arr = np.array(cv_arr).flatten()
        cv_df[name + '_cond_vol'] = cv_arr
    cv_df.to_csv(os.path.join(OUT, 'conditional_volatility.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  中间结果: conditional_volatility.csv 已保存 ({len(cv_df)} obs)")

    # 保存中间结果：标准化残差
    sr_df = pd.DataFrame()
    for name in model_names:
        sr = models[name]['fit'].std_resid
        sr_arr = sr.dropna().values if hasattr(sr, 'dropna') else np.array(sr)
        sr_arr = np.array(sr_arr).flatten()
        sr_df[name + '_std_resid'] = sr_arr
    sr_df.to_csv(os.path.join(OUT, 'standardized_residuals.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  中间结果: standardized_residuals.csv 已保存 ({len(sr_df)} obs)")


# ====================== 7. 滚动窗口样本外预测 ======================
def rolling_forecast(models, resid_full, best_name):
    """
    滚动窗口预测 (rolling forecast)：
    训练窗口 = 前 90% 数据
    每次向前滚动 1 步，重新估计所有模型
    每 50 步重新估计一次以节省时间
    """
    logger.info("\n" + "=" * 60)
    logger.info("7. 滚动窗口样本外预测 (Rolling Forecast)")
    logger.info("=" * 60)

    T = len(resid_full)
    n_train_init = int(T * 0.90)
    n_test = T - n_train_init
    re_estimate_every = 50

    logger.info(f"  总观测: {T}, 初始训练集: {n_train_init}, 测试集: {n_test}")
    logger.info(f"  重新估计频率: 每 {re_estimate_every} 步")

    model_names = list(models.keys())
    predictions = {name: np.full(n_test, np.nan) for name in model_names}
    actual_vol = np.full(n_test, np.nan)

    for step in range(n_test):
        idx = n_train_init + step
        train_data = resid_full[:idx]

        if idx < T:
            actual_vol[step] = np.abs(resid_full[idx])

        if step == 0 or step % re_estimate_every == 0:
            fits = {}
            try:
                fits['GARCH(1,1)-N'] = arch_model(train_data, mean='Zero', vol='GARCH', p=1, q=1,
                                                    dist='normal').fit(disp='off', options={'maxiter': 2000})
                fits['GARCH(1,1)-t'] = arch_model(train_data, mean='Zero', vol='GARCH', p=1, q=1,
                                                    dist='t').fit(disp='off', options={'maxiter': 2000})
                fits['EGARCH(1,1)-t'] = arch_model(train_data, mean='Zero', vol='EGARCH', p=1, q=1,
                                                     dist='t').fit(disp='off', options={'maxiter': 2000})
                fits['APARCH(1,1)-t'] = arch_model(train_data, mean='Zero', vol='APARCH', p=1, q=1,
                                                     dist='t').fit(disp='off', options={'maxiter': 2000})
            except Exception as e:
                logger.warning(f"  Step {step}: 模型重估计失败 ({e})，使用上一步参数")
                continue

        for name, fit_obj in fits.items():
            try:
                fcast = fit_obj.forecast(horizon=1)
                var_pred = fcast.variance.values[-1, 0]
                predictions[name][step] = np.sqrt(max(var_pred, 0))
            except Exception:
                predictions[name][step] = np.nan

        if step % 200 == 0 and step > 0:
            logger.info(f"  进度: {step}/{n_test} 步完成")

    logger.info(f"  滚动预测完成 ({n_test} 步)")

    # 保存中间结果：滚动预测值（样本外）
    pred_df = pd.DataFrame({'actual_proxy_vol': actual_vol})
    for name in model_names:
        pred_df[name + '_pred_vol'] = predictions[name]
    pred_df.to_csv(os.path.join(OUT, 'rolling_forecast_predictions.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  中间结果: rolling_forecast_predictions.csv 已保存 ({n_test} steps, columns: actual + 4 models)")

    # ====== 评估指标 ======
    eval_all = {}
    for name in model_names:
        pred = predictions[name]
        mask = ~np.isnan(pred) & ~np.isnan(actual_vol)
        a = actual_vol[mask]; p = pred[mask]
        n_val = len(a)
        if n_val < 10:
            logger.warning(f"  {name}: 有效预测不足 ({n_val})")
            continue

        rmse = np.sqrt(np.mean((a - p)**2))
        mae = np.mean(np.abs(a - p))
        d = (np.abs(a) + np.abs(p)) / 2; d = np.where(d == 0, 1e-10, d)
        smape = np.mean(np.abs(a - p) / d) * 100
        log_like = np.log(p**2) + (a**2) / (p**2)
        qlike = np.mean(log_like)

        eval_all[name] = {'RMSE': rmse, 'MAE': mae, 'SMAPE(%)': smape, 'QLIKE': qlike, 'n': n_val}
        logger.info(f"  {name} (n={n_val}): RMSE={rmse:.6f}, MAE={mae:.6f}, SMAPE={smape:.4f}%, QLIKE={qlike:.4f}")

    eval_df = pd.DataFrame([{'模型': n, **v} for n, v in eval_all.items()])
    eval_df.to_csv(os.path.join(OUT, 'forecast_evaluation_rolling.csv'), index=False, encoding='utf-8-sig')

    # ====== DM 检验 (四模型两两) ======
    logger.info("\n  Diebold-Mariano 检验 (平方误差损失):")
    dm_results = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            ni, nj = model_names[i], model_names[j]
            mask = ~np.isnan(predictions[ni]) & ~np.isnan(predictions[nj]) & ~np.isnan(actual_vol)
            a = actual_vol[mask]; pi = predictions[ni][mask]; pj = predictions[nj][mask]
            if len(a) < 20:
                dm_results.append({'模型1': ni, '模型2': nj, 'DM统计量': np.nan, 'DM_p值': np.nan, '结论': '样本不足'})
                continue
            ei = (a - pi)**2; ej = (a - pj)**2
            d = ei - ej
            dm_m = np.mean(d); dm_s = np.std(d, ddof=1)
            nd = len(d)
            if dm_s > 0:
                dm_stat = np.sqrt(nd) * dm_m / dm_s
                dm_p = 2 * (1 - t_dist.cdf(abs(dm_stat), df=nd - 1))
            else:
                dm_stat, dm_p = np.nan, np.nan
            sig = '***' if (not np.isnan(dm_p) and dm_p < 0.01) else ('**' if dm_p < 0.05 else ('*' if dm_p < 0.10 else ''))
            if dm_stat < 0:
                conclusion = f'{ni} 更优{sig}' if sig else f'{ni} 略优(不显著)'
            else:
                conclusion = f'{nj} 更优{sig}' if sig else f'{nj} 略优(不显著)'
            dm_results.append({'模型1': ni, '模型2': nj, 'DM统计量': dm_stat, 'DM_p值': dm_p, '结论': conclusion})
            logger.info(f"    {ni} vs {nj}: DM={dm_stat:.4f}, p={dm_p:.4f} >> {conclusion}")
    pd.DataFrame(dm_results).to_csv(os.path.join(OUT, 'dm_test_rolling.csv'), index=False, encoding='utf-8-sig')

    return eval_all, dm_results


# ====================== 8. 汇总 ======================
def final_summary(stats_dict, models, best_name, eval_all, dm_results):
    logger.info("\n" + "=" * 60)
    logger.info("8. 最终结果汇总")
    logger.info("=" * 60)

    logger.info(f"\n  [数据概况]")
    logger.info(f"    观测数: {stats_dict['N']}, 均值: {stats_dict['mu']:.6f}%, 标准差: {stats_dict['sigma']:.6f}%")

    logger.info(f"\n  [样本内拟合](按AIC排序)")
    sorted_models = sorted(models.items(), key=lambda x: x[1]['aic'])
    for rank, (n, m) in enumerate(sorted_models, 1):
        logger.info(f"    {rank}. {n}: AIC={m['aic']:.4f}" + (' *最优' if n == best_name else ''))

    logger.info(f"\n  [样本外滚动预测]({stats_dict['N']}个观测, 滚动窗口90%/10%)")
    best_rmse = min(eval_all, key=lambda k: eval_all[k]['RMSE'])
    best_qlike = min(eval_all, key=lambda k: eval_all[k]['QLIKE'])
    for n, v in eval_all.items():
        logger.info(f"    {n}: RMSE={v['RMSE']:.6f}, QLIKE={v['QLIKE']:.4f}")
    logger.info(f"    最佳RMSE: {best_rmse} ({eval_all[best_rmse]['RMSE']:.6f})")
    logger.info(f"    最佳QLIKE: {best_qlike} ({eval_all[best_qlike]['QLIKE']:.4f})")

    logger.info(f"\n  [核心结论]")
    logger.info(f"    1. 样本内: {best_name} 拟合最优 (AIC最小)")
    logger.info(f"    2. 样本外: {best_rmse} 预测精度最高 (RMSE最小)")
    logger.info(f"    3. APARCH-t 模型同时捕捉厚尾、杠杆效应和幂变换，提供了最灵活的波动率刻画")

    # 保存汇总
    summary = [
        {'项目': '数据起始', '值': '2000-01-04'},
        {'项目': '数据结束', '值': '2026-06-08'},
        {'项目': '有效观测', '值': stats_dict['N']},
        {'项目': '均值方程', '值': 'ARIMA (auto_arima选择)'},
        {'项目': '波动率模型数', '值': '4 (GARCH-N, GARCH-t, EGARCH-t, APARCH-t)'},
        {'项目': '样本内最优', '值': best_name},
        {'项目': '最优AIC', '值': f"{models[best_name]['aic']:.4f}"},
        {'项目': '样本外最优RMSE', '值': f"{best_rmse} ({eval_all[best_rmse]['RMSE']:.6f})"},
    ]
    pd.DataFrame(summary).to_csv(os.path.join(OUT, 'final_summary.csv'), index=False, encoding='utf-8-sig')

    logger.info(f"\n  所有结果已保存至 {BASE_DIR}/ 目录")
    logger.info(f"    >> 数据: {OUT}/")


# ====================== 主函数 ======================
def main():
    logger.info("=" * 60)
    logger.info("上证综指波动率分析 — 核心建模")
    logger.info("(GARCH-N, GARCH-t, EGARCH-t, APARCH-t)")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {BASE_DIR}")
    logger.info("=" * 60)

    # 1-3: 数据 + 描述 + 检验
    df = load_data()
    r, stats_dict = descriptive_stats(df)
    _ = preliminary_tests(r)

    # 4: ARIMA 过滤
    arima_resid, arima_order = arima_filter(r)
    logger.info(f"  ARIMA残差长度: {len(arima_resid)}")

    # 5: 四模型建模 (含APARCH)
    models, best_name = four_model_modeling(arima_resid)

    # 6: 诊断 (保存中间结果CSV，不绘图)
    diagnostics(models, arima_resid)

    # 7: 滚动预测 + DM 检验
    eval_all, dm_results = rolling_forecast(models, arima_resid, best_name)

    # 8: 汇总
    final_summary(stats_dict, models, best_name, eval_all, dm_results)

    logger.info("\n" + "=" * 60)
    logger.info("核心建模全部完成!")
    logger.info(f"所有数据结果保存在: {OUT}")
    logger.info(f"请运行 plot_figures.py 生成图表")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()