import optuna
import pandas as pd
import numpy as np
from optuna import TrialPruned
import sys
import os
from portfolio_strategy import fetch_stock_data
from portfolio_strategy import SynchronizedPortfolio
from datetime import datetime, timedelta

# 假设下面这些函数和类已经从新的策略代码中引入：
# - SynchronizedPortfolio
# - compute_indicators（在 SynchronizedPortfolio.add_account 中会调用）
# - get_available_funds
# - 以及其他辅助函数


# 计算回测指标函数（你可以直接使用你原来的 compute_metrics）
def compute_metrics(trade_df, initial_balance=10000):
    """
    计算回测指标：
      - ROI：最终权益相对于初始资金的收益率（百分比）
      - 最大回撤：权益曲线的最大跌幅（百分比）
      - 夏普比率：基于权益收益率的年化夏普比率
    假设 trade_df 包含的字段有：
      Date, Action, Reason, Price, Shares, Balance, Position, Realized_Profit, Stock
    其中 Total_Equity = Balance + Position * Price
    """
    if trade_df.empty:
        return None, None, None

    # 根据现有字段计算 Total_Equity
    trade_df = trade_df.copy()  # 避免修改原始数据
    trade_df['Total_Equity'] = trade_df['Balance'] + trade_df['Position'] * trade_df['Price']

    final_equity = trade_df['Total_Equity'].iloc[-1]
    total_profit = final_equity - initial_balance
    roi = (total_profit / initial_balance) * 100

    # 计算权益变化率，假设每个记录代表一个时间点（例如小时），
    # 可以简单地计算每个记录的权益百分比变化
    trade_df['Equity_Return'] = trade_df['Total_Equity'].pct_change().fillna(0)
    std_ret = trade_df['Equity_Return'].std()
    sharpe = (trade_df['Equity_Return'].mean() / std_ret) * (252 ** 0.5) if std_ret != 0 else 0

    cumulative = trade_df['Total_Equity'].cummax()
    drawdown = (trade_df['Total_Equity'] - cumulative) / cumulative
    max_drawdown = drawdown.min() * 100  # 以百分比表示

    return roi, max_drawdown, sharpe


# 修改后的 objective_new：在采样参数时加入 bollinger_window 参数
def objective(trial, df, strategy_type, initial_balance=10000):
    if strategy_type == 'trend':
        bollinger_window = trial.suggest_int("bollinger_window", 30, 100)
        atr_thresh = trial.suggest_float("atr_pct_threshold", 0.002, 0.02)
        adx_thresh = trial.suggest_int("adx_threshold", 20, 40)
        vol_thresh = trial.suggest_float("vol_threshold", 1.1, 2.0)
        trailing_stop_pct = trial.suggest_float("trailing_stop_pct", 0.90, 0.99)
        params = {
            'bollinger_window': bollinger_window,
            'trailing_stop_pct': trailing_stop_pct,
            'adx_threshold': adx_thresh,
            'vol_threshold': vol_thresh,
            'atr_pct_threshold': atr_thresh
        }
    elif strategy_type == 'range':
        # 对于 range 策略，同样加入 bollinger_window 参数
        params = {
            'bollinger_window': trial.suggest_int("window", 60, 120),  # 布林带窗口，小时级别，偏短适合抓中短期波动
            'min_strength': trial.suggest_float("min_strength", 0.2, 0.7),  # 突破强度，防止假突破
            'min_drop_pct': trial.suggest_float("min_drop_pct", 0.005, 0.02),  # 创新低买入，0.5%-2%
            'min_rise_pct': trial.suggest_float("min_rise_pct", 0.005, 0.02),  # 创新高卖出，0.5%-2%
            'float_position_pct': trial.suggest_float("float_position_pct", 0.1, 0.3),  # 浮动仓比例 10%-30%
        }
    else:
        raise ValueError("未知的策略类型")

    # 创建组合时，将参数传递到 add_account 内部，在那里会调用 compute_indicators(df, bollinger_window=params['bollinger_window'])
    portfolio = SynchronizedPortfolio(total_capital=initial_balance)
    portfolio.add_account("TUNE", strategy_type, params, df, allocation_pct=100)
    portfolio.run()
    trade_log_df = portfolio.combined_trade_log()

    if trade_log_df.empty:
        return 1e6, 1e6, 1e6

    roi, max_dd, sharpe = compute_metrics(trade_log_df, initial_balance=initial_balance)
    print(roi, max_dd, sharpe)
    if roi is None or max_dd is None or sharpe is None:
        return 1e6, 1e6, 1e6

    return -roi, max_dd, -sharpe


# 优化器函数（单只股票的参数调优）
def optimize_strategy(df, ticker, strategy_type='trend', n_trials=50, initial_balance=10000, verbose=False):
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"]  # 目标分别为 -ROI, Max Drawdown, -Sharpe
    )

    if verbose:
        study.optimize(lambda trial: objective(trial, df, strategy_type, initial_balance), n_trials=n_trials)
    else:
        with HiddenPrints():
            study.optimize(lambda trial: objective(trial, df, strategy_type, initial_balance), n_trials=n_trials)

    pareto_trials = study.best_trials
    print(f"\n✅ 共找到 {len(pareto_trials)} 个 Pareto 最优解")

    # 获取所有 ROI、MDD、Sharpe 数据
    roi_values = np.array([-t.values[0] for t in pareto_trials])  # 负号转换回正值
    mdd_values = np.array([t.values[1] for t in pareto_trials])
    sharpe_values = np.array([-t.values[2] for t in pareto_trials])  # 负号转换回正值

    # 归一化指标 (防止某个指标的数量级过大影响最终选择)
    roi_norm = (roi_values - roi_values.min()) / (roi_values.max() - roi_values.min() + 1e-8)
    sharpe_norm = (sharpe_values - sharpe_values.min()) / (sharpe_values.max() - sharpe_values.min() + 1e-8)
    mdd_norm = (mdd_values.max() - mdd_values) / (mdd_values.max() - mdd_values.min() + 1e-8)  # 反向归一化

    # 计算综合评分 (自定义权重)
    weights = [0.55, 0.3, 0.15]  # ROI 50%, Sharpe 30%, MDD 20%
    scores = weights[0] * roi_norm + weights[1] * sharpe_norm + weights[2] * mdd_norm

    # 选择综合评分最高的 trial 作为最佳参数
    best_index = np.argmax(scores)
    best_trial = pareto_trials[best_index]
    best_params = best_trial.params

    # 重新跑一次回测
    portfolio = SynchronizedPortfolio(total_capital=initial_balance)
    portfolio.add_account("TUNE", strategy_type, best_params, df, allocation_pct=100)
    portfolio.run()
    trade_log_df = portfolio.combined_trade_log()
    roi, max_dd, sharpe = compute_metrics(trade_log_df, initial_balance=initial_balance)

    print("\n🎯 [最佳综合解]")
    print(f"参数: {best_params}")
    print(f"ROI: {roi:.2f}%, Max Drawdown: {max_dd:.2f}%, Sharpe Ratio: {sharpe:.2f}")

    # 保存交易日志
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    trade_log_df.to_csv(f'{output_dir}/{ticker}-{strategy_type}-tuning.csv', index=False)
    print(f"✅ {ticker} 最佳策略交易日志已保存至 {output_dir}/{ticker}-{strategy_type}-tuning.csv")

    return best_params, roi, max_dd, sharpe, pareto_trials


# 屏蔽打印的辅助类（和原来一样）
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def batch_backtest(ticker_list, result_file='strategy_results.csv', n_trials=50, verbose=False,
                   roi_threshold=0, max_dd_threshold=-30, sharpe_threshold=1):
    results = []
    allocation_scores = []

    for ticker in ticker_list:
        print(f"🚀 正在优化 {ticker}...")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = (datetime.now()).strftime("%Y-%m-%d")
        # === 获取数据 ===
        try:
            df = fetch_stock_data(ticker, start_date, end_date, "hour")
        except Exception as e:
            print(f"❌ 获取 {ticker} 数据失败: {e}")
            continue

        # === Trend 策略 ===
        try:
            trend_params, trend_roi, trend_dd, trend_sharpe, trend_pareto = optimize_strategy(
                df, ticker, 'trend', n_trials, verbose=verbose
            )
            print(f"✅ {ticker} TREND ROI: {trend_roi:.2f}%, MaxDD: {trend_dd:.2f}%, Sharpe: {trend_sharpe:.2f}")
        except (Exception, TrialPruned) as e:
            print(f"❌ {ticker} TREND 策略失败: {e}")
            trend_params, trend_roi, trend_dd, trend_sharpe, trend_pareto = {}, None, None, None, None

        # === Range 策略 ===
        try:
            range_params, range_roi, range_dd, range_sharpe, range_pareto = optimize_strategy(
                df, ticker, 'range', n_trials, verbose=verbose
            )
            print(f"✅ {ticker} RANGE ROI: {range_roi:.2f}%, MaxDD: {range_dd:.2f}%, Sharpe: {range_sharpe:.2f}")
        except (Exception, TrialPruned) as e:
            print(f"❌ {ticker} RANGE 策略失败: {e}")
            range_params, range_roi, range_dd, range_sharpe, range_pareto = {}, None, None, None, None

        # === 选择更优策略 ===
        chosen_strategy, chosen_roi, chosen_dd, chosen_sharpe, chosen_params, chosen_pareto = None, None, None, None, None, None
        if trend_roi is not None and range_roi is not None:
            if trend_roi >= range_roi:
                chosen_strategy, chosen_roi, chosen_dd, chosen_sharpe, chosen_params, chosen_pareto = 'Trend', trend_roi, trend_dd, trend_sharpe, trend_params, trend_pareto
            else:
                chosen_strategy, chosen_roi, chosen_dd, chosen_sharpe, chosen_params, chosen_pareto = 'Range', range_roi, range_dd, range_sharpe, range_params, range_pareto
        elif trend_roi is not None:
            chosen_strategy, chosen_roi, chosen_dd, chosen_sharpe, chosen_params, chosen_pareto = 'Trend', trend_roi, trend_dd, trend_sharpe, trend_params, trend_pareto
        elif range_roi is not None:
            chosen_strategy, chosen_roi, chosen_dd, chosen_sharpe, chosen_params, chosen_pareto = 'Range', range_roi, range_dd, range_sharpe, range_params, range_pareto

        # === 筛选达标逻辑 ===
        eligible = (
            chosen_roi is not None and chosen_roi >= roi_threshold and
            chosen_dd is not None and chosen_dd >= max_dd_threshold and
            chosen_sharpe is not None and chosen_sharpe >= sharpe_threshold
        )

        # === 记录结果 ===
        results.append({
            'Stock': ticker,
            'Chosen Strategy': chosen_strategy,
            'ROI (%)': chosen_roi,
            'Max Drawdown (%)': chosen_dd,
            'Sharpe Ratio': chosen_sharpe,
            'Best Params': chosen_params,
            'Eligible': eligible
        })

    # === 归一化并计算 allocation score ===
    df_results = pd.DataFrame(results)

    # 只考虑达标的股票
    df_results = df_results[df_results["Eligible"] == True]

    if not df_results.empty:
        # 标准化 ROI、Sharpe、MDD
        roi_values = df_results["ROI (%)"].astype(float)
        sharpe_values = df_results["Sharpe Ratio"].astype(float)
        mdd_values = df_results["Max Drawdown (%)"].astype(float)

        roi_norm = (roi_values - roi_values.min()) / (roi_values.max() - roi_values.min() + 1e-8)
        sharpe_norm = (sharpe_values - sharpe_values.min()) / (sharpe_values.max() - sharpe_values.min() + 1e-8)
        mdd_norm = (mdd_values.max() - mdd_values) / (mdd_values.max() - mdd_values.min() + 1e-8)  # 反向归一化

        # 计算 allocation score
        weights = [0.5, 0.3, 0.2]  # ROI 50%, Sharpe 30%, MDD 20%
        df_results["Allocation Score"] = weights[0] * roi_norm + weights[1] * sharpe_norm + weights[2] * mdd_norm

        # 计算 allocation (%)，我们想确保总分配加起来等于 20%
        total_score = df_results["Allocation Score"].sum()
        df_results["Allocation (%)"] = (df_results["Allocation Score"] / total_score) * 200 if total_score > 0 else 0
    else:
        df_results["Allocation (%)"] = 0

    # === 保存结果 ===
    df_results.to_csv(result_file, index=False)
    print(f"✅ 筛选并分配比例后的组合表已保存至 {result_file}")

    return df_results


# 示例：假设我们对某个股票进行调优
if __name__ == "__main__":
    # 这里用 fetch_stock_data 获取数据
    ticker_list = ["AMZN", "META", "NVDA", "TSLA", "TQQQ", "PLTR", "SNOW", "CRM",
                   "GOOGL", "AAPL", "AVGO", "KO", "SPY", "VRT", "AMD", "COST",
                   "JNJ", "JPM", 'MSFT', "ASML", "BRK.B", "CRWD", "PYPL",
                   "DIS", "AXP", "ROKU", "COIN", "SHOP", "INTC"]
    batch_backtest(ticker_list, result_file='strategy_results.csv', n_trials=30, verbose=False)
