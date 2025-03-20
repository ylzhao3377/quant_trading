import optuna
import pandas as pd
import numpy as np
from optuna import TrialPruned
import sys
import os
from datetime import datetime, timedelta
import pytz

# Import modules from strategy_logic.py (renamed from simple_strategy.py)
from strategy_logic import (
    fetch_stock_data,
    compute_metrics,
    HiddenPrints
)

# Import SynchronizedPortfolio class from portfolio_strategy.py
from portfolio_strategy import SynchronizedPortfolio


def convert_to_pst(dt):
    """
    Convert a UTC datetime to PST (Pacific Standard Time).

    Args:
        dt: datetime object or timestamp string

    Returns:
        datetime: PST datetime object
    """
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)

    # If datetime is naive (no timezone), assume it's UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)

    # Convert to PST
    pacific = pytz.timezone('US/Pacific')
    return dt.astimezone(pacific)


def objective(trial, df, strategy_type, initial_balance=10000):
    """
    Objective function for Optuna optimization framework.
    Defines parameters to optimize and evaluates strategy performance.

    Args:
        trial: Optuna trial object
        df: Stock data DataFrame
        strategy_type: 'trend' or 'range'
        initial_balance: Starting capital

    Returns:
        tuple: (-ROI, max_drawdown, -Sharpe ratio) for multi-objective optimization
    """
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
        params = {
            'bollinger_window': trial.suggest_int("window", 60, 120),
            'min_strength': trial.suggest_float("min_strength", 0.2, 0.7),
            'min_drop_pct': trial.suggest_float("min_drop_pct", 0.005, 0.02),
            'min_rise_pct': trial.suggest_float("min_rise_pct", 0.005, 0.02),
        }
    else:
        raise ValueError("Unknown strategy type")

    # Create portfolio and run backtest
    portfolio = SynchronizedPortfolio(total_capital=initial_balance)
    portfolio.add_account("TUNE", strategy_type, params, df, allocation_pct=100)
    portfolio.run()
    trade_log_df = portfolio.combined_trade_log()

    if trade_log_df.empty:
        return 1e6, 1e6, 1e6

    # Calculate performance metrics
    roi, max_dd, sharpe = compute_metrics(trade_log_df, initial_balance=initial_balance)
    print(roi, max_dd, sharpe)
    if roi is None or max_dd is None or sharpe is None:
        return 1e6, 1e6, 1e6

    # Return negative ROI and Sharpe since Optuna minimizes by default
    return -roi, max_dd, -sharpe


def optimize_strategy(df, ticker, strategy_type='trend', n_trials=50, initial_balance=10000, verbose=False):
    """
    Optimize strategy parameters using Optuna.

    Args:
        df: Stock data DataFrame
        ticker: Stock ticker symbol
        strategy_type: 'trend' or 'range'
        n_trials: Number of optimization trials
        initial_balance: Starting capital
        verbose: Whether to print optimization progress

    Returns:
        tuple: (best_params, roi, max_drawdown, sharpe_ratio, pareto_trials)
    """
    # Create multi-objective study
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"]  # -ROI, Max Drawdown, -Sharpe
    )

    # Run optimization
    if verbose:
        study.optimize(lambda trial: objective(trial, df, strategy_type, initial_balance), n_trials=n_trials)
    else:
        with HiddenPrints():
            study.optimize(lambda trial: objective(trial, df, strategy_type, initial_balance), n_trials=n_trials)

    pareto_trials = study.best_trials
    print(f"\n✅ Found {len(pareto_trials)} Pareto optimal solutions")

    # Get performance metrics from all Pareto-optimal trials
    roi_values = np.array([-t.values[0] for t in pareto_trials])  # Convert back to positive
    mdd_values = np.array([t.values[1] for t in pareto_trials])
    sharpe_values = np.array([-t.values[2] for t in pareto_trials])  # Convert back to positive

    # Filter for trials that meet risk criteria: max_dd >= -20% and Sharpe >= 2
    valid_indices = []
    for i in range(len(pareto_trials)):
        if mdd_values[i] >= -20 and sharpe_values[i] >= 2:
            valid_indices.append(i)

    # Check if any trials meet the criteria
    if valid_indices:
        # Select the trial with highest ROI among those meeting risk criteria
        best_roi = -float('inf')
        best_index = -1
        for i in valid_indices:
            if roi_values[i] > best_roi:
                best_roi = roi_values[i]
                best_index = i

        best_trial = pareto_trials[best_index]
        best_params = best_trial.params

        print(f"\n🔍 Selected parameters with highest ROI ({best_roi:.2f}%) while meeting risk criteria:")
        print(f"   Max Drawdown: {mdd_values[best_index]:.2f}%, Sharpe Ratio: {sharpe_values[best_index]:.2f}")
    else:
        # If no trials meet the strict criteria, fall back to the original weighted scoring method
        print("\n⚠️ No trials meet strict risk criteria (max_dd >= -20% AND Sharpe >= 2)")
        print("   Falling back to weighted scoring method...")

        # Normalize metrics for comparable scaling
        roi_norm = (roi_values - roi_values.min()) / (roi_values.max() - roi_values.min() + 1e-8)
        sharpe_norm = (sharpe_values - sharpe_values.min()) / (sharpe_values.max() - sharpe_values.min() + 1e-8)
        mdd_norm = (mdd_values.max() - mdd_values) / (mdd_values.max() - mdd_values.min() + 1e-8)  # Reverse for MDD

        # Calculate composite score with custom weights
        weights = [0.7, 0.2, 0.1]  # ROI: 70%, Sharpe: 20%, MDD: 10%
        scores = weights[0] * roi_norm + weights[1] * sharpe_norm + weights[2] * mdd_norm

        # Select best parameter set by composite score
        best_index = np.argmax(scores)
        best_trial = pareto_trials[best_index]
        best_params = best_trial.params

    # Verify performance with best parameters
    portfolio = SynchronizedPortfolio(total_capital=initial_balance)
    portfolio.add_account("TUNE", strategy_type, best_params, df, allocation_pct=100)
    portfolio.run()
    trade_log_df = portfolio.combined_trade_log()

    # Convert dates to PST
    trade_log_df['Date'] = pd.to_datetime(trade_log_df['Date'])
    trade_log_df['Date_PST'] = trade_log_df['Date'].apply(convert_to_pst)
    trade_log_df['Date_PST_Str'] = trade_log_df['Date_PST'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

    roi, max_dd, sharpe = compute_metrics(trade_log_df, initial_balance=initial_balance)

    # Current time in PST for log messages
    now_pst = convert_to_pst(datetime.now()).strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"\n🎯 [Best Combined Solution] - {now_pst}")
    print(f"Parameters: {best_params}")
    print(f"ROI: {roi:.2f}%, Max Drawdown: {max_dd:.2f}%, Sharpe Ratio: {sharpe:.2f}")

    # Save trade log
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    trade_log_df.to_csv(f'{output_dir}/{ticker}-{strategy_type}-tuning.csv', index=False)
    print(f"✅ {ticker} best strategy trade log saved to {output_dir}/{ticker}-{strategy_type}-tuning.csv")

    return best_params, roi, max_dd, sharpe, pareto_trials


def batch_backtest(ticker_list, result_file='strategy_results.csv', n_trials=50, verbose=False):
    """
    Run optimization and backtest for multiple stocks.
    First finds all eligible strategies, then selects the best one.

    Args:
        ticker_list: List of stock tickers
        result_file: Output CSV file for results
        n_trials: Number of optimization trials per stock
        verbose: Whether to print progress

    Returns:
        pd.DataFrame: Results DataFrame
    """
    results = []

    # Get current time in PST for logging
    now_utc = datetime.now(pytz.UTC)
    now_pst = convert_to_pst(now_utc)
    pst_time_str = now_pst.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Starting batch backtest at {pst_time_str}")

    for ticker in ticker_list:
        print(f"🚀 Optimizing {ticker}...")
        # Calculate dates in PST
        end_date_pst = now_pst.strftime("%Y-%m-%d")
        start_date_pst = (now_pst - timedelta(days=365)).strftime("%Y-%m-%d")

        # Get stock data
        try:
            df = fetch_stock_data(ticker, start_date_pst, end_date_pst, "hour")
        except Exception as e:
            print(f"❌ Failed to get {ticker} data: {e}")
            continue

        # Optimize trend strategy
        try:
            trend_params, trend_roi, trend_dd, trend_sharpe, trend_pareto = optimize_strategy(
                df, ticker, 'trend', n_trials, verbose=verbose
            )
            print(f"✅ {ticker} TREND ROI: {trend_roi:.2f}%, MaxDD: {trend_dd:.2f}%, Sharpe: {trend_sharpe:.2f}")

            # Check if trend strategy meets eligibility criteria
            trend_eligible = (
                    trend_roi is not None and trend_roi > 0 and
                    trend_dd is not None and trend_dd >= -20 and
                    trend_sharpe is not None and trend_sharpe >= 2
            )
        except (Exception, TrialPruned) as e:
            print(f"❌ {ticker} TREND strategy failed: {e}")
            trend_params, trend_roi, trend_dd, trend_sharpe, trend_pareto = {}, None, None, None, None
            trend_eligible = False

        # Optimize range strategy
        try:
            range_params, range_roi, range_dd, range_sharpe, range_pareto = optimize_strategy(
                df, ticker, 'range', n_trials, verbose=verbose
            )
            print(f"✅ {ticker} RANGE ROI: {range_roi:.2f}%, MaxDD: {range_dd:.2f}%, Sharpe: {range_sharpe:.2f}")

            # Check if range strategy meets eligibility criteria
            range_eligible = (
                    range_roi is not None and range_roi > 0 and
                    range_dd is not None and range_dd >= -20 and
                    range_sharpe is not None and range_sharpe >= 2
            )
        except (Exception, TrialPruned) as e:
            print(f"❌ {ticker} RANGE strategy failed: {e}")
            range_params, range_roi, range_dd, range_sharpe, range_pareto = {}, None, None, None, None
            range_eligible = False

        # Select better strategy ONLY from eligible strategies
        chosen_strategy, chosen_roi, chosen_dd, chosen_sharpe, chosen_params, chosen_pareto = None, None, None, None, None, None
        eligible = False

        if trend_eligible and range_eligible:
            # Both strategies are eligible, choose the better one
            if trend_roi >= range_roi:
                chosen_strategy = 'Trend'
                chosen_roi, chosen_dd, chosen_sharpe = trend_roi, trend_dd, trend_sharpe
                chosen_params, chosen_pareto = trend_params, trend_pareto
            else:
                chosen_strategy = 'Range'
                chosen_roi, chosen_dd, chosen_sharpe = range_roi, range_dd, range_sharpe
                chosen_params, chosen_pareto = range_params, range_pareto
            eligible = True
        elif trend_eligible:
            # Only trend strategy is eligible
            chosen_strategy = 'Trend'
            chosen_roi, chosen_dd, chosen_sharpe = trend_roi, trend_dd, trend_sharpe
            chosen_params, chosen_pareto = trend_params, trend_pareto
            eligible = True
        elif range_eligible:
            # Only range strategy is eligible
            chosen_strategy = 'Range'
            chosen_roi, chosen_dd, chosen_sharpe = range_roi, range_dd, range_sharpe
            chosen_params, chosen_pareto = range_params, range_pareto
            eligible = True
        elif trend_roi is not None and range_roi is not None:
            # Neither strategy is eligible, but we have results for both
            # Record the better one but mark as not eligible
            if trend_roi >= range_roi:
                chosen_strategy = 'Trend'
                chosen_roi, chosen_dd, chosen_sharpe = trend_roi, trend_dd, trend_sharpe
                chosen_params, chosen_pareto = trend_params, trend_pareto
            else:
                chosen_strategy = 'Range'
                chosen_roi, chosen_dd, chosen_sharpe = range_roi, range_dd, range_sharpe
                chosen_params, chosen_pareto = range_params, range_pareto
            eligible = False
        elif trend_roi is not None:
            # Only have trend results, but not eligible
            chosen_strategy = 'Trend'
            chosen_roi, chosen_dd, chosen_sharpe = trend_roi, trend_dd, trend_sharpe
            chosen_params, chosen_pareto = trend_params, trend_pareto
            eligible = False
        elif range_roi is not None:
            # Only have range results, but not eligible
            chosen_strategy = 'Range'
            chosen_roi, chosen_dd, chosen_sharpe = range_roi, range_dd, range_sharpe
            chosen_params, chosen_pareto = range_params, range_pareto
            eligible = False

        # Record timestamp in PST
        timestamp_pst = now_pst.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Record results with eligibility status
        results.append({
            'Stock': ticker,
            'Chosen Strategy': chosen_strategy,
            'ROI (%)': chosen_roi,
            'Max Drawdown (%)': chosen_dd,
            'Sharpe Ratio': chosen_sharpe,
            'Best Params': chosen_params,
            'Eligible': eligible,
            'Trend Eligible': trend_eligible if 'trend_eligible' in locals() else False,
            'Range Eligible': range_eligible if 'range_eligible' in locals() else False,
            'Timestamp PST': timestamp_pst
        })

    # Calculate allocation scores for portfolio construction
    df_results = pd.DataFrame(results)

    # Filter eligible stocks
    df_results = df_results[df_results["Eligible"] == True]

    if not df_results.empty:
        # Normalize performance metrics
        roi_values = df_results["ROI (%)"].astype(float)
        sharpe_values = df_results["Sharpe Ratio"].astype(float)
        mdd_values = df_results["Max Drawdown (%)"].astype(float)

        roi_norm = (roi_values - roi_values.min()) / (roi_values.max() - roi_values.min() + 1e-8)
        sharpe_norm = (sharpe_values - sharpe_values.min()) / (sharpe_values.max() - sharpe_values.min() + 1e-8)
        mdd_norm = (mdd_values.max() - mdd_values) / (mdd_values.max() - mdd_values.min() + 1e-8)  # Reverse for MDD

        # Calculate allocation score with weighted metrics
        weights = [0.7, 0.2, 0.1]  # ROI: 50%, Sharpe: 30%, MDD: 20%
        df_results["Allocation Score"] = weights[0] * roi_norm + weights[1] * sharpe_norm + weights[2] * mdd_norm

        # Calculate allocation percentages (total = 100%)
        total_score = df_results["Allocation Score"].sum()
        df_results["Allocation (%)"] = (df_results["Allocation Score"] / total_score) * 100 if total_score > 0 else 0
    else:
        df_results["Allocation (%)"] = 0

    # Save results
    df_results.to_csv(result_file, index=False)
    print(f"✅ Portfolio allocation results saved to {result_file}")

    return df_results


# Example usage
if __name__ == "__main__":
    ticker_list = ["AMZN", "META", "NVDA", "TSLA", "TQQQ", "PLTR", "SNOW", "CRM",
                   "GOOGL", "AAPL", "AVGO", "KO", "SPY", "VRT", "AMD", "COST",
                   "JNJ", "JPM", 'MSFT', "ASML", "BRK.B", "CRWD", "PYPL", "SBUX",
                   "DIS", "AXP", "ROKU", "COIN", "SHOP", "INTC", "LULU", "PANW"]
    batch_backtest(ticker_list, result_file='strategy_results.csv', n_trials=30, verbose=False)