import optuna
import pandas as pd
import numpy as np
from optuna import TrialPruned
import sys
import os
from portfolio_strategy import SynchronizedPortfolio
from portfolio_strategy import DataFetcher  # Import the DataFetcher class instead
from datetime import datetime, timedelta


class PerformanceMetrics:
    """
    Calculates performance metrics for trading strategy evaluation.
    """

    @staticmethod
    def compute_metrics(trade_df, initial_balance=10000):
        """
        Calculate backtest metrics:
          - ROI: Final equity relative to initial capital (%)
          - Maximum drawdown: Maximum equity curve decline (%)
          - Sharpe ratio: Annualized Sharpe ratio based on equity returns

        Args:
            trade_df (pd.DataFrame): Trade log with OHLCV and position data
            initial_balance (float): Initial account balance

        Returns:
            tuple: (roi, max_drawdown, sharpe_ratio) or (None, None, None) if empty
        """
        if trade_df.empty:
            return None, None, None

        # Calculate Total_Equity from existing fields
        trade_df = trade_df.copy()  # Avoid modifying original data
        trade_df['Total_Equity'] = trade_df['Balance'] + trade_df['Position'] * trade_df['Price']

        # Calculate ROI
        final_equity = trade_df['Total_Equity'].iloc[-1]
        total_profit = final_equity - initial_balance
        roi = (total_profit / initial_balance) * 100

        # Calculate equity returns and Sharpe ratio
        trade_df['Equity_Return'] = trade_df['Total_Equity'].pct_change().fillna(0)
        std_ret = trade_df['Equity_Return'].std()
        # Annualize assuming hourly data - multiply by sqrt(252 trading days)
        sharpe = (trade_df['Equity_Return'].mean() / std_ret) * (252 ** 0.5) if std_ret != 0 else 0

        # Calculate maximum drawdown
        cumulative = trade_df['Total_Equity'].cummax()
        drawdown = (trade_df['Total_Equity'] - cumulative) / cumulative
        max_drawdown = drawdown.min() * 100  # Convert to percentage

        return roi, max_drawdown, sharpe


class HiddenPrints:
    """
    Context manager to suppress print output.
    Used for hiding Optuna internal trial outputs.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class StrategyOptimizer:
    """
    Optimizes trading strategy parameters using Optuna.
    """

    @staticmethod
    def objective(trial, df, strategy_type, initial_balance=10000):
        """
        Objective function for Optuna optimization.

        Args:
            trial (optuna.Trial): Optuna trial object
            df (pd.DataFrame): Price data
            strategy_type (str): Strategy type ('trend' or 'range')
            initial_balance (float): Initial account balance

        Returns:
            tuple: (-roi, max_drawdown, -sharpe_ratio) for multi-objective optimization
        """
        if strategy_type == 'trend':
            # Suggest parameters for trend strategy
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
            # Suggest parameters for range strategy
            params = {
                'bollinger_window': trial.suggest_int("window", 60, 120),
                # Bollinger window, hourly, shorter for capturing medium-term fluctuations
                'min_strength': trial.suggest_float("min_strength", 0.2, 0.7),
                # Breakout strength, prevents false breakouts
                'min_drop_pct': trial.suggest_float("min_drop_pct", 0.005, 0.02),  # Buy on new low, 0.5%-2%
                'min_rise_pct': trial.suggest_float("min_rise_pct", 0.005, 0.02),  # Sell on new high, 0.5%-2%
                'float_position_pct': trial.suggest_float("float_position_pct", 0.1, 0.3),
                # Floating position ratio 10%-30%
            }
        else:
            raise ValueError("Unknown strategy type")

        # Create portfolio and add account with parameters
        portfolio = SynchronizedPortfolio(total_capital=initial_balance)
        portfolio.add_account("TUNE", strategy_type, params, df, allocation_pct=100)
        portfolio.run()
        trade_log_df = portfolio.combined_trade_log()

        if trade_log_df.empty:
            return 1e6, 1e6, 1e6

        roi, max_dd, sharpe = PerformanceMetrics.compute_metrics(trade_log_df, initial_balance=initial_balance)
        print(roi, max_dd, sharpe)
        if roi is None or max_dd is None or sharpe is None:
            return 1e6, 1e6, 1e6

        # Return negative ROI and Sharpe for minimization (multi-objective)
        return -roi, max_dd, -sharpe

    @staticmethod
    def optimize_strategy(df, ticker, strategy_type='trend', n_trials=50, initial_balance=10000, verbose=False):
        """
        Optimize strategy parameters for a single stock.

        Args:
            df (pd.DataFrame): Price data
            ticker (str): Stock ticker symbol
            strategy_type (str): Strategy type ('trend' or 'range')
            n_trials (int): Number of optimization trials
            initial_balance (float): Initial account balance
            verbose (bool): Whether to print trial results

        Returns:
            tuple: (best_params, roi, max_drawdown, sharpe, pareto_trials)
        """
        # Create multi-objective study
        study = optuna.create_study(
            directions=["minimize", "minimize", "minimize"]  # Objectives: -ROI, Max Drawdown, -Sharpe
        )

        # Run optimization
        if verbose:
            study.optimize(lambda trial: StrategyOptimizer.objective(trial, df, strategy_type, initial_balance),
                           n_trials=n_trials)
        else:
            # Suppress trial output
            with HiddenPrints():
                study.optimize(lambda trial: StrategyOptimizer.objective(trial, df, strategy_type, initial_balance),
                               n_trials=n_trials)

        # Get Pareto-optimal solutions
        pareto_trials = study.best_trials
        print(f"\n✅ Found {len(pareto_trials)} Pareto optimal solutions")

        # Extract metrics from Pareto-optimal solutions
        roi_values = np.array([-t.values[0] for t in pareto_trials])  # Convert back to positive
        mdd_values = np.array([t.values[1] for t in pareto_trials])
        sharpe_values = np.array([-t.values[2] for t in pareto_trials])  # Convert back to positive

        # Normalize metrics to prevent any single metric from dominating
        roi_norm = (roi_values - roi_values.min()) / (roi_values.max() - roi_values.min() + 1e-8)
        sharpe_norm = (sharpe_values - sharpe_values.min()) / (sharpe_values.max() - sharpe_values.min() + 1e-8)
        mdd_norm = (mdd_values.max() - mdd_values) / (mdd_values.max() - mdd_values.min() + 1e-8)  # Reverse for MDD

        # Calculate combined score with weights
        weights = [0.55, 0.30, 0.15]  # ROI 55%, Sharpe 30%, MDD 15%
        scores = weights[0] * roi_norm + weights[1] * sharpe_norm + weights[2] * mdd_norm

        # Select best trial based on combined score
        best_index = np.argmax(scores)
        best_trial = pareto_trials[best_index]
        best_params = best_trial.params

        # Run backtest with optimal parameters
        portfolio = SynchronizedPortfolio(total_capital=initial_balance)
        portfolio.add_account("TUNE", strategy_type, best_params, df, allocation_pct=100)
        portfolio.run()
        trade_log_df = portfolio.combined_trade_log()
        roi, max_dd, sharpe = PerformanceMetrics.compute_metrics(trade_log_df, initial_balance=initial_balance)

        # Print results
        print("\n🎯 [Best combined solution]")
        print(f"Parameters: {best_params}")
        print(f"ROI: {roi:.2f}%, Max Drawdown: {max_dd:.2f}%, Sharpe Ratio: {sharpe:.2f}")

        # Save trade log
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        trade_log_df.to_csv(f'{output_dir}/{ticker}-{strategy_type}-tuning.csv', index=False)
        print(f"✅ {ticker} best strategy trade log saved to {output_dir}/{ticker}-{strategy_type}-tuning.csv")

        return best_params, roi, max_dd, sharpe, pareto_trials


class PortfolioBuilder:
    """
    Builds optimized portfolio across multiple stocks.
    """

    @staticmethod
    def batch_backtest(ticker_list, result_file='strategy_results.csv', n_trials=50, verbose=False,
                       roi_threshold=0, max_dd_threshold=-30, sharpe_threshold=1):
        """
        Batch backtest and optimize multiple stocks.

        Args:
            ticker_list (list): List of stock tickers
            result_file (str): Output file for results
            n_trials (int): Number of optimization trials per strategy
            verbose (bool): Whether to print detailed output
            roi_threshold (float): Minimum ROI threshold for eligibility (%)
            max_dd_threshold (float): Maximum drawdown threshold for eligibility (%)
            sharpe_threshold (float): Minimum Sharpe ratio threshold for eligibility

        Returns:
            pd.DataFrame: Results DataFrame
        """
        results = []

        # Optimize each ticker
        for ticker in ticker_list:
            print(f"🚀 Optimizing {ticker}...")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = (datetime.now()).strftime("%Y-%m-%d")

            # Get data
            try:
                df = DataFetcher.fetch_stock_data(ticker, start_date, end_date, "hour")
            except Exception as e:
                print(f"❌ Failed to fetch {ticker} data: {e}")
                continue

            # Optimize trend strategy
            try:
                trend_params, trend_roi, trend_dd, trend_sharpe, trend_pareto = StrategyOptimizer.optimize_strategy(
                    df, ticker, 'trend', n_trials, verbose=verbose
                )
                print(f"✅ {ticker} TREND ROI: {trend_roi:.2f}%, MaxDD: {trend_dd:.2f}%, Sharpe: {trend_sharpe:.2f}")
            except (Exception, TrialPruned) as e:
                print(f"❌ {ticker} TREND strategy failed: {e}")
                trend_params, trend_roi, trend_dd, trend_sharpe, trend_pareto = {}, None, None, None, None

            # Optimize range strategy
            try:
                range_params, range_roi, range_dd, range_sharpe, range_pareto = StrategyOptimizer.optimize_strategy(
                    df, ticker, 'range', n_trials, verbose=verbose
                )
                print(f"✅ {ticker} RANGE ROI: {range_roi:.2f}%, MaxDD: {range_dd:.2f}%, Sharpe: {range_sharpe:.2f}")
            except (Exception, TrialPruned) as e:
                print(f"❌ {ticker} RANGE strategy failed: {e}")
                range_params, range_roi, range_dd, range_sharpe, range_pareto = {}, None, None, None, None

            # Choose the better strategy
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

            # Check eligibility against thresholds
            eligible = (
                    chosen_roi is not None and chosen_roi >= roi_threshold and
                    chosen_dd is not None and chosen_dd >= max_dd_threshold and
                    chosen_sharpe is not None and chosen_sharpe >= sharpe_threshold
            )

            # Store results
            results.append({
                'Stock': ticker,
                'Chosen Strategy': chosen_strategy,
                'ROI (%)': chosen_roi,
                'Max Drawdown (%)': chosen_dd,
                'Sharpe Ratio': chosen_sharpe,
                'Best Params': chosen_params,
                'Eligible': eligible
            })

        # Create DataFrame and apply eligibility filter
        df_results = pd.DataFrame(results)
        filtered_results = df_results[df_results["Eligible"] == True]

        # Calculate allocation percentages for eligible strategies
        if not filtered_results.empty:
            # Normalize metrics for allocation calculation
            roi_values = filtered_results["ROI (%)"].astype(float)
            sharpe_values = filtered_results["Sharpe Ratio"].astype(float)
            mdd_values = filtered_results["Max Drawdown (%)"].astype(float)

            # Handle edge cases to prevent division by zero
            roi_range = roi_values.max() - roi_values.min()
            sharpe_range = sharpe_values.max() - sharpe_values.min()
            mdd_range = mdd_values.max() - mdd_values.min()

            # Normalize with epsilon to prevent division by zero
            roi_norm = (roi_values - roi_values.min()) / (roi_range + 1e-8) if roi_range > 0 else 1.0
            sharpe_norm = (sharpe_values - sharpe_values.min()) / (sharpe_range + 1e-8) if sharpe_range > 0 else 1.0
            # Reverse normalization for drawdown (smaller is better)
            mdd_norm = (mdd_values.max() - mdd_values) / (mdd_range + 1e-8) if mdd_range > 0 else 1.0

            # Calculate allocation score using the same weights as strategy selection
            weights = [0.55, 0.30, 0.15]  # ROI 55%, Sharpe 30%, MDD 15%
            filtered_results["Allocation Score"] = (
                    weights[0] * roi_norm +
                    weights[1] * sharpe_norm +
                    weights[2] * mdd_norm
            )

            # Calculate allocation percentages (total = 200%)
            total_score = filtered_results["Allocation Score"].sum()
            if total_score > 0:
                filtered_results["Allocation (%)"] = (filtered_results["Allocation Score"] / total_score) * 200
            else:
                filtered_results["Allocation (%)"] = 0

            # Update the original results with allocation information
            df_results.update(filtered_results)
        else:
            # Add allocation columns even if no strategies are eligible
            df_results["Allocation Score"] = 0
            df_results["Allocation (%)"] = 0

        # Save results to CSV
        df_results.to_csv(result_file, index=False)
        print(f"✅ Portfolio with allocations saved to {result_file}")

        return df_results


# Main execution
if __name__ == "__main__":
    # List of stocks to test
    ticker_list = [
        "AMZN", "META", "NVDA", "TSLA", "TQQQ", "PLTR", "SNOW", "CRM",
        "GOOGL", "AAPL", "AVGO", "KO", "SPY", "VRT", "AMD", "COST",
        "JNJ", "JPM", 'MSFT', "ASML", "BRK.B", "CRWD", "PYPL",
        "DIS", "AXP", "ROKU", "COIN", "SHOP", "INTC"
    ]

    # Run batch backtest
    PortfolioBuilder.batch_backtest(
        ticker_list,
        result_file='strategy_results.csv',
        n_trials=30,
        verbose=False
    )