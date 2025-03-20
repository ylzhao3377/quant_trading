import pandas as pd
import numpy as np
import os
import sys
from polygon import RESTClient
from datetime import datetime, timedelta


# ----------------------------------------
# Utility Functions
# ----------------------------------------
class HiddenPrints:
    """
    Context manager to suppress print output.
    Useful for hiding Optuna internal trial outputs.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ----------------------------------------
# Data Fetching Functions
# ----------------------------------------
def fetch_stock_data(ticker: str, start_date: str, end_date: str, timespan: str):
    """
    Fetch historical stock data from Polygon.io.

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        timespan (str): Time interval (e.g., 'hour', 'day')

    Returns:
        pd.DataFrame: DataFrame containing the historical data
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Please set the POLYGON_API_KEY environment variable.")

    client = RESTClient(api_key)
    aggs = []

    for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
    ):
        aggs.append(agg)

    df = pd.DataFrame(aggs)
    if df.empty:
        print(f"{ticker} has no data in the specified date range.")
        return df

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# ----------------------------------------
# Technical Indicators
# ----------------------------------------
def compute_indicators(df, bollinger_window=50):
    """
    Compute various technical indicators for the given price data.

    Args:
        df (pd.DataFrame): Price data containing OHLCV
        bollinger_window (int): Window size for Bollinger Bands calculation

    Returns:
        pd.DataFrame: DataFrame with additional technical indicators
    """
    # EMA calculations
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()

    # ATR calculation
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # Bollinger Bands
    df['Middle'] = df['close'].rolling(window=bollinger_window).mean()
    df['STD'] = df['close'].rolling(window=bollinger_window).std()
    df['Upper'] = df['Middle'] + 2 * df['STD']
    df['Lower'] = df['Middle'] - 2 * df['STD']

    # ADX calculation
    df['+DM'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    df['-DM'] = np.where(df['low'] < df['low'].shift(1), df['low'].shift(1) - df['low'], 0)
    df['+DI'] = 100 * (df['+DM'].rolling(14).sum() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(14).sum() / df['ATR'])
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, np.nan)) * 100
    df['ADX'] = df['DX'].rolling(14).mean()

    # Volume analysis
    df['Volume_MA'] = df['volume'].rolling(20).mean()

    return df


# ----------------------------------------
# Strategy Logic
# ----------------------------------------
def compute_trend_strength(row, params):
    """
    Calculate trend strength based on technical indicators.

    Args:
        row (pd.Series): Row of data with technical indicators
        params (dict): Strategy parameters

    Returns:
        float: Trend strength (0.0 to 1.0)
    """
    price = row['close']
    adx = row['ADX']
    atr = row['ATR']
    atr_pct = atr / price if price > 0 else 0

    volume_spike = row['volume'] > row['Volume_MA'] * params.get("vol_threshold", 1.5)
    breakout_confirmed = (adx > params.get("adx_threshold", 30)) or volume_spike
    trend_up = (row['EMA50'] > row['EMA200']) and (adx > 20)

    if trend_up and (price > row['Upper']) and breakout_confirmed and (
            atr_pct > params.get("atr_pct_threshold", 0.005)):
        return 1.0

    return 0.0


def compute_range_strength(row, params):
    """
    Calculate range strength based on price deviation from mean.

    Args:
        row (pd.Series): Row of data with technical indicators
        params (dict): Strategy parameters

    Returns:
        float: Range strength (0.0 to 1.0)
    """
    price = row['close']
    middle = row['Middle']
    std = row['STD']

    if std <= 0:
        return 0.0

    z = (price - middle) / std
    k, c = 3, 2
    strength = 1 / (1 + np.exp(-k * (abs(z) - c)))

    return strength


def run_trend_sell_logic(account, row, params):
    """
    Execute sell logic for trend following strategy.

    Args:
        account (object): Trading account with position and balance information
        row (pd.Series): Current price data with indicators
        params (dict): Strategy parameters

    Returns:
        dict or None: Trade log entry if a sell occurred, None otherwise
    """
    if account.in_position and account.position > 0:
        price = row['close']
        trailing_stop_pct = params.get("trailing_stop_pct", 0.95)
        account.max_profit_price = max(account.max_profit_price, price)
        trailing_stop_price = account.max_profit_price * trailing_stop_pct

        sell_shares = 0
        reason = ""

        # Take profit: Below Bollinger middle band
        if price < row['Middle']:
            sell_shares = account.position
            reason = "Price below Bollinger Middle (Take Profit)"
        # Stop loss: Below EMA200 or trailing stop triggered
        elif price < row['EMA200'] or price <= trailing_stop_price:
            sell_shares = account.position
            reason = "Below EMA200 or Trailing Stop triggered"

        if sell_shares > 0:
            profit = (price - account.entry_price) * account.position
            account.realized_profit += profit
            account.balance += account.position * price
            log_entry = {
                "Date": row["datetime"],
                "Action": "SELL",
                "Reason": reason,
                "Price": price,
                "Shares": sell_shares,
                "Balance": account.balance,
                "Position": 0,
                "Realized_Profit": account.realized_profit
            }
            account.position = 0
            account.in_position = False
            account.entry_price = 0
            account.max_profit_price = 0
            return log_entry
    return None


def run_range_sell_logic(account, row, params):
    """
    Execute sell logic for range trading strategy.

    Args:
        account (object): Trading account with position and balance information
        row (pd.Series): Current price data with indicators
        params (dict): Strategy parameters

    Returns:
        dict or None: Trade log entry if a sell occurred, None otherwise
    """
    price = row['close']
    date = row['datetime']
    lower_band = row['Lower']
    upper_band = row['Upper']
    middle_band = row['Middle']
    std = row['STD']
    z_score = (price - middle_band) / std if std > 0 else 0
    k, c = 3, 2
    breakout_strength = 1 / (1 + np.exp(-k * (abs(z_score) - c)))
    min_rise_pct = params.get("min_rise_pct", 0.02)
    min_strength = params.get("min_strength", 0.3)
    init_bal = params.get("initial_balance", 10000)
    max_lot_value = init_bal / 5

    # Position sell logic
    if price > upper_band and breakout_strength >= min_strength and account.position > 0:
        if account.last_sell_price is None or price > account.last_sell_price * (1 + min_rise_pct):
            sell_value = min(max_lot_value * breakout_strength, account.position * price)
            shares = int(sell_value / price)
            if shares > 0:
                total_cost = sum(account.entry_prices[:shares])
                profit = shares * price - total_cost
                account.realized_profit += profit
                account.balance += shares * price
                account.position -= shares
                account.entry_prices = account.entry_prices[shares:]
                account.last_sell_price = price
                action = "SELL"
                reason = f"Sell on break upper (Z={z_score:.2f}, Strength={breakout_strength:.2f})"
                if account.position == 0:
                    account.last_buy_price = None
                    account.last_sell_price = None
                return {"Date": date, "Action": action, "Reason": reason, "Price": price, "Shares": shares,
                        "Balance": account.balance, "Position": account.position,
                        "Realized_Profit": account.realized_profit}
    return None


# ----------------------------------------
# Performance Metrics
# ----------------------------------------
def compute_metrics(trade_df, initial_balance=10000):
    """
    Calculate backtest metrics:
      - ROI: Return on investment (percentage)
      - Maximum drawdown: Maximum equity curve decline (percentage)
      - Sharpe ratio: Risk-adjusted return measure

    Args:
        trade_df (pd.DataFrame): Trade log DataFrame
        initial_balance (float): Initial account balance

    Returns:
        tuple: (ROI, max_drawdown, sharpe_ratio)
    """
    if trade_df.empty:
        return None, None, None

    # Calculate Total_Equity if not present
    trade_df = trade_df.copy()  # Avoid modifying original data
    if 'Total_Equity' not in trade_df.columns:
        trade_df['Total_Equity'] = trade_df['Balance'] + trade_df['Position'] * trade_df['Price']

    final_equity = trade_df['Total_Equity'].iloc[-1]
    total_profit = final_equity - initial_balance
    roi = (total_profit / initial_balance) * 100

    # Calculate equity returns for Sharpe ratio
    trade_df['Equity_Return'] = trade_df['Total_Equity'].pct_change().fillna(0)
    std_ret = trade_df['Equity_Return'].std()
    sharpe = (trade_df['Equity_Return'].mean() / std_ret) * (252 ** 0.5) if std_ret != 0 else 0

    # Calculate maximum drawdown
    cumulative = trade_df['Total_Equity'].cummax()
    drawdown = (trade_df['Total_Equity'] - cumulative) / cumulative
    max_drawdown = drawdown.min() * 100  # as percentage

    return roi, max_drawdown, sharpe


# ----------------------------------------
# Account Management
# ----------------------------------------
class CapitalPool:
    """
    Manages the allocation and tracking of capital across different strategies.
    """

    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.available_capital = total_capital

    def allocate(self, amount):
        if self.available_capital >= amount:
            self.available_capital -= amount
            return amount
        else:
            amount -= 1
            self.available_capital -= amount
            return amount

    def release(self, amount):
        self.available_capital += amount

    def status(self):
        return self.available_capital


def get_available_funds(account, idle_multiplier=4):
    """
    Calculate available funds for trading based on account type.

    Args:
        account: Trading account object
        idle_multiplier: Multiplier for idle cash

    Returns:
        float: Available funds
    """
    if account.strategy_type == "trend":
        return account.balance * idle_multiplier if account.position == 0 else account.balance
    elif account.strategy_type == "range":
        return account.balance * idle_multiplier if account.position == 0 else account.balance