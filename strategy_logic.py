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
            limit=50000,
            adjusted="true"
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
            # Always sell all shares when selling (round up)
            sell_shares = account.position

            # Debug log to verify entry price and calculate profit
            entry_price = account.entry_price if hasattr(account, 'entry_price') else 0
            if entry_price <= 0:
                print(f"WARNING: Missing entry price for {account.name}, using fallback calculation")
                # Fallback calculation if entry price isn't available
                entry_price = price * 0.95  # Assume 5% profit as fallback

            profit = (price - entry_price) * sell_shares

            print(f"SELL: {account.position} shares @ ${price:.2f}, entry was ${entry_price:.2f}")
            print(f"Profit: ${profit:.2f} ({((price / entry_price) - 1) * 100:.2f}%)")

            account.realized_profit += profit
            account.balance += account.position * price
            log_entry = {
                "Date": row["datetime"],
                "Action": "SELL",
                "Reason": reason,
                "Price": price,
                "Shares": sell_shares,
                "Entry_Price": entry_price,
                "Profit": profit,
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
    Execute sell logic for range trading strategy with dynamic trailing stop loss.

    Phased selling approach:
    1. Initial Breakout Sell: Sell 1/3 of position when price first breaks upper Bollinger Band
    2. Continued Uptrend Sell: Sell 1/2 of remaining position as price continues higher
    3. Trailing Stop: Only activate trailing stop when price has peaked and starts dropping
       - Trailing stop level is max(middle_band, peak_price * trailing_stop_pct)

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
    trailing_stop_pct = params.get("trailing_stop_pct", 0.95)

    # Initialize tracking attributes if they don't exist
    if not hasattr(account, 'peak_price'):
        account.peak_price = None

    if not hasattr(account, 'price_peaked'):
        account.price_peaked = False

    if not hasattr(account, 'breakout_sell_done'):
        account.breakout_sell_done = False

    if not hasattr(account, 'initial_position_size'):
        account.initial_position_size = account.position

    # Update peak price tracking logic
    if account.last_sell_price is not None:
        # If we have a new high after selling has started
        if account.peak_price is None or price > account.peak_price:
            account.peak_price = price
            account.price_peaked = False  # Reset peaked flag when setting a new peak
        # If price has dropped from peak by a small margin, mark as peaked
        elif price < account.peak_price * 0.99 and not account.price_peaked:
            account.price_peaked = True

    # Sell logic
    if account.position > 0:
        sell_shares = 0
        reason = ""

        # Case 1: Initial Breakout Sell - round up to 1/3 of position
        if price > upper_band and breakout_strength >= min_strength and not account.breakout_sell_done:
            # Calculate 1/3 of initial position
            if account.initial_position_size == 0:  # Safeguard
                account.initial_position_size = account.position

            one_third_position = int(account.initial_position_size / 3)
            # Round up if position is small
            if account.position <= 3:
                one_third_position = account.position  # Sell all if 3 or fewer shares
            elif one_third_position == 0 and account.position > 0:
                one_third_position = 1  # Sell at least 1 share

            if one_third_position > 0:
                sell_shares = min(one_third_position, account.position)
                reason = f"Initial breakout sell (1/3 of position) - (Z={z_score:.2f}, Strength={breakout_strength:.2f})"
                account.breakout_sell_done = True

        # Case 2: Continued Uptrend Sell - round up to 1/2 of remaining position
        elif price > upper_band and breakout_strength >= min_strength and account.breakout_sell_done:
            # Only trigger if price has risen significantly from last sell
            if account.last_sell_price is not None and price > account.last_sell_price * (1 + min_rise_pct):
                # Calculate 1/2 of remaining position
                half_remaining = int(account.position / 2)
                # Round up if position is small
                if account.position <= 2:
                    half_remaining = account.position  # Sell all if 2 or fewer shares
                elif half_remaining == 0 and account.position > 0:
                    half_remaining = 1  # Sell at least 1 share

                if half_remaining > 0:
                    sell_shares = half_remaining
                    reason = f"Continued uptrend sell (1/2 of remaining) - New high ${price:.2f}"

        # Case 3: Trailing Stop Logic - ONLY activated after price has peaked and started to drop
        elif account.peak_price is not None and account.price_peaked:
            # Calculate trailing stop price based on peak price, with middle band as floor
            trailing_stop_price = max(account.peak_price * trailing_stop_pct, middle_band)

            # Check if trailing stop is triggered
            if price <= trailing_stop_price:
                sell_shares = account.position  # Sell entire remaining position
                if trailing_stop_price == middle_band:
                    reason = f"Trailing stop at middle band (${middle_band:.2f})"
                else:
                    reason = f"Trailing stop triggered ({trailing_stop_pct * 100:.0f}% of peak ${account.peak_price:.2f})"

        # Execute the sell if shares to sell > 0
        if sell_shares > 0:
            # Make sure we don't try to sell more than we have
            sell_shares = min(sell_shares, account.position)

            # Calculate profit
            total_profit = 0
            avg_entry_price = 0

            if account.entry_prices and len(account.entry_prices) >= sell_shares:
                # Get the entry prices for the shares we're selling
                sold_entry_prices = account.entry_prices[:sell_shares]
                total_cost = sum(sold_entry_prices)
                avg_entry_price = total_cost / len(sold_entry_prices)
                total_profit = (sell_shares * price) - total_cost
            else:
                # Fallback if entry_prices doesn't contain enough data
                print(f"WARNING: Missing complete entry prices for {account.name}, using average-based calculation")
                avg_entry_price = sum(account.entry_prices) / len(
                    account.entry_prices) if account.entry_prices else price * 0.95
                total_cost = avg_entry_price * sell_shares
                total_profit = (price - avg_entry_price) * sell_shares

            # Log the sell operation details
            print(
                f"SELL (Range): {sell_shares}/{account.position} shares @ ${price:.2f}, avg entry: ${avg_entry_price:.2f}")
            print(f"Profit: ${total_profit:.2f} ({((price / avg_entry_price) - 1) * 100:.2f}%)")
            print(f"Reason: {reason}")

            # Update account state
            account.realized_profit += total_profit
            account.balance += sell_shares * price
            account.position -= sell_shares

            # Update entry prices array
            if sell_shares < len(account.entry_prices):
                account.entry_prices = account.entry_prices[sell_shares:]
            else:
                account.entry_prices = []

            account.last_sell_price = price

            # Reset tracking variables if position is closed
            if account.position == 0:
                account.last_buy_price = None
                account.last_sell_price = None
                account.peak_price = None
                account.price_peaked = False
                account.breakout_sell_done = False
                account.initial_position_size = 0

            return {
                "Date": date,
                "Action": "SELL",
                "Price": price,
                "Shares": sell_shares,
                "Avg_Entry_Price": avg_entry_price,
                "Profit": total_profit,
                "Balance": account.balance,
                "Position": account.position,
                "Realized_Profit": account.realized_profit,
                "Reason": reason
            }

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