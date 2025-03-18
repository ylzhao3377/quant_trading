import pandas as pd
import numpy as np
import ast
import os
import sys
from polygon import RESTClient
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage

# Configuration
os.environ["POLYGON_API_KEY"] = "0Fp6qkxgz6QugnvLPiR6d9cEMpK3hxFF"


class EmailNotifier:
    """
    Handles email notifications with optional attachments.
    """

    @staticmethod
    def send_email(subject, message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password,
                   attachments=None):
        """
        Send an email with optional attachments.

        Args:
            subject (str): Email subject
            message (str): Email body content
            from_addr (str): Sender email address
            to_addrs (list): List of recipient email addresses
            smtp_server (str): SMTP server address
            smtp_port (int): SMTP server port
            smtp_user (str): SMTP username
            smtp_password (str): SMTP password
            attachments (list, optional): List of file paths to attach
        """
        # Create email object
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.set_content(message)

        # Add attachments
        if attachments:
            for file_path in attachments:
                try:
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                        file_name = os.path.basename(file_path)
                    # Add attachment
                    msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)
                except Exception as e:
                    print(f"Unable to attach file {file_path}: {e}")

        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Enable TLS encryption
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")


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


class DataFetcher:
    """
    Handles fetching stock data from Polygon.io API.
    """

    @staticmethod
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


class TechnicalAnalysis:
    """
    Computes technical indicators for trading strategies.
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


class CapitalPool:
    """
    Manages the allocation and tracking of capital across different strategies.
    """

    def __init__(self, total_capital):
        """
        Initialize the capital pool.

        Args:
            total_capital (float): Total available capital
        """
        self.total_capital = total_capital
        self.available_capital = total_capital

    def allocate(self, amount):
        """
        Allocate capital from the pool.

        Args:
            amount (float): Amount to allocate

        Returns:
            float: Actually allocated amount (may be less if insufficient funds)
        """
        if self.available_capital >= amount:
            self.available_capital -= amount
            return amount
        return 0

    def release(self, amount):
        """
        Release capital back to the pool.

        Args:
            amount (float): Amount to release
        """
        self.available_capital += amount

    def status(self):
        """
        Get current available capital.

        Returns:
            float: Available capital
        """
        return self.available_capital


class SubAccount:
    """
    Represents a trading account for a specific asset and strategy.
    """

    def __init__(self, name, strategy_type, allocation_pct, capital_pool, params):
        """
        Initialize a sub-account.

        Args:
            name (str): Account name (typically ticker symbol)
            strategy_type (str): Strategy type ('trend' or 'range')
            allocation_pct (float): Capital allocation percentage
            capital_pool (CapitalPool): Reference to the capital pool
            params (dict): Strategy parameters
        """
        self.name = name
        self.strategy_type = strategy_type.lower()
        self.allocation_pct = allocation_pct
        self.capital_pool = capital_pool
        self.params = params
        self.balance = capital_pool.total_capital * allocation_pct / 100
        self.trade_log = []
        self.last_price = None  # For handling missing data

        # Initialize strategy-specific attributes
        if self.strategy_type == 'trend':
            self.in_position = False
            self.position = 0
            self.entry_price = 0
            self.max_profit_price = 0
            self.realized_profit = 0
        elif self.strategy_type == 'range':
            self.base_position = 0
            self.float_position = 0
            self.base_entry_prices = []
            self.float_entry_prices = []
            self.last_buy_price = None
            self.last_sell_price = None
            self.realized_profit = 0


class StrategyExecutor:
    """
    Executes trading strategies with specific buy/sell logic.
    """

    @staticmethod
    def get_available_funds(account, idle_multiplier=4):
        """
        Calculate available funds for trading based on account type.

        Args:
            account (SubAccount): Trading account
            idle_multiplier (int): Multiplier for idle cash

        Returns:
            float: Available funds
        """
        if account.strategy_type == "trend":
            return account.balance * idle_multiplier if account.position == 0 else account.balance
        elif account.strategy_type == "range":
            total_pos = account.base_position + account.float_position
            return account.balance * idle_multiplier if total_pos == 0 else account.balance

    @staticmethod
    def run_trend_sell_logic(account, row, params):
        """
        Execute sell logic for trend following strategy.

        Args:
            account (SubAccount): Trading account
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

    @staticmethod
    def run_range_sell_logic(account, row, params):
        """
        Execute sell logic for range trading strategy.

        Args:
            account (SubAccount): Trading account
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

        # Base position sell logic
        if price > upper_band and breakout_strength >= min_strength and account.base_position > 0:
            if account.last_sell_price is None or price > account.last_sell_price * (1 + min_rise_pct):
                sell_value = min(max_lot_value * breakout_strength, account.base_position * price)
                shares = int(sell_value / price)
                if shares > 0:
                    total_cost = sum(account.base_entry_prices[:shares])
                    profit = shares * price - total_cost
                    account.realized_profit += profit
                    account.balance += shares * price
                    account.base_position -= shares
                    account.base_entry_prices = account.base_entry_prices[shares:]
                    account.last_sell_price = price
                    action = "SELL"
                    reason = f"Base sell on break upper (Z={z_score:.2f}, Strength={breakout_strength:.2f})"
                    if account.base_position == 0:
                        account.last_buy_price = None
                        account.last_sell_price = None
                    return {"Date": date, "Action": action, "Reason": reason, "Price": price, "Shares": shares,
                            "Balance": account.balance, "Position": account.base_position,
                            "Realized_Profit": account.realized_profit}

        # Float position sell logic
        if price > upper_band and breakout_strength >= min_strength and account.float_position > 0:
            float_sell_value = min(init_bal * params.get("float_position_pct", 0.1) * breakout_strength,
                                   account.float_position * price)
            shares = int(float_sell_value / price)
            if shares > 0:
                float_total_cost = sum(account.float_entry_prices[:shares])
                profit = shares * price - float_total_cost
                account.realized_profit += profit
                account.balance += shares * price
                account.float_position -= shares
                account.float_entry_prices = account.float_entry_prices[shares:]
                action = "SELL"
                reason = f"Float sell on break upper (Z={z_score:.2f}, Strength={breakout_strength:.2f})"
                return {"Date": date, "Action": action, "Reason": reason, "Price": price, "Shares": shares,
                        "Balance": account.balance, "Position": account.float_position,
                        "Realized_Profit": account.realized_profit}
        return None


class SynchronizedPortfolio:
    """
    Manages a portfolio of multiple trading accounts with synchronized execution.
    """

    def __init__(self, total_capital):
        """
        Initialize a synchronized portfolio.

        Args:
            total_capital (float): Total capital for the portfolio
        """
        self.capital_pool = CapitalPool(total_capital)
        self.accounts = {}  # {ticker: {'account': SubAccount, 'data': DataFrame}}
        self.trade_log = []
        self.snapshot_log = []

    def add_account(self, name, strategy_type, params, df, allocation_pct):
        """
        Add a trading account to the portfolio.

        Args:
            name (str): Account name (typically ticker symbol)
            strategy_type (str): Strategy type ('trend' or 'range')
            params (dict): Strategy parameters
            df (pd.DataFrame): Price data
            allocation_pct (float): Capital allocation percentage
        """
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Compute technical indicators
        window = params.get("bollinger_window", 50)
        df = TechnicalAnalysis.compute_indicators(df, bollinger_window=window)
        df.sort_values('datetime', inplace=True)

        # Allocate capital
        allocated = self.capital_pool.allocate(self.capital_pool.total_capital * allocation_pct / 100)
        account = SubAccount(name, strategy_type, allocation_pct, self.capital_pool, params)
        account.balance = allocated

        # Initialize strategy-specific attributes
        if strategy_type.lower() == 'range':
            account.base_position = 0
            account.float_position = 0
            account.base_entry_prices = []
            account.float_entry_prices = []
            account.last_buy_price = None
            account.last_sell_price = None
            account.realized_profit = 0
        elif strategy_type.lower() == 'trend':
            account.in_position = False
            account.position = 0
            account.entry_price = 0
            account.max_profit_price = 0
            account.realized_profit = 0

        self.accounts[name] = {'account': account, 'data': df}

    def run(self):
        """
        Run the portfolio simulation across all accounts.
        """
        # Build unified timeline by merging all datetime columns
        all_times = sorted(set().union(*[set(data['data']['datetime']) for data in self.accounts.values()]))
        n_times = len(all_times)

        for i, current_time in enumerate(all_times):
            # Calculate pre-operation total balance
            pre_total_balance = self.capital_pool.status() + sum(
                [data['account'].balance for data in self.accounts.values()])

            # === SELL PHASE ===
            for name, data in self.accounts.items():
                df = data['data']
                account = data['account']
                current_rows = df[df['datetime'] == current_time]

                if not current_rows.empty:
                    row = current_rows.iloc[0]

                    # Record pre-operation position
                    if account.strategy_type == 'trend':
                        pre_position = account.position
                        log_entry = StrategyExecutor.run_trend_sell_logic(account, row, account.params)
                    elif account.strategy_type == 'range':
                        pre_position = account.base_position + account.float_position
                        log_entry = StrategyExecutor.run_range_sell_logic(account, row, account.params)
                    else:
                        log_entry = None

                    if log_entry and log_entry["Action"] != "HOLD":
                        # Add pre-operation total balance
                        log_entry["Pre_Total_Balance"] = pre_total_balance

                        # Record sell ratio
                        if pre_position > 0:
                            log_entry["Operation_Ratio"] = log_entry["Shares"] / pre_position
                        else:
                            log_entry["Operation_Ratio"] = 0

                        log_entry["Stock"] = name
                        self.trade_log.append(log_entry)
                        account.trade_log.append(log_entry)

            # === BUY PHASE: First collect all valid buy orders ===
            valid_orders = []
            total_strength = 0.0

            for name, data in self.accounts.items():
                df = data['data']
                account = data['account']
                current_rows = df[df['datetime'] == current_time]

                if not current_rows.empty:
                    row = current_rows.iloc[0]

                    if account.strategy_type == 'trend' and not account.in_position:
                        strength = TechnicalAnalysis.compute_trend_strength(row, account.params)
                        if strength > 0:
                            valid_orders.append((account, row, strength))
                            total_strength += strength
                    elif account.strategy_type == 'range':
                        if row['close'] < row['Lower']:
                            strength = TechnicalAnalysis.compute_range_strength(row, account.params)
                            if strength >= account.params.get("min_strength", 0.3):
                                valid_orders.append((account, row, strength))
                                total_strength += strength

            # Execute buy orders with weight-based allocation
            for (account, row, strength) in valid_orders:
                weight = strength / total_strength if total_strength > 0 else 0
                allocated_funds = pre_total_balance * weight

                # Limit to current account balance
                funds_to_use = min(allocated_funds, account.balance)
                price = row['close']
                balance_before = account.balance
                shares_to_buy = int(funds_to_use / price)

                if shares_to_buy > 0:
                    money_spent = shares_to_buy * price
                    # Calculate buy ratio: amount spent / pre-operation total balance
                    buy_ratio = money_spent / pre_total_balance

                    if account.strategy_type == 'trend':
                        account.entry_price = price
                        account.max_profit_price = price
                        account.position = shares_to_buy
                        account.balance -= money_spent
                        account.in_position = True
                        log_entry = {
                            "Date": row['datetime'],
                            "Action": "BUY",
                            "Reason": "Dynamic Buy Allocation (Trend)",
                            "Price": price,
                            "Shares": shares_to_buy,
                            "Balance": account.balance,
                            "Position": account.position,
                            "Realized_Profit": account.realized_profit,
                            "Pre_Total_Balance": pre_total_balance,
                            "Operation_Ratio": buy_ratio
                        }
                    elif account.strategy_type == 'range':
                        account.base_position += shares_to_buy
                        account.base_entry_prices.extend([price] * shares_to_buy)
                        account.balance -= money_spent
                        log_entry = {
                            "Date": row['datetime'],
                            "Action": "BUY",
                            "Reason": "Dynamic Buy Allocation (Range)",
                            "Price": price,
                            "Shares": shares_to_buy,
                            "Balance": account.balance,
                            "Position": account.base_position + account.float_position,
                            "Realized_Profit": account.realized_profit,
                            "Pre_Total_Balance": pre_total_balance,
                            "Operation_Ratio": buy_ratio
                        }
                    log_entry["Stock"] = account.name
                    self.trade_log.append(log_entry)
                    account.trade_log.append(log_entry)

            # === SNAPSHOT RECORDING ===
            current_day = pd.to_datetime(current_time).date()
            is_last_of_day = (i == n_times - 1) or (pd.to_datetime(all_times[i + 1]).date() != current_day)

            if valid_orders or is_last_of_day:
                snapshot = self.take_snapshot(current_time)
                self.snapshot_log.append(snapshot)

        # --- Add final position state (Final HOLD) ---
        final_time = all_times[-1]

        # Calculate final pre-operation total balance
        final_pre_total_balance = self.capital_pool.status() + sum(
            data['account'].balance for data in self.accounts.values()
        )

        for name, data in self.accounts.items():
            account = data['account']
            df = data['data']

            # Get latest data row
            latest_row = df.iloc[-1]
            price = latest_row['close']

            if account.strategy_type == 'trend':
                final_position = account.position
                final_balance = account.balance
            elif account.strategy_type == 'range':
                final_position = account.base_position + account.float_position
                final_balance = account.balance

            final_log = {
                "Date": latest_row['datetime'],
                "Action": "HOLD_FINAL",
                "Reason": "Final holding state",
                "Price": price,
                "Shares": final_position,
                "Balance": final_balance,
                "Position": final_position,
                "Realized_Profit": account.realized_profit,
                "Pre_Total_Balance": final_pre_total_balance,
                "Operation_Ratio": 0,  # No trade ratio for final state
                "Stock": name
            }
            self.trade_log.append(final_log)

    def take_snapshot(self, current_time):
        """
        Take a snapshot of the portfolio state at the current time.

        Args:
            current_time (datetime): Current time for the snapshot

        Returns:
            dict: Portfolio snapshot
        """
        snapshot = {"Date": current_time}
        total_position_value = 0
        total_realized_profit = 0

        for name, data in self.accounts.items():
            account = data['account']
            df = data['data']
            current_rows = df[df['datetime'] == current_time]

            if not current_rows.empty:
                current_price = current_rows.iloc[0]['close']
                # Update last price
                account.last_price = current_price
            else:
                # Use last recorded price if data is missing
                current_price = account.last_price if account.last_price is not None else np.nan

            pos = account.position if account.strategy_type == "trend" else account.base_position
            snapshot[f"Position_{name}"] = pos
            snapshot[f"Price_{name}"] = current_price
            total_position_value += pos * (current_price if pd.notna(current_price) else 0)
            total_realized_profit += account.realized_profit

        total_account_balance = sum(data['account'].balance for data in self.accounts.values())
        snapshot["Total_Balance"] = self.capital_pool.status() + total_account_balance
        snapshot["Total_Equity"] = snapshot["Total_Balance"] + total_position_value
        snapshot["Total_Realized_Profit"] = total_realized_profit

        return snapshot

    def combined_trade_log(self):
        """
        Get combined trade log for all accounts.

        Returns:
            pd.DataFrame: Combined trade log
        """
        if self.trade_log:
            combined_df = pd.DataFrame(self.trade_log)
            combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
            combined_df = combined_df.sort_values("Date").reset_index(drop=True)
            return combined_df
        return pd.DataFrame()

    def snapshot_log_df(self):
        """
        Get snapshot log as DataFrame.

        Returns:
            pd.DataFrame: Snapshot log
        """
        df = pd.DataFrame(self.snapshot_log)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        return df


# Main execution block
if __name__ == "__main__":
    # Main process: Read strategy results and build portfolio
    results_df = pd.read_csv("strategy_results.csv")
    ticker_list = results_df["Stock"]
    filtered_results = results_df[
        (results_df['Stock'].isin(ticker_list)) &
        (results_df['Eligible'] == True) &
        (results_df['Allocation (%)'] > 0)
    ]

    print("Filtered results:")
    print(filtered_results[['Stock', 'Chosen Strategy', 'Best Params', 'Allocation (%)']])

    total_capital = 100000  # Set according to actual capital
    portfolio = SynchronizedPortfolio(total_capital=total_capital)

    for idx, row in filtered_results.iterrows():
        ticker = row['Stock']
        strategy_type = row['Chosen Strategy'].lower().strip()  # 'trend' or 'range'
        best_params = ast.literal_eval(row['Best Params'])
        allocation_pct = float(row['Allocation (%)'])
        now = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        df = DataFetcher.fetch_stock_data(ticker, "2024-03-12", now, "hour")
        if df.empty:
            print(f"{ticker} has no data, skipping.")
            continue
        portfolio.add_account(ticker, strategy_type, best_params, df, allocation_pct)
        print(f"Added {ticker}: strategy={strategy_type}, Allocation={allocation_pct}%")

    portfolio.run()

    combined_log = portfolio.combined_trade_log()
    combined_log.to_csv("combined_trade_log.csv", index=False)
    print("✅ Combined trade log saved to combined_trade_log.csv")

    snapshot_df = portfolio.snapshot_log_df()
    snapshot_df.to_csv("snapshot_trade_log.csv", index=False)
    print("✅ Snapshot log saved to snapshot_trade_log.csv")

    # Email configuration (replace smtp_password with your app-specific password)
    from_addr = "ylzhao3377@gmail.com"
    to_addrs = ["ylzhao3377@gmail.com"]
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "ylzhao3377@gmail.com"
    smtp_password = "pntr minq hlcb uikz"  # Recommend using app-specific password

    # Email subject and content
    subject = "Strategy Backtest Results"
    message = "Please find attached the combined_trade_log.csv and snapshot_trade_log.csv files."

    # Attachment paths
    attachments = ["combined_trade_log.csv", "snapshot_trade_log.csv"]

    EmailNotifier.send_email(subject, message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password, attachments)