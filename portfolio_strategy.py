import pandas as pd
import numpy as np
import ast
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
import pytz

# Import shared components from strategy_logic.py (renamed from simple_strategy.py)
from strategy_logic import (
    fetch_stock_data,
    compute_indicators,
    compute_trend_strength,
    compute_range_strength,
    run_trend_sell_logic,
    run_range_sell_logic,
    CapitalPool,
    get_available_funds,
    HiddenPrints
)

# Environment configuration
os.environ["POLYGON_API_KEY"] = "0Fp6qkxgz6QugnvLPiR6d9cEMpK3hxFF"


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

        # Compute technical indicators using the function from simple_strategy.py
        window = params.get("bollinger_window", 50)
        df = compute_indicators(df, bollinger_window=window)
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
                        # Use the shared sell logic function from simple_strategy.py
                        log_entry = run_trend_sell_logic(account, row, account.params)
                    elif account.strategy_type == 'range':
                        pre_position = account.base_position + account.float_position
                        # Use the shared sell logic function from simple_strategy.py
                        log_entry = run_range_sell_logic(account, row, account.params)
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
                        # Use the shared trend strength function from simple_strategy.py
                        strength = compute_trend_strength(row, account.params)
                        if strength > 0:
                            valid_orders.append((account, row, strength))
                            total_strength += strength
                    elif account.strategy_type == 'range':
                        if row['close'] < row['Lower']:
                            # Use the shared range strength function from simple_strategy.py
                            strength = compute_range_strength(row, account.params)
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
        # Add the PST version of time
        current_time_pst = convert_to_pst(current_time)
        snapshot["Date_PST"] = current_time_pst

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

            # Add PST date column
            combined_df["Date_PST"] = combined_df["Date"].apply(convert_to_pst)
            combined_df["Date_PST_Str"] = combined_df["Date_PST"].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

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

        # Add PST date column
        df["Date_PST"] = df["Date"].apply(convert_to_pst)
        df["Date_PST_Str"] = df["Date_PST"].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

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
        now = datetime.now().strftime("%Y-%m-%d")
        # Use fetch_stock_data from simple_strategy.py
        df = fetch_stock_data(ticker, "2025-03-13", now, "hour")
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

    EmailNotifier.send_email(subject, message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password,
                             attachments)